from typing import Optional, Tuple, Union

import torch

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    get_attention_cp_rank,
    get_attention_cp_size,
)
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnKernelBackend,
    get_linear_attn_decode_backend,
    get_linear_attn_prefill_backend,
)
from sglang.srt.layers.attention.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.utils import is_cpu, is_cuda, is_npu
from sglang.srt.utils.common import rank0_log

if not is_cpu():
    from sglang.srt.layers.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )

if is_cuda():
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
    ):
        triton_kernel = TritonGDNKernel()

        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            self.decode_kernel = CuteDSLGDNKernel()
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            raise ValueError(
                "CuTe DSL backend only supports decode, not prefill. "
                "Use --linear-attn-prefill-backend triton instead."
            )
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel: use FlashInfer if either decode or prefill selected it
        if decode_backend.is_flashinfer() or prefill_backend.is_flashinfer():
            self.verify_kernel = flashinfer_kernel
        else:
            self.verify_kernel = triton_kernel

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    def __init__(self, model_runner: ModelRunner):
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert (
                self.conv_states_shape[-1] < FLA_CHUNK_SIZE
            ), f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"

        decode_backend = get_linear_attn_decode_backend()
        prefill_backend = get_linear_attn_prefill_backend()
        self.kernel_dispatcher = GDNKernelDispatcher(decode_backend, prefill_backend)
        self.verify_intermediate_state_indices = torch.arange(
            self.req_to_token_pool.size, dtype=torch.int32, device=model_runner.device
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        # === V15 (B3) — CP-aware GDN forward ===
        # Read custom _gdn_cp_metadata (NOT attn_cp_metadata, which would activate
        # other downstream CP code paths in the MoE communicator that we can't
        # satisfy without splitting hidden_states).
        # Fall through to non-CP path when seq_len doesn't divide evenly (warmup,
        # short prompts) — CP code path requires seq_len % (2*cp_size) == 0 for
        # the equal-segment PoC algorithm.
        cp_meta = getattr(forward_batch, "_gdn_cp_metadata", None)
        cp_size = getattr(forward_batch, "_gdn_cp_size", 1)
        if (
            cp_meta is not None
            and cp_size > 1
            and not forward_batch.forward_mode.is_target_verify()
            and seq_len % (2 * cp_size) == 0
            and seq_len >= 2 * cp_size * 64  # need at least one chunk per segment (CHUNK_SIZE=64)
        ):
            cp_rank = getattr(forward_batch, "_gdn_cp_rank", get_attention_cp_rank())
            # one-shot print so we can confirm CP path fired in production
            if not getattr(self.__class__, "_v15_cp_path_printed", False):
                print(
                    f"[V15 CP FIRE] layer={layer.layer_id} seq_len={seq_len} "
                    f"cp_size={cp_size} cp_rank={cp_rank} "
                    f"split_list={cp_meta.split_list}",
                    flush=True,
                )
                self.__class__._v15_cp_path_printed = True
            return self._forward_extend_cp(
                layer, forward_batch, mixed_qkv, a, b, cp_meta, cp_size, cp_rank
            )

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = forward_batch.extend_prefix_lens > 0

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                mixed_qkv_to_track = mixed_qkv[
                    :, forward_metadata.track_conv_indices
                ].transpose(0, 1)
                conv_states[forward_metadata.conv_states_mask_indices] = (
                    mixed_qkv_to_track
                )

            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                activation=layer.activation,
                conv_states=conv_states,
                has_initial_state=has_initial_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
            ).transpose(0, 1)[:seq_len]

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )

        actual_seq_len = query.shape[0]
        query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            core_attn_out = self.kernel_dispatcher.target_verify(
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                q=query,
                k=key,
                v=value,
                a=a,
                b=b,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
                intermediate_states_buffer=intermediate_state_cache,
                intermediate_state_indices=intermediate_state_indices,
                cache_steps=forward_batch.spec_info.draft_token_num,
                retrieve_parent_token=retrieve_parent_token,
            )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                query_start_loc=query_start_loc,
            )

            if (is_npu() or is_cpu()) and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if h is not None:
                self._track_mamba_state_extend(
                    forward_batch, h, ssm_states, forward_metadata
                )

        # === V20: dump ssm_state after non-CP prefill (gated by DUMP_SSM=1 env) ===
        # Captures the kernel's in-place writeback for cp=1 baseline + noise floor.
        # One-shot per (layer_id, world_rank) per server boot to avoid filesystem spam.
        import os as _os
        if _os.getenv("DUMP_SSM") == "1" and forward_batch.forward_mode.is_extend():
            try:
                import torch.distributed as _dist
                _world_rank = _dist.get_rank() if _dist.is_initialized() else 0
                _seq_len = mixed_qkv.shape[0] if isinstance(mixed_qkv, torch.Tensor) else seq_len
                _path = (
                    f"/tmp/v20_ssm_cp1_layer{layer.layer_id}_rank{_world_rank}"
                    f"_T{_seq_len}.pt"
                )
                if not _os.path.exists(_path):
                    _slot = int(cache_indices[0].item())
                    torch.save(
                        ssm_states[_slot].detach().float().cpu(),
                        _path,
                    )
                    if layer.layer_id == 0:
                        print(
                            f"[V20 SSM DUMP cp=1] layer={layer.layer_id} slot={_slot} "
                            f"path={_path}",
                            flush=True,
                        )
            except Exception as _e:
                print(f"[V20 SSM DUMP cp=1 ERROR] {_e}", flush=True)
        return core_attn_out

    # ===========================================================================
    # V15 (B3): CP-aware GDN forward.
    #
    # Contract:
    #   - mixed_qkv arrives FULL-T (model file did NOT split). This avoids the
    #     unbounded OOB chain on aiter_backend (which is not CP-aware).
    #   - cp_meta is set by Qwen3_5ForCausalLM.forward when CP is active.
    #   - We do real CP work in Pass 1 + (b,M) all-gather + chain reduce + Pass 2,
    #     mirroring poc-a/cp_gdn_zigzag.py:cp_zigzag_two_pass.
    #   - Returns FULL-T core_attn_out so downstream layers (full-attn / MLP)
    #     keep operating on full sequence.
    #
    # Caveats (B3 v1, prefill-correctness only):
    #   - conv1d runs on the FULL sequence on every rank (redundant). No conv1d
    #     boundary error (ranks compute the SAME conv1d).
    #   - SSM state writeback is SKIPPED. Decode after this prefill is broken;
    #     prefill-correctness test (V5/V6 hidden-state dump methodology) is
    #     unaffected.
    #   - Prefix cache (extend_prefix_lens > 0) not supported. PoC scope.
    # ===========================================================================
    def _forward_extend_cp(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        cp_meta,
        cp_size: int,
        cp_rank: int,
    ):
        SEG_PER_RANK = 2  # zigzag invariant: each rank holds {head, tail} pair
        seq_len = mixed_qkv.shape[0]
        device = mixed_qkv.device

        forward_metadata = self.forward_metadata
        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        has_initial_states = forward_batch.extend_prefix_lens > 0

        # === V17 DIAG: check if ssm_states[cache_indices[0]] is non-zero for first prefill
        # If non-zero → cp=1's "initial_state=ssm_states" reads stale data ≠ my zeros.
        if not getattr(self.__class__, "_v17_ssm_diag_printed", False):
            try:
                slot = int(cache_indices[0].item())
                ssm_slice = ssm_states[slot]
                print(
                    f"[V17 SSM DIAG] layer={layer.layer_id} slot={slot} "
                    f"ssm_states[slot] abs max={ssm_slice.abs().max().item():.3e} "
                    f"mean abs={ssm_slice.abs().mean().item():.3e} "
                    f"shape={tuple(ssm_slice.shape)}",
                    flush=True,
                )
                if layer.layer_id >= 5:
                    self.__class__._v17_ssm_diag_printed = True
            except Exception as _e:
                print(f"[V17 SSM DIAG ERROR] {_e}", flush=True)

        # ----- conv1d on FULL mixed_qkv (every rank computes same result) -----
        mixed_qkv_t = mixed_qkv.transpose(0, 1)
        if forward_metadata.has_mamba_track_mask:
            mixed_qkv_to_track = mixed_qkv_t[
                :, forward_metadata.track_conv_indices
            ].transpose(0, 1)
            conv_states[forward_metadata.conv_states_mask_indices] = mixed_qkv_to_track
        mixed_qkv = causal_conv1d_fn(
            mixed_qkv_t,
            layer.conv_weights,
            layer.bias,
            activation=layer.activation,
            conv_states=conv_states,
            has_initial_state=has_initial_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            seq_lens_cpu=forward_batch.extend_seq_lens_cpu,
        ).transpose(0, 1)[:seq_len]

        # ----- Split q/k/v + gating (FULL-T) -----
        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        Hq = layer.num_q_heads
        Hk = layer.num_k_heads
        Hv = layer.num_v_heads
        Kdim = layer.head_k_dim
        Vdim = layer.head_v_dim
        query = query.view(1, seq_len, Hq, layer.head_q_dim)
        key = key.view(1, seq_len, Hk, Kdim)
        value = value.view(1, seq_len, Hv, Vdim)
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        # ----- CP segment plan (zigzag) -----
        num_segs = SEG_PER_RANK * cp_size
        assert seq_len % num_segs == 0, (
            f"V15 B3 PoC requires seq_len divisible by 2*cp_size; "
            f"got seq_len={seq_len} num_segs={num_segs}"
        )
        seg_len = seq_len // num_segs
        cu_seg = torch.tensor([0, seg_len], dtype=torch.int32, device=device)
        state_idx = torch.tensor([0], dtype=torch.int64, device=device)
        owned = [cp_rank, num_segs - 1 - cp_rank]  # zigzag rule

        def _slice_seg(t, seg_idx):
            s = seg_idx * seg_len
            return t.narrow(1, s, seg_len).contiguous()

        # ----- Pass 1: only owned segments, init=0, capture (b, M) -----
        b_pair = torch.zeros(
            SEG_PER_RANK, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        M_pair = torch.zeros(
            SEG_PER_RANK, Hv, Kdim, Kdim, dtype=torch.float32, device=device
        )
        for slot, seg_idx in enumerate(owned):
            q_seg = _slice_seg(query, seg_idx)
            k_seg = _slice_seg(key, seg_idx)
            v_seg = _slice_seg(value, seg_idx)
            g_seg = _slice_seg(g, seg_idx)
            beta_seg = _slice_seg(beta, seg_idx)
            state = torch.zeros(
                1, Hv, Vdim, Kdim, dtype=torch.float32, device=device
            )
            M_buf = torch.zeros(
                1, Hv, Kdim, Kdim, dtype=torch.float32, device=device
            )
            chunk_gated_delta_rule(
                q=q_seg,
                k=k_seg,
                v=v_seg,
                g=g_seg,
                beta=beta_seg,
                initial_state=state,
                initial_state_indices=state_idx,
                cu_seqlens=cu_seg,
                use_qk_l2norm_in_kernel=True,
                M_out=M_buf,
            )
            b_pair[slot] = state[0]
            M_pair[slot] = M_buf[0]

        # ----- All-gather (b, M) over cp_group (use sglang's GroupCoordinator wrapper) -----
        b_gathered = torch.empty(
            num_segs, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        M_gathered = torch.empty(
            num_segs, Hv, Kdim, Kdim, dtype=torch.float32, device=device
        )
        attn_cp_all_gather_into_tensor(b_gathered, b_pair.contiguous())
        attn_cp_all_gather_into_tensor(M_gathered, M_pair.contiguous())

        # ----- Reorder gather → causal index (PoC: causal_to_gather_perm) -----
        perm = []
        for i in range(num_segs):
            owner = min(i, num_segs - 1 - i)
            slot = 0 if i == owner else 1
            perm.append(owner * SEG_PER_RANK + slot)
        perm_t = torch.tensor(perm, dtype=torch.long, device=device)
        b_causal = b_gathered[perm_t]
        M_causal = M_gathered[perm_t]

        # ----- fp32 chain reduce (LOCAL): S_init[i+1] = S_init[i] @ M[i] + b[i] -----
        S_init = [None] * num_segs
        S_init[0] = torch.zeros(
            1, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        for i in range(1, num_segs):
            # PoC: torch.einsum("nhvk,hkj->nhvj", S, M) — affine fp32 update.
            S_init[i] = (
                torch.einsum("nhvk,hkj->nhvj", S_init[i - 1], M_causal[i - 1])
                + b_causal[i - 1].unsqueeze(0)
            )

        # ----- Pass 2: only owned segments with correct S_init -----
        # V19.1: capture in-place writeback of S_init_seg after Pass 2 of seg_(N-1)
        #         (the rank that owns the last causal segment = rank 0 by zigzag rule).
        #         This is the kernel's true final state for that segment, ensuring
        #         bit-exact match with cp=1's writeback (which uses the same kernel).
        out_pair = torch.empty(
            SEG_PER_RANK, seg_len, Hv, Vdim, dtype=value.dtype, device=device
        )
        captured_final_state = None  # only rank owning seg_(N-1) sets this
        last_seg_idx = num_segs - 1
        for slot, seg_idx in enumerate(owned):
            q_seg = _slice_seg(query, seg_idx)
            k_seg = _slice_seg(key, seg_idx)
            v_seg = _slice_seg(value, seg_idx)
            g_seg = _slice_seg(g, seg_idx)
            beta_seg = _slice_seg(beta, seg_idx)
            S_init_seg = S_init[seg_idx].clone().contiguous()
            o, _, _ = chunk_gated_delta_rule(
                q=q_seg,
                k=k_seg,
                v=v_seg,
                g=g_seg,
                beta=beta_seg,
                initial_state=S_init_seg,
                initial_state_indices=state_idx,
                cu_seqlens=cu_seg,
                use_qk_l2norm_in_kernel=True,
                M_out=None,
            )
            out_pair[slot] = o[0]  # [seg_len, Hv, Vdim]
            # On the rank owning seg_(N-1), capture the kernel's in-place state writeback.
            if seg_idx == last_seg_idx:
                captured_final_state = S_init_seg[0].clone().contiguous()

        # ----- All-gather Pass-2 outputs and reorder to causal full-T -----
        out_gathered = torch.empty(
            num_segs, seg_len, Hv, Vdim, dtype=value.dtype, device=device
        )
        attn_cp_all_gather_into_tensor(out_gathered, out_pair.contiguous())
        out_causal = out_gathered[perm_t]  # [num_segs, seg_len, Hv, Vdim]
        core_attn_out = out_causal.reshape(1, num_segs * seg_len, Hv, Vdim)

        # ===========================================================================
        # V19.1: ssm_state writeback for decode-after-CP-prefill support.
        #
        # Mirrors sglang's full-attn pattern (cp_allgather_and_save_kv_cache):
        # every CP rank ends prefill with the request's full state in its local
        # pool. Decode then runs through the standard non-CP path (scheduler
        # clears attn_cp_metadata on extend→decode transition; GDN forward_decode
        # reads ssm_states[cache_indices[0]] symmetrically per CP rank).
        #
        # The rank owning seg_(N-1) (= rank 0 by zigzag rule, since rank 0 owns
        # {0, 2*cp_size-1}) ran Pass 2 on that segment and captured the kernel's
        # in-place final-state writeback into `captured_final_state`. We broadcast
        # this to all CP ranks and write to local ssm_states[slot]. Using the
        # kernel's actual writeback (rather than reconstructing via einsum
        # extension of the chain reduce) ensures bit-exact equivalence with
        # cp=1's automatic in-place writeback in the non-CP path.
        #
        # Conv_state writeback: already handled by causal_conv1d_fn above (every
        # CP rank runs full-T conv1d on identical inputs → identical writes,
        # idempotent across ranks).
        # ===========================================================================
        if captured_final_state is None:
            # ranks not owning seg_(N-1) allocate buffer to receive broadcast
            captured_final_state = torch.empty(
                Hv, Vdim, Kdim, dtype=torch.float32, device=device
            )
        # Broadcast from the rank that captured (= rank 0 in the cp group).
        # attn_cp_all_gather_into_tensor with output sized cp_size × input gives
        # us all ranks' values; rank 0's slot is at offset 0. Each rank then
        # uses gathered[0] as the authoritative S_final.
        gathered_states = torch.empty(
            cp_size, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        attn_cp_all_gather_into_tensor(gathered_states, captured_final_state)
        S_final = gathered_states[0]  # rank 0 owns seg_(N-1) → its captured state is the truth
        ssm_states[cache_indices[0]] = S_final.to(ssm_states.dtype)

        # === V20: dump ssm_state after V19.1 CP writeback (gated by DUMP_SSM=1) ===
        import os as _os
        if _os.getenv("DUMP_SSM") == "1":
            try:
                import torch.distributed as _dist
                _world_rank = _dist.get_rank() if _dist.is_initialized() else 0
                _path = (
                    f"/tmp/v20_ssm_cp{cp_size}_layer{layer.layer_id}"
                    f"_rank{_world_rank}_T{seq_len}.pt"
                )
                if not _os.path.exists(_path):
                    _slot = int(cache_indices[0].item())
                    torch.save(
                        ssm_states[_slot].detach().float().cpu(),
                        _path,
                    )
                    if layer.layer_id == 0:
                        print(
                            f"[V20 SSM DUMP cp={cp_size}] layer={layer.layer_id} "
                            f"slot={_slot} path={_path}",
                            flush=True,
                        )
            except Exception as _e:
                print(f"[V20 SSM DUMP cp ERROR] {_e}", flush=True)
        return core_attn_out
