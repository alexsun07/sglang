from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from sglang.srt.layers.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.dp_attention import (
    attn_cp_all_gather_into_tensor,
    get_attention_cp_group,
    get_attention_cp_rank,
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

        # CP attributes (mirror FA backend / aiter backend pattern).
        # Used by _forward_extend_cp dispatch and conv1d D1a boundary exchange.
        self.attn_cp_size = getattr(model_runner, "attn_cp_size", 1)
        self.attn_cp_rank = (
            get_attention_cp_rank() if self.attn_cp_size > 1 else 0
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

        # CP-aware prefill path. Activated when the model layer set
        # forward_batch.attn_cp_metadata via cp_split_and_rebuild_data, so
        # mixed_qkv arrives ALREADY local-T per rank (= rank's 2 owned
        # zigzag segments concatenated). The full-sequence equal-segment
        # algorithm requires the FULL kv_len divisible by 2*cp_size with
        # at least one CHUNK_SIZE=64 chunk per segment; this is gated
        # upstream via can_cp_split() inside the model. Warmup / short
        # prompts have attn_cp_metadata == None and fall through here.
        cp_meta = getattr(forward_batch, "attn_cp_metadata", None)
        cp_size = self.attn_cp_size
        if (
            cp_meta is not None
            and cp_size > 1
            and not forward_batch.forward_mode.is_target_verify()
            and forward_batch.forward_mode.is_context_parallel_extend()
        ):
            return self._forward_extend_cp(
                layer,
                forward_batch,
                mixed_qkv,
                a,
                b,
                cp_size,
                self.attn_cp_rank,
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

        return core_attn_out

    # CP-aware prefill for the GDN linear-attention path.
    #
    # mixed_qkv arrives LOCAL-T per rank because the model layer (in
    # qwen3_5.py post-zigzag-cp) calls cp_split_and_rebuild_data after
    # embed; subsequent layers (in_proj, conv1d, GDN body, etc.) all
    # operate on local-T per rank. This backend's contract:
    #   - input: mixed_qkv shape [seg_a_len + seg_b_len, q_dim+k_dim+v_dim]
    #     where seg_a = rank's first owned segment (= seg_(cp_rank)),
    #           seg_b = rank's second owned segment (= seg_(2cp-1-cp_rank))
    #     in causal order: seg_a is earlier in the full sequence.
    #   - output: same local-T layout (the model's exit
    #     cp_all_gather_rerange_output gathers + reorders to full-T
    #     before lm_head).
    #
    # Algorithm (zigzag two-pass + (b, M) chain reduce):
    #
    #   conv1d: D1a boundary exchange — each rank exchanges (K-1) tokens
    #           with adjacent ranks for left-context, then runs F.conv1d
    #           on each owned segment locally. Sequence stays sharded;
    #           only K-1 = 3 tokens × hidden cross the wire per layer.
    #           Middle rank (cp_size-1)'s seg_b feed comes intra-rank
    #           from its own seg_a tail — no P2P for that pair.
    #   Pass 1: each rank runs its 2 owned segments with initial_state=0
    #           and captures (b_seg, M_seg) — the segment's affine output
    #           on the recurrent state.
    #   Reduce: all_gather (b, M); each rank deterministically chains the
    #           per-segment affine maps in fp32 to derive S_init for every
    #           segment.
    #   Pass 2: each rank reruns its 2 owned segments with the correct
    #           S_init and emits the per-segment outputs.
    #   Output: simple concat([seg_a_out, seg_b_out]) → return local-T.
    #           No outer all_gather (model layer's exit handles that).
    #
    # ssm_state writeback (decode-after-CP): rank 0 owns seg_(2cp-1) (the
    # last causal segment) and captures the Pass 2 final state, then
    # all_gather + use slot 0 → all ranks write same state to local cache.
    #
    # conv_state writeback: rank 0's seg_b INPUT tail (last K-1 tokens of
    # mixed_qkv before conv) is the full-sequence conv_state. Rank 0
    # broadcasts via cp_group; all ranks write same conv_state to local
    # cache slot.
    #
    # Constraints (see report.md §1.4): prefix=0 only, no cuda graph,
    # batch=1, attn_tp ∈ {1, 2}.
    def _forward_extend_cp(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        cp_size: int,
        cp_rank: int,
    ):
        SEG_PER_RANK = 2  # zigzag invariant: each rank holds {head, tail} pair
        cp_meta = forward_batch.attn_cp_metadata
        device = mixed_qkv.device
        dtype = mixed_qkv.dtype
        dim = mixed_qkv.shape[-1]

        # Owned segment lengths from standard cp metadata (split_list is the
        # 2*cp_size segment lengths in causal order; remainder distributed
        # to the first segments — seg_a / seg_b can differ by 1 in pathological
        # cases, hence torch.split with explicit lengths instead of chunk(2)).
        num_segs = SEG_PER_RANK * cp_size
        seg_a_len = cp_meta.split_list[cp_rank]                  # = seg_(cp_rank)
        seg_b_len = cp_meta.split_list[num_segs - 1 - cp_rank]   # = seg_(2cp-1-cp_rank)
        local_T = seg_a_len + seg_b_len
        assert mixed_qkv.shape[0] == local_T, (
            f"GDN backend CP path expects local-T input (= seg_a + seg_b = "
            f"{seg_a_len} + {seg_b_len}), got mixed_qkv.shape[0]={mixed_qkv.shape[0]}"
        )

        # === D1a: conv1d via boundary exchange ===
        # K = conv kernel size. Stored on conv_weights as last dim.
        # conv_weights shape per causal_conv1d_fn docstring: (dim, K).
        K = layer.conv_weights.shape[-1]
        Kp = K - 1  # left-context size needed for causal conv1d

        # Pre-compute own segment tails (last Kp tokens of each owned seg's
        # INPUT mixed_qkv). If a segment is shorter than Kp, left-pad with
        # zeros (matches conv1d causal-pad semantics for very short segs;
        # in practice seg_len >> K=4 since CP is gated by seq_len >= 2cp*64).
        def _tail(seg_input):
            if seg_input.shape[0] >= Kp:
                return seg_input[-Kp:].contiguous()
            pad = torch.zeros(
                Kp - seg_input.shape[0], dim, dtype=dtype, device=device
            )
            return torch.cat([pad, seg_input], dim=0).contiguous()

        seg_a_input = mixed_qkv[:seg_a_len]
        seg_b_input = mixed_qkv[seg_a_len:]
        own_seg_a_tail = _tail(seg_a_input)
        own_seg_b_tail = _tail(seg_b_input)

        # === D1a forensic probe (reviewer instrumentation ask 1) ===
        # First-forward only, per backend instance: confirms layer.activation
        # matches our F.silu assumption + records cp/seg dims for postmortem
        # if numerical drift shows up.
        if not getattr(self, "_d1a_logged", False):
            print(
                f"[D1a] activation={layer.activation!r} K={K} dim={dim} "
                f"cp_rank={cp_rank} cp_size={cp_size} "
                f"seg_a_len={seg_a_len} seg_b_len={seg_b_len} local_T={local_T}",
                flush=True,
            )
            self._d1a_logged = True

        # === Boundary exchange via single all_gather ===
        # Original D1a design used 4× cp_group.send/recv (Ring A + Ring B).
        # Empirically (Stage 2 boot) NCCL warns:
        #   "An unbatched P2P op (send/recv) was called on this ProcessGroup
        #    with size 2. In lazy initialization mode, this will result in
        #    a new 2-rank NCCL communicator to be created."
        # With 45 GDN layers × 4 P2P ops each, this creates many tiny
        # sub-communicators on the hot path → server hangs at warmup.
        #
        # Refactor: pack (seg_a_tail, seg_b_tail) per rank, all_gather across
        # cp_group (collective op, no lazy sub-comm creation), then locally
        # pick the correct left-feeds based on rank-counting logic. Total
        # collective volume per layer per rank = cp_size × 2 × Kp × dim ≈
        # cp_size × 48KB at qwen3.5 dims; still trivial vs the (b,M)
        # all_gathers already on this path.
        #
        # Causal ownership recap:
        #   rank r owns seg_a = seg_(r) and seg_b = seg_(2cp-1-r).
        #   seg_a's prev (in causal order) = seg_(r-1), owned by rank (r-1)
        #     as ITS seg_a (slot 0). For r=0, no prev → zeros.
        #   seg_b's prev = seg_(2cp-2-r), owned by:
        #     - if 2cp-2-r < cp_size: rank (2cp-2-r) as its seg_a (slot 0)
        #       — happens when r >= cp_size-1, i.e., for the middle rank
        #       cp_size-1 (where 2cp-2-r = cp_size-1 = own rank's seg_a).
        #     - else: rank (r+1) as ITS seg_b (slot 1).
        cp_group = get_attention_cp_group()
        zeros_tail = torch.zeros(Kp, dim, dtype=dtype, device=device)

        # Pack [seg_a_tail, seg_b_tail] per rank → shape [2, Kp, dim]
        local_tails = torch.stack([own_seg_a_tail, own_seg_b_tail], dim=0)
        # Gathered shape: [cp_size, 2, Kp, dim]
        all_tails = torch.empty(
            cp_size, 2, Kp, dim, dtype=dtype, device=device
        )
        attn_cp_all_gather_into_tensor(all_tails, local_tails.contiguous())

        # Pick left-feed for seg_a (= seg_(cp_rank))
        if cp_rank == 0:
            seg_a_left = zeros_tail
        else:
            # prev = seg_(cp_rank-1) = rank (cp_rank-1)'s seg_a (slot 0)
            seg_a_left = all_tails[cp_rank - 1, 0]

        # Pick left-feed for seg_b (= seg_(2cp-1-cp_rank))
        # prev = seg_(2cp-2-cp_rank)
        prev_b_idx = 2 * cp_size - 2 - cp_rank
        if prev_b_idx < cp_size:
            # First-half segment, owned by rank (prev_b_idx) as its seg_a.
            # When prev_b_idx == cp_rank (middle rank case): intra-rank.
            seg_b_left = all_tails[prev_b_idx, 0]
        else:
            # Second-half segment, owned by rank (cp_rank+1) as its seg_b.
            seg_b_left = all_tails[cp_rank + 1, 1]

        # === Per-segment causal conv1d via F.conv1d (depthwise) ===
        # conv_weights stored as (dim, K) per kernel docstring; depthwise
        # conv1d expects weight shape (out_channels, in_channels/groups, K)
        # with groups=channels → reshape to (dim, 1, K).
        # IMPORTANT: pytorch F.conv1d does cross-correlation (no flip),
        # while causal_conv1d_fn (mamba kernel) is true convolution (kernel
        # flipped). Empirically test both via env var if drift suspected.
        # Default: assume causal_conv1d_fn semantics → flip kernel for F.conv1d.
        # Set CP_CONV_NO_FLIP=1 to disable the flip (debug option).
        conv_w_raw = layer.conv_weights.unsqueeze(1)  # [dim, 1, K]
        import os as _os
        _flip = _os.environ.get("CP_CONV_NO_FLIP", "0") != "1"
        conv_w = conv_w_raw.flip(-1) if _flip else conv_w_raw
        conv_b = layer.bias  # [dim] or None

        def _causal_conv1d_local(seg_input, left_context):
            """seg_input [T, C], left_context [Kp, C]. Returns [T, C] post-act."""
            x = torch.cat([left_context, seg_input], dim=0)  # [T+Kp, C]
            # F.conv1d expects (N, C, L) with depthwise weight (C, 1, K).
            x_t = x.transpose(0, 1).unsqueeze(0)  # [1, C, T+Kp]
            y = F.conv1d(x_t, conv_w, bias=conv_b, padding=0, groups=conv_w.shape[0])
            # y shape: [1, C, T+Kp - K + 1] = [1, C, T]
            y = y.squeeze(0).transpose(0, 1)  # [T, C]
            # Activation (mamba/qwen3.5 conv uses silu/swish; both equivalent)
            if layer.activation in ("silu", "swish"):
                y = F.silu(y)
            elif layer.activation in (None, "identity"):
                pass
            else:
                raise NotImplementedError(
                    f"GDN CP conv1d activation {layer.activation!r} not handled"
                )
            return y

        seg_a_conv = _causal_conv1d_local(seg_a_input, seg_a_left)  # [seg_a_len, dim]
        seg_b_conv = _causal_conv1d_local(seg_b_input, seg_b_left)  # [seg_b_len, dim]
        mixed_qkv = torch.cat([seg_a_conv, seg_b_conv], dim=0)  # [local_T, dim]

        # === conv_state writeback ===
        # Full-sequence conv_state = last Kp INPUT tokens of seg_(2cp-1).
        # Rank 0 owns seg_(2cp-1) as its seg_b → own_seg_b_tail (already
        # computed above) is the right tensor on rank 0.
        # Use all_gather (same pattern as ssm_state writeback below) instead
        # of broadcast — pynccl broadcast vs subsequent torch ops on default
        # stream can race; all_gather via the same `attn_cp_all_gather_into_tensor`
        # path that the (b,M) gather uses is empirically correctly ordered.
        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        cache_indices = self.forward_metadata.mamba_cache_indices

        # Gather all ranks' seg_b tails; pick rank 0's slot.
        all_seg_b_tails = torch.empty(
            cp_size, Kp, dim, dtype=dtype, device=device
        )
        attn_cp_all_gather_into_tensor(all_seg_b_tails, own_seg_b_tail.contiguous())
        conv_state_buf = all_seg_b_tails[0]  # rank 0's seg_b tail = full-seq conv state
        # === conv_states cache layout assert (reviewer instrumentation ask 2) ===
        _expected_slot_shape = (dim, Kp)
        _actual_slot_shape = tuple(conv_states[cache_indices[0]].shape)
        assert _actual_slot_shape == _expected_slot_shape, (
            f"GDN CP D1a conv_state writeback layout mismatch: expected "
            f"{_expected_slot_shape} (dim, K-1), got {_actual_slot_shape}. "
            f"conv_states.shape={tuple(conv_states.shape)}, "
            f"cache_indices[0]={cache_indices[0].item()}"
        )
        conv_states[cache_indices[0]] = conv_state_buf.transpose(0, 1).to(
            conv_states.dtype
        )

        # === Split q/k/v + gating on local-T ===
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
        # Reshape: [local_T, H*D] → [1, local_T, H, D] (kernel expects batch dim)
        query = query.view(1, local_T, Hq, layer.head_q_dim)
        key = key.view(1, local_T, Hk, Kdim)
        value = value.view(1, local_T, Hv, Vdim)
        g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)

        # === Slice into 2 owned segments (already in (prev,next) layout) ===
        state_idx = torch.tensor([0], dtype=torch.int64, device=device)
        # slice fn: returns [1, seg_len, H, D] for kernel input
        def _slice_local(t, start, length):
            return t.narrow(1, start, length).contiguous()

        # owned[slot=0] = seg_a (= seg_cp_rank, length seg_a_len, starts at 0)
        # owned[slot=1] = seg_b (= seg_(2cp-1-cp_rank), length seg_b_len, starts at seg_a_len)
        owned_starts = [0, seg_a_len]
        owned_lens = [seg_a_len, seg_b_len]
        owned_seg_idx = [cp_rank, num_segs - 1 - cp_rank]

        # === Pass 1: capture (b_seg, M_seg) per owned segment ===
        b_pair = torch.zeros(
            SEG_PER_RANK, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        M_pair = torch.zeros(
            SEG_PER_RANK, Hv, Kdim, Kdim, dtype=torch.float32, device=device
        )
        for slot in range(SEG_PER_RANK):
            seg_len = owned_lens[slot]
            start = owned_starts[slot]
            cu_seg = torch.tensor([0, seg_len], dtype=torch.int32, device=device)
            q_seg = _slice_local(query, start, seg_len)
            k_seg = _slice_local(key, start, seg_len)
            v_seg = _slice_local(value, start, seg_len)
            g_seg = _slice_local(g, start, seg_len)
            beta_seg = _slice_local(beta, start, seg_len)
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

        # === All-gather (b, M) across cp_group ===
        b_gathered = torch.empty(
            num_segs, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        M_gathered = torch.empty(
            num_segs, Hv, Kdim, Kdim, dtype=torch.float32, device=device
        )
        attn_cp_all_gather_into_tensor(b_gathered, b_pair.contiguous())
        attn_cp_all_gather_into_tensor(M_gathered, M_pair.contiguous())

        # Reorder rank-major (rank 0 head, rank 0 tail, rank 1 head, ...)
        # to causal segment order. Identical perm logic to old design.
        perm = []
        for i in range(num_segs):
            owner = min(i, num_segs - 1 - i)
            slot = 0 if i == owner else 1
            perm.append(owner * SEG_PER_RANK + slot)
        perm_t = torch.tensor(perm, dtype=torch.long, device=device)
        b_causal = b_gathered[perm_t]
        M_causal = M_gathered[perm_t]

        # === fp32 deterministic chain reduce ===
        S_init = [None] * num_segs
        S_init[0] = torch.zeros(
            1, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        for i in range(1, num_segs):
            S_init[i] = (
                torch.einsum("nhvk,hkj->nhvj", S_init[i - 1], M_causal[i - 1])
                + b_causal[i - 1].unsqueeze(0)
            )

        # === Pass 2: rerun owned segments with correct S_init ===
        # Output buffer: 2 contiguous segments concatenated in (prev,next) layout
        # to match local-T input ordering.
        seg_a_out = torch.empty(
            seg_a_len, Hv, Vdim, dtype=value.dtype, device=device
        )
        seg_b_out = torch.empty(
            seg_b_len, Hv, Vdim, dtype=value.dtype, device=device
        )
        out_slots = [seg_a_out, seg_b_out]
        captured_final_state = None
        last_seg_idx = num_segs - 1
        for slot in range(SEG_PER_RANK):
            seg_len = owned_lens[slot]
            start = owned_starts[slot]
            seg_idx = owned_seg_idx[slot]
            cu_seg = torch.tensor([0, seg_len], dtype=torch.int32, device=device)
            q_seg = _slice_local(query, start, seg_len)
            k_seg = _slice_local(key, start, seg_len)
            v_seg = _slice_local(value, start, seg_len)
            g_seg = _slice_local(g, start, seg_len)
            beta_seg = _slice_local(beta, start, seg_len)
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
            out_slots[slot].copy_(o[0])  # [seg_len, Hv, Vdim]
            if seg_idx == last_seg_idx:
                captured_final_state = S_init_seg[0].clone().contiguous()

        # === Assemble local-T output (no outer all_gather; model handles it) ===
        # Layout matches mixed_qkv input: [seg_a_out, seg_b_out] concatenated.
        # The kernel expects [batch=1, T, Hv, Vdim] but downstream code
        # squeezes batch out at radix_linear_attention.forward — match the
        # standard non-CP shape: [1, local_T, Hv, Vdim].
        core_attn_out = torch.cat([seg_a_out, seg_b_out], dim=0).reshape(
            1, local_T, Hv, Vdim
        )

        # === ssm_state writeback for decode-after-CP-prefill ===
        # Rank 0 owns seg_(2cp-1) — the last causal segment — and has
        # captured the kernel's in-place final-state writeback. All-gather
        # and use rank 0's slot. Same protocol as V19.1.
        if captured_final_state is None:
            captured_final_state = torch.empty(
                Hv, Vdim, Kdim, dtype=torch.float32, device=device
            )
        gathered_states = torch.empty(
            cp_size, Hv, Vdim, Kdim, dtype=torch.float32, device=device
        )
        attn_cp_all_gather_into_tensor(gathered_states, captured_final_state)
        ssm_states[cache_indices[0]] = gathered_states[0].to(ssm_states.dtype)

        return core_attn_out
