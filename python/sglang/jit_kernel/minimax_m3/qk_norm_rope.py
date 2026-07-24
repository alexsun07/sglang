# SPDX-License-Identifier: Apache-2.0
"""Fused MiniMax-M3 per-head Gemma Q/K RMSNorm + partial RoPE for ROCm."""

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _qk_gemma_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    q_out_ptr,
    k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_m,
    q_stride_d,
    k_stride_m,
    k_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    is_neox_style: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_program = tl.program_id(1)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < head_dim
    half_rotary: tl.constexpr = rotary_dim // 2

    is_q = head_program < q_heads
    head_id = tl.where(is_q, head_program, head_program - q_heads)
    in_ptr = tl.where(is_q, q_ptr, k_ptr)
    out_ptr = tl.where(is_q, q_out_ptr, k_out_ptr)
    weight_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)
    stride_m = tl.where(is_q, q_stride_m, k_stride_m)
    stride_d = tl.where(is_q, q_stride_d, k_stride_d)
    n_heads = tl.where(is_q, q_heads, k_heads)

    base_in = in_ptr + token_id * stride_m + head_id * head_dim * stride_d
    x = tl.load(base_in + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * (1.0 + w)
    # Match the unfused path: GemmaRMSNorm writes bf16/fp16, then RoPE reads
    # that rounded value in the following kernel.
    normed = normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    rotary_mask = cols < rotary_dim
    if is_neox_style:
        partner_cols = tl.where(
            cols < half_rotary, cols + half_rotary, cols - half_rotary
        )
        cos_cols = tl.where(cols < half_rotary, cols, cols - half_rotary)
        sign = tl.where(cols < half_rotary, -1.0, 1.0)
    else:
        partner_cols = tl.where((cols % 2) == 0, cols + 1, cols - 1)
        cos_cols = cols // 2
        sign = tl.where((cols % 2) == 0, -1.0, 1.0)

    partner_mask = partner_cols < head_dim
    x_partner = tl.load(
        base_in + partner_cols * stride_d,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    w_partner = tl.load(
        weight_ptr + partner_cols,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    partner_normed = x_partner * rstd * (1.0 + w_partner)
    partner_normed = partner_normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    pos = tl.load(positions_ptr + token_id).to(tl.int64)
    cos_sin_base = cos_sin_cache_ptr + pos * rotary_dim
    cos = tl.load(cos_sin_base + cos_cols, mask=rotary_mask, other=1.0).to(tl.float32)
    sin = tl.load(
        cos_sin_base + half_rotary + cos_cols,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = normed * cos + sign * partner_normed * sin
    out = tl.where(rotary_mask, rotated, normed)

    base_out = out_ptr + token_id * n_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out.to(out_ptr.dtype.element_ty), mask=mask)


def qk_gemma_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return normalized+rotated Q/K tensors with the same shapes as ``q``/``k``."""
    assert q.dim() == 2 and k.dim() == 2
    assert positions.dim() == 1
    assert q.shape[0] == k.shape[0] == positions.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    block_hd = triton.next_power_of_2(head_dim)

    _qk_gemma_rmsnorm_rope_kernel[(q.shape[0], q_heads + k_heads)](
        q,
        k,
        q_out,
        k_out,
        q_weight,
        k_weight,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        q_heads,
        k_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    return q_out, k_out


@triton.jit
def _sparse_qk_index_gemma_rmsnorm_rope_kernel(
    q_ptr,
    k_ptr,
    idx_q_ptr,
    idx_k_ptr,
    q_out_ptr,
    k_out_ptr,
    idx_q_out_ptr,
    idx_k_out_ptr,
    q_weight_ptr,
    k_weight_ptr,
    idx_q_weight_ptr,
    idx_k_weight_ptr,
    positions_ptr,
    cos_sin_cache_ptr,
    q_stride_m,
    q_stride_d,
    k_stride_m,
    k_stride_d,
    idx_q_stride_m,
    idx_q_stride_d,
    idx_k_stride_m,
    idx_k_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    is_neox_style: tl.constexpr,
    BLOCK_HD: tl.constexpr,
):
    token_id = tl.program_id(0)
    head_program = tl.program_id(1)
    cols = tl.arange(0, BLOCK_HD)
    mask = cols < head_dim
    half_rotary: tl.constexpr = rotary_dim // 2

    main_heads: tl.constexpr = q_heads + k_heads
    idx_k_program: tl.constexpr = q_heads + k_heads + idx_q_heads

    is_q = head_program < q_heads
    is_k = (head_program >= q_heads) & (head_program < main_heads)
    is_idx_q = (head_program >= main_heads) & (head_program < idx_k_program)

    head_id = tl.where(
        is_q,
        head_program,
        tl.where(
            is_k,
            head_program - q_heads,
            tl.where(is_idx_q, head_program - main_heads, 0),
        ),
    )

    in_ptr = tl.where(
        is_q,
        q_ptr,
        tl.where(is_k, k_ptr, tl.where(is_idx_q, idx_q_ptr, idx_k_ptr)),
    )
    out_ptr = tl.where(
        is_q,
        q_out_ptr,
        tl.where(is_k, k_out_ptr, tl.where(is_idx_q, idx_q_out_ptr, idx_k_out_ptr)),
    )
    weight_ptr = tl.where(
        is_q,
        q_weight_ptr,
        tl.where(
            is_k,
            k_weight_ptr,
            tl.where(is_idx_q, idx_q_weight_ptr, idx_k_weight_ptr),
        ),
    )
    stride_m = tl.where(
        is_q,
        q_stride_m,
        tl.where(
            is_k,
            k_stride_m,
            tl.where(is_idx_q, idx_q_stride_m, idx_k_stride_m),
        ),
    )
    stride_d = tl.where(
        is_q,
        q_stride_d,
        tl.where(
            is_k,
            k_stride_d,
            tl.where(is_idx_q, idx_q_stride_d, idx_k_stride_d),
        ),
    )
    out_heads = tl.where(
        is_q,
        q_heads,
        tl.where(is_k, k_heads, tl.where(is_idx_q, idx_q_heads, 1)),
    )

    base_in = in_ptr + token_id * stride_m + head_id * head_dim * stride_d
    x = tl.load(base_in + cols * stride_d, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / head_dim
    rstd = tl.rsqrt(var + eps)
    normed = x * rstd * (1.0 + w)
    normed = normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    rotary_mask = cols < rotary_dim
    if is_neox_style:
        partner_cols = tl.where(
            cols < half_rotary, cols + half_rotary, cols - half_rotary
        )
        cos_cols = tl.where(cols < half_rotary, cols, cols - half_rotary)
        sign = tl.where(cols < half_rotary, -1.0, 1.0)
    else:
        partner_cols = tl.where((cols % 2) == 0, cols + 1, cols - 1)
        cos_cols = cols // 2
        sign = tl.where((cols % 2) == 0, -1.0, 1.0)

    partner_mask = partner_cols < head_dim
    x_partner = tl.load(
        base_in + partner_cols * stride_d,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    w_partner = tl.load(
        weight_ptr + partner_cols,
        mask=partner_mask,
        other=0.0,
    ).to(tl.float32)
    partner_normed = x_partner * rstd * (1.0 + w_partner)
    partner_normed = partner_normed.to(q_out_ptr.dtype.element_ty).to(tl.float32)

    pos = tl.load(positions_ptr + token_id).to(tl.int64)
    cos_sin_base = cos_sin_cache_ptr + pos * rotary_dim
    cos = tl.load(cos_sin_base + cos_cols, mask=rotary_mask, other=1.0).to(tl.float32)
    sin = tl.load(
        cos_sin_base + half_rotary + cos_cols,
        mask=rotary_mask,
        other=0.0,
    ).to(tl.float32)
    rotated = normed * cos + sign * partner_normed * sin
    out = tl.where(rotary_mask, rotated, normed)

    base_out = out_ptr + token_id * out_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out.to(q_out_ptr.dtype.element_ty), mask=mask)


def sparse_qk_index_gemma_rmsnorm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    idx_q_weight: torch.Tensor,
    idx_k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
    is_neox_style: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse main and sparse-index Gemma Q/K RMSNorm + RoPE into one launch."""
    assert q.dim() == k.dim() == idx_q.dim() == idx_k.dim() == 2
    assert positions.dim() == 1
    assert q.shape[0] == k.shape[0] == idx_q.shape[0] == idx_k.shape[0]
    assert q.shape[0] == positions.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert idx_q.shape[1] % head_dim == 0
    assert idx_k.shape[1] == head_dim
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_q_heads = idx_q.shape[1] // head_dim
    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    idx_q_out = torch.empty(idx_q.shape, dtype=idx_q.dtype, device=idx_q.device)
    idx_k_out = torch.empty(idx_k.shape, dtype=idx_k.dtype, device=idx_k.device)
    block_hd = triton.next_power_of_2(head_dim)

    _sparse_qk_index_gemma_rmsnorm_rope_kernel[
        (q.shape[0], q_heads + k_heads + idx_q_heads + 1)
    ](
        q,
        k,
        idx_q,
        idx_k,
        q_out,
        k_out,
        idx_q_out,
        idx_k_out,
        q_weight,
        k_weight,
        idx_q_weight,
        idx_k_weight,
        positions,
        cos_sin_cache,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        idx_q.stride(0),
        idx_q.stride(1),
        idx_k.stride(0),
        idx_k.stride(1),
        q_heads,
        k_heads,
        idx_q_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        num_warps=4,
    )
    return q_out, k_out, idx_q_out, idx_k_out


@triton.jit
def _fused_qkv_norm_rope_cache_kernel(
    # Separate Q, K, V input pointers (may be strided views from torch.split)
    q_ptr,
    q_stride_t,
    k_ptr,
    k_stride_t,
    v_ptr,
    v_stride_t,
    # Contiguous output pointers
    q_out_ptr,
    k_out_ptr,
    # Norm weights
    qw_ptr,
    kw_ptr,
    # RoPE cos/sin caches: [max_pos, rotary_dim//2]
    cos_cache_ptr,
    sin_cache_ptr,
    cos_sin_stride_pos,
    # Positions
    pos_ptr,
    # KV cache pointers (SHUFFLE layout)
    k_cache_ptr,
    v_cache_ptr,
    # KV cache strides (5D SHUFFLE: [num_blocks, num_kv_heads, D//X, block_size, X])
    kc_stride_block,
    kc_stride_head,
    kc_stride_dx,
    kc_stride_slot,
    kc_stride_x,
    # V cache strides (5D SHUFFLE: [num_blocks, num_kv_heads, block_size//X, head_dim, X])
    vc_stride_block,
    vc_stride_head,
    vc_stride_sc,
    vc_stride_d,
    vc_stride_x,
    # KV scale pointers (per-token scales: [num_blocks, num_kv_heads, block_size])
    k_scale_ptr,
    v_scale_ptr,
    ks_stride_block,
    ks_stride_head,
    vs_stride_block,
    vs_stride_head,
    # Slot mapping
    slot_mapping_ptr,
    # M-RoPE parameters
    pos_stride_row,  # stride between position rows (for 2D positions)
    # Dimensions
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    eps: tl.constexpr,
    # Cache layout
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    ROTARY_DIM_HALF: tl.constexpr,
    IS_FP8: tl.constexpr,
    # M-RoPE section boundaries (cumulative)
    MROPE_S0: tl.constexpr = 0,
    MROPE_S1: tl.constexpr = 0,
    IS_MROPE: tl.constexpr = False,
):
    # Grid: (num_tokens * (num_heads + num_kv_heads),)
    pid = tl.program_id(0)
    total_heads = num_heads + num_kv_heads
    token_id = pid // total_heads
    head_id = pid % total_heads

    d_offs = tl.arange(0, BLOCK_D)

    if head_id < num_heads:
        # ============ Q head processing: GemmaRMSNorm + RoPE ============
        h = head_id
        # Input offset uses strided layout
        q_in_offset = token_id * q_stride_t + h * BLOCK_D

        # Load q from strided input
        q = tl.load(q_ptr + q_in_offset + d_offs).to(tl.float32)

        # GemmaRMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + weight)
        variance = tl.sum(q * q, axis=0) / BLOCK_D
        q_normed = q * tl.math.rsqrt(variance + eps)
        qw = tl.load(qw_ptr + d_offs).to(tl.float32)
        q_normed = q_normed * (1.0 + qw)

        # RoPE (neox-style, partial rotary)
        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(
                d_offs < ROTARY_DIM,
                d_offs - ROTARY_DIM_HALF,
                tl.zeros_like(d_offs),
            ),
        )

        if IS_MROPE:
            # M-RoPE: per-dim position selection based on mrope_section
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(
            cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0
        ).to(tl.float32)
        sin_vals = tl.load(
            sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0
        ).to(tl.float32)

        # Neox RoPE rotation in registers (no global memory scratch)
        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        q_gathered_raw = tl.load(q_ptr + q_in_offset + gather_idx).to(tl.float32)
        qw_gathered = tl.load(qw_ptr + gather_idx).to(tl.float32)
        q_gathered_normed = (
            q_gathered_raw * tl.math.rsqrt(variance + eps) * (1.0 + qw_gathered)
        )
        q_rot = tl.where(first_half_mask, -q_gathered_normed, q_gathered_normed)
        q_rot = tl.where(rot_mask, q_rot, 0.0)

        q_roped = q_normed * cos_vals + q_rot * sin_vals

        # Write to contiguous q_out: [T, num_heads * BLOCK_D]
        q_out_offset = token_id * (num_heads * BLOCK_D) + h * BLOCK_D
        tl.store(
            q_out_ptr + q_out_offset + d_offs,
            q_roped.to(q_out_ptr.dtype.element_ty),
        )
    else:
        # ============ KV head processing ============
        kv_h = head_id - num_heads

        # --- K: GemmaRMSNorm + RoPE → contiguous k_out + cache write ---
        # Input offset uses strided layout
        k_in_offset = token_id * k_stride_t + kv_h * BLOCK_D
        k = tl.load(k_ptr + k_in_offset + d_offs).to(tl.float32)

        # GemmaRMSNorm on k
        k_variance = tl.sum(k * k, axis=0) / BLOCK_D
        k_normed = k * tl.math.rsqrt(k_variance + eps)
        kw = tl.load(kw_ptr + d_offs).to(tl.float32)
        k_normed = k_normed * (1.0 + kw)

        # RoPE on k (neox-style, partial rotary)
        rot_mask = d_offs < ROTARY_DIM
        first_half_mask = d_offs < ROTARY_DIM_HALF
        d_cos_idx = tl.where(
            first_half_mask,
            d_offs,
            tl.where(
                d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, tl.zeros_like(d_offs)
            ),
        )

        if IS_MROPE:
            pos_t = tl.load(pos_ptr + 0 * pos_stride_row + token_id)
            pos_h = tl.load(pos_ptr + 1 * pos_stride_row + token_id)
            pos_w = tl.load(pos_ptr + 2 * pos_stride_row + token_id)
            pos_per_dim = tl.where(
                d_cos_idx < MROPE_S0,
                pos_t,
                tl.where(d_cos_idx < MROPE_S1, pos_h, pos_w),
            )
            cos_base = pos_per_dim * cos_sin_stride_pos
        else:
            pos = tl.load(pos_ptr + token_id)
            cos_base = pos * cos_sin_stride_pos

        cos_vals = tl.load(
            cos_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=1.0
        ).to(tl.float32)
        sin_vals = tl.load(
            sin_cache_ptr + cos_base + d_cos_idx, mask=rot_mask, other=0.0
        ).to(tl.float32)

        # Neox RoPE rotation in registers
        gather_idx = tl.where(
            first_half_mask,
            d_offs + ROTARY_DIM_HALF,
            tl.where(d_offs < ROTARY_DIM, d_offs - ROTARY_DIM_HALF, d_offs),
        )
        k_gathered_raw = tl.load(k_ptr + k_in_offset + gather_idx).to(tl.float32)
        kw_gathered = tl.load(kw_ptr + gather_idx).to(tl.float32)
        k_gathered_normed = (
            k_gathered_raw * tl.math.rsqrt(k_variance + eps) * (1.0 + kw_gathered)
        )
        k_rot = tl.where(first_half_mask, -k_gathered_normed, k_gathered_normed)
        k_rot = tl.where(rot_mask, k_rot, 0.0)

        k_roped = k_normed * cos_vals + k_rot * sin_vals

        # Write to contiguous k_out: [T, num_kv_heads * BLOCK_D]
        k_out_offset = token_id * (num_kv_heads * BLOCK_D) + kv_h * BLOCK_D
        tl.store(
            k_out_ptr + k_out_offset + d_offs,
            k_roped.to(k_out_ptr.dtype.element_ty),
        )

        # --- V: load for cache write (no norm, no RoPE) ---
        v_in_offset = token_id * v_stride_t + kv_h * BLOCK_D
        v = tl.load(v_ptr + v_in_offset + d_offs)

        # === KV cache write (SHUFFLE layout) ===
        slot = tl.load(slot_mapping_ptr + token_id).to(tl.int64)
        if slot >= 0:
            block_idx = slot // BLOCK_SIZE
            slot_in_block = slot % BLOCK_SIZE

            if IS_FP8:
                # FP8 per-token quantization for k
                k_abs_max = tl.max(tl.abs(k_roped), axis=0)
                k_scale = k_abs_max / 240.0
                k_scale = tl.where(k_scale == 0.0, 1.0, k_scale)
                k_quant = (k_roped / k_scale).to(k_cache_ptr.dtype.element_ty)

                tl.store(
                    k_scale_ptr
                    + block_idx * ks_stride_block
                    + kv_h * ks_stride_head
                    + slot_in_block,
                    k_scale,
                )
            else:
                k_quant = k_roped.to(k_cache_ptr.dtype.element_ty)

            # K cache SHUFFLE write: [num_blocks, num_kv_heads, head_dim//X, block_size, X]
            k_quant_2d = tl.reshape(k_quant, (BLOCK_D // X_SIZE, X_SIZE))
            dx_offs = tl.arange(0, BLOCK_D // X_SIZE).to(tl.int64)
            x_offs = tl.arange(0, X_SIZE).to(tl.int64)
            k_cache_ptrs = (
                k_cache_ptr
                + block_idx * kc_stride_block
                + kv_h * kc_stride_head
                + dx_offs[:, None] * kc_stride_dx
                + slot_in_block * kc_stride_slot
                + x_offs[None, :] * kc_stride_x
            )
            tl.store(k_cache_ptrs, k_quant_2d)

            if IS_FP8:
                # FP8 per-token quantization for v
                v_f32 = v.to(tl.float32)
                v_abs_max = tl.max(tl.abs(v_f32), axis=0)
                v_scale = v_abs_max / 240.0
                v_scale = tl.where(v_scale == 0.0, 1.0, v_scale)
                v_quant = (v_f32 / v_scale).to(v_cache_ptr.dtype.element_ty)

                tl.store(
                    v_scale_ptr
                    + block_idx * vs_stride_block
                    + kv_h * vs_stride_head
                    + slot_in_block,
                    v_scale,
                )
            else:
                v_quant = v.to(v_cache_ptr.dtype.element_ty)

            # V cache SHUFFLE write: [num_blocks, num_kv_heads, block_size//X, head_dim, X]
            slot_chunk = slot_in_block // X_SIZE
            x_off = slot_in_block % X_SIZE
            v_cache_ptrs = (
                v_cache_ptr
                + block_idx * vc_stride_block
                + kv_h * vc_stride_head
                + slot_chunk * vc_stride_sc
                + d_offs.to(tl.int64) * vc_stride_d
                + x_off * vc_stride_x
            )
            tl.store(v_cache_ptrs, v_quant)




def atom_main_norm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    eps: float,
    head_dim: int,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ATOM's main-only fused GemmaRMSNorm + partial NeoX RoPE + 5D SHUFFLE KV
    cache write (``_fused_qkv_norm_rope_cache_kernel``), adapted for sglang.

    Used by MiniMax-M3 index-topk "skip" sparse layers: they reuse the group
    source layer's top-k, so only the MAIN q/k/v need norm+rope+cache (no index).

    sglang keeps a single merged ``cos_sin_cache`` [max_pos, rotary_dim] (first
    half cos, second half sin); ATOM's kernel takes separate cos/sin pointers
    with a shared per-position row stride, so pass zero-copy column slices with
    stride(0) == rotary_dim. bf16 KV cache only (IS_FP8=False).
    """
    assert q.dim() == k.dim() == v.dim() == 2
    assert k_cache.dim() == 5 and v_cache.dim() == 5
    assert rotary_dim % 2 == 0 and rotary_dim <= head_dim
    T = q.shape[0]
    num_heads = q.shape[1] // head_dim
    num_kv_heads = k.shape[1] // head_dim
    half = rotary_dim // 2

    # merged cache -> separate cos/sin views (zero-copy); row stride = rotary_dim
    cos_sin_cache = cos_sin_cache.to(q.dtype)
    cos_cache = cos_sin_cache[:, :half]
    sin_cache = cos_sin_cache[:, half:rotary_dim]
    cos_sin_stride_pos = cos_sin_cache.stride(0)

    block_size = k_cache.shape[3]
    x_size = k_cache.shape[4]

    q_out = q.new_empty((T, num_heads * head_dim))
    k_out = k.new_empty((T, num_kv_heads * head_dim))

    total_heads = num_heads + num_kv_heads
    grid = (T * total_heads,)
    _fused_qkv_norm_rope_cache_kernel[grid](
        q,
        q.stride(0),
        k,
        k.stride(0),
        v,
        v.stride(0),
        q_out,
        k_out,
        q_weight,
        k_weight,
        cos_cache,
        sin_cache,
        cos_sin_stride_pos,
        positions,
        k_cache,
        v_cache,
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        k_cache.stride(3),
        k_cache.stride(4),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        v_cache.stride(3),
        v_cache.stride(4),
        q,  # k_scale dummy (bf16, unused)
        q,  # v_scale dummy
        0,
        0,
        0,
        0,
        out_cache_loc,
        0,  # pos_stride_row (not M-RoPE)
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        eps=eps,
        BLOCK_SIZE=block_size,
        X_SIZE=x_size,
        BLOCK_D=head_dim,
        ROTARY_DIM=rotary_dim,
        ROTARY_DIM_HALF=half,
        IS_FP8=False,
        MROPE_S0=0,
        MROPE_S1=0,
        IS_MROPE=False,
    )
    return q_out, k_out
