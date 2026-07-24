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
def _sparse_qk_index_gemma_rmsnorm_rope_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    idx_q_ptr,
    idx_k_ptr,
    q_out_ptr,
    k_out_ptr,
    idx_q_out_ptr,
    idx_k_out_ptr,
    k_cache_ptr,
    v_cache_ptr,
    idx_k_cache_ptr,
    loc_ptr,
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
    v_stride_m,
    v_stride_d,
    idx_q_stride_m,
    idx_q_stride_d,
    idx_k_stride_m,
    idx_k_stride_d,
    k_cache_stride_s,
    k_cache_stride_h,
    k_cache_stride_d,
    v_cache_stride_s,
    v_cache_stride_h,
    v_cache_stride_d,
    idx_k_cache_stride_s,
    idx_k_cache_stride_h,
    idx_k_cache_stride_d,
    q_heads: tl.constexpr,
    k_heads: tl.constexpr,
    idx_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps: tl.constexpr,
    is_neox_style: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    USE_5D: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X: tl.constexpr,
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
    out_typed = out.to(q_out_ptr.dtype.element_ty)

    base_out = out_ptr + token_id * out_heads * head_dim + head_id * head_dim
    tl.store(base_out + cols, out_typed, mask=mask)

    loc = tl.load(loc_ptr + token_id).to(tl.int64)
    v_base = v_ptr + token_id * v_stride_m + head_id * head_dim * v_stride_d
    v_val = tl.load(v_base + cols * v_stride_d, mask=mask & is_k, other=0.0)

    if USE_5D:
        # SHUFFLE 5D KV cache (contiguous), mirroring
        # sglang.srt.layers.attention.utils.reshape_and_cache_shuffle_5d:
        #   K: (num_blocks, H, head_dim // X, block_size, X)
        #   V: (num_blocks, H, block_size // X, head_dim, X)
        block_idx = loc // BLOCK_SIZE
        slot_in_page = loc % BLOCK_SIZE
        layer_stride = k_heads * head_dim * BLOCK_SIZE
        head_stride = head_dim * BLOCK_SIZE
        d_outer = cols // X
        d_inner = cols % X
        k_tgt = (
            k_cache_ptr
            + block_idx * layer_stride
            + head_id * head_stride
            + d_outer * (BLOCK_SIZE * X)
            + slot_in_page * X
            + d_inner
        )
        tl.store(k_tgt, out_typed, mask=mask & is_k)

        page_outer = slot_in_page // X
        page_inner = slot_in_page % X
        v_tgt = (
            v_cache_ptr
            + block_idx * layer_stride
            + head_id * head_stride
            + page_outer * (head_dim * X)
            + cols * X
            + page_inner
        )
        tl.store(v_tgt, v_val, mask=mask & is_k)
    else:
        cache_k_base = (
            k_cache_ptr
            + loc * k_cache_stride_s
            + head_id * k_cache_stride_h
            + cols * k_cache_stride_d
        )
        tl.store(cache_k_base, out_typed, mask=mask & is_k)

        cache_v_base = (
            v_cache_ptr
            + loc * v_cache_stride_s
            + head_id * v_cache_stride_h
            + cols * v_cache_stride_d
        )
        tl.store(cache_v_base, v_val, mask=mask & is_k)

    is_idx_k = head_program == idx_k_program
    idx_cache_base = (
        idx_k_cache_ptr + loc * idx_k_cache_stride_s + cols * idx_k_cache_stride_d
    )
    tl.store(idx_cache_base, out_typed, mask=mask & is_idx_k)


def sparse_qk_index_gemma_rmsnorm_rope_cache(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    idx_k_cache: torch.Tensor,
    out_cache_loc: torch.Tensor,
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
    skip_index: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse sparse Q/K/index norm+RoPE with main KV and index-K cache stores.

    When ``skip_index`` is True (an index-topk "skip" layer that reuses the
    group source layer's top-k, so its idx_q/idx_k are never consumed and its
    idx_k cache is never read in prefill or decode), the grid drops the
    idx_q/idx_k program rows: the kernel does ONLY main q/k norm+rope + main KV
    cache write (ATOM's main-only ``_fused_qkv_norm_rope_cache_kernel`` shape).
    idx_q/idx_k are returned unchanged (unused downstream for skip layers).

    Supports two main-KV cache layouts:
      - 3D NHD: k_cache/v_cache are (num_slots, k_heads, head_dim).
      - 5D SHUFFLE (vectorized_5d): k_cache is
        (num_blocks, k_heads, head_dim // X, block_size, X) and v_cache is
        (num_blocks, k_heads, block_size // X, head_dim, X). Matches
        sglang.srt.layers.attention.utils.reshape_and_cache_shuffle_5d.
    The index-K cache stays 3D in both modes.
    """
    assert q.dim() == k.dim() == v.dim() == idx_q.dim() == idx_k.dim() == 2
    use_5d = k_cache.dim() == 5
    if use_5d:
        assert v_cache.dim() == 5
    else:
        assert k_cache.dim() == v_cache.dim() == 3
    assert idx_k_cache.dim() == 3
    assert out_cache_loc.dim() == positions.dim() == 1
    assert q.shape[0] == k.shape[0] == v.shape[0] == idx_q.shape[0] == idx_k.shape[0]
    assert q.shape[0] == positions.shape[0] == out_cache_loc.shape[0]
    assert q.shape[1] % head_dim == 0
    assert k.shape[1] % head_dim == 0
    assert v.shape[1] == k.shape[1]
    assert idx_q.shape[1] % head_dim == 0
    assert idx_k.shape[1] == head_dim
    assert rotary_dim <= head_dim and rotary_dim % 2 == 0

    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_q_heads = idx_q.shape[1] // head_dim
    if use_5d:
        # k_cache: (num_blocks, k_heads, head_dim // X, block_size, X)
        # v_cache: (num_blocks, k_heads, block_size // X, head_dim, X)
        assert k_cache.shape[1] == v_cache.shape[1] == k_heads
        block_size_5d = k_cache.shape[3]
        x_5d = k_cache.shape[4]
        assert k_cache.shape[2] * x_5d == head_dim
        assert v_cache.shape[2] * x_5d == block_size_5d and v_cache.shape[3] == head_dim
        assert head_dim % x_5d == 0 and block_size_5d % x_5d == 0
        assert k_cache.is_contiguous() and v_cache.is_contiguous()
    else:
        assert k_cache.shape[1] == v_cache.shape[1] == k_heads
        block_size_5d = 0
        x_5d = 0
    assert idx_k_cache.shape[1] == 1

    q_out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    k_out = torch.empty(k.shape, dtype=k.dtype, device=k.device)
    idx_q_out = torch.empty(idx_q.shape, dtype=idx_q.dtype, device=idx_q.device)
    idx_k_out = torch.empty(idx_k.shape, dtype=idx_k.dtype, device=idx_k.device)
    block_hd = triton.next_power_of_2(head_dim)

    # skip_index: drop the idx_q (+idx_k) program rows so only the main q/k arms
    # run. Every index op in the kernel is guarded by is_idx_q/is_idx_k, which are
    # false for program rows < q_heads+k_heads, so a shorter grid = main-only.
    _n_head_programs = (
        q_heads + k_heads if skip_index else q_heads + k_heads + idx_q_heads + 1
    )
    _sparse_qk_index_gemma_rmsnorm_rope_cache_kernel[
        (q.shape[0], _n_head_programs)
    ](
        q,
        k,
        v,
        idx_q,
        idx_k,
        q_out,
        k_out,
        idx_q_out,
        idx_k_out,
        k_cache,
        v_cache,
        idx_k_cache,
        out_cache_loc,
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
        v.stride(0),
        v.stride(1),
        idx_q.stride(0),
        idx_q.stride(1),
        idx_k.stride(0),
        idx_k.stride(1),
        k_cache.stride(0) if not use_5d else 0,
        k_cache.stride(1) if not use_5d else 0,
        k_cache.stride(2) if not use_5d else 0,
        v_cache.stride(0) if not use_5d else 0,
        v_cache.stride(1) if not use_5d else 0,
        v_cache.stride(2) if not use_5d else 0,
        idx_k_cache.stride(0),
        idx_k_cache.stride(1),
        idx_k_cache.stride(2),
        q_heads,
        k_heads,
        idx_q_heads,
        head_dim,
        rotary_dim,
        eps,
        is_neox_style,
        BLOCK_HD=block_hd,
        USE_5D=use_5d,
        BLOCK_SIZE=block_size_5d,
        X=x_5d,
        num_warps=4,
    )
    return q_out, k_out, idx_q_out, idx_k_out


# ---------------------------------------------------------------------------
# ATOM-style single-program-per-token fused kernel (ported from ATOM
# atom/model_ops/minimax_m3/sparse_attn.py _fused_qknorm_rope_kv_insert_shuffle
# + _gemma_norm_rope_head). One program handles ALL heads of one token via
# tl.static_range -> far fewer/leaner launches than the per-(token,head)
# where-dispatch kernel above. Writes main K/V into the 5D SHUFFLE cache
# (page_size parameterized; math identical to reshape_and_cache_shuffle_5d)
# and index-K into the 3D (flat page) index cache.
# ---------------------------------------------------------------------------
@triton.jit
def _gemma_norm_rope_head_atom(
    row_ptr,
    w_ptr,
    cos_ptr,
    sin_ptr,
    HEAD_DIM: tl.constexpr,
    ROT_HALF: tl.constexpr,
    eps,
):
    d = tl.arange(0, HEAD_DIM)
    vals = tl.load(row_ptr + d).to(tl.float32)
    w = tl.load(w_ptr + d).to(tl.float32)
    var = tl.sum(vals * vals, axis=0) / HEAD_DIM
    normed = vals * tl.rsqrt(var + eps) * (1.0 + w)

    dh = tl.arange(0, HEAD_DIM)
    is_low = dh < ROT_HALF
    in_rot = dh < (2 * ROT_HALF)
    partner_idx = tl.where(is_low, dh + ROT_HALF, dh - ROT_HALF)
    pvals = tl.load(row_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    pw = tl.load(w_ptr + partner_idx, mask=in_rot, other=0.0).to(tl.float32)
    p_normed = pvals * tl.rsqrt(var + eps) * (1.0 + pw)

    j = tl.where(is_low, dh, dh - ROT_HALF)
    # cos/sin cache may be bf16 in sglang (ATOM assumed fp32); upcast so the
    # rope multiply is done in fp32 (matches the reference / old kernel).
    cos = tl.load(cos_ptr + j, mask=in_rot, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + j, mask=in_rot, other=0.0).to(tl.float32)
    sign = tl.where(is_low, -1.0, 1.0)
    roped = normed * cos + sign * p_normed * sin
    return tl.where(in_rot, roped, normed)


@triton.jit
def _fused_qknorm_rope_kv_insert_5d_kernel(
    q_ptr,          # [num_tokens, q_stride_m] main q view
    k_ptr,          # [num_tokens, k_stride_m] main k view
    v_ptr,          # [num_tokens, v_stride_m] main v view
    iq_ptr,         # [num_tokens, iq_stride_m] index_q view
    ik_ptr,         # [num_tokens, ik_stride_m] index_k view
    q_norm_w_ptr,
    k_norm_w_ptr,
    iq_norm_w_ptr,
    ik_norm_w_ptr,
    cos_sin_ptr,
    positions_ptr,
    loc_ptr,        # [num_tokens] int (logical slot)
    q_out_ptr,      # [num_tokens, num_heads*head_dim]
    iq_out_ptr,     # [num_tokens, num_index_heads*idx_head_dim]
    kc_ptr,         # SHUFFLE K [nb, nkv, head_dim//x, page, x]
    vc_ptr,         # SHUFFLE V [nb, nkv, page//x, head_dim, x]
    idx_k_cache_ptr,  # [*, idx_head_dim] flat (3D page cache, contiguous)
    q_stride_m,
    k_stride_m,
    v_stride_m,
    iq_stride_m,
    ik_stride_m,
    num_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_index_heads: tl.constexpr,
    head_dim: tl.constexpr,
    idx_head_dim: tl.constexpr,
    rotary_dim: tl.constexpr,
    eps,
    x: tl.constexpr,       # 16 // itemsize
    PAGE: tl.constexpr,    # physical page size (16 or 64)
):
    tok = tl.program_id(0)
    half = rotary_dim // 2
    pos = tl.load(positions_ptr + tok)
    cos_row = cos_sin_ptr + pos * rotary_dim
    sin_row = cos_sin_ptr + pos * rotary_dim + half
    d = tl.arange(0, head_dim)

    # ----- (1) q heads -> q_out -----
    q_row = q_ptr + tok * q_stride_m
    for h in tl.static_range(num_heads):
        out = _gemma_norm_rope_head_atom(
            q_row + h * head_dim, q_norm_w_ptr, cos_row, sin_row, head_dim, half, eps
        )
        tl.store(
            q_out_ptr + tok * (num_heads * head_dim) + h * head_dim + d,
            out.to(q_out_ptr.dtype.element_ty),
        )

    # ----- (2) index_q heads -> iq_out -----
    di = tl.arange(0, idx_head_dim)
    iq_row = iq_ptr + tok * iq_stride_m
    for h in tl.static_range(num_index_heads):
        out = _gemma_norm_rope_head_atom(
            iq_row + h * idx_head_dim, iq_norm_w_ptr, cos_row, sin_row, idx_head_dim, half, eps
        )
        tl.store(
            iq_out_ptr + tok * (num_index_heads * idx_head_dim) + h * idx_head_dim + di,
            out.to(iq_out_ptr.dtype.element_ty),
        )

    slot = tl.load(loc_ptr + tok).to(tl.int64)
    page = slot // PAGE
    s = slot % PAGE
    valid_slot = slot >= 0

    # ----- (3) k heads -> SHUFFLE K, (4) v heads -> SHUFFLE V -----
    k_row = k_ptr + tok * k_stride_m
    v_row = v_ptr + tok * v_stride_m
    for h in tl.static_range(num_kv_heads):
        kout = _gemma_norm_rope_head_atom(
            k_row + h * head_dim, k_norm_w_ptr, cos_row, sin_row, head_dim, half, eps
        )
        k_off = (
            ((page * num_kv_heads + h) * (head_dim // x) + d // x) * (PAGE * x)
            + s * x
            + (d % x)
        )
        tl.store(kc_ptr + k_off, kout.to(kc_ptr.dtype.element_ty), mask=valid_slot)

        vvals = tl.load(v_row + h * head_dim + d)  # raw
        v_off = (
            ((page * num_kv_heads + h) * (PAGE // x) + s // x) * (head_dim * x)
            + d * x
            + (s % x)
        )
        tl.store(vc_ptr + v_off, vvals.to(vc_ptr.dtype.element_ty), mask=valid_slot)

    # ----- (5) index_k -> flat index cache -----
    ik_row = ik_ptr + tok * ik_stride_m
    ikout = _gemma_norm_rope_head_atom(
        ik_row, ik_norm_w_ptr, cos_row, sin_row, idx_head_dim, half, eps
    )
    tl.store(
        idx_k_cache_ptr + slot * idx_head_dim + di,
        ikout.to(idx_k_cache_ptr.dtype.element_ty),
        mask=valid_slot,
    )


def sparse_qk_index_gemma_rmsnorm_rope_cache_atom(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    idx_q: torch.Tensor,
    idx_k: torch.Tensor,
    k_cache: torch.Tensor,   # 5D SHUFFLE (num_blocks, k_heads, head_dim//x, page, x)
    v_cache: torch.Tensor,   # 5D SHUFFLE (num_blocks, k_heads, page//x, head_dim, x)
    idx_k_cache: torch.Tensor,  # 3D (flat page) index cache
    out_cache_loc: torch.Tensor,
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
    """ATOM-structured fused qk/idx norm+rope + 5D KV cache write.

    Drop-in replacement for sparse_qk_index_gemma_rmsnorm_rope_cache with the
    ATOM one-program-per-token kernel. Only q_out / idx_q_out are materialized;
    k / idx_k live in the cache (downstream reads them from cache), so the
    original k / idx_k tensors are returned unchanged as shape placeholders.
    """
    assert q.dim() == k.dim() == v.dim() == idx_q.dim() == idx_k.dim() == 2
    assert k_cache.dim() == 5 and v_cache.dim() == 5
    assert idx_k_cache.dim() == 3
    assert is_neox_style, "ATOM fused kernel assumes NeoX-style RoPE"
    num_tokens = q.shape[0]
    q_heads = q.shape[1] // head_dim
    k_heads = k.shape[1] // head_dim
    idx_head_dim = idx_k.shape[1]
    idx_q_heads = idx_q.shape[1] // idx_head_dim
    assert k_cache.shape[1] == v_cache.shape[1] == k_heads
    page = k_cache.shape[3]
    x = k_cache.shape[4]
    assert head_dim % x == 0 and page % x == 0
    assert k_cache.is_contiguous() and v_cache.is_contiguous() and idx_k_cache.is_contiguous()

    # q / idx_q may be non-contiguous VIEWS (row stride != width). The kernel
    # writes q_out / idx_q_out with a CONTIGUOUS row stride (num_heads*head_dim),
    # so allocate contiguous outputs — empty_like(view) would inherit the padded
    # stride and mis-place every row.
    q_out = torch.empty((num_tokens, q_heads * head_dim), dtype=q.dtype, device=q.device)
    idx_q_out = torch.empty(
        (num_tokens, idx_q_heads * idx_head_dim), dtype=idx_q.dtype, device=idx_q.device
    )
    idx_k_cache_flat = idx_k_cache.reshape(-1, idx_head_dim)

    _fused_qknorm_rope_kv_insert_5d_kernel[(num_tokens,)](
        q, k, v, idx_q, idx_k,
        q_weight, k_weight, idx_q_weight, idx_k_weight,
        cos_sin_cache, positions, out_cache_loc,
        q_out, idx_q_out,
        k_cache, v_cache, idx_k_cache_flat,
        q.stride(0), k.stride(0), v.stride(0), idx_q.stride(0), idx_k.stride(0),
        num_heads=q_heads,
        num_kv_heads=k_heads,
        num_index_heads=idx_q_heads,
        head_dim=head_dim,
        idx_head_dim=idx_head_dim,
        rotary_dim=rotary_dim,
        eps=eps,
        x=x,
        PAGE=page,
        num_warps=4,
    )
    # k / idx_k returned unchanged (ignored downstream; attention reads cache).
    return q_out, k, idx_q_out, idx_k
