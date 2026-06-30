# Copyright 2025 SGLang Team
"""Experimental ATOM-style MiniMax-M3 sparse prefill.

This is intentionally narrow and env-gated. It replaces only the main sparse
attention step after SGLang has already produced ``topk_idx`` with the index
attention path. The implementation builds the page-16 sparse block table that
ATOM feeds into AITER's Gluon paged-attention kernel, then treats each prefill
query token as an independent length-1 decode sequence.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.attention.aiter_utils import (
    get_recommended_splits,
    pa_decode_gluon,
)

SPARSE_BLOCK_SIZE = 128
ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE


@triton.jit
def _build_atom_sparse_bt_prefill_kernel(
    topk_ptr,  # [Hkv, total_q, topk] int32
    req_to_token_ptr,  # [max_reqs, max_kv_len] int32
    req_pool_indices_ptr,  # [batch] int32
    req_id_ptr,  # [total_q] int32
    abs_pos_ptr,  # [total_q] int32
    sparse_bt_ptr,  # [total_q * Hkv, topk * 8] int32
    sparse_ctx_ptr,  # [total_q * Hkv] int32
    max_topk,
    stride_topk_h,
    stride_topk_n,
    stride_topk_t,
    stride_req_to_token_b,
    stride_sparse_bt_n,
    num_kv_heads: tl.constexpr,
    page_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_h = tl.program_id(1)

    req_id = tl.load(req_id_ptr + pid_n)
    req_pool_id = tl.load(req_pool_indices_ptr + req_id)
    abs_pos = tl.load(abs_pos_ptr + pid_n)
    causal_len = abs_pos + 1
    self_blk = abs_pos // page_size

    topk_row = topk_ptr + pid_h * stride_topk_h + pid_n * stride_topk_n
    out_row = sparse_bt_ptr + (pid_n * num_kv_heads + pid_h) * stride_sparse_bt_n
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_t, mask=off_t < max_topk, other=-1)

    valid = (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)
    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    sparse_slot = tl.where(is_full, earlier_full, n_full)

    # Convert logical 128-token sparse block -> physical 16-page IDs in the
    # collapsed (phys_page, kv_head) page space consumed by pa_decode_gluon.
    token_pos = blk * page_size
    token_slot = tl.load(
        req_to_token_ptr + req_pool_id * stride_req_to_token_b + token_pos,
        mask=valid,
        other=0,
    ).to(tl.int32)
    logical_page = token_slot // page_size
    phys_base = logical_page * pages_per_block
    dst_base = sparse_slot * pages_per_block

    for j in tl.static_range(0, pages_per_block):
        encoded_page = (phys_base + j) * num_kv_heads + pid_h
        tl.store(out_row + dst_base + j, encoded_page, mask=valid)

    n_used = n_valid * pages_per_block
    off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
    tl.store(out_row + off_w, tl.zeros_like(off_w), mask=off_w >= n_used)

    tail_tokens = causal_len - self_blk * page_size
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * page_size + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * page_size, causal_len))
    tl.store(sparse_ctx_ptr + pid_n * num_kv_heads + pid_h, ctx)


def _build_atom_sparse_bt_prefill(
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    total_q = topk_idx.shape[1]
    num_kv_heads = topk_idx.shape[0]
    topk = topk_idx.shape[2]
    pos = torch.arange(total_q, dtype=torch.int32, device=topk_idx.device)
    req_id = torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True).to(
        torch.int32
    )
    abs_pos = (prefix_lens[req_id] + (pos - cu_seqlens[req_id])).to(torch.int32)

    sparse_bt = torch.empty(
        (total_q * num_kv_heads, topk * PAGES_PER_SPARSE_BLOCK),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    sparse_ctx = torch.empty(
        (total_q * num_kv_heads,), dtype=torch.int32, device=topk_idx.device
    )
    _build_atom_sparse_bt_prefill_kernel[(total_q, num_kv_heads)](
        topk_idx,
        req_to_token,
        req_pool_indices,
        req_id,
        abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(0),
        topk_idx.stride(1),
        topk_idx.stride(2),
        req_to_token.stride(0),
        sparse_bt.stride(0),
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        pages_per_block=PAGES_PER_SPARSE_BLOCK,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


def can_use_atom_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sink: Optional[torch.Tensor],
    block_size_k: int,
) -> bool:
    return (
        pa_decode_gluon is not None
        and get_recommended_splits is not None
        and sink is None
        and block_size_k == SPARSE_BLOCK_SIZE
        and k_cache.dim() == 5
        and v_cache.dim() == 5
        and k_cache.shape[3] == SPARSE_BLOCK_SIZE
        and v_cache.shape[2] * v_cache.shape[4] == SPARSE_BLOCK_SIZE
        and q.shape[-1] == 128
    )


def repaginate_kv_128_to_16(
    k_cache: torch.Tensor, v_cache: torch.Tensor, num_kv_heads: int, head_dim: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Re-paginate SGLang vectorized_5d page-128 KV cache to page-16 for the Gluon
    paged-attention kernel (which reads page size from key_cache.shape[-2] and only
    supports 16/64/1024).

    SGLang vectorized_5d layouts:
        K: (num_blocks, H, head_dim//x, page=128, x)   page axis interior (4th dim)
        V: (num_blocks, H, page//x,    head_dim, x)
    For K the 128 page axis is INTERIOR, so a plain leading-dim .view() scrambles
    head/d_outer vs page. Split 128 -> (8, 16) and transpose the outer 8 to the
    front (a contiguous copy). For V a plain leading-dim view is numerically
    correct (its interior nesting differs).
    Returns page-16 views collapsed to (num_phys16*H, 1, ...) for the kernel.
    """
    x = k_cache.shape[-1]
    num_blocks = k_cache.shape[0]
    k16 = (
        k_cache.view(
            num_blocks, num_kv_heads, head_dim // x, PAGES_PER_SPARSE_BLOCK, ASM_PAGE_SIZE, x
        )
        .permute(0, 3, 1, 2, 4, 5)
        .contiguous()
        .view(num_blocks * PAGES_PER_SPARSE_BLOCK, num_kv_heads, head_dim // x, ASM_PAGE_SIZE, x)
    )
    v16 = v_cache.view(
        num_blocks * PAGES_PER_SPARSE_BLOCK, num_kv_heads, ASM_PAGE_SIZE // x, head_dim, x
    )
    k_view = k16.view(num_blocks * PAGES_PER_SPARSE_BLOCK * num_kv_heads, 1, *k16.shape[2:])
    v_view = v16.view(num_blocks * PAGES_PER_SPARSE_BLOCK * num_kv_heads, 1, *v16.shape[2:])
    return k_view, v_view


def vectorized_5d_index_cache_to_nhd(cache: torch.Tensor) -> torch.Tensor:
    """Materialize a SHUFFLE 5D index cache as NHD for the existing top-k kernel.

    This is a temporary bridge for the first POC. It copies the index cache, so it
    is not the final performance path. The final version should port ATOM's
    index-topk kernel to read SHUFFLE 5D directly.
    """
    assert cache.dim() == 5
    num_blocks, num_heads, d_outer, page_size, x = cache.shape
    return (
        cache.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(num_blocks * page_size, num_heads, d_outer * x)
    )


@torch.no_grad()
def atom_gluon_sparse_prefill(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    block_size_k: int,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """Run ATOM-style Gluon sparse prefill over SGLang SHUFFLE 5D KV cache."""
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    num_kv_heads = k_cache.shape[1]
    assert num_q_heads % num_kv_heads == 0
    q_group = num_q_heads // num_kv_heads

    sparse_bt, sparse_ctx = _build_atom_sparse_bt_prefill(
        topk_idx,
        req_to_token,
        req_pool_indices,
        cu_seqlens,
        prefix_lens,
        block_size_k,
    )

    k_view, v_view = repaginate_kv_128_to_16(k_cache, v_cache, num_kv_heads, head_dim)

    q_view = q.view(total_q * num_kv_heads, q_group, head_dim)
    out = torch.empty_like(q)
    out_view = out.view(total_q * num_kv_heads, q_group, head_dim)
    num_seqs = total_q * num_kv_heads
    max_part_num = get_recommended_splits(num_seqs, 1)
    ctx_part = 256
    intermediate_shape = (num_seqs, 1, max_part_num, q_group)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (*intermediate_shape, head_dim), dtype=q.dtype, device=q.device
    )

    pa_decode_gluon(
        output=out_view,
        query=q_view,
        key_cache=k_view,
        value_cache=v_view,
        context_lengths=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        query_length=1,
        max_context_partition_num=max_part_num,
        context_partition_size=ctx_part,
        compute_type=q.dtype,
        key_scale=None,
        value_scale=None,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        sinks=None,
        sliding_window=0,
        ps=True,
    )
    return out
