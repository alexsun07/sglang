# Copyright 2025 SGLang Team
"""Experimental ATOM-style MiniMax-M3 sparse prefill.

This is intentionally narrow and env-gated. It replaces only the main sparse
attention step after SGLang has already produced ``topk_idx`` with the index
attention path. The implementation builds the sparse page table that ATOM feeds
into AITER's Gluon paged-attention kernel, then treats each prefill query token
as an independent length-1 decode sequence.

Requires the KV pool to be allocated with ``--page-size`` equal to the Gluon
page size (16 or 64); the vectorized_5d K/V buffer is then fed to the kernel as
a zero-copy view (no per-layer repagination copy). The sparse selection block
size (``sparse_block_size``, 128) is decoupled from the physical page size: each
selected 128-token block maps to ``sparse_block_size // page_size`` physical
pages, looked up individually from ``req_to_token`` (the pages of one block are
not guaranteed to be physically contiguous when page < block).
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
# Gluon kernel supports kv_block_size in {16, 64, 1024}; the KV pool must be
# launched with --page-size set to one of the small ones so the 5D buffer is
# consumed zero-copy.
GLUON_PAGE_SIZES = (16, 64)


@triton.jit
def _build_atom_sparse_bt_prefill_kernel(
    topk_ptr,  # [Hkv=1, total_q, topk] int32
    req_to_token_ptr,  # [max_reqs, max_kv_len] int32
    req_pool_indices_ptr,  # [batch] int32
    req_id_ptr,  # [total_q] int32
    abs_pos_ptr,  # [total_q] int32
    sparse_bt_ptr,  # [total_q, topk * pages_per_block] int32
    sparse_ctx_ptr,  # [total_q] int32
    max_topk,
    stride_topk_n,
    stride_topk_t,
    stride_req_to_token_b,
    stride_sparse_bt_n,
    sparse_block_size: tl.constexpr,
    page_size: tl.constexpr,
    pages_per_block: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)

    req_id = tl.load(req_id_ptr + pid_n)
    req_pool_id = tl.load(req_pool_indices_ptr + req_id)
    abs_pos = tl.load(abs_pos_ptr + pid_n)
    causal_len = abs_pos + 1
    self_blk = abs_pos // sparse_block_size

    topk_row = topk_ptr + pid_n * stride_topk_n
    out_row = sparse_bt_ptr + pid_n * stride_sparse_bt_n
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_t, mask=off_t < max_topk, other=-1)

    # Causal: a query at abs_pos may attend its own block and all earlier ones.
    valid = (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)
    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    # Pack full blocks first (each contributes a whole sparse_block_size span),
    # the tail/current block last (partial, causal). This keeps the kernel's
    # sequential page walk aligned with sparse_ctx.
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    sparse_slot = tl.where(is_full, earlier_full, n_full)

    # For each selected sparse block, emit its pages_per_block physical pages.
    # The pages of one block are NOT guaranteed contiguous when page < block,
    # so look up req_to_token at every page boundary.
    dst_base = sparse_slot * pages_per_block
    for j in tl.static_range(0, pages_per_block):
        token_pos = blk * sparse_block_size + j * page_size
        slot = tl.load(
            req_to_token_ptr + req_pool_id * stride_req_to_token_b + token_pos,
            mask=valid,
            other=0,
        ).to(tl.int32)
        phys_page = slot // page_size
        tl.store(out_row + dst_base + j, phys_page, mask=valid)

    n_used = n_valid * pages_per_block
    off_w = tl.arange(0, BLOCK_SIZE_T * pages_per_block)
    tl.store(out_row + off_w, tl.zeros_like(off_w), mask=off_w >= n_used)

    # Effective KV length the kernel will walk across the packed pages.
    tail_tokens = causal_len - self_blk * sparse_block_size
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * sparse_block_size + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(
        has_tail, ctx, tl.minimum(n_valid * sparse_block_size, causal_len)
    )
    tl.store(sparse_ctx_ptr + pid_n, ctx)


def _build_atom_sparse_bt_prefill(
    topk_idx: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    prefix_lens: torch.Tensor,
    sparse_block_size: int,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # topk_idx: [num_kv_heads=1, total_q, topk]
    total_q = topk_idx.shape[1]
    topk = topk_idx.shape[2]
    pages_per_block = sparse_block_size // page_size
    pos = torch.arange(total_q, dtype=torch.int32, device=topk_idx.device)
    req_id = torch.searchsorted(cu_seqlens[1:].contiguous(), pos, right=True).to(
        torch.int32
    )
    abs_pos = (prefix_lens[req_id] + (pos - cu_seqlens[req_id])).to(torch.int32)

    sparse_bt = torch.empty(
        (total_q, topk * pages_per_block),
        dtype=torch.int32,
        device=topk_idx.device,
    )
    sparse_ctx = torch.empty((total_q,), dtype=torch.int32, device=topk_idx.device)
    _build_atom_sparse_bt_prefill_kernel[(total_q,)](
        topk_idx,
        req_to_token,
        req_pool_indices,
        req_id,
        abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        req_to_token.stride(0),
        sparse_bt.stride(0),
        sparse_block_size=sparse_block_size,
        page_size=page_size,
        pages_per_block=pages_per_block,
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
    # k_cache: vectorized_5d (num_blocks, num_kv_heads, head_dim//x, page, x)
    if (
        pa_decode_gluon is None
        or get_recommended_splits is None
        or sink is not None
        or block_size_k != SPARSE_BLOCK_SIZE
        or k_cache.dim() != 5
        or v_cache.dim() != 5
        or q.shape[-1] != 128
    ):
        return False
    num_kv_heads = k_cache.shape[1]
    page_size = k_cache.shape[3]
    # Gluon addresses heads via the cache's head dim internally; the block table
    # carries pure page ids. The single-KV-head case (M3 TP>=4) needs no head
    # folding. >1 KV heads would require per-head block tables -> fall back.
    return (
        num_kv_heads == 1
        and page_size in GLUON_PAGE_SIZES
        and SPARSE_BLOCK_SIZE % page_size == 0
        and v_cache.shape[2] * v_cache.shape[4] == page_size
    )


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
    """Run ATOM-style Gluon sparse prefill over SGLang SHUFFLE 5D KV cache.

    The KV cache is fed to the Gluon kernel as a zero-copy view: the pool is
    allocated page-16/64 so its native (num_blocks, 1, head_dim//x, page, x)
    layout already matches what the kernel expects. ``num_kv_heads == 1`` is
    enforced by ``can_use_atom_prefill``.
    """
    if sm_scale is None:
        sm_scale = q.shape[-1] ** -0.5

    q = q.contiguous()
    total_q, num_q_heads, head_dim = q.shape
    page_size = k_cache.shape[3]

    sparse_bt, sparse_ctx = _build_atom_sparse_bt_prefill(
        topk_idx,
        req_to_token,
        req_pool_indices,
        cu_seqlens,
        prefix_lens,
        block_size_k,
        page_size,
    )

    out = torch.empty_like(q)
    num_seqs = total_q
    max_part_num = get_recommended_splits(num_seqs, 1)
    ctx_part = 256
    intermediate_shape = (num_seqs, 1, max_part_num, num_q_heads)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        (*intermediate_shape, head_dim), dtype=q.dtype, device=q.device
    )

    pa_decode_gluon(
        output=out,
        query=q,
        key_cache=k_cache,
        value_cache=v_cache,
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
