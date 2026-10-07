"""A KV source's compressor: pooled latents, their index keys and compressed entries, packed and stored by position."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

from tensorfold.families.glm5_next.cuda import qmm

from . import norms, quant, rope
from .buffers import Buffers, State
from .norms import _normed
from .weights import LayerW


@triton.jit
def _pool(CMP, TAIL, W, LAT, EPOS, POS, LANE, SEG0, tail_lane, eps, D: tl.constexpr, BLOCK: tl.constexpr,
          LANES: tl.constexpr):
    """Pooling (M:473-485): row r at an odd q pools group q // 2 from row r - 1 (or the committed tail) and itself;
    LANES: q is POS[r], and a segment's first row reads its lane's tail."""

    r = tl.program_id(0)
    if LANES:
        q = tl.load(POS + r)
    else:
        q = tl.load(POS) + r
    d = tl.arange(0, BLOCK)
    ok = d < D
    cur = CMP + r.to(tl.int64) * (2 * D)
    kv1 = tl.load(cur + d, mask=ok, other=0.0)
    z1 = tl.load(cur + D + d, mask=ok, other=0.0)
    if LANES:
        if r == tl.load(SEG0 + r):
            prev = TAIL + tl.load(LANE + r).to(tl.int64) * tail_lane
        else:
            prev = cur - 2 * D
        kv0 = tl.load(prev + d, mask=ok, other=0.0)
        z0 = tl.load(prev + D + d, mask=ok, other=0.0)
    elif r > 0:
        kv0 = tl.load(cur - 2 * D + d, mask=ok, other=0.0)
        z0 = tl.load(cur - D + d, mask=ok, other=0.0)
    else:
        kv0 = tl.load(TAIL + d, mask=ok, other=0.0)
        z0 = tl.load(TAIL + D + d, mask=ok, other=0.0)
    m = tl.maximum(z0, z1)                          # softmax over the group's two scores, per channel
    e0 = libdevice.exp(z0 - m)
    e1 = libdevice.exp(z1 - m)
    s = e0 + e1
    pooled = kv0 * tl.math.div_rn(e0, s) + kv1 * tl.math.div_rn(e1, s)
    pooled = tl.where(q % 2 == 1, pooled, 0.0)
    lat = pooled.to(tl.bfloat16).to(tl.float32)
    tl.store(LAT + r.to(tl.int64) * D + d, _normed(lat, W, d, ok, eps, D), mask=ok)
    tl.store(EPOS + r, q - q % 2)                   # the group's first position, where its entry is rotated


@triton.jit
def _store(LAT, KI, COMP, INDEX_K, POS, LANE, comp_lane, key_lane, RATIO: tl.constexpr, D: tl.constexpr,
           DK: tl.constexpr, BLOCK: tl.constexpr, BLOCK_K: tl.constexpr, LANES: tl.constexpr):
    """Row r's packed entry (D bytes) and index key (DK bytes) into slot (pos + r) // RATIO when its position
    completes a group; LANES: at POS[r], into lane LANE[r]'s caches."""

    r = tl.program_id(0)
    if LANES:
        q = tl.load(POS + r)
        lane = tl.load(LANE + r).to(tl.int64)
        COMP += lane * comp_lane
        INDEX_K += lane * key_lane
    else:
        q = tl.load(POS) + r
    if q % RATIO == RATIO - 1:
        j = (q // RATIO).to(tl.int64)
        d = tl.arange(0, BLOCK)
        tl.store(COMP + j * D + d, tl.load(LAT + r.to(tl.int64) * D + d, mask=d < D), mask=d < D)
        k = tl.arange(0, BLOCK_K)
        tl.store(INDEX_K + j * DK + k, tl.load(KI + r.to(tl.int64) * DK + k, mask=k < DK), mask=k < DK)


def compress(lw: LayerW, xa: torch.Tensor, state: State, buf: Buffers, table: torch.Tensor, eps: float,
             first: int = 0, tables: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None) -> None:
    """Pool, index-key and entry rows of KV source ``lw`` from ``xa`` [R, D], the forward's rows first.., at
    ``state.pos_dev`` + first + r; a row before first opens the group of an odd first row, as the tail does at 0.

    ``tables`` (lane, rpos, seg0): ``state`` is a ``Lanes``, row r of lane lane[r] at rpos[r], its segment from
    seg0[r]."""

    layer, comp, idx = lw.index, lw.attn.comp, lw.attn.idx
    rows, hd = xa.shape[0], comp.norm.shape[0]
    if comp.ratio != state.ratio.get(layer) or idx is None or idx.wk is None:
        raise ValueError(f"layer {layer}: not a KV source of ratio {comp.ratio} with an index-K projection")
    lanes = tables is not None
    if lanes and (first or not 0 < rows <= min(buf.rows, *(t.shape[0] for t in tables))):
        raise ValueError(f"compress: {rows} lane rows from {first} for {buf.rows}-row buffers")
    if not lanes and (not 0 < rows <= buf.rows - first or state.pos + first + rows > state.capacity):
        raise ValueError(f"compress: rows {first}..{first + rows} at {state.pos} for {buf.rows}-row buffers, "
                         f"{state.capacity} slots")
    lane, pos, seg0 = tables if lanes else (state.pos_dev, state.pos_dev if not first else state.pos_dev + first,
                                            state.pos_dev)
    caches = (state.comp[layer][0], state.index_k[layer][0]) if lanes else (state.comp[layer], state.index_k[layer])
    lat, ki = buf.lat[:rows], buf.kI[:rows]
    block = triton.next_power_of_2(hd)
    if comp.ratio == 2:
        slot = state.pooled.index(layer)
        cmp = buf.cmp[slot, first:first + rows]
        qmm.matmul(xa, comp.wkv, out=cmp.view(rows, 2 * hd), f32=True)
        tail = state.tail[0, slot] if lanes else state.tail[slot] if not first else buf.cmp[slot, first - 1]
        _pool[(rows,)](cmp, tail, comp.norm, lat, buf.epos, pos, lane, seg0, state.tail.stride(0) if lanes else 0,
                       eps, D=hd, BLOCK=block, LANES=lanes, num_warps=4, enable_fp_fusion=False)
        positions = buf.epos[:rows]
    else:
        qmm.matmul(xa, comp.wkv, out=lat)
        norms.rmsnorm(lat, comp.norm, eps, lat)
        positions = pos[:rows] if lanes else pos    # entry j = q, rotated at q
    qmm.matmul(lat, idx.wk, out=ki)                 # the index key reads the latent before RoPE (M:744, 749-750)
    norms.rmsnorm(ki, idx.k_norm, eps, ki)
    kp = quant.fp4_qdq_1x32_e8m0(rope.apply(ki, positions, table), buf.kIp[:rows])
    lp = quant.fp4_qdq_1x16_e4m3(rope.apply(lat, positions, table), buf.latp[:rows])
    d, dk = lp.shape[1], kp.shape[1]
    strides = (state.comp[layer].stride(0), state.index_k[layer].stride(0)) if lanes else (0, 0)
    _store[(rows,)](lp, kp, *caches, pos, lane, *strides, RATIO=comp.ratio, D=d, DK=dk,
                    BLOCK=triton.next_power_of_2(d), BLOCK_K=triton.next_power_of_2(dk), LANES=lanes, num_warps=4)


def commit_tail(state: State, buf: Buffers, pos: int, keep: int) -> None:
    """After a forward at ``pos`` keeps ``keep`` rows: an open group's first row becomes each ratio-2 source's tail."""

    if not state.pooled:
        return
    if (pos + keep) % 2:
        state.tail.copy_(buf.cmp[:, keep - 1])
        state.tail_valid.fill_(1)
    else:
        state.tail_valid.zero_()
