"""DeepSeek-V4.1's lightning indexer: index Q to each row's ascending top-k entries, each row alone."""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import triton
import triton.language as tl

from tensorfold.families.glm5_next.cuda import qmm

from . import mx8, quant, rope
from .buffers import SPLIT_BINS, SPLIT_SPAN, score_rows
from .quant import as_loaded, unpack_fp4

if TYPE_CHECKING:
    from ..config import Config
    from .weights import IdxW

TILE = 128                  # entries (or candidate slots) a score item covers
ROWS = 16                   # rows a score item computes; a decode window is one item a tile
WAVES = 4                   # persistent score programs per SM
PICK = 1024                 # keys a select step reads (blocks: PICK // 2 of them)
STEPS = SPLIT_SPAN // PICK  # decode: select steps one program of a row covers

_SMS: dict[int, int] = {}


def index_q(w: IdxW, qr: torch.Tensor, table: torch.Tensor, positions: torch.Tensor, out: torch.Tensor, *,
            prompt: bool = False) -> torch.Tensor:
    """M:550-552: out [rows, heads, head_dim] = FP4 QDQ of RoPE(wq_b(qr)) in bf16, rows at ``positions``."""

    rows, heads, dim = out.shape
    if not out.is_contiguous() or heads * dim != w.wq_b.n:
        raise ValueError(f"index_q: out {tuple(out.shape)} is not a contiguous [rows, heads, dim] of {w.wq_b.n}")
    flat = out.view(rows, heads * dim)
    y = mx8.mm(w.wq_b, qr, flat, prompt=prompt)
    if y.data_ptr() != flat.data_ptr():
        flat.copy_(y)
    rope.apply(out, positions, table)
    return quant.fp4_qdq_1x32_e8m0(out)


def index_weights(w: IdxW, xa: torch.Tensor, out: torch.Tensor, part: torch.Tensor | None = None) -> torch.Tensor:
    """M:555: out [rows, heads] = fp32 weights_proj(xa) * head_dim^-0.5 * heads^-0.5 (fp32, never rounded to bf16)."""

    heads = out.shape[1]
    qmm.matmul(xa, w.wproj, out=out, f32=True, part=part)
    return out.mul_((w.wq_b.n // heads) ** -0.5 * heads ** -0.5)


@triton.jit
def _row_scores(Q, W, k, h, hok, d, D: tl.constexpr):
    """sum_h relu(q_h . k_t) * w_h over one row's heads (the MMA's M), t over the tile: fp32 [TILE]."""

    q = tl.load(Q + h[:, None] * D + d[None, :], mask=hok[:, None], other=0.0)
    w = tl.load(W + h, mask=hok, other=0.0)
    return tl.sum(tl.maximum(tl.dot(q, tl.trans(k)), 0.0) * w[:, None], axis=0)


@triton.jit
def _keys(KEYS, e, ok, d, D: tl.constexpr, BT: tl.constexpr, PACKED: tl.constexpr):
    """bf16 [BT, D] index keys of entries ``e``: bf16 rows, or packed FP4 rows dequantized; zeros where not ``ok``."""

    if PACKED:
        k = as_loaded(unpack_fp4(KEYS + e * (D // 2 + D // 32), ok, BT, D, 32).to(tl.bfloat16))
    else:
        k = tl.load(KEYS + e[:, None] * D + d[None, :], mask=ok[:, None], other=0.0)
    return k


@triton.jit
def _scores(Q, W, KEYS, OUT, POS, CAND, CANDN, LANE, SEGS, NSEG, R, row0, ratio, slots, key_lane, sq, sw, so, sc,
            H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr, BT: tl.constexpr, RB: tl.constexpr,
            G: tl.constexpr, CANDIDATES: tl.constexpr, PACKED: tl.constexpr, LANES: tl.constexpr,
            SP: tl.constexpr):
    """Items (rows, tile) over a persistent grid; row r's entry t is at absolute position pos + row0 + r. LANES: POS
    holds each row's position, row r reads lane LANE[r]'s keys, and items are (segment, tile) of the NSEG segments
    (start row, rows) in SEGS."""

    if not LANES:
        first = tl.load(POS).to(tl.int64) + row0
    h = tl.arange(0, HP)
    hok = h < H
    d = tl.arange(0, D)
    j = tl.arange(0, BT)
    if CANDIDATES:                                  # slot c of row r: entry cand[r, c // G] * G + c % G
        tiles = tl.cdiv(slots, BT)
        for i in range(tl.program_id(0), R * tiles, tl.num_programs(0)):
            r = i // tiles
            c = (i % tiles) * BT + j
            n = tl.load(CANDN + r) * G
            if (i % tiles) * BT < n:
                blk = tl.load(CAND + r * sc + c // G, mask=c < n, other=0).to(tl.int64)
                e = blk * G + c % G
                if LANES:
                    ok = (c < n) & (e < (tl.load(POS + r).to(tl.int64) + 1) // ratio)
                    k = _keys(KEYS + tl.load(LANE + r).to(tl.int64) * key_lane, e, ok, d, D, BT, PACKED)
                else:
                    ok = (c < n) & (e < (first + r + 1) // ratio)
                    k = _keys(KEYS, e, ok, d, D, BT, PACKED)
                s = _row_scores(Q + r * sq, W + r * sw, k, h, hok, d, D)
                tl.store(OUT + r * so + c, s, mask=ok)     # slots past the row's entries are never read
    elif LANES:                                     # a tile is one segment's, so never read across lanes
        g = tl.arange(0, SP)
        nseg = tl.load(NSEG)
        last = tl.load(SEGS + 2 * g, mask=g < nseg, other=0) + tl.load(SEGS + 2 * g + 1, mask=g < nseg, other=1) - 1
        tops = (tl.load(POS + last, mask=g < nseg, other=-1).to(tl.int64) + 1) // ratio
        tiles = tl.cdiv(tl.max(tops, 0), BT)        # the most entries a segment's last row sees
        for i in range(tl.program_id(0), nseg * tiles, tl.num_programs(0)):
            lo = tl.load(SEGS + 2 * (i // tiles))
            hi = lo + tl.load(SEGS + 2 * (i // tiles) + 1)
            top = (tl.load(POS + hi - 1).to(tl.int64) + 1) // ratio
            e = (i % tiles) * BT + j
            if (i % tiles) * BT < top:
                k = _keys(KEYS + tl.load(LANE + lo).to(tl.int64) * key_lane, e, e < top, d, D, BT, PACKED)
                for r in range(lo, hi):
                    s = _row_scores(Q + r * sq, W + r * sw, k, h, hok, d, D)
                    s = tl.where(e < (tl.load(POS + r).to(tl.int64) + 1) // ratio, s, float("-inf"))
                    tl.store(OUT + r * so + e, s, mask=e < top)
    else:
        tiles = tl.cdiv((first + R) // ratio, BT)
        for i in range(tl.program_id(0), tl.cdiv(R, RB) * tiles, tl.num_programs(0)):
            lo = (i // tiles) * RB
            hi = tl.minimum(R, lo + RB)
            top = (first + hi) // ratio             # entries the item's last row sees
            e = (i % tiles) * BT + j
            if (i % tiles) * BT < top:
                k = _keys(KEYS, e, e < top, d, D, BT, PACKED)
                for r in range(lo, hi):             # the tile read once for all the item's rows
                    s = _row_scores(Q + r * sq, W + r * sw, k, h, hok, d, D)
                    s = tl.where(e < (first + r + 1) // ratio, s, float("-inf"))
                    tl.store(OUT + r * so + e, s, mask=e < top)


def _sms(device: torch.device) -> int:
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _SMS:
        _SMS[i] = torch.cuda.get_device_properties(i).multi_processor_count
    return _SMS[i]


class LaneArgs(NamedTuple):
    """A shared decode forward's tables: each row's lane and position, the segments (start row, rows) and their
    count, the keys' lane stride in elements."""

    lane: torch.Tensor
    rpos: torch.Tensor
    segs: torch.Tensor
    nseg: torch.Tensor
    stride: int


def scores(qI: torch.Tensor, wI: torch.Tensor, keys: torch.Tensor, ratio: int, pos: torch.Tensor,
           out: torch.Tensor, *, row0: int = 0, cand: torch.Tensor | None = None,
           cand_n: torch.Tensor | None = None, block: int = 8, tables: LaneArgs | None = None) -> torch.Tensor:
    """Each row's fp32 sum_h relu(q_h . k_t) * w_h over the entries it sees (M:556-567), -inf past them; ``keys``:
    bf16 rows, or packed FP4 rows per 32 (the same bits). ``tables`` (decode): ``keys`` are lane 0's, row r is lane
    lane[r]'s at rpos[r]."""

    R, H, D = qI.shape
    packed = keys.dtype == torch.uint8
    width = quant.width(D, quant.FP4_E8M0) if packed else D
    if qI.dtype != torch.bfloat16 or keys.dtype not in (torch.bfloat16, torch.uint8) or not qI.is_contiguous() \
            or keys.shape[1] != width or not keys.is_contiguous():
        raise ValueError("scores: contiguous bf16 qI [rows, heads, dim] and keys [entries, dim] (bf16 or packed)")
    if wI.dtype != torch.float32 or tuple(wI.shape) != (R, H) or wI.stride(1) != 1:
        raise ValueError(f"scores: fp32 weights [{R}, {H}] with unit-stride heads")
    if out.dtype != torch.float32 or out.shape[0] < R or out.stride(1) != 1 or out.shape[1] < keys.shape[0]:
        raise ValueError(f"scores: fp32 out of {R} rows and at least {keys.shape[0]} columns")
    _check_tables(tables, R, row0)
    if R == 0:
        return out
    if cand is None:
        blocks = tables.segs.shape[0] if tables is not None else triton.cdiv(R, ROWS)
        items, slots = blocks * triton.cdiv(keys.shape[0], TILE), 0
    else:
        if cand.dtype != torch.int32 or cand.shape[0] < R or cand.stride(1) != 1 or cand_n is None:
            raise ValueError("scores: int32 candidate blocks [rows, blocks] and their counts")
        slots = cand.shape[1] * block
        items = R * triton.cdiv(slots, TILE)
    grid = (max(1, min(items, WAVES * _sms(qI.device))),)
    lane, segs, nseg, stride = (tables.lane, tables.segs, tables.nseg, tables.stride) if tables is not None else \
        (pos, pos, pos, 0)
    _scores[grid](qI, wI, keys, out, tables.rpos if tables is not None else pos, cand if cand is not None else qI,
                  cand_n if cand is not None else qI, lane, segs, nseg, R, row0, ratio, slots, stride, qI.stride(0),
                  wI.stride(0), out.stride(0), cand.stride(0) if cand is not None else 0, H=H,
                  HP=max(16, triton.next_power_of_2(H)), D=D, BT=TILE, RB=ROWS, G=block, CANDIDATES=cand is not None,
                  PACKED=packed, LANES=tables is not None,
                  SP=triton.next_power_of_2(segs.shape[0]) if tables is not None else 1, num_warps=4)
    return out


def _check_tables(tables: LaneArgs | None, R: int, row0: int) -> None:
    if tables is not None and (row0 or tables.lane.shape[0] < R or tables.rpos.shape[0] < R):
        raise ValueError(f"indexer: lane tables {tuple(tables.lane.shape)}, {tuple(tables.rpos.shape)} for {R} "
                         f"decode rows from {row0}")


@triton.jit
def _key(s):
    """A float32 as a uint32 whose unsigned order is the floats' order, -0 counted as +0."""

    bits = tl.where(s == 0.0, 0.0, s).to(tl.int32, bitcast=True)
    return (bits ^ ((bits >> 31) | -2147483648)).to(tl.uint32, bitcast=True)


@triton.jit
def _items(S, CAND, c0, n, vis, G: tl.constexpr, BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr,
           B: tl.constexpr):
    """Keys of items c0.. of one row, whether each takes part, and the id written for it."""

    i = c0 + tl.arange(0, B)
    if BLOCKS:                                      # M:593-604: a block's best visible entry, the newest pinned
        e = i[:, None] * G + tl.arange(0, G)[None, :]
        v = tl.max(tl.load(S + e, mask=e < vis, other=float("-inf")), axis=1)
        v = tl.where(i == (vis - 1) // G, float("inf"), v)
        ok = (i < n) & (v > float("-inf"))
        val = i
    elif CANDIDATES:
        live = i < n
        val = tl.load(CAND + i // G, mask=live, other=0) * G + i % G
        ok = live & (val < vis)
        v = tl.load(S + i, mask=ok, other=0.0)
    else:
        ok = i < n
        v = tl.load(S + i, mask=ok, other=0.0)
        val = i
    return _key(v), ok, val


@triton.jit
def _extent(POS, CANDN, r, row0, ratio, G: tl.constexpr, BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr,
            LANES: tl.constexpr):
    """(entries row r sees, items it selects among); LANES: POS holds each row's position."""

    if LANES:
        vis = (tl.load(POS + r).to(tl.int64) + 1) // ratio
    else:
        vis = (tl.load(POS).to(tl.int64) + row0 + r + 1) // ratio
    if BLOCKS:
        n = tl.cdiv(vis, G)
    elif CANDIDATES:
        n = tl.load(CANDN + r).to(tl.int64) * G
    else:
        n = vis
    return vis, n


@triton.jit
def _select(S, OUT, OUTN, POS, CAND, CANDN, row0, ratio, K, ss, so, sc, G: tl.constexpr,
            BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr, B: tl.constexpr, LANES: tl.constexpr):
    """Program r: radix select of the K-th best key, then the items above it and the lowest-numbered ties."""

    r = tl.program_id(0).to(tl.int64)
    vis, n = _extent(POS, CANDN, r, row0, ratio, G, BLOCKS, CANDIDATES, LANES)
    row = S + r * ss
    cand = CAND + r * sc
    bins = tl.arange(0, 256)
    prefix = tl.zeros((), dtype=tl.uint32)
    fixed = tl.zeros((), dtype=tl.uint32)
    need = K
    count = K
    for p in tl.static_range(4):
        hist = tl.zeros((256,), dtype=tl.int32)
        for c in range(0, n, B):
            u, ok, val = _items(row, cand, c, n, vis, G, BLOCKS, CANDIDATES, B)
            match = ok & ((u & fixed) == prefix)
            hist += tl.histogram(((u >> (24 - 8 * p)) & 0xFF).to(tl.int32), 256, mask=match)
        if p == 0:
            need = tl.minimum(need, tl.sum(hist, 0))
            count = need
        at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
        digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
        need -= tl.sum(tl.where(bins > digit, hist, 0), 0)
        prefix = prefix | (digit.to(tl.uint32) << (24 - 8 * p))
        fixed = fixed | (tl.full((), 0xFF, tl.uint32) << (24 - 8 * p))
    written = 0
    equal_seen = 0
    for c in range(0, n, B):
        u, ok, val = _items(row, cand, c, n, vis, G, BLOCKS, CANDIDATES, B)
        eq = (ok & (u == prefix)).to(tl.int32)
        take = (ok & (u > prefix)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
        t = take.to(tl.int32)
        tl.store(OUT + r * so + written + tl.cumsum(t, 0) - t, val.to(tl.int32), mask=take)
        written += tl.sum(t, 0)
        equal_seen += tl.sum(eq, 0)
    tl.store(OUTN + r, count)


@triton.jit
def _digit(hist, need, bins):
    """The largest digit whose items and the larger digits' reach ``need``, and what is left to take at it."""

    at_or_above = tl.sum(hist, 0) - tl.cumsum(hist, 0) + hist
    digit = tl.max(tl.where(at_or_above >= need, bins, 0), 0)
    return digit, need - tl.sum(tl.where(bins > digit, hist, 0), 0)


@triton.jit
def _split_hist(S, POS, CAND, CANDN, HIST, COUNTS, row0, ratio, ss, sc, P, G: tl.constexpr, BLOCKS: tl.constexpr,
                CANDIDATES: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr, NB: tl.constexpr,
                LANES: tl.constexpr):
    """Program (r, p): the first digit's histogram of items [p STEPS B, (p + 1) STEPS B) added into row r's."""

    r = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1)
    vis, n = _extent(POS, CANDN, r, row0, ratio, G, BLOCKS, CANDIDATES, LANES)
    tl.store(COUNTS + (r * 3 + 1) * P + p, 0)       # the counts _split_refine adds to, left by the last selection
    tl.store(COUNTS + (r * 3 + 2) * P + p, 0)
    c0 = p * (STEPS * B)
    if c0 < n:
        hist = tl.zeros((NB,), dtype=tl.int32)
        for c in range(c0, tl.minimum(n, c0 + STEPS * B), B):
            u, ok, _ = _items(S + r * ss, CAND + r * sc, c, n, vis, G, BLOCKS, CANDIDATES, B)
            hist += tl.histogram((u >> 21).to(tl.int32), NB, mask=ok)
        tl.atomic_add(HIST + r * NB + tl.arange(0, NB), hist, mask=hist > 0)


@triton.jit
def _split_gather(S, POS, CAND, CANDN, HIST, BN, BKEY, BITEM, COUNTS, row0, ratio, K, ss, sc, sb, P,
                  G: tl.constexpr, BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr, B: tl.constexpr,
                  STEPS: tl.constexpr, NB: tl.constexpr, LANES: tl.constexpr):
    """Program (r, p): its items past the first digit's bin counted, those in the bin appended to row r's list."""

    r = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1)
    vis, n = _extent(POS, CANDN, r, row0, ratio, G, BLOCKS, CANDIDATES, LANES)
    hist = tl.load(HIST + r * NB + tl.arange(0, NB))
    digit = _digit(hist, tl.minimum(K, tl.sum(hist, 0)), tl.arange(0, NB))[0]
    above = 0
    c0 = p * (STEPS * B)
    for c in range(c0, tl.minimum(n, c0 + STEPS * B), B):
        u, ok, _ = _items(S + r * ss, CAND + r * sc, c, n, vis, G, BLOCKS, CANDIDATES, B)
        top = (u >> 21).to(tl.int32)
        above += tl.sum((ok & (top > digit)).to(tl.int32), 0)
        inbin = (ok & (top == digit)).to(tl.int32)
        k = tl.sum(inbin, 0)
        if k > 0:
            at = tl.atomic_add(BN + r, k) + tl.cumsum(inbin, 0) - inbin
            tl.store(BKEY + r * sb + at, u.to(tl.int32, bitcast=True), mask=inbin == 1)
            tl.store(BITEM + r * sb + at, (c + tl.arange(0, B)).to(tl.int32), mask=inbin == 1)
    tl.store(COUNTS + r * 3 * P + p, above)


@triton.jit
def _refine(KEYS, m, prefix, fixed, need, SHIFT: tl.constexpr, WIDTH: tl.constexpr, B: tl.constexpr,
            NB: tl.constexpr):
    """The next digit (WIDTH bits at SHIFT) of the K-th best key among the m keys matching ``prefix``."""

    j = tl.arange(0, B)
    h = tl.zeros((NB,), dtype=tl.int32)
    for c in range(0, m, B):
        u = tl.load(KEYS + c + j, mask=c + j < m, other=0).to(tl.uint32, bitcast=True)
        match = (c + j < m) & ((u & fixed) == prefix)
        h += tl.histogram(((u >> SHIFT) & ((1 << WIDTH) - 1)).to(tl.int32), NB, mask=match)
    d, need = _digit(h, need, tl.arange(0, NB))
    return prefix | (d.to(tl.uint32) << SHIFT), fixed | (tl.full((), (1 << WIDTH) - 1, tl.uint32) << SHIFT), need


@triton.jit
def _split_refine(OUTN, HIST, BN, BKEY, BITEM, COUNTS, THR, K, sb, P, B: tl.constexpr, SPAN: tl.constexpr,
                  NB: tl.constexpr):
    """Program r: the K-th best key from the first digit's bin, then each program's items above and at it; at worst
    (the whole row in that bin) three passes over it, where ``_select`` makes five."""

    r = tl.program_id(0).to(tl.int64)
    bins = tl.arange(0, NB)
    hist = tl.load(HIST + r * NB + bins)
    count = tl.minimum(K, tl.sum(hist, 0))
    digit, need = _digit(hist, count, bins)
    m = tl.load(BN + r)
    prefix = digit.to(tl.uint32) << 21
    fixed = tl.full((), 0x7FF, tl.uint32) << 21
    j = tl.arange(0, B)
    prefix, fixed, need = _refine(BKEY + r * sb, m, prefix, fixed, need, 10, 11, B, NB)   # the next 11 bits
    prefix, fixed, need = _refine(BKEY + r * sb, m, prefix, fixed, need, 0, 10, B, NB)    # the last 10
    for c in range(0, m, B):
        ok = c + j < m
        u = tl.load(BKEY + r * sb + c + j, mask=ok, other=0).to(tl.uint32, bitcast=True)
        at = COUNTS + r * 3 * P + tl.load(BITEM + r * sb + c + j, mask=ok, other=0) // SPAN
        tl.atomic_add(at + P, 1, mask=ok & (u > prefix))
        tl.atomic_add(at + 2 * P, 1, mask=ok & (u == prefix))
    tl.store(THR + 2 * r, prefix.to(tl.int32, bitcast=True))
    tl.store(THR + 2 * r + 1, need)
    tl.store(OUTN + r, count)
    tl.store(HIST + r * NB + bins, tl.zeros((NB,), dtype=tl.int32))     # zero for the next selection
    tl.store(BN + r, 0)


@triton.jit
def _split_write(S, OUT, POS, CAND, CANDN, COUNTS, THR, row0, ratio, ss, so, sc, P, G: tl.constexpr,
                 BLOCKS: tl.constexpr, CANDIDATES: tl.constexpr, B: tl.constexpr, STEPS: tl.constexpr,
                 PP: tl.constexpr, LANES: tl.constexpr):
    """Program (r, p): its items above the K-th best key and its share of the lowest-numbered ties, after the
    items the earlier programs write."""

    r = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1)
    vis, n = _extent(POS, CANDN, r, row0, ratio, G, BLOCKS, CANDIDATES, LANES)
    c0 = p * (STEPS * B)
    if c0 < n:
        q = tl.arange(0, PP)
        before = q < p
        counts = COUNTS + r * 3 * P + q
        equal_seen = tl.sum(tl.load(counts + 2 * P, mask=before, other=0), 0)
        thr = tl.load(THR + 2 * r).to(tl.uint32, bitcast=True)
        need = tl.load(THR + 2 * r + 1)
        written = tl.sum(tl.load(counts, mask=before, other=0) + tl.load(counts + P, mask=before, other=0), 0) + \
            tl.minimum(equal_seen, need)
        for c in range(c0, tl.minimum(n, c0 + STEPS * B), B):
            u, ok, val = _items(S + r * ss, CAND + r * sc, c, n, vis, G, BLOCKS, CANDIDATES, B)
            eq = (ok & (u == thr)).to(tl.int32)
            take = (ok & (u > thr)) | ((eq == 1) & (tl.cumsum(eq, 0) - eq + equal_seen < need))
            t = take.to(tl.int32)
            tl.store(OUT + r * so + written + tl.cumsum(t, 0) - t, val.to(tl.int32), mask=take)
            written += tl.sum(t, 0)
            equal_seen += tl.sum(eq, 0)


def _check(s: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor) -> int:
    R = s.shape[0]
    if s.dtype != torch.float32 or s.stride(1) != 1:
        raise ValueError("select: fp32 scores with unit-stride columns")
    if out.dtype != torch.int32 or out_n.dtype != torch.int32 or out.shape[0] < R or out_n.shape[0] < R \
            or out.stride(1) != 1:
        raise ValueError(f"select: int32 outputs of at least {R} rows")
    return R


def _split(s: torch.Tensor, ratio: int, pos: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor, row0: int,
           cand: torch.Tensor | None, cand_n: torch.Tensor | None, block: int, work: dict[str, torch.Tensor],
           blocks: bool, lanes: bool) -> None:
    """``_select``'s lists from a decode window's few rows, each row spread over programs: one histogram pass of
    the first digit, the rest resolved on that digit's bin, then each program writes its share in item order."""

    R, K = s.shape[0], out.shape[1]
    B = PICK // 2 if blocks else PICK
    items = triton.cdiv(s.shape[1], block) if blocks else s.shape[1] if cand is None else cand.shape[1] * block
    hist, bn, bkey, bitem, counts, thr = (work[k] for k in ("hist", "bn", "bkey", "bitem", "counts", "thr"))
    P = triton.cdiv(items, STEPS * B)
    if hist.shape[0] < R or bkey.shape[1] < min(items, s.shape[1]) or counts.shape[2] < P \
            or bkey.stride(0) != bitem.stride(0):
        raise ValueError(f"select: split scratch for {hist.shape[0]} rows, {bkey.shape[1]} entries and "
                         f"{counts.shape[2]} programs, not {R} rows of {items} items")
    cand_, cand_n_, sc = (cand, cand_n, cand.stride(0)) if cand is not None else (s, s, 0)
    mode = {"G": block, "BLOCKS": blocks, "CANDIDATES": cand is not None, "B": B, "STEPS": STEPS, "LANES": lanes}
    Pw = counts.shape[2]
    _split_hist[(R, P)](s, pos, cand_, cand_n_, hist, counts, row0, ratio, s.stride(0), sc, Pw, NB=SPLIT_BINS, **mode,
                        num_warps=4)
    _split_gather[(R, P)](s, pos, cand_, cand_n_, hist, bn, bkey, bitem, counts, row0, ratio, K, s.stride(0), sc,
                          bkey.stride(0), Pw, NB=SPLIT_BINS, **mode, num_warps=4)
    _split_refine[(R,)](out_n, hist, bn, bkey, bitem, counts, thr, K, bkey.stride(0), Pw, B=PICK, SPAN=STEPS * B,
                        NB=SPLIT_BINS, num_warps=4)
    _split_write[(R, P)](s, out, pos, cand_, cand_n_, counts, thr, row0, ratio, s.stride(0), out.stride(0), sc, Pw,
                         PP=triton.next_power_of_2(Pw), **mode, num_warps=4)


def candidates(s: torch.Tensor, ratio: int, pos: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor, *,
               block: int = 8, row0: int = 0, work: dict[str, torch.Tensor] | None = None,
               tables: LaneArgs | None = None) -> torch.Tensor:
    """Each row's best blocks of ``block`` entries by their best score (M:583-610), ascending into out; ``work``:
    a decode window's split selection (the same lists); ``tables``: row r at rpos[r]."""

    R = _check(s, out, out_n)
    _check_tables(tables, R, row0)
    lanes, pos = tables is not None, tables.rpos if tables is not None else pos
    if R and work is not None:
        _split(s, ratio, pos, out, out_n, row0, None, None, block, work, True, lanes)
    elif R:
        _select[(R,)](s, out, out_n, pos, s, s, row0, ratio, out.shape[1], s.stride(0), out.stride(0), 0,
                      G=block, BLOCKS=True, CANDIDATES=False, B=PICK // 2, LANES=lanes, num_warps=4)
    return out


def topk(s: torch.Tensor, ratio: int, pos: torch.Tensor, out: torch.Tensor, out_n: torch.Tensor, *,
         row0: int = 0, cand: torch.Tensor | None = None, cand_n: torch.Tensor | None = None,
         block: int = 8, work: dict[str, torch.Tensor] | None = None, tables: LaneArgs | None = None) -> torch.Tensor:
    """Each row's best visible entries (M:577-580), ascending into out; with ``cand``, of its candidates; ``work``:
    a decode window's split selection (the same lists); ``tables``: row r at rpos[r]."""

    R = _check(s, out, out_n)
    _check_tables(tables, R, row0)
    lanes, pos = tables is not None, tables.rpos if tables is not None else pos
    if R and work is not None:
        _split(s, ratio, pos, out, out_n, row0, cand, cand_n, block, work, False, lanes)
    elif R:
        _select[(R,)](s, out, out_n, pos, cand if cand is not None else s, cand_n if cand is not None else s, row0,
                      ratio, out.shape[1], s.stride(0), out.stride(0), cand.stride(0) if cand is not None else 0,
                      G=block, BLOCKS=False, CANDIDATES=cand is not None, B=PICK, LANES=lanes, num_warps=4)
    return out


def select(cfg: Config, qI: torch.Tensor, wI: torch.Tensor, keys: torch.Tensor, ratio: int, pos_dev: torch.Tensor,
           buf: torch.Tensor, lists: torch.Tensor, list_n: torch.Tensor, *, pos: int | None = None,
           source: tuple[torch.Tensor, torch.Tensor] | None = None,
           within: tuple[torch.Tensor, torch.Tensor] | None = None,
           work: dict[str, torch.Tensor] | None = None, tables: LaneArgs | None = None) -> torch.Tensor:
    """Scores, candidate blocks and top-k lists for rows at pos_dev + r; ``pos`` marks a prompt chunk; ``work``:
    a decode window's split-selection scratch; ``tables``: a shared decode forward's rows over lane 0's ``keys``."""

    R = qI.shape[0]
    if tables is not None and pos is not None:
        raise ValueError("select: lane tables are for decode rows, not a prompt chunk")
    step = R if pos is None else min(score_rows((pos + R) // ratio), buf.shape[0])
    blk = cfg.candidate_block_size
    for a in range(0, R, max(step, 1)):
        n = min(step, R - a)
        part = buf[:n]
        cand = (within[0][a:a + n], within[1][a:a + n]) if within is not None else (None, None)
        scores(qI[a:a + n], wI[a:a + n], keys, ratio, pos_dev, part, row0=a, cand=cand[0], cand_n=cand[1],
               block=blk, tables=tables)
        if source is not None:
            candidates(part, ratio, pos_dev, source[0][a:a + n], source[1][a:a + n], block=blk, row0=a, work=work,
                       tables=tables)
        topk(part, ratio, pos_dev, lists[a:a + n], list_n[a:a + n], row0=a, cand=cand[0], cand_n=cand[1], block=blk,
             work=work, tables=tables)
    return lists
