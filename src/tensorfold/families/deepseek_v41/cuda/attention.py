"""One attention sublayer (M:765-789); an index layer always scores its own KV source's index keys."""

from __future__ import annotations

import torch

from . import PREFILL_ROWS, attn_kernel, compressor, indexer, mx8, norms, quant, rope
from .buffers import Buffers, State
from .lanes import Lanes
from .weights import LayerW, Weights

_ROWS: dict[torch.device, torch.Tensor] = {}


def _anchors(pos: torch.Tensor, rows: int) -> torch.Tensor:
    """int32 [rows]: pos + r, each row's last window position, computed on the device so a graph replays it."""

    ar = _ROWS.get(pos.device)
    if ar is None or ar.numel() < rows:
        ar = _ROWS[pos.device] = torch.arange(max(rows, PREFILL_ROWS), dtype=torch.int32, device=pos.device)
    return pos + ar[:rows]


def _index(w: Weights, lw: LayerW, xa: torch.Tensor, qr: torch.Tensor, state: State | Lanes, buf: Buffers,
           start: int, rows: int, table: torch.Tensor, at: torch.Tensor, prompt: bool) -> None:
    """M:550-580: lists of rows start.. (at positions ``at``) into ``buf.lists``; candidates as in ``select``."""

    role, idx = lw.role, lw.attn.idx
    qI, wI = buf.qI[:rows - start], buf.wI[:rows - start]
    indexer.index_q(idx, qr, table, at, qI, prompt=prompt)
    indexer.index_weights(idx, xa, wI)
    cand = (buf.cand[start:rows], buf.cand_n[start:rows])
    keys, tables = state.index_k[role.kv_src], None
    if isinstance(state, Lanes):
        tables = indexer.LaneArgs(buf.lane[:rows], buf.rpos[:rows], buf.segs, buf.nseg, keys.stride(0))
        keys = keys[0]
    indexer.select(w.cfg, qI, wI, keys, role.ratio, at[:1], buf.scores, buf.lists[start:rows], buf.list_n[start:rows],
                   pos=state.pos + start if prompt else None, source=cand if role.candidate_source else None,
                   within=cand if role.uses_candidates else None, work=None if prompt else buf.split, tables=tables)


def attention(lw: LayerW, w: Weights, state: State | Lanes, buf: Buffers, rows: int, prompt: bool, start: int = 0,
              kv_from: int = 0, first: int = 0) -> torch.Tensor | None:
    """``buf.X`` rows start.. of ``rows`` -> the rank's fp32 share ``buf.part[start:rows]`` after ``wo_b`` (None when
    start is ``rows``); rows kv_from.. write their window KV, and a KV source pools rows first.. (the rows before
    pooled already). With ``Lanes``, a shared decode forward: row r is lane ``buf.lane[r]``'s at ``buf.rpos[r]``."""

    cfg, a, role, L = w.cfg, lw.attn, lw.role, lw.index
    lanes = isinstance(state, Lanes)
    if role.mode == "dspark" or not 0 < rows <= buf.rows or not 0 <= first <= kv_from <= start <= rows or \
            (lanes and (prompt or start or getattr(buf, "lane", None) is None)):
        raise ValueError(f"attention: layer {L} ({role.mode}) on rows {first}/{kv_from}/{start}..{rows} of a "
                         f"{buf.rows}-row buffer{' over lanes' if lanes else ''}")
    eps, table, pos = cfg.rms_norm_eps, w.rope[role.rope], state.pos_dev
    source = bool(role.ratio) and role.kv_src == L
    anchors = buf.rpos[:rows] if lanes else _anchors(pos, rows)
    lo = first if source else kv_from
    norms.collapse_norm(buf.X[lo:rows].view(rows - lo, -1), buf.pre_in[lo:rows], lw.attn_norm, buf.xn[lo:rows], eps)
    if kv_from < rows:
        qakv = buf.qakv[kv_from:rows]
        mx8.mm(a.wqa_kv, buf.xn[kv_from:rows], qakv, prompt=prompt)
        quant.norm_rope_fp8(qakv[:, cfg.q_lora_rank:], a.kv_norm, eps, anchors[kv_from:] if lanes or kv_from else pos,
                            table, buf.kvw[L, kv_from:rows])
    if source:
        compressor.compress(lw, buf.xn[first:rows], state, buf, table, eps, first,
                            tables=(buf.lane[:rows], buf.rpos[:rows], buf.seg0[:rows]) if lanes else None)
    if start == rows:
        return None
    n, at = rows - start, anchors[start:] if lanes or start else pos
    xa, qr, q = buf.xn[start:rows], buf.qr[start:rows], buf.q[start:rows]
    norms.rmsnorm(buf.qakv[start:rows, :cfg.q_lora_rank], a.q_norm, eps, qr)
    mx8.mm(a.wq_b, qr, q.view(n, -1), prompt=prompt)
    rope.apply(q, at, table)
    extra = lists = counts = tables = None
    if role.ratio:
        if role.idx_src == L:
            _index(w, lw, xa, qr, state, buf, start, rows, table, at, prompt)
        extra, lists, counts = state.comp[role.kv_src], buf.lists[start:rows], buf.list_n[start:rows]
    if lanes:                                   # lane 0's slices, each row's lane a stride away
        tables = attn_kernel.LaneArgs(buf.lane, buf.seg0, pos, state.rings.stride(0),
                                      extra.stride(0) if extra is not None else 0)
        ring, extra = state.rings[0, L], extra[0] if extra is not None else None
    else:
        ring = state.rings[L]
    o = attn_kernel.attention(q, ring, buf.kvw[L], pos, anchors[start:], extra, lists, counts, a.sink,
                              buf.o[start:rows], prompt=prompt, part=buf.attn_part, lanes=tables)
    rope.apply(o, at, table, inverse=True)
    u = mx8.grouped(a.wo_a, o.view(n, -1), buf.u[start:rows], prompt=prompt)
    return mx8.mm(a.wo_b, u, buf.part[start:rows], f32=True, prompt=prompt)
