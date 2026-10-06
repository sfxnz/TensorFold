"""Keyed sampling over the ranks' vocabulary halves: GLM's rule for the target, a device draw for drafts."""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.comm import fast_gather
from tensorfold.engine.exact_sampling import MARGIN, Sampling, _mix
from tensorfold.families.glm5_next.cuda.decode import sample_rows

from .weights import Weights

DRAFT_TOP_K = 1024      # a draft's largest top_k, and its top_k when the request's is off: never whole shards
CANDIDATES = DRAFT_TOP_K + MARGIN   # a rank's candidates for a keyed draft, so values tied at the cut resolve by id
NO_CUT = 2.0            # the top_p of a request without one: no cumulative mass reaches it


def target_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int],
                sampling: Sampling | None) -> tuple[list[int], float]:
    """Rows of this rank's head logits at their absolute positions -> (tokens, host seconds), same on every rank."""

    start = time.perf_counter()
    tokens = sample_rows(w, logits, positions, sampling)
    return tokens, time.perf_counter() - start


class Draws:
    """One rank's device scratch and request knobs for draft draws: the host writes the knobs, graphs read them."""

    def __init__(self, vocab: int, world: int, device: torch.device | str = "cuda") -> None:
        dev, i64 = torch.device(device), torch.int64
        n = self.width = min(CANDIDATES, vocab // world)
        self.world = world
        self.cols = torch.zeros((n,), dtype=i64, device=dev)
        self.vals = torch.empty((n,), dtype=torch.float32, device=dev)
        self.keys = torch.zeros((n,), dtype=i64, device=dev)
        self.best = torch.zeros((1,), dtype=i64, device=dev)
        self.got = torch.zeros((world * n,), dtype=i64, device=dev)
        self.got_best = torch.zeros((world,), dtype=i64, device=dev)
        self.order = torch.empty((world * n,), dtype=i64, device=dev)
        self.at = torch.empty((world * n,), dtype=i64, device=dev)
        self.knobs = torch.zeros((5,), dtype=i64, device=dev)    # key, top_k, float64 bits of t, top_p, ln min_p
        self.set_for: tuple | None = None

    def set(self, sampling: Sampling | None) -> bool:
        """Write ``sampling``'s knobs when they changed -> whether its drafts are keyed draws (else greedy)."""

        if sampling is None or sampling.temperature <= 0:
            return False
        want = (sampling.seed, sampling.temperature, sampling.top_k, sampling.top_p, sampling.min_p)
        if want != self.set_for:
            with np.errstate(over="ignore"):         # ``exact_sampling.uniform``'s first mix: the seed's alone
                key = _mix(np.uint64(sampling.seed & 0xFFFFFFFFFFFFFFFF) + np.uint64(0x9E3779B97F4A7C15))
            k = min(int(sampling.top_k) or DRAFT_TOP_K, DRAFT_TOP_K, self.world * self.width)
            top_p = sampling.top_p if 0.0 < sampling.top_p < 1.0 else NO_CUT
            floats = np.array([max(float(sampling.temperature), 1e-6), top_p, sampling.min_log], dtype=np.float64)
            knobs = np.concatenate([np.array([key], dtype=np.uint64).view(np.int64), np.array([k], dtype=np.int64),
                                    floats.view(np.int64)])
            self.knobs.copy_(torch.from_numpy(knobs))
            self.set_for = want
        return True


@triton.jit
def _keys(ROW, COLS, H, ME, KEYS, offset, n, RK: tl.constexpr, BIAS: tl.constexpr, B: tl.constexpr):
    """Each candidate column's int64 key, ascending in the order (-value, id) as ``sampling._order``; with BIAS the
    value first gains the Markov bias ``H[col] . ME`` as one fp32 sum (sparse Markov)."""

    at = tl.program_id(0) * B + tl.arange(0, B)
    ok = at < n
    col = tl.load(COLS + at, ok, other=0)
    v = tl.load(ROW + col, ok, other=0.0)
    if BIAS:
        r = tl.arange(0, RK)
        h = tl.load(H + col[:, None] * RK + r[None, :], ok[:, None], other=0.0).to(tl.float32)
        v = v + tl.sum(h * tl.load(ME + r).to(tl.float32)[None, :], axis=1)
    bits = tl.where(v == 0.0, 0.0, v).to(tl.int32, bitcast=True).to(tl.int64)          # -0 ties +0
    tl.store(KEYS + at, ((-1 - (bits ^ ((bits >> 31) & 0x7FFFFFFF))) << 32) + col + offset, ok)


@triton.jit
def _mix64(x):
    """``exact_sampling._mix`` (the splitmix64 finalizer) on uint64."""

    x = x ^ (x >> 30)
    x = x * 0xBF58476D1CE4E5B9
    x = x ^ (x >> 27)
    x = x * 0x94D049BB133111EB
    return x ^ (x >> 31)


@triton.jit
def _value(key):
    """A candidate key's value back as float64 (the key's high word inverts ``_keys``' order map)."""

    t = -1 - (key >> 32)
    return (t ^ ((t >> 31) & 0x7FFFFFFF)).to(tl.int32).to(tl.float32, bitcast=True).to(tl.float64)


@triton.jit
def _draw(KEYS, n, KNOBS, POS, OUT, ahead, KEYED: tl.constexpr, B: tl.constexpr, G: tl.constexpr):
    """OUT[0] = the draw at position POS[0] + ahead from every rank's candidate keys: greedy, the least key; keyed,
    ``exact_sampling.choose_rows`` over the keys sorted ascending: the top_k (at most B) give top_p's cumulative mass,
    and the keyed Gumbel max runs over the kept prefix, G candidates at a time."""

    if KEYED:
        k = tl.minimum(tl.load(KNOBS + 1), n)
        temp = tl.load(KNOBS + 2).to(tl.float64, bitcast=True)
        at = tl.arange(0, B)
        live = at < k
        scaled = tl.where(live, _value(tl.load(KEYS + at, live, other=0)) / temp, -float("inf"))
        top = tl.max(scaled, 0)
        e = tl.where(live, tl.exp(scaled - top), 0.0)
        cum = tl.cumsum(e / tl.sum(e, 0), 0)
        keep = tl.sum((live & (cum < tl.load(KNOBS + 3).to(tl.float64, bitcast=True))).to(tl.int64), 0) + 1
        last = tl.minimum(keep, k)                          # the top_p cut ends the prefix; min_p cuts inside it
        floor = top + tl.load(KNOBS + 4).to(tl.float64, bitcast=True)
        pos = (tl.load(POS).to(tl.int64) + ahead).to(tl.uint64, bitcast=True)
        key = _mix64(tl.load(KNOBS).to(tl.uint64, bitcast=True) ^ (pos * 0xD1B54A32D192ED03))
        best = tl.full([], float("-inf"), tl.float64)
        tok = tl.full([], 0, tl.int64)
        for start in range(0, last, G):
            g = start + tl.arange(0, G)
            c = tl.load(KEYS + g, g < last, other=0)
            v = _value(c) / temp
            ids = c & 0xFFFFFFFF
            x = _mix64(key ^ ids.to(tl.uint64, bitcast=True))
            u = (x >> 11).to(tl.float64) * 1.1102230246251565e-16 + 5.551115123125783e-17
            score = tl.where((g < last) & (v >= floor), v - tl.log(-tl.log(u)), -float("inf"))
            m = tl.max(score, 0)
            pick = tl.sum(tl.where(tl.arange(0, G) == tl.argmax(score, 0), ids, 0), 0)
            tok = tl.where(m > best, pick, tok)                 # an earlier chunk keeps a tie, as argmax would
            best = tl.maximum(m, best)
    else:
        at = tl.arange(0, G)
        tok = tl.min(tl.load(KEYS + at, at < n, other=0x7FFFFFFFFFFFFFFF), 0) & 0xFFFFFFFF
    tl.store(OUT, tok.to(tl.int32))


def draft(w: Weights, d: Draws, row: torch.Tensor, pos: torch.Tensor, ahead: int, keyed: bool, out: torch.Tensor,
          sparse: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
    """The draft from this rank's fp32 logits ``row`` [V / world] at position ``pos[0] + ahead`` into ``out[0]``, alike
    on every rank; ``sparse`` (head bf16 [V / world, rank], me [rank]) adds the Markov bias to the candidates only."""

    n = d.width
    if keyed or sparse is not None:
        torch.topk(row, n, sorted=False, out=(d.vals, d.cols))
    else:
        torch.argmax(row, dim=0, keepdim=True, out=d.cols[:1])      # the first of equal maxima: the lowest id
        n = 1
    head, me = sparse if sparse is not None else (row, row)         # read only with the bias
    _keys[(triton.cdiv(n, 64),)](row, d.cols, head, me, d.keys, w.vocab_offset, n, RK=me.shape[0] if sparse else 1,
                                 BIAS=sparse is not None, B=64, num_warps=4)
    if keyed:
        got = d.keys
        if w.world > 1:
            fast_gather(w.comm, d.keys, d.got)
            got = d.got
        torch.sort(got, out=(d.order, d.at))
        _draw[(1,)](d.order, got.numel(), d.knobs, pos, out, ahead, KEYED=True, B=DRAFT_TOP_K, G=64, num_warps=8)
        return
    best = d.keys[:1] if n == 1 else torch.amin(d.keys, dim=0, keepdim=True, out=d.best)
    if w.world > 1:
        fast_gather(w.comm, best, d.got_best)
        best = d.got_best
    _draw[(1,)](best, best.numel(), d.knobs, pos, out, ahead, KEYED=False, B=DRAFT_TOP_K, G=64, num_warps=1)
