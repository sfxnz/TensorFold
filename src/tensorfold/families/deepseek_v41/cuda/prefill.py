"""A prompt on one rank in chunks cut at the keep point, the decoder on the rows that matter, ending in the bits of
the whole prompt at once."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import wait

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling

from ..config import Config
from ..engram_hash import rank_columns
from . import dspark, sample, snapshot
from . import forward as F
from .buffers import Buffers
from .snapshot import Snapshot


def chunks(begin: int, n: int, keep_at: int | None, rows: int) -> list[tuple[int, int]]:
    """``[begin, K)`` then ``[K, n)`` in pieces of at most ``rows``, K the keep point (n without one); none empty."""

    k = n if keep_at is None else keep_at
    return [(a, min(a + rows, end)) for lo, end in ((begin, k), (k, n)) for a in range(lo, end, rows)]


def decoder_rows(cfg: Config, layers: Sequence[int], begin: int, anchor: int, n: int, a: int,
                 z: int) -> tuple[dict[int, tuple[int, int]], int]:
    """Chunk [a, z) of an ``n``-token prompt from ``begin``: each decoder layer's (first block row, first window KV
    row) and the first DSpark tap row, relative to a, that leave the head row and the state at ``anchor`` and n exact.

    Past the last KV source (CED) older rows reach a row only through its window: the last layer runs the head row,
    each layer before it the rows the next one's window KV needs, and every layer keeps the ring rows before anchor.
    """

    src = max(cfg.kv_source_layer_ids, default=-1)
    dec = [L for L in layers if L >= src]
    if src not in layers or any(cfg.roles[L].engram for L in dec):
        return {}, 0
    win = cfg.sliding_window

    def rel(p: int) -> int:
        return min(max(p, begin, a), z) - a

    rows, full = {}, n - 1
    for L in reversed(dec):
        kv = min(full - (win - 1), anchor - win)
        rows[L] = (rel(full), rel(kv))
        full = kv
    return rows, rel(anchor - win)


def _stage_ids(b: Buffers, tokens: Sequence[int]) -> None:
    """A chunk's token ids through the pinned twin, once its previous copy has finished."""

    R = len(tokens)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = tokens
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()


class _Rows:
    """The prompt's Engram rows, each chunk's read on the reader's thread into pinned half ``i % 2``."""

    def __init__(self, e, prompt: Sequence[int], begin: int, spans: list[tuple[int, int]]) -> None:
        w, st, b = e.w, e.st, e.pbuf
        ids = e.hasher.ids(np.asarray(st.history, dtype=np.int64), np.asarray(prompt[begin:], dtype=np.int64))
        ids = rank_columns(ids, w.rank, w.world)
        starts = np.asarray(e.reader.layout.starts[:ids.shape[1]], dtype=np.int64)
        self.flat = (ids + starts[None, :, None]).reshape(len(ids), -1)     # [n, layers * columns]
        self.reader, self.b, self.begin, self.spans = e.reader, b, begin, spans
        self.advised, self.pending = 0, None

    def _ids(self, i: int) -> np.ndarray:
        a, z = self.spans[i]
        return self.flat[a - self.begin:z - self.begin].reshape(-1)

    def advise(self, upto: int) -> None:
        """WILLNEED on the pages of chunks up to ``upto`` (exclusive) not advised yet."""

        for i in range(self.advised, min(upto, len(self.spans))):
            self.reader.advise(self._ids(i))
        self.advised = max(self.advised, upto)

    def read(self, i: int) -> None:
        """Start chunk i's rows into half i % 2 once that half's last copy to the device has finished."""

        b, half = self.b, i % 2
        b.eraw_done[half].synchronize()
        ids = self._ids(i)
        rec = b.eraw_host[half].numpy().reshape(-1, b.eraw_host[half].shape[-1])[:ids.size]
        self.pending = self.reader.gather_async(ids, rec[:, :self.reader.wrow], rec[:, self.reader.wrow:])

    def land(self, i: int) -> None:
        """Chunk i's rows, once read, into ``eraw`` on the compute stream (after the chunk before it computes)."""

        b, half = self.b, i % 2
        R = self.spans[i][1] - self.spans[i][0]
        fut, self.pending = self.pending, None
        fut.result()
        b.eraw[:R].copy_(b.eraw_host[half][:R], non_blocking=True)
        b.eraw_done[half].record()

    def drain(self) -> None:
        """Wait for a read still writing a pinned half (an exception left it running)."""

        if self.pending is not None:
            wait([self.pending])


@torch.no_grad()
def prefill(e, prompt: Sequence[int], sampling: Sampling | None, resume: Snapshot | None = None,
            keep_at: int | None = None, keep: Callable[[Snapshot], object] | None = None,
            space: torch.Tensor | None = None) -> int:
    """Commit ``prompt`` from 0 or ``resume`` and draw its first token; ``keep`` takes the ``keep_at`` snapshot."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.pbuf
    n = len(prompt)
    if n > st.capacity:
        raise ValueError(f"a {n}-token prompt in a state of {st.capacity} positions")
    begin = 0 if resume is None else len(resume.ids)
    if resume is not None and (begin >= n or list(prompt[:begin]) != resume.ids):
        raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
    if keep_at is not None and (keep is None or not max(1, begin) <= keep_at <= n):
        raise ValueError(f"a kept prefix needs a callback and a point in {max(1, begin)}..{n}, not {keep_at}")
    if resume is None:
        st.reset()
    else:
        snapshot.restore(e, resume)
    spans = chunks(begin, n, keep_at, b.rows)
    anchor = min(keep_at, n - 1) if keep_at is not None and keep_at > begin else n - 1
    layers = [lw.index for lw in w.layers]
    kept = resume if keep_at == begin else None
    rows = _Rows(e, prompt, begin, spans) if w.engram else None
    logits = None
    try:
        if rows is not None:
            rows.advise(2)
            rows.read(0)
        for i, (a, z) in enumerate(spans):
            R = z - a
            _stage_ids(b, prompt[a:z])
            if rows is not None:
                rows.land(i)
                rows.advise(i + 3)
                if i + 1 < len(spans):
                    rows.read(i + 1)
            plan, taps = decoder_rows(w.cfg, layers, begin, anchor, n, a, z)
            logits = F.compute(w, st, b, R, prompt=True, head_rows=1 if i == len(spans) - 1 else 0, rows=plan,
                               taps_from=taps)
            F.commit(w, st, b, R, R)
            if w.dspark is not None and taps < R:
                dspark.absorb(e, b, min(R - taps, b.taps.shape[0]), prompt=True)
            if z == keep_at:
                kept = snapshot.take(e, prompt[:keep_at], space)
    finally:
        if rows is not None:
            rows.drain()
    if kept is not None:
        keep(kept)
    (first,), _ = sample.target_rows(w, logits, [n], sampling)
    return first
