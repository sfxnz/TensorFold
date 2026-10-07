"""A prompt on one rank in chunks cut at a keep point before its last row (one at the last row is kept from inside
its chunk), the decoder on the rows that matter, ending in the bits of the whole prompt at once; one chunk a step, so
prompts on other states may take turns on the chunk buffers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import wait

import numpy as np
import torch

from tensorfold.engine.exact_sampling import Sampling

from ..config import Config
from ..engram_hash import rank_columns
from . import dspark, engram, sample, snapshot
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


def _absorb(e, b: Buffers, R: int, taps: int, since: int, upto: int) -> None:
    """DSpark absorb of the chunk's tap rows since..upto (committed through upto), the last ring's worth of them."""

    end = R - min(R - taps, b.taps.shape[0])        # tap row t is the chunk's row end + t
    lo = max(end, since, upto - e.st.rings.shape[1])
    if e.w.dspark is not None and lo < upto:
        dspark.absorb(e, b, upto - lo, prompt=True, first=lo - end)


def span_ids(hasher, history: Sequence[int], prompt: Sequence[int], begin: int, a: int, z: int) -> np.ndarray:
    """Engram ids of ``prompt[a:z]`` for a prompt from ``begin`` after ``history``: the n-gram context before a."""

    back = hasher.max_ngram - 1
    before = [*history, *prompt[max(begin, a - back):a]]
    return hasher.ids(np.asarray(before, dtype=np.int64), np.asarray(prompt[a:z], dtype=np.int64))


def _stage_ids(b: Buffers, tokens: Sequence[int]) -> None:
    """A chunk's token ids through the pinned twin, once its previous copy has finished."""

    R = len(tokens)
    b.staged.synchronize()
    b.ids_host[:R].numpy()[:] = tokens
    b.ids[:R].copy_(b.ids_host[:R], non_blocking=True)
    b.staged.record()


class _Rows:
    """The prompt's Engram rows, each chunk's ids hashed when first needed and read on the reader's thread into pinned
    half ``i % 2``."""

    def __init__(self, e, prompt: Sequence[int], begin: int, spans: list[tuple[int, int]]) -> None:
        w = e.w
        self.hasher, self.rank, self.world, self.history = e.hasher, w.rank, w.world, list(e.st.history)
        self.starts = np.asarray(e.reader.layout.starts, dtype=np.int64)
        self.scales = w.engram_scales               # resident: only the weight bytes are read
        self.idx = self.scales is not None          # a read also stages each row's place in the resident scales
        self.reader, self.b, self.prompt, self.begin, self.spans = e.reader, e.pbuf, prompt, begin, spans
        self.advised, self.pending, self.cache = 0, None, {}

    def _ids(self, i: int, idx: bool = False) -> np.ndarray:
        """Chunk i's file rows, or (``idx``) their places in the resident scales."""

        if i not in self.cache:
            a, z = self.spans[i]
            ids = rank_columns(span_ids(self.hasher, self.history, self.prompt, self.begin, a, z), self.rank,
                               self.world)
            flat = (ids + self.starts[None, :ids.shape[1], None]).reshape(-1)
            self.cache[i] = flat, engram.resident(ids, self.scales).reshape(-1) if self.idx else None
        return self.cache[i][int(idx)]

    def advise(self, upto: int) -> None:
        """WILLNEED on the pages of chunks up to ``upto`` (exclusive) not advised yet."""

        for i in range(self.advised, min(upto, len(self.spans))):
            self.reader.advise(self._ids(i), scales=self.scales is None)
        self.advised = max(self.advised, upto)

    def read(self, i: int) -> None:
        """Start chunk i's rows into half i % 2 once that half's last copy to the device has finished."""

        b, half = self.b, i % 2
        b.eraw_done[half].synchronize()
        ids = self._ids(i)
        rec = b.eraw_host[half].numpy().reshape(-1, b.eraw_host[half].shape[-1])[:ids.size]
        if self.idx:
            b.eidx_host[half].numpy().reshape(-1)[:ids.size] = self._ids(i, True)
            self.pending = self.reader.gather_async(ids, rec[:, :self.reader.wrow])
        else:
            self.pending = self.reader.gather_async(ids, rec[:, :self.reader.wrow], rec[:, self.reader.wrow:])

    def land(self, i: int) -> None:
        """Chunk i's rows, once read, into ``eraw`` on the compute stream (after the chunk before it computes)."""

        b, half = self.b, i % 2
        R = self.spans[i][1] - self.spans[i][0]
        fut, self.pending = self.pending, None
        fut.result()
        b.eraw[:R].copy_(b.eraw_host[half][:R], non_blocking=True)
        if self.idx:
            b.eidx[:R].copy_(b.eidx_host[half][:R], non_blocking=True)
        b.eraw_done[half].record()
        self.cache = {k: v for k, v in self.cache.items() if k > i}

    def drain(self) -> None:
        """Wait for a read still writing a pinned half (an exception or a pause left it running)."""

        if self.pending is not None:
            wait([self.pending])


class Fill:
    """A prompt's prefill one chunk a step: ``first`` once done; between steps ``pause`` lends ``pbuf`` to another."""

    @torch.no_grad()
    def __init__(self, e, prompt: Sequence[int], sampling: Sampling | None, resume: Snapshot | None = None,
                 keep_at: int | None = None, keep: Callable[[Snapshot], object] | None = None,
                 space: torch.Tensor | None = None, rows: int | None = None) -> None:
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
        rows = b.rows if rows is None else rows
        if not 0 < rows <= b.rows:
            raise ValueError(f"{rows}-row chunks in {b.rows}-row prompt buffers")
        if resume is None:
            st.reset()
        else:
            snapshot.restore(e, resume)
        self.spans = chunks(begin, n, keep_at if keep_at is None or keep_at < n - 1 else None, rows)
        self.anchor = min(keep_at, n - 1) if keep_at is not None and keep_at > begin else n - 1
        self.layers = [lw.index for lw in w.layers]
        self.kept = resume if keep_at == begin else None
        self.reads = _Rows(e, prompt, begin, self.spans) if w.engram else None
        self.e, self.prompt, self.sampling, self.begin = e, prompt, sampling, begin
        self.keep_at, self.keep, self.space = keep_at, keep, space
        self.i, self.first = 0, None

    @property
    def rows_left(self) -> int:
        """Prompt rows not yet committed."""

        return len(self.prompt) - self.spans[self.i][0] if self.i < len(self.spans) else 0

    @torch.no_grad()
    def step(self) -> bool:
        """Run the next chunk; True once the prompt is committed, ``keep`` has its snapshot and ``first`` is drawn."""

        e, i, reads = self.e, self.i, self.reads
        w, st, b = e.w, e.st, e.pbuf
        n, (a, z), last = len(self.prompt), self.spans[i], i == len(self.spans) - 1
        R = z - a
        if reads is not None and reads.pending is None:     # the first chunk, or the first after a pause
            reads.advise(i + 2)
            reads.read(i)
        _stage_ids(b, self.prompt[a:z])
        if reads is not None:
            reads.land(i)
            reads.advise(i + 3)
            if not last:
                reads.read(i + 1)
        plan, taps = decoder_rows(w.cfg, self.layers, self.begin, self.anchor, n, a, z)
        logits = F.compute(w, st, b, R, prompt=True, head_rows=1 if last else 0, rows=plan, taps_from=taps)
        keep_at = self.keep_at
        k = keep_at - a if keep_at is not None and a < keep_at < z else R    # a snapshot inside: at row k
        F.commit(w, st, b, R, k)
        _absorb(e, b, R, taps, 0, k)
        if a + k == keep_at:
            self.kept = snapshot.take(e, self.prompt[:keep_at], self.space)
        if k < R:
            F.commit(w, st, b, R, R, first=k)
            _absorb(e, b, R, taps, k, R)
        self.i += 1
        if not last:
            return False
        if self.kept is not None:
            self.keep(self.kept)
        (self.first,), _ = sample.target_rows(w, logits, [n], self.sampling)
        return True

    def pause(self) -> None:
        """Let another prompt use ``pbuf``'s pinned halves: the read ahead finishes and the next step reads it again."""

        if self.reads is not None:
            self.reads.drain()
            self.reads.pending = None

    def close(self) -> None:
        """Wait for a read a failed step left writing a pinned half."""

        if self.reads is not None:
            self.reads.drain()


@torch.no_grad()
def prefill(e, prompt: Sequence[int], sampling: Sampling | None, resume: Snapshot | None = None,
            keep_at: int | None = None, keep: Callable[[Snapshot], object] | None = None,
            space: torch.Tensor | None = None) -> int:
    """Commit ``prompt`` from 0 or ``resume`` and draw its first token; ``keep`` takes the ``keep_at`` snapshot."""

    f = Fill(e, prompt, sampling, resume, keep_at, keep, space)
    try:
        while not f.step():
            pass
    finally:
        f.close()
    return f.first
