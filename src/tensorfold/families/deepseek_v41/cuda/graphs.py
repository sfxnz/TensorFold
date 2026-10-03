"""CUDA graphs of one engine's decode-side device work: a verify forward per window size, DSpark's absorb per kept
row count, its block and each Markov step.

Each reads static buffers and ``st.pos_dev`` only, so one graph serves every position and replays the eager path's
kernels with its launch parameters; the exchanges run on the compute stream, so the graphs capture them too.
"""

from __future__ import annotations

import gc

import torch

from . import GRAPH_ROWS, MAX_ROWS, dspark
from . import forward as F


class Graphs:
    def __init__(self, e) -> None:
        self.e = e
        self.pool = torch.cuda.graph_pool_handle()
        self.verify: dict[int, torch.cuda.CUDAGraph] = {}      # by window rows
        self.absorb: dict[int, torch.cuda.CUDAGraph] = {}      # by kept rows
        self.block: torch.cuda.CUDAGraph | None = None
        self.inputs: list[torch.cuda.CUDAGraph] = []           # Markov step i's embedding and confidence
        self.steps: list[torch.cuda.CUDAGraph] = []            # Markov step i's head bias

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        """``fn`` run once eagerly (compiling its kernels outside the capture), then captured into the pool."""

        fn()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        enabled = gc.isenabled()
        gc.disable()            # collecting an old graph mid-capture destroys it and invalidates the capture
        try:
            # thread-local: NCCL's helper threads may call CUDA while this thread captures
            with torch.cuda.graph(g, pool=self.pool, capture_error_mode="thread_local"):
                fn()
        finally:
            if enabled:
                gc.enable()
        torch.cuda.synchronize()
        return g

    @torch.no_grad()
    def warm(self) -> int:
        """Capture every graph -> how many. Each runs eagerly first at positions from MAX_ROWS, which writes the
        sequence's state: the engine resets it after."""

        e = self.e
        w, st, b = e.w, e.st, e.dbuf
        if st.capacity < 2 * MAX_ROWS:
            raise ValueError(f"graphs warm at positions {MAX_ROWS}..{2 * MAX_ROWS - 1}: a state of {st.capacity}")
        st.reset()
        st.set_pos(MAX_ROWS)
        token = w.cfg.bos_token_id
        for R in GRAPH_ROWS:
            F.stage(w, st, b, [token] * R, e.hasher, e.reader)
            self.verify[R] = self._capture(lambda R=R: F.compute(w, st, b, R, prompt=False, head_rows=R))
        if w.dspark is not None:
            for n in GRAPH_ROWS:
                self.absorb[n] = self._capture(lambda n=n: dspark.absorb(e, b, n, prompt=False))
            e.dwork.bids[:1].fill_(token)
            e.dwork.mids.fill_(token)
            self.block = self._capture(lambda: dspark.block(e))
            B = w.cfg.dspark_block_size
            self.inputs = [self._capture(lambda i=i: dspark.markov_input(e, i)) for i in range(B)]
            self.steps = [self._capture(lambda i=i: dspark.markov_step(e, i)) for i in range(B)]
        return len(self.verify) + len(self.absorb) + (self.block is not None) + len(self.inputs) + len(self.steps)
