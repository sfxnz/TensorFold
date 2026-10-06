"""CUDA graphs of decode's device work: a verify forward per window size, DSpark's absorb, and its proposals."""

from __future__ import annotations

import gc

import torch

from tensorfold.engine.exact_sampling import Sampling

from . import GRAPH_ROWS, MAX_ROWS, dspark
from . import forward as F


class Graphs:
    def __init__(self, e) -> None:
        self.e = e
        self.pool = torch.cuda.graph_pool_handle()
        self.verify: dict[int, torch.cuda.CUDAGraph] = {}      # by window rows
        self.absorb: dict[int, torch.cuda.CUDAGraph] = {}      # by kept rows
        self.drafts: dict[tuple[int, bool], torch.cuda.CUDAGraph] = {}     # block + d Markov steps, by (d, keyed)

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
        """Capture every graph -> how many; each runs eagerly first, so the engine resets the sequence after."""

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
            e.dwork.chain.fill_(token)
            e.dwork.draws.set(Sampling(seed=0, temperature=1.0, top_k=0))
            for keyed in (False, True):
                for d in range(1, w.cfg.dspark_block_size + 1):
                    self.drafts[d, keyed] = self._capture(lambda d=d, keyed=keyed: dspark.chain(e, d, keyed))
        return len(self.verify) + len(self.absorb) + len(self.drafts)
