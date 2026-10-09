"""CUDA graphs of decode's device work: a verify forward per window size, DSpark's absorb, and its proposals."""

from __future__ import annotations

import gc

import torch

from tensorfold.engine.exact_sampling import Sampling

from . import GRAPH_ROWS, MAX_ROWS, dspark, proposals, sample
from . import forward as F
from .lanes import stage_tables


def verify_window(w, st, b, R: int) -> None:
    """A verify window's forward over its R staged rows, then (with ``b``'s greedy scratch) each row's greedy key."""

    F.compute(w, st, b, R, prompt=False, head_rows=R)
    if b.gkeys is not None:
        sample.window_keys(w, b, R)


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

    def _proposals(self, e) -> tuple[dict[int, torch.cuda.CUDAGraph], dict[tuple[int, bool], torch.cuda.CUDAGraph]]:
        """Engine ``e``'s absorb graphs from tap row 0 by kept rows, then its proposals' by (d, keyed); e sits at
        MAX_ROWS or later."""

        w, b, token = e.w, e.dbuf, e.w.cfg.bos_token_id
        absorb = {n: self._capture(lambda n=n: dspark.absorb(e, b, n, prompt=False)) for n in GRAPH_ROWS}
        e.dwork.bids[:1].fill_(token)
        e.dwork.chain.fill_(token)
        e.dwork.draws.set(Sampling(seed=0, temperature=1.0, top_k=0))
        drafts = {}
        for keyed in (False, True):
            for d in range(1, w.cfg.dspark_block_size + 1):
                drafts[d, keyed] = self._capture(lambda d=d, keyed=keyed: dspark.chain(e, d, keyed))
        return absorb, drafts

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
            self.verify[R] = self._capture(lambda R=R: verify_window(w, st, b, R))
        if w.dspark is not None:
            self.absorb, self.drafts = self._proposals(e)
        return len(self.verify) + len(self.absorb) + len(self.drafts)


class LaneGraphs(Graphs):
    """Graphs of the shared forward over ``lanes``: verify by total rows T (the row tables are data, so one graph serves
    every layout), each lane Engine's absorb from forward row 0 and its proposals, and with ``batch`` the proposals of
    2.. lanes in one block by (lanes, keyed of them, d), in one pool."""

    def __init__(self, w, lanes, mbuf, engines, batch=None) -> None:
        if len(engines) != lanes.slots or any(e.st is not lanes.view(k) or e.dbuf is not mbuf
                                              for k, e in enumerate(engines)):
            raise ValueError(f"lane graphs: {len(engines)} engines for {lanes.slots} lanes, each on its lane's view "
                             "and the shared buffers")
        self.w, self.lanes, self.mbuf, self.engines = w, lanes, mbuf, engines
        self.pool = torch.cuda.graph_pool_handle()
        self.verify: dict[int, torch.cuda.CUDAGraph] = {}      # by total rows T
        self.absorb: list[dict[int, torch.cuda.CUDAGraph]] = [{} for _ in engines]     # per lane, by kept rows
        self.drafts: list[dict[tuple[int, bool], torch.cuda.CUDAGraph]] = [{} for _ in engines]
        self.batch, self.batched = batch, {}

    @torch.no_grad()
    def warm(self) -> int:
        """Capture every graph -> how many: each lane at MAX_ROWS, every T laid out round-robin; then every lane is
        reset."""

        w, lanes, b, S = self.w, self.lanes, self.mbuf, self.lanes.slots
        if lanes.capacity < 2 * MAX_ROWS:
            raise ValueError(f"graphs warm at positions {MAX_ROWS}..{2 * MAX_ROWS - 1}: lanes of {lanes.capacity}")
        for e in self.engines:
            e.st.reset()
            e.st.set_pos(MAX_ROWS)
        b.eraw.zero_()          # the Engram rows the warm forwards read: any finite bytes
        b.eidx.zero_()
        token = w.cfg.bos_token_id
        for T in range(1, MAX_ROWS * S + 1):
            stage_tables(b, [(k, MAX_ROWS, [token] * (T // S + (k < T % S))) for k in range(min(S, T))])
            self.verify[T] = self._capture(lambda T=T: verify_window(w, lanes, b, T))
        if w.dspark is not None:
            for k, e in enumerate(self.engines):
                self.absorb[k], self.drafts[k] = self._proposals(e)
            if self.batch is not None:
                self._batched(token)
        for e in self.engines:
            e.reset()
        return len(self.verify) + sum(map(len, self.absorb)) + sum(map(len, self.drafts)) + len(self.batched)

    def _batched(self, token: int) -> None:
        """The block of lanes 0.. drafting together, by (lanes, keyed of them, d)."""

        k, sampling = self.batch, Sampling(seed=0, temperature=1.0, top_k=0)
        for L in range(2, min(k.slots, self.lanes.slots) + 1):
            for keyed in range(L + 1):
                proposals.set_slots(k, [(j, token, sampling if j >= L - keyed else None) for j in range(L)])
                for d in range(1, self.w.cfg.dspark_block_size + 1):
                    self.batched[L, keyed, d] = self._capture(
                        lambda L=L, keyed=keyed, d=d: proposals.chain(self.w, self.lanes, self.mbuf, k, L, keyed, d))
