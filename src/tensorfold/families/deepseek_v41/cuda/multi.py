"""Concurrent requests over lanes: shared verify rounds over the decoding lanes, prompts filling between them."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from functools import partial
from types import SimpleNamespace

import numpy as np
import torch

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.streams import Stream

from ..engram_hash import rank_columns
from . import BLOCK, dspark, engram, sample
from . import forward as F
from .buffers import Buffers
from .decode import Engine, _clock, accept
from .lanes import Lanes, stage_tables
from .multi_fill import FillPlan, span_rows
from .multi_tp import TwoRanks
from .prefill import Fill

STAGES_ENV = "TF_DSV41_STAGES"      # "1": each stream's stats carry its rounds' mean ms by part as ``stages_ms``


def _elapsed(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    """Seconds between two recorded events, once the second has run."""

    end.synchronize()
    return start.elapsed_time(end) / 1e3


class LaneDecoder(TwoRanks):
    """Upstream's decoder contract (``tensorfold.cuda.scheduler``) over ``lanes``, ``engines[k]`` lane k's Engine on the
    shared buffers; ``policy`` is (drafts a round, confidence stop or None), ``graphs`` LaneGraphs or None, ``kept`` a
    Kept over ``engines`` or None. On two ranks rank 0 sets ``link`` and rank 1 runs ``follow``."""

    def __init__(self, w, lanes: Lanes, engines: Sequence[Engine], mbuf: Buffers, pbuf: Buffers,
                 policy: tuple[int, float | None], eos: Sequence[int], graphs, share: float, kept=None) -> None:
        if kept is not None and [id(e) for e in getattr(kept, "lanes", ())] != [id(e) for e in engines]:
            raise ValueError("lane prompts resume from kept snapshots of these lanes' engines only")
        if len(engines) != lanes.slots or any(e.st is not lanes.view(k) or e.dbuf is not mbuf or e.pbuf is not pbuf
                                              for k, e in enumerate(engines)):
            raise ValueError(f"lane decoder: {len(engines)} engines for {lanes.slots} lanes, each on its lane's view "
                             "and the shared buffers")
        if not 0 <= policy[0] <= BLOCK:
            raise ValueError(f"{policy[0]} drafts a round: DSpark proposes 0 to {BLOCK}")
        self.w, self.lanes, self.engines, self.mbuf, self.pbuf = w, lanes, list(engines), mbuf, pbuf
        self.drafts, self.confidence = int(policy[0]), policy[1]
        self.eos, self.graphs, self.kept = tuple(eos), graphs, kept
        for k, e in enumerate(self.engines):        # each lane's absorb and proposals replay its own graphs
            e.graphs = None if graphs is None else SimpleNamespace(verify={}, absorb=graphs.absorb[k],
                                                                   drafts=graphs.drafts[k])
        self.plan = FillPlan(share)
        self.streams: dict[int, Stream] = {}        # decoding, by stream id
        self.filling: list[Stream] = []             # admitted, prompts filling, in arrival order
        self.arrived = lambda: False                # the Scheduler's: whether a foreground request waits
        self.current: Stream | None = None          # the prompt that ran the last span
        self.next_id = 0
        self.used = [-1] * lanes.slots              # the stream id each lane last took
        self.stages = os.environ.get(STAGES_ENV) == "1"

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def _free(self) -> list[int]:
        busy = {s.lane for s in (*self.streams.values(), *self.filling)}
        return [k for k in range(self.lanes.slots) if k not in busy]

    def _ends(self, s: Stream) -> tuple[int, ...]:
        return self.eos if s.stop_eos else ()

    def _drafting(self, s: Stream) -> bool:
        return s.draft and self.drafts > 0 and self.w.dspark is not None

    def _lane(self, free: list[int], hit) -> int:
        """The hit's lane if free, else the lowest free lane holding no kept rows, else the free lane used least
        recently."""

        if self.kept is None:
            return free[0]
        at = self.kept.lane(hit)
        if at in free:
            return at
        empty = [k for k in free if not self.kept.holds(k)]
        return empty[0] if empty else min(free, key=lambda k: self.used[k])

    def _rows_in(self, hit) -> int | str | None:
        """Where ``hit``'s rows are: a lane, "arena" once saved, or None without a hit."""

        if hit is None:
            return None
        lane = self.kept.lane(hit)
        return "arena" if lane is None else lane

    def _valid(self, s: Stream) -> None:
        prompt, C, vocab = s.prompt, self.lanes.capacity, self.w.cfg.vocab_size
        if not prompt:
            raise ValueError("an empty prompt")
        if len(prompt) >= C:
            raise ValueError(f"prompt of {len(prompt)} tokens: lanes hold {C} positions")
        if min(prompt) < 0 or max(prompt) >= vocab:
            raise ValueError(f"prompt token ids must lie in [0, {vocab})")

    def _plan(self, s: Stream) -> tuple[dict, object]:
        """Rank 0's admission of ``s``: {lane, cached, rows_in, rows}, and the kept snapshot it resumes."""

        self._valid(s)
        free = self._free()
        if not free:
            raise NoRoom(f"all {self.lanes.slots} lanes are busy")
        s.count = max(1, min(s.count, self.lanes.capacity - len(s.prompt)))
        hit = self.kept.lookup(s.prompt) if self.kept is not None and s.draft else None
        rows = span_rows(self.pbuf.rows, any(not x.done for x in self.streams.values()))
        plan = {"lane": self._lane(free, hit), "cached": len(hit.ids) if hit is not None else 0,
                "rows_in": self._rows_in(hit), "rows": rows}
        return plan, hit

    @torch.no_grad()
    def admit(self, s: Stream, told: dict | None = None) -> None:
        """Queue ``s``'s prompt in a free lane, running no forward: later steps fill it a span at a time, from its
        longest kept prefix when it drafts; rank 1 is ``told`` rank 0's plan."""

        plan, hit = self._admission(s, told)
        lane, rows = plan["lane"], plan["rows"]
        e = self.engines[lane]
        e.drop_reads()
        s.fill = Fill(e, s.prompt, s.sampling, rows=rows) if self.kept is None else self._resume(s, lane, hit, rows)
        s.lane, s.sid, s.prefill_s, s.cached, s.span_rows = lane, self.next_id, 0.0, plan["cached"], rows
        self.used[lane] = self.next_id
        self.next_id += 1
        self.filling.append(s)

    def _resume(self, s: Stream, lane: int, hit, rows: int) -> Fill:
        """``s``'s fill in ``lane`` from ``hit``, its rows put in the lane, the snapshot before its last token kept
        when it drafts; the hit stays pinned and the span reserved until the lane's fill settles."""

        kept, prompt = self.kept, s.prompt
        keep_at = max(1, len(prompt) - 1) if s.draft else None
        fresh = keep_at is not None and (hit is None or len(hit.ids) != keep_at)
        space = kept.reserve(hit, lane, room=fresh)
        if fresh and space is None:
            keep_at = None
        try:
            kept.take_over(hit, lane)
            return Fill(self.engines[lane], prompt, s.sampling, resume=hit, keep_at=keep_at, space=space, rows=rows,
                        keep=partial(kept.remember, lane=lane) if keep_at is not None else None)
        except BaseException:
            kept.settle(lane)
            raise

    @torch.no_grad()
    def round(self, told: dict | None = None) -> list[Stream]:
        """Up to FILL_SPANS spans of a prompt when no lane decodes, else a span or a decode round -> streams ended;
        rank 1 is ``told`` rank 0's plan."""

        s, spans, decode = self._round_plan(told)
        if s is not None:
            done = self._fill(s, spans)
        elif decode:
            start = time.perf_counter()
            done = self._decode()
            if self.filling and self.follower is None:
                self.plan.spent(time.perf_counter() - start)
        else:
            done = []
        assert self._invariants()
        return done

    def _fill(self, s: Stream, spans: int) -> list[Stream]:
        """Up to ``spans`` spans of ``s``'s prompt, the prompt that ran the last span paused first if it was another."""

        if self.current is not None and self.current is not s and self.current.fill is not None:
            self.current.fill.pause()
        self.current, f = s, s.fill
        for _ in range(spans):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t = time.perf_counter()
            start.record()
            try:
                last = f.step()
            except Exception as exc:                 # noqa: BLE001  (this request fails, the others go on)
                f.close()
                s.error, s.done = exc, True
                self._unfill(s)
                return [s]
            end.record()
            s.prefill_s += time.perf_counter() - t
            if self.follower is None:               # rank 0's credit; rank 1 is told
                self.plan.spanned(s, self.filling, lambda start=start, end=end: _elapsed(start, end))
            if last:
                return self._join(s)
        return []

    def _unfill(self, s: Stream) -> None:
        self.filling = [x for x in self.filling if x is not s]
        s.fill = None
        if self.kept is not None:
            self.kept.settle(s.lane)
        if self.current is s:
            self.current = None

    def _join(self, s: Stream) -> list[Stream]:
        """``s``'s prompt is committed: it takes its first token and decodes, its first window's rows read ahead."""

        first = s.fill.first
        self._unfill(s)
        if self.kept is not None:
            self.kept.live[s.lane] = list(s.prompt)
        s.context = list(s.prompt)
        s.started = time.perf_counter()
        s.take([first], self._ends(s))
        if s.done:
            return [s]
        self.streams[s.sid] = s
        if self._drafting(s):
            e = self.engines[s.lane]
            e.fetch(0, e.st.history, first)
        return []

    def _stage(self, e: Engine, seg0: int, tokens: Sequence[int]) -> None:
        """Lane ``e``'s window's Engram rows into mbuf rows seg0.., those read ahead through its pinned block."""

        w, b, a = self.w, self.mbuf, e.ahead
        if not w.engram:
            return
        t = time.perf_counter()
        ready, R = e._ready(tokens), len(tokens)
        ids = np.empty((0, *b.eraw.shape[1:3]), dtype=np.int64)
        if ready < R:
            back = w.cfg.engram_max_ngram_size - 1
            before = (list(e.st.history) + list(tokens[:ready]))[-back:] if back > 0 else []
            ids = rank_columns(e.hasher.ids(np.asarray(before, dtype=np.int64),
                                            np.asarray(tokens[ready:], dtype=np.int64)), w.rank, w.world)
        engram.stage_rows(ids, e.reader, a.eraw_host[0], a.eraw_done[0], b.eraw[seg0:seg0 + R], first=ready,
                          scales=w.engram_scales, idx_host=a.eidx_host[0], idx=b.eidx[seg0:seg0 + R])
        e._timed("engram", t)

    def _decode(self) -> list[Stream]:
        """One verify forward over every decoding lane's window, then each lane's rows as alone -> streams ended."""

        w, b = self.w, self.mbuf
        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.lane)
        engines = [self.engines[s.lane] for s in live]
        for s, e in zip(live, engines):             # 1. each lane's proposal: the solo block on its view
            e.clock = _clock()
            d = min(self.drafts, s.count - len(s.out) - 1) if self._drafting(s) else 0
            s.drafts = list(e.propose(s.out[-1], e.st.pos - 1, s.sampling, d, self.confidence)[0][:d]) if d > 0 else []
        windows = [(s.lane, e.st.pos, [s.out[-1], *s.drafts]) for s, e in zip(live, engines)]
        T, seg0 = stage_tables(b, windows), 0       # 2. ids and row tables
        for s, e, (_, _, tokens) in zip(live, engines, windows):     # 3. each lane's Engram rows
            s.seg0, s.R = seg0, len(tokens)
            self._stage(e, seg0, tokens)
            seg0 += s.R
        t = time.perf_counter()                     # 4. the shared forward
        g = self.graphs.verify.get(T) if self.graphs is not None else None
        if g is not None:
            g.replay()
        else:
            F.compute(w, self.lanes, b, T, prompt=False, head_rows=T)
        launch = time.perf_counter() - t
        torch.cuda.current_stream().synchronize()
        device = time.perf_counter() - t - launch
        draws, seconds = self._draws(live)          # 5. every lane's draws
        kept = []
        for s, e, sampled in zip(live, engines, draws):              # 6. each lane's kept rows
            e.clock["launch"] += launch
            e.clock["device"] += device
            e.clock["sampling"] += seconds
            kept.append(sampled[:accept(sampled, s.drafts, self._ends(s))])
        for s, e, new, (_, _, tokens) in zip(live, engines, kept, windows):     # 7. commit
            F.commit(w, e.st, b, s.R, len(new), row0=s.seg0)
            if self.kept is not None:
                self.kept.live[s.lane] += tokens[:len(new)]
        for s, e, new in zip(live, engines, kept):  # 8-9. absorb and read ahead, for lanes that go on drafting
            if not self._drafting(s) or len(s.out) + len(new) >= s.count or new[-1] in self._ends(s):
                continue
            if s.seg0 == 0:                         # its graph's rows
                e.absorb(len(new))
            else:
                t = time.perf_counter()
                dspark.absorb(e, b, len(new), prompt=False, first=s.seg0)
                e._timed("launch", t)
            e.fetch(0, e.st.history, new[-1])
        for s, e, new in zip(live, engines, kept):  # 10. emit
            s.counted(s.R)
            s.take(new, self._ends(s))
            if self.stages:
                _clocked(s, e.clock)
        return [s for s in live if s.done]

    def _draws(self, live: list[Stream]) -> tuple[list[list[int]], float]:
        """Each live lane's target draws on its rows of the forward -> (their tokens, host seconds)."""

        rows = [self.mbuf.logits[s.seg0:s.seg0 + s.R] for s in live]
        at = [self.engines[s.lane].st.pos + 1 for s in live]
        return sample.lane_rows(self.w, rows, [list(range(a, a + s.R)) for a, s in zip(at, live)],
                                [s.sampling for s in live])

    def finish(self, done: list[Stream]) -> None:
        """Free the lanes of streams that ended: done, a client gone, or a background stream yielding."""

        if done:
            self._send(["finish", [s.sid for s in done]])
        for s in done:
            filling = any(x is s for x in self.filling)
            if not filling and self.streams.get(s.sid) is not s:
                continue
            if filling:
                s.fill.close()
                self._unfill(s)
            else:
                del self.streams[s.sid]
            self.engines[s.lane].drop_reads()

    def drop(self) -> list[Stream]:
        """Every live stream, all lanes freed; a failed round may leave the decode selection's scratch dirty: zeroed."""

        self._send(["drop"])
        live = [s for s in self.streams.values() if not s.done] + self.filling
        for s in self.filling:
            s.fill.close()
            s.fill = None
        if self.kept is not None:
            for k in list(self.kept.pinned):
                self.kept.settle(k)
            self.kept.live.clear()
        for e in self.engines:
            e.drop_reads()
        self.streams, self.filling, self.current = {}, [], None
        self.plan.reset()
        for t in self.mbuf.split.values():
            t.zero_()
        return live

    def _invariants(self) -> bool:
        """Decoding streams in ``streams``, filling ones in ``filling``, one lane each, contexts prompt then reply."""

        filling = {id(s) for s in self.filling}
        assert all(s.fill is not None for s in self.filling), "a filling stream without its fill"
        for s in self.streams.values():
            assert id(s) not in filling and s.fill is None, f"stream {s.sid} both decodes and fills"
            assert len(s.context) == len(s.prompt) + len(s.out), f"stream {s.sid}'s context is not prompt + reply"
        lanes = [s.lane for s in (*self.streams.values(), *self.filling)]
        assert len(set(lanes)) == len(lanes), f"lanes {lanes} shared"
        return True


def _clocked(s: Stream, clock: dict[str, float]) -> None:
    """Add a round's seconds by part to ``s``, whose stats then carry the mean ms a round as ``stages_ms``."""

    sums = getattr(s, "clocks", None)
    if sums is None:
        sums = s.clocks = _clock()
        own = s.stats
        s.stats = lambda: {**own(), "stages_ms": {k: round(v * 1e3 / max(s.rounds, 1), 3) for k, v in sums.items()}}
    for k, v in clock.items():
        sums[k] += v
