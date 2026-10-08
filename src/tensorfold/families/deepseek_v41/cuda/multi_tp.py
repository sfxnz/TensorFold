"""The lane decoder on two ranks: rank 0's ordered ops over a link, both ranks' state checked before each step's
collectives, and rank 1's ``follow``."""

from __future__ import annotations

import base64
import hashlib
import json
import struct

import numpy as np
import torch

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import multi_tp as upstream
from tensorfold.families.qwen4_exp.cuda.multi_tp import OutOfStep, _pack, _unpack

from . import protocol
from .multi_fill import FILL_SPANS

__all__ = ["Link", "OutOfStep", "TwoRanks", "admit_op", "digest", "pack_prompt", "read_admit", "shape",
           "unpack_prompt"]


class Link(upstream.Link):
    """Upstream's ordered link, its store keys in this family's namespace."""

    KEY = "tf_dsv41/lanes/socket"

    def _key(self) -> str:
        return f"tf_dsv41/lanes/{self.n}"


def pack_prompt(prompt) -> tuple[str, str]:
    """A prompt as base64 little-endian int32 bytes, and their sha256."""

    raw = np.asarray(prompt, dtype="<i4").tobytes()
    return base64.b64encode(raw).decode("ascii"), hashlib.sha256(raw).hexdigest()


def unpack_prompt(text: str) -> tuple[list[int], str]:
    """``pack_prompt``'s prompt, and the sha256 of the bytes that arrived."""

    raw = base64.b64decode(text)
    return np.frombuffer(raw, dtype="<i4").tolist(), hashlib.sha256(raw).hexdigest()


def admit_op(sid: int, s: Stream, text: str, plan: dict) -> list:
    """Rank 0's admission of ``s`` (its prompt packed as ``text``, hashed as ``s.digest``) with ``plan``."""

    return ["admit", sid, text, s.digest, s.count, _pack(s.sampling), bool(s.draft), bool(s.stop_eos),
            bool(s.background), plan]


def read_admit(op: list) -> tuple[int, Stream, dict]:
    """``admit_op``'s (sid, stream, plan); the stream's ``digest`` hashes what arrived, ``sent`` is rank 0's."""

    sid, text, sent, count, sampling, draft, stop_eos, background, plan = op[1:]
    prompt, here = unpack_prompt(text)
    s = Stream(prompt, count, _unpack(sampling), draft=draft, stop_eos=stop_eos, background=background)
    s.digest, s.sent = here, sent
    return sid, s, plan


def shape(dec) -> list:
    """The logical state both ranks must share, from lengths and cached digests only (never ids)."""

    kept = dec.kept
    lanes = [[e.st.pos, len(e.st.history)] for e in dec.engines]
    streams = [[s.sid, s.lane, len(s.out), s.out[-1] if s.out else None, s.count, bool(s.done), bool(s.draft),
                bool(s.stop_eos), bool(s.background), _pack(s.sampling),
                None if s.fill is None else s.fill.i, None if s.fill is None else s.span_rows]
               for s in [*dec.streams.values(), *dec.filling]]
    entries = [] if kept is None else [[len(x.snap.ids), x.digest, "arena" if x.lane is None else x.lane]
                                       for x in kept.cache]
    settings = [dec.lanes.slots, dec.lanes.capacity, dec.drafts, dec.confidence, list(dec.eos), dec.pbuf.rows,
                None if kept is None else kept.entries, dec.batch is not None]
    return [dec.next_id, lanes, streams, entries, dec._free(), settings]


def digest(state) -> list[int]:
    """The first 8 bytes of the sha256 of ``state``'s JSON, as two int32."""

    return list(struct.unpack("<2i", hashlib.sha256(json.dumps(state).encode()).digest()[:8]))


class TwoRanks:
    """LaneDecoder's two-rank half: ``link`` on rank 0 sends each op before acting, ``follower`` on rank 1 is set
    while it follows."""

    link: Link | None = None
    follower: Link | None = None

    def _agree(self, what: str, plan: list, checks: tuple = ()) -> tuple[bool, ...]:
        """One small all-gather of [digest of shape and ``plan``, *checks] before either rank's step collectives ->
        each check held on both."""

        if self.w.comm is None:
            return tuple(bool(c) for c in checks)
        width = 2 + len(checks)
        ranks = protocol.gather_ints(self.w.comm, [*digest([shape(self), plan]), *map(int, checks)])
        if len(ranks) != 2 or any(len(r) != width for r in ranks):
            raise OutOfStep("the shared plan check returned an invalid rank count")
        if ranks[0][:2] != ranks[1][:2]:
            raise OutOfStep(f"the two ranks planned different {what}s; the requests in it fail, serving goes on")
        return tuple(all(bool(r[i]) for r in ranks) for i in range(2, width))

    def _admission(self, s: Stream, told: dict | None):
        """Rank 0 plans ``s``'s admission and sends it, rank 1 checks the plan it was told; both agree before acting
        -> (plan, the snapshot it resumes)."""

        if told is None:
            plan, hit = self._plan(s)               # a refusal here reaches neither the link nor rank 1
            found = holds = True
        else:
            plan, (found, holds, hit) = told, self._told(s, told)
        if self.w.comm is None:
            return plan, hit
        if told is None:
            text, s.digest = pack_prompt(s.prompt)
            if self.link is not None:
                self.link.send(admit_op(self.next_id, s, text, plan))
        found, holds = self._agree("admission", [plan, self.next_id, s.digest, s.count, _pack(s.sampling),
                                                 bool(s.draft), bool(s.stop_eos), bool(s.background)],
                                   (found, holds))
        if not found:
            raise OutOfStep(f"rank 1 has no snapshot of the {plan['cached']} tokens rank 0 resumes from; the request "
                            "fails, serving goes on")
        if not holds:
            raise OutOfStep("the admission plan does not hold on both ranks; the request fails, serving goes on")
        return plan, hit

    def _told(self, s: Stream, plan: dict) -> tuple[bool, bool, object]:
        """Rank 1: (it has the snapshot, the plan holds here, that snapshot)."""

        try:
            hit = self.kept.find(s.prompt, plan["cached"]) if plan["cached"] else None
        except Exception:                           # noqa: BLE001  (refused on both ranks in the agreement)
            return False, True, None
        try:
            self._valid(s)
            holds = (s.digest == s.sent and plan["lane"] in self._free() and (hit is None or s.draft)
                     and plan["rows_in"] == self._rows_in(hit) and 0 < plan["rows"] <= self.pbuf.rows)
        except Exception:                           # noqa: BLE001
            holds = False
        return True, holds, hit

    def _round_plan(self, told: dict | None):
        """Rank 0 plans the round and sends it, rank 1 checks the plan it was told; both agree before acting ->
        (the prompt to fill or None, its spans, whether lanes decode)."""

        decoding = any(not s.done for s in self.streams.values())
        if told is None:
            arrived = bool(self.filling) and not decoding and self.arrived()
            s, spans = self.plan.next(self.filling, decoding, arrived)
            plan, holds = {"fill": None if s is None else s.sid, "spans": spans, "decode": s is None and decoding}, True
        else:
            plan, (s, holds) = told, self._told_round(told, decoding)
        if self.w.comm is not None:
            if self.link is not None:
                self.link.send(["round", [x.sid for x in [*self.streams.values(), *self.filling] if x.done], plan])
            (holds,) = self._agree("round", [plan], (holds,))
            if not holds:
                raise OutOfStep("the round plan does not hold on both ranks; the live requests fail, serving goes on")
        return s, plan["spans"], plan["decode"]

    def _told_round(self, plan: dict, decoding: bool) -> tuple[Stream | None, bool]:
        """Rank 1: (the prompt the plan fills, whether the plan holds here); several spans only when no lane decodes."""

        try:
            fill, spans, decode = plan["fill"], plan["spans"], plan["decode"]
            s = next((x for x in self.filling if x.sid == fill), None)
            if fill is None:
                return None, spans == 0 and decode is decoding
            return s, s is not None and decode is False and 1 <= spans <= (1 if decoding else FILL_SPANS)
        except Exception:                           # noqa: BLE001
            return None, False

    def _send(self, op: list) -> None:
        if self.link is not None:
            self.link.send(op)

    @torch.no_grad()
    def follow(self, link: Link) -> None:
        """Rank 1: rank 0's ops in order until ``stop`` or a closed link; a refused admission or a failed round is
        printed and serving goes on."""

        self.follower, finished = link, {}
        try:
            while True:
                op = link.receive()
                if op is None or op[0] == "stop":
                    return
                if op[0] == "admit":
                    sid, s, plan = read_admit(op)
                    try:
                        self.admit(s, told=plan)
                    except Exception as exc:        # noqa: BLE001  (rank 0 refused it too)
                        print(f"[tensorfold] rank 1: stream {sid} refused: {exc}", flush=True)
                elif op[0] == "round":
                    for s in [*self.streams.values(), *self.filling]:
                        if s.sid in op[1]:
                            s.done = True
                    try:
                        finished.update({s.sid: s for s in self.round(told=op[2])})
                    except Exception as exc:        # noqa: BLE001  (rank 0 drops the same failed round)
                        print(f"[tensorfold] rank 1: a round failed: {exc}", flush=True)
                elif op[0] == "finish":
                    known = {**{s.sid: s for s in self.filling}, **self.streams, **finished}
                    self.finish([known[sid] for sid in op[1] if sid in known])
                    for sid in op[1]:
                        finished.pop(sid, None)
                elif op[0] == "drop":
                    self.drop()
                    finished.clear()
        finally:
            self.follower = None
