"""Kept prompt snapshots of one rank in one arena allocated at startup, kept and dropped alike on both ranks."""

from __future__ import annotations

import hashlib
import math
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch

from . import snapshot
from .snapshot import ALIGN, Snapshot

ENTRIES_ENV = "TF_DSV41_CACHE_ENTRIES"
ENTRIES = 8                 # kept snapshots at most, unless ENTRIES_ENV says otherwise


def entries_wanted(environ: Mapping[str, str] | None = None, lanes: int = 0) -> int:
    """Snapshots ENTRIES_ENV allows (ENTRIES + ``lanes`` unset); ValueError naming the variable for a negative or
    non-integer value."""

    value = (os.environ if environ is None else environ).get(ENTRIES_ENV, "").strip()
    try:
        n = int(value) if value else ENTRIES + lanes
    except ValueError:
        n = -1
    if n < 0:
        raise ValueError(f"{ENTRIES_ENV}={value}: a whole number of snapshots, 0 or more")
    return n


class Arena:
    """One uint8 device buffer handed out first fit in ALIGN-rounded spans; nothing is allocated after construction."""

    def __init__(self, nbytes: int, device: torch.device | str) -> None:
        self.buf = torch.empty((nbytes // ALIGN * ALIGN,), dtype=torch.uint8, device=device)
        self.spans: dict[int, int] = {}     # start -> length

    def take(self, nbytes: int) -> int | None:
        """The lowest free span of ``nbytes`` (rounded up to ALIGN) -> its start, or None when no gap holds it."""

        size, at = math.ceil(nbytes / ALIGN) * ALIGN, 0
        for start in sorted(self.spans):
            if start - at >= size:
                break
            at = start + self.spans[start]
        if at + size > self.buf.numel():
            return None
        self.spans[at] = size
        return at

    def view(self, at: int) -> torch.Tensor:
        return self.buf[at:at + self.spans[at]]

    def give(self, at: int | None) -> None:
        if at is not None:
            del self.spans[at]


@dataclass(eq=False)
class Entry:
    snap: Snapshot
    state: int                  # arena span of the rings and tails
    rows: int | None = None     # arena span of the saved cache rows; None while they are live
    lane: int | None = None     # the lane whose caches hold the rows; None once they are saved
    digest: str = ""            # ids_digest of the snapshot's ids


@dataclass(eq=False)
class Pin:
    """A lane's fill in flight: the snapshot it resumes and the span its keep point's snapshot takes."""

    hit: Snapshot | None
    at: int | None = None


def ids_digest(ids: Sequence[int]) -> str:
    """sha256 of the ids as int32 bytes."""

    return hashlib.sha256(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


class Kept:
    """Kept snapshots of ``e``'s state (lane 0), or of lane Engines ``e[k]``'s, least recently used first, and the ids
    each lane's caches hold; a lane is the token of its fill in flight."""

    def __init__(self, e, nbytes: int, entries: int) -> None:
        self.lanes = list(e) if isinstance(e, Sequence) else [e]
        self.entries = entries
        self.arena = Arena(nbytes, self.lanes[0].st.device)
        self.state = snapshot.state_bytes(self.lanes[0])
        self.cache: list[Entry] = []
        self._live: dict[int, list[int]] = {}
        self.pinned: dict[int, Pin] = {}

    @property
    def live(self) -> dict[int, list[int]]:
        """Lane -> the ids whose rows its caches hold final."""

        return self._live

    @live.setter
    def live(self, ids: Mapping[int, list[int]] | Sequence[int]) -> None:
        self._live = dict(ids) if isinstance(ids, Mapping) else {0: list(ids)}    # a list: lane 0's, the legacy

    def _entry(self, snap: Snapshot) -> Entry | None:
        return next((x for x in self.cache if x.snap is snap), None)

    def _usable(self, x: Entry) -> bool:
        """Its rows are saved, or final in its lane."""

        n = len(x.snap.ids)
        return x.lane is None or self._live.get(x.lane, [])[:n] == x.snap.ids

    def _pinned(self, x: Entry) -> bool:
        return any(p.hit is x.snap for p in self.pinned.values())

    def lookup(self, prompt: Sequence[int]) -> Snapshot | None:
        """Rank 0: the longest kept snapshot of a strict prefix of ``prompt``."""

        best = None
        for x in self.cache:
            n = len(x.snap.ids)
            if (n < len(prompt) and list(prompt[:n]) == x.snap.ids and (best is None or n > len(best.ids))
                    and self._usable(x)):
                best = x.snap
        return best

    def find(self, prompt: Sequence[int], cached: int) -> Snapshot:
        """Rank 1: the snapshot of ``prompt[:cached]`` rank 0 resumes from."""

        for x in self.cache:
            if len(x.snap.ids) == cached and list(prompt[:cached]) == x.snap.ids and self._usable(x):
                return x.snap
        raise RuntimeError(f"rank 1 has no snapshot of the {cached} tokens rank 0 resumes from")

    def lane(self, snap: Snapshot | None) -> int | None:
        """The lane whose caches hold ``snap``'s rows, or None."""

        x = None if snap is None else self._entry(snap)
        return None if x is None else x.lane

    def holds(self, lane: int) -> bool:
        """Whether a kept snapshot's rows live in ``lane``."""

        return any(x.lane == lane for x in self.cache)

    def _drop(self, x: Entry) -> None:
        self.arena.give(x.state)
        self.arena.give(x.rows)
        x.snap.rows = None
        self.cache.remove(x)

    def _room(self, nbytes: int, keep: Callable[[Entry], bool]) -> int | None:
        """A span of ``nbytes``, dropping the least recently used entries neither ``keep`` nor a pin spares."""

        while (at := self.arena.take(nbytes)) is None:
            old = next((x for x in self.cache if not keep(x) and not self._pinned(x)), None)
            if old is None:
                return None
            self._drop(old)
        return at

    def reserve(self, hit: Snapshot | None, lane: int = 0, room: bool = True) -> torch.Tensor | None:
        """Pin ``lane``'s fill from ``hit`` -> the span its keep point's snapshot is copied into, or None when the
        arena cannot hold one (``room`` False: the keep point is the resume point, no span)."""

        if lane in self.pinned:
            raise RuntimeError(f"lane {lane} already has a fill in flight")
        pin = self.pinned[lane] = Pin(hit)
        if self.entries == 0 or not room:
            return None
        while len(self.cache) + sum(p.at is not None for p in self.pinned.values()) >= self.entries:
            old = next((x for x in self.cache if not self._pinned(x)), None)
            if old is None:
                return None
            self._drop(old)
        pin.at = self._room(self.state, lambda x: False)
        return None if pin.at is None else self.arena.view(pin.at)

    def remember(self, snap: Snapshot, lane: int = 0) -> None:
        """A fill's keep: a new snapshot takes its lane's reserved span; a resumed one moves last."""

        x, pin = self._entry(snap), self.pinned.get(lane)
        if x is None:
            if pin is not None and pin.at is None and snap is pin.hit:
                return                                  # a resumed snapshot dropped while its fill ran
            if pin is None or pin.at is None:
                raise RuntimeError("a kept snapshot without a reserved span")
            x, pin.at = Entry(snap, pin.at, lane=lane, digest=ids_digest(snap.ids)), None
        else:
            self.cache.remove(x)
        self.cache.append(x)

    def settle(self, lane: int = 0) -> None:
        """Unpin ``lane``'s fill and give back a reservation no snapshot took (the keep point was the resume point,
        or the fill failed or ended)."""

        pin = self.pinned.pop(lane, None)
        if pin is not None:
            self.arena.give(pin.at)

    def take_over(self, hit: Snapshot | None, lane: int = 0) -> None:
        """Before a fill from ``hit`` in ``lane``: save the rows of that lane's snapshots it does not resume, then put
        its own rows in the lane, copied from the lane holding them or loaded from the arena."""

        keep, e = hit.ids if hit is not None else [], self.lanes[lane]
        live = self._live.get(lane, [])

        def resumes(x: Entry) -> bool:
            n = len(x.snap.ids)
            return n <= len(keep) and keep[:n] == x.snap.ids

        for x in list(self.cache):
            if x not in self.cache or x.lane != lane or resumes(x):
                continue
            n = len(x.snap.ids)
            if live[:n] != x.snap.ids:
                self._drop(x)
                continue
            at = self._room(snapshot.row_bytes(e, x.snap), lambda y, x=x: y is x or resumes(y))
            if at is None:
                self._drop(x)
                continue
            x.rows, x.lane = at, None
            snapshot.save_rows(e, x.snap, self.arena.view(at))
        if hit is not None:
            x = self._entry(hit)
            if x is None:
                raise RuntimeError("a resume from a snapshot that is not kept")
            if x.lane is None:
                snapshot.load_rows(e, hit)
                self.arena.give(x.rows)
                x.rows, x.lane, hit.rows = None, lane, None
            elif x.lane != lane:
                snapshot.copy_rows(self.lanes[x.lane].st, e.st, len(hit.ids))
        self._live[lane] = list(keep)

    def held(self) -> list[tuple[list[int], int, int | None]]:
        """(ids, state span, rows span) of each kept snapshot, least recently used first."""

        return [(x.snap.ids, x.state, x.rows) for x in self.cache]
