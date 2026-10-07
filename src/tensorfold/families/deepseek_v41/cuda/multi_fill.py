"""Rank 0's choice between a prompt's span and a decode round over the lanes, and which filling prompt goes next."""

from __future__ import annotations

from collections.abc import Callable, Sequence

FILL_GUARD = 8          # a prompt passed over this many spans goes next
FILL_SPANS = 4          # spans of one prompt a step when no lane decodes
BUSY_ROWS = 512         # span rows of a prompt admitted while lanes decode


def span_rows(rows: int, decoding: bool) -> int:
    """A new prompt's span rows for prompt buffers of ``rows``: BUSY_ROWS at most while lanes decode."""

    return min(BUSY_ROWS, rows) if decoding else rows


class FillPlan:
    """Decode-share credit: after a span of s seconds credit = min(credit, 0) + share * s, rounds spend their seconds,
    and spans run while credit <= 0 (``share`` 0: whole prompts first)."""

    def __init__(self, share: float) -> None:
        if not share >= 0:
            raise ValueError(f"a decode share of {share}: 0 or more")
        self.share = float(share)
        self.reset()

    def reset(self) -> None:
        self.credit = 0.0
        self.passed: dict[int, int] = {}              # stream id -> spans other prompts took since its last
        self.timers: list[Callable[[], float]] = []   # spans run, their seconds read at the next decision

    def order(self, filling: Sequence) -> list:
        """Overdue prompts first (most passed over), then foreground, fewest rows left, and arrival order."""

        def key(s):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return not due, -passed if due else 0, s.background, s.fill.rows_left

        return sorted(filling, key=key)

    def next(self, filling: Sequence, decoding: bool, arrived: bool = False) -> tuple[object | None, int]:
        """The prompt to fill and how many spans, or (None, 0) for a decode round; ``arrived``: a request waits."""

        for timer in self.timers:
            self.credit = min(self.credit, 0.0) + self.share * timer()
        self.timers = []
        if not filling:
            self.credit = 0.0                         # no prompt waits: rounds owe nothing
            return None, 0
        if not decoding:
            return self.order(filling)[0], 1 if arrived else FILL_SPANS
        return (self.order(filling)[0], 1) if self.credit <= 0.0 else (None, 0)

    def spanned(self, s, filling: Sequence, timer: Callable[[], float]) -> None:
        """``s`` ran a span, ``timer`` its seconds once it has run; the other prompts were passed over."""

        for x in filling:
            self.passed[x.sid] = 0 if x is s else self.passed.get(x.sid, 0) + 1
        for sid in self.passed.keys() - {x.sid for x in filling}:
            del self.passed[sid]
        self.timers.append(timer)

    def spent(self, seconds: float) -> None:
        """A decode round while prompts fill."""

        self.credit -= seconds
