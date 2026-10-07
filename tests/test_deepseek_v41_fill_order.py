"""Rank 0's fill plan over the lanes: which prompt fills next, span or decode round by the decode-share credit (span
seconds read at the next decision), FILL_SPANS when no lane decodes, and BUSY_ROWS spans for a prompt admitted while
lanes decode."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tensorfold.families.deepseek_v41.cuda.multi_fill import BUSY_ROWS, FILL_GUARD, FILL_SPANS, FillPlan, span_rows


def _stream(sid: int, rows: int, background: bool = False) -> SimpleNamespace:
    return SimpleNamespace(sid=sid, background=background, fill=SimpleNamespace(rows_left=rows))


def _sids(streams) -> list[int]:
    return [s.sid for s in streams]


def test_order_is_foreground_then_fewest_rows_then_arrival():
    plan = FillPlan(0.5)
    filling = [_stream(0, 900, True), _stream(1, 300), _stream(2, 900), _stream(3, 300), _stream(4, 10, True)]
    assert _sids(plan.order(filling)) == [1, 3, 2, 4, 0]


def test_a_prompt_passed_over_fill_guard_spans_goes_first():
    plan = FillPlan(0.0)
    big, small = _stream(0, 5000, True), _stream(1, 100)
    filling = [big, small]
    for i in range(FILL_GUARD):
        s, spans = plan.next(filling, decoding=True)
        assert s is small and spans == 1, f"span {i}: the foreground prompt with fewer rows"
        plan.spanned(s, filling, lambda: 0.1)
    assert plan.passed == {0: FILL_GUARD, 1: 0}
    s, _ = plan.next(filling, decoding=True)
    assert s is big, "overdue: passed over FILL_GUARD spans"
    plan.spanned(s, filling, lambda: 0.1)
    assert plan.passed == {0: 0, 1: 1} and plan.next(filling, decoding=True)[0] is small
    plan.spanned(small, [small], lambda: 0.1)
    assert plan.passed == {1: 0}, "a prompt that left the filling list is forgotten"


def test_the_most_passed_over_overdue_prompt_goes_first():
    plan = FillPlan(0.0)
    filling = [_stream(0, 50), _stream(1, 60), _stream(2, 10)]
    plan.passed = {0: FILL_GUARD, 1: FILL_GUARD + 2, 2: 0}
    assert _sids(plan.order(filling)) == [1, 0, 2]


@pytest.mark.parametrize("share", [0.25, 0.5, 4.0])
def test_credit_after_a_span_pays_for_rounds(share):
    plan, filling = FillPlan(share), [_stream(0, 4000)]
    s, spans = plan.next(filling, decoding=True)
    assert s is filling[0] and spans == 1, "credit starts at 0: a span"
    plan.spanned(s, filling, lambda: 2.0)
    rounds = 0
    while plan.next(filling, decoding=True) == (None, 0):
        plan.spent(0.125)                               # exact in binary: no rounding decides a step
        rounds += 1
    assert rounds == share * 2.0 / 0.125 and plan.credit == 0.0, "rounds spend share x the span's seconds"


def test_credit_rule_with_given_span_times():
    plan, filling = FillPlan(0.5), [_stream(0, 4000)]
    assert plan.next(filling, decoding=True)[0] is not None
    plan.spanned(filling[0], filling, lambda: 2.0)
    assert plan.next(filling, decoding=True) == (None, 0) and plan.credit == pytest.approx(1.0)
    plan.spent(0.6)
    assert plan.next(filling, decoding=True) == (None, 0) and plan.credit == pytest.approx(0.4)
    plan.spent(0.6)
    assert plan.next(filling, decoding=True)[0] is filling[0] and plan.credit == pytest.approx(-0.2)
    plan.spanned(filling[0], filling, lambda: 1.0)
    assert plan.next(filling, decoding=True) == (None, 0)
    assert plan.credit == pytest.approx(0.3), "a debt carries over: min(-0.2, 0) + 0.5 * 1.0"
    plan.spent(0.1)
    plan.spanned(filling[0], filling, lambda: 0.0)     # an unspent credit does not carry over
    plan.next(filling, decoding=True)
    assert plan.credit == pytest.approx(0.0)


def test_share_zero_fills_whole_prompts_first():
    plan, filling = FillPlan(0.0), [_stream(0, 4000)]
    for _ in range(20):
        s, spans = plan.next(filling, decoding=True)
        assert s is filling[0] and spans == 1
        plan.spanned(s, filling, lambda: 3.0)


def test_span_seconds_are_read_at_the_next_decision():
    plan, filling, reads = FillPlan(0.5), [_stream(0, 4000)], []

    def timer():
        reads.append(1)
        return 1.0

    plan.spanned(filling[0], filling, timer)
    plan.spanned(filling[0], filling, timer)
    assert reads == [], "a span's seconds are not read where it launches"
    plan.next(filling, decoding=True)
    assert reads == [1, 1] and plan.credit == pytest.approx(0.5) and plan.timers == []
    plan.next(filling, decoding=True)
    assert reads == [1, 1], "each span's seconds count once"


def test_no_lane_decoding_runs_fill_spans_and_one_when_a_request_waits():
    plan = FillPlan(0.5)
    filling = [_stream(0, 4000), _stream(1, 20)]
    plan.credit = 5.0
    s, spans = plan.next(filling, decoding=False)
    assert s is filling[1] and spans == FILL_SPANS, "nothing decodes: the credit does not hold a prompt back"
    assert plan.next(filling, decoding=False, arrived=True) == (filling[1], 1)


def test_nothing_filling_decodes_and_owes_nothing():
    plan = FillPlan(0.5)
    plan.credit = -7.0
    assert plan.next([], decoding=True) == (None, 0) and plan.credit == 0.0
    assert plan.next([], decoding=False) == (None, 0)
    plan.passed, plan.timers = {3: 2}, [lambda: 1.0]
    plan.reset()
    assert (plan.credit, plan.passed, plan.timers) == (0.0, {}, [])


def test_span_rows_while_lanes_decode():
    assert span_rows(2048, decoding=False) == 2048
    assert span_rows(2048, decoding=True) == BUSY_ROWS == 512
    assert span_rows(129, decoding=True) == 129 and span_rows(7, decoding=False) == 7


def test_a_negative_share_is_refused():
    with pytest.raises(ValueError, match="decode share"):
        FillPlan(-0.1)
    with pytest.raises(ValueError, match="decode share"):
        FillPlan(float("nan"))
