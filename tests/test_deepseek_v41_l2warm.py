"""DeepSeek-V4.1's L2 warm on the host: the switch, the ranges a gather warms in reading order, and only decode
forwards warm, each share before its gather and joined after the head."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import l2warm

MB = 1 << 20


@pytest.mark.parametrize("value, want", [("", None), ("0", None), (" 0 ", None), ("150", (150.0, l2warm.CAP_MB * MB)),
                                         ("100:4", (100.0, 4 * MB)), (" 200:0.5 ", (200.0, MB // 2))])
def test_the_switch_reads_a_rate_and_a_cap(value, want):
    assert l2warm.wanted({l2warm.WARM_ENV: value}) == want
    assert l2warm.wanted({}) is None


@pytest.mark.parametrize("value", ["yes", "-1", "0.5", "0:8", "150:", "150:0", "150:-2", "nan", "inf", "150:inf", "1:2:3"])
def test_another_value_is_refused_naming_the_variable(value):
    with pytest.raises(ValueError, match=l2warm.WARM_ENV):
        l2warm.wanted({l2warm.WARM_ENV: value})


def _covers(got, tensors, cap):
    """Each range is the whole 32-byte sectors inside its tensor's bytes up to ``cap``, in order -> bytes left."""

    left, S = cap, l2warm.SECTOR
    for (lo, sectors), t in zip(got, tensors):
        a, take = t.data_ptr(), min(t.numel() * t.element_size(), left)
        assert lo % S == 0 and a <= lo < a + S and a + take - S < lo + sectors * S <= a + take
        left -= take
    return left


def test_ranges_cover_the_first_cap_bytes_in_order():
    whole = torch.zeros(1 << 16, dtype=torch.uint8)
    parts = [whole[3:1003], whole[2048:2048 + 64], whole[5000:9001], torch.zeros(10, 4)]
    total = sum(t.numel() * t.element_size() for t in parts)
    for cap in (100, 1000, 1040, 1064, 3000, total, total + 99):
        got = l2warm.ranges(parts, cap)
        assert len(got) == (1 + sum(cap > s for s in (1000, 1064, 5065)) if cap < total else len(parts)), cap
        assert _covers(got, parts, cap) == max(0, cap - total)
    assert l2warm.ranges(parts, 31) == [] and l2warm.ranges([whole[1:40]], 1 << 20) == []   # no whole sector


def test_ranges_skip_empty_tensors_and_stop_at_the_table_size():
    t = torch.zeros(64, dtype=torch.uint8)
    assert l2warm.ranges([t[:0], t], 1 << 20) == l2warm.ranges([t], 1 << 20) != []
    many = [torch.zeros(64, dtype=torch.uint8) for _ in range(l2warm.MAX_RANGES + 3)]
    assert len(l2warm.ranges(many, 1 << 20)) == l2warm.MAX_RANGES


def _lin(tag):
    return SimpleNamespace(bs=f"{tag}.bs", w8=f"{tag}.w8")


def _layer(i):
    return SimpleNamespace(index=i, hc_attn=SimpleNamespace(fn=f"{i}.hc_attn"), hc_ffn=SimpleNamespace(fn=f"{i}.hc_ffn"),
                           attn=SimpleNamespace(wqa_kv=_lin(f"{i}.wqa_kv")),
                           moe=SimpleNamespace(gate=f"{i}.gate", shared_gu=_lin(f"{i}.shared_gu")))


def test_each_gather_warms_what_is_read_after_it():
    w = SimpleNamespace(layers=[_layer(0), _layer(1), _layer(2)], engram={1: SimpleNamespace(wkv=_lin("1.engram"))},
                        norm="norm", head=_lin("head"))
    assert l2warm.reads(w) == [
        (["0.hc_ffn", "0.gate", "0.shared_gu.bs", "0.shared_gu.w8"],
         ["1.engram.bs", "1.engram.w8", "1.hc_attn", "1.wqa_kv.bs", "1.wqa_kv.w8"]),
        (["1.hc_ffn", "1.gate", "1.shared_gu.bs", "1.shared_gu.w8"], ["2.hc_attn", "2.wqa_kv.bs", "2.wqa_kv.w8"]),
        (["2.hc_ffn", "2.gate", "2.shared_gu.bs", "2.shared_gu.w8"], ["norm", "head.bs", "head.w8"])]


class _Warm:
    def __init__(self, log):
        self.log = log

    def ahead(self, index, k):
        self.log.append(("warm", index, k))

    def join(self):
        self.log.append("join")


@pytest.mark.parametrize("on", [False, True])
@pytest.mark.parametrize("prompt", [False, True])
def test_only_decode_forwards_warm_each_share_before_its_gather(monkeypatch, on, prompt):
    log = []

    def block(lw, w, st, b, R, e, prompt, start, kv_from, taps_from):
        assert (yield start) == "gathered"
        assert (yield start) == "gathered"

    monkeypatch.setattr(F, "_block", block)
    monkeypatch.setattr(F, "_gather", lambda w, b, R, start: log.append("gather") or "gathered")
    F.layer(SimpleNamespace(index=7), SimpleNamespace(warm=_Warm(log) if on else None), None, None, 3, None, prompt)
    if on and not prompt:
        assert log == [("warm", 7, 0), "gather", ("warm", 7, 1), "gather"]
    else:
        assert log == ["gather", "gather"]


@pytest.mark.parametrize("prompt", [False, True])
def test_a_decode_forward_joins_the_warm_after_its_head(monkeypatch, prompt):
    log = []
    w = SimpleNamespace(cfg=SimpleNamespace(hidden_size=4, hc_mult=1), layers=["L0", "L1"], engram={}, world=1, embed=None,
                        warm=_Warm(log))
    b = SimpleNamespace(rows=4, logits=torch.zeros(4, 2), ids=torch.zeros(4, dtype=torch.int64),
                        X=torch.zeros(4, 4), pre_in=torch.zeros(4, 1))
    monkeypatch.setattr(F.glue, "embed", lambda *a: None)
    monkeypatch.setattr(F, "layer", lambda lw, *a: log.append(lw))
    monkeypatch.setattr(F, "head", lambda *a, **k: log.append("head") or "logits")
    assert F.compute(w, None, b, 2, prompt=prompt, head_rows=1) == "logits"
    assert log == ["L0", "L1", "head"] + ([] if prompt else ["join"])
