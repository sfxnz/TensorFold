"""DeepSeek-V4.1-Flash's startup on the host, the pack's headers and a fake rank pair: each --parallel lane holds a whole
window in admission, and an engine with lanes serves through a scheduler of that many streams."""

from __future__ import annotations

import json
import shutil
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from test_deepseek_v41_split import FIXTURE, inventory

from tensorfold.cuda import build, capacity
from tensorfold.families.deepseek_v41.cuda import engine as E
from tensorfold.families.deepseek_v41.cuda import lanes, loader, protocol

GIB = capacity.GIB
NATIVE = 1 << 20
BUDGET = int(103.8 * GIB)       # an idle GB10's MemAvailable less the admission reserve
TIGHT = int(81.5 * GIB)         # one lane's native window fits, four lanes' do not


class Loaded(Exception):
    pass


@pytest.fixture
def start(tmp_path, monkeypatch):
    """``start(lanes, context, budget)`` builds rank 0 against the pack's headers up to its weights, which raise Loaded
    (``load``: fake weights instead) -> the engine object."""

    shutil.copy(FIXTURE, tmp_path / "config.json")
    names = inventory(json.loads(FIXTURE.read_text()))
    for name in ("TF_DSV41_CACHE_GIB", "TF_DSV41_CACHE_ENTRIES"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TF_DSV41_L2WARM", "0")         # fake weights: nothing to warm
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a: None)
    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "headers", lambda model_dir, **kw: names)
    monkeypatch.setattr(capacity, "unified", lambda t: False)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: None)
    monkeypatch.setattr(protocol, "gather_ints", lambda comm, values, device="cuda": [list(values), list(values)])
    monkeypatch.setattr(E.DeepSeekV41Engine, "_engram", staticmethod(lambda model_dir, cfg: (None, None)))

    def run(lanes, context=None, budget=BUDGET, load=None):
        def weights(model_dir, cfg, rank, world, comm, **kw):
            if load is None:
                raise Loaded
            return load(cfg, comm)

        monkeypatch.setattr(capacity, "available_bytes", lambda t: budget)
        monkeypatch.setattr(loader, "load", weights)
        comm = SimpleNamespace(barrier=lambda: None, ready=lambda *a, **kw: None, store=None)
        obj = E.DeepSeekV41Engine.__new__(E.DeepSeekV41Engine)
        try:
            obj.__init__(tmp_path, rank=0, master="", port=0, policy=(5, 0.15), context=context,
                         context_explicit=context is not None, lanes=lanes, comm=comm, graphs=False)
        except Loaded:
            pass
        return obj

    return run


def test_one_lane_keeps_the_native_estimate(start, capsys):
    obj = start(1)
    assert obj.limit == NATIVE and not obj.concurrent
    assert "CUDA rank 0 startup estimate 81.32 GiB" in capsys.readouterr().out


def test_four_lanes_fit_the_native_window_in_a_larger_estimate(start, capsys):
    obj = start(4)
    assert obj.limit == NATIVE and obj.concurrent
    assert "CUDA rank 0 startup estimate 82.55 GiB" in capsys.readouterr().out


def test_without_context_the_window_shrinks_to_fit_every_lane(start):
    assert start(1, budget=TIGHT).limit == NATIVE
    window = start(4, budget=TIGHT).limit
    assert 0 < window < NATIVE


def test_an_explicit_context_too_big_for_the_lanes_is_refused_naming_parallel(start):
    assert start(1, NATIVE, budget=TIGHT).limit == NATIVE
    with pytest.raises(ValueError, match=r"largest fitting .* With --parallel 4, each of the 4 lanes holds a whole "
                                         r"window: lower --parallel or --context"):
        start(4, NATIVE, budget=TIGHT)
    fits = start(4, budget=TIGHT).limit
    assert start(4, fits, budget=TIGHT).limit == fits


def test_an_engine_with_lanes_serves_through_a_scheduler_of_that_many_streams(start, monkeypatch):
    def load(cfg, comm):
        return SimpleNamespace(cfg=cfg, world=2, rank=0, device=torch.device("meta"), dspark=None, engram=False,
                               comm=comm)

    monkeypatch.setattr(lanes, "Ahead", lambda cfg, world: SimpleNamespace())
    monkeypatch.setattr(E.DeepSeekV41Engine, "_warm_lanes", lambda self: None)
    obj = start(4, 65536, load=load)
    try:
        assert obj.concurrent and obj.scheduler.max_streams == 4 and obj.scheduler.decoder is obj.decoder
        assert obj.decoder.lanes.slots == 4 and obj.decoder.plan.share == 0.5 and obj.decoder.link is not None
        assert [e.st for e in obj.engines] == [obj.lanes.view(k) for k in range(4)]
        assert len({id(e.reader) for e in obj.engines}) == 1, "every lane reads Engram through one reader"
    finally:
        obj.close()
    assert obj.scheduler is None
    one = start(1, 65536)
    assert not one.concurrent and one.scheduler is None
