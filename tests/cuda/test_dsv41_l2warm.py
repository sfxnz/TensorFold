"""DeepSeek-V4.1's L2 warm kernel: it writes nothing, keeps its pace, and replays from a graph beside its stream."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.deepseek_v41.cuda import l2warm


def _table(tensors, cap=1 << 40) -> torch.Tensor:
    return torch.tensor(l2warm.ranges(tensors, cap), dtype=torch.int64, device="cuda")


def test_warming_writes_nothing():
    g = torch.Generator(device="cuda").manual_seed(0)
    whole = torch.randint(0, 256, (3 << 20,), generator=g, dtype=torch.uint8, device="cuda")
    keep = whole.clone()
    parts = [whole[1:1000], whole[4093:(1 << 20) + 7], whole[(2 << 20) + 31:]]    # unaligned starts and ends
    l2warm._ext().warm(_table(parts), 200.0, l2warm.CTAS)
    l2warm._ext().warm(_table(parts[:1]), 50.0, 1)
    torch.cuda.synchronize()
    assert torch.equal(whole, keep)


@pytest.mark.parametrize("gb_per_s", [50.0, 100.0])
def test_warming_keeps_its_pace(gb_per_s):
    data = torch.empty(32 << 20, dtype=torch.uint8, device="cuda")
    table = _table([data])
    l2warm._ext().warm(table, gb_per_s, l2warm.CTAS)        # loaded and launched once
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    l2warm._ext().warm(table, gb_per_s, l2warm.CTAS)
    end.record()
    end.synchronize()
    want_ms = data.numel() / gb_per_s / 1e6
    took = start.elapsed_time(end)
    assert 0.95 * want_ms <= took <= 5 * want_ms + 1.0, f"{data.numel()} bytes at {gb_per_s} GB/s in {took:.3f} ms"


def test_a_captured_warm_replays_beside_its_stream():
    w8 = torch.ones(4 << 20, dtype=torch.uint8, device="cuda")
    bs = torch.ones(1 << 17, dtype=torch.uint8, device="cuda")
    fn = torch.ones(24, 4096, device="cuda")
    lin = SimpleNamespace(bs=bs, w8=w8)
    lw = SimpleNamespace(index=3, hc_ffn=SimpleNamespace(fn=fn), hc_attn=SimpleNamespace(fn=fn),
                         moe=SimpleNamespace(gate=fn, shared_gu=lin))
    w = SimpleNamespace(layers=[lw], engram={}, norm=fn, head=lin, device="cuda")
    warm = l2warm.Warm(w, 150.0, 2 << 20)
    assert sorted(warm.tables) == [(3, 0), (3, 1)]
    x = torch.arange(1024, dtype=torch.float32, device="cuda")
    y = torch.empty_like(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        y.copy_(x * 2)
        warm.ahead(3, 0)
        y.add_(1)
        warm.ahead(3, 1)
        warm.ahead(9, 0)                # no table: nothing launched
        warm.join()
    x.fill_(5.0)
    graph.replay()
    torch.cuda.synchronize()
    assert bool((y == 11.0).all())
