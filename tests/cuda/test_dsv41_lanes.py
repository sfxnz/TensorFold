"""The shared verify forward over 2, 3 and 4 lanes: each row's logits, window KV, taps, pooled rows and entry position
equal the legacy forward of its lane's window alone, for every total of rows and random layouts, and each lane's state
after its own commit and absorb (or the lanes' absorbs in one pass) equals legacy's; rows of a 24-row window equal each
row run alone; the lane graphs (verify by total rows, each lane's absorb and proposals) replay eager's bits.

Lanes are prefilled on their views with prompts across the tiny config's transitions (16 visible entries, 32-entry
candidate pools, the 128-slot ring) and up to 5000 tokens. The pack's check needs ``TF_DSV41_MODEL``.
"""

from __future__ import annotations

import hashlib
import os
import random
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_pair import pair, run_pair
from dsv41_ref_weights import RefWeights

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, GRAPH_ROWS, MAX_ROWS, dspark, engram, loader, proposals
from tensorfold.families.deepseek_v41.cuda import decode as D
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import prefill as P
from tensorfold.families.deepseek_v41.cuda.buffers import Buffers
from tensorfold.families.deepseek_v41.cuda.graphs import LaneGraphs
from tensorfold.families.deepseek_v41.cuda.lanes import Ahead, Lanes, stage_tables
from tensorfold.families.deepseek_v41.engram_hash import rank_columns
from tensorfold.families.deepseek_v41.engram_table import Reader

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
MODEL = os.environ.get("TF_DSV41_MODEL", "")
needs_model = pytest.mark.skipif(not MODEL or not Path(MODEL).is_dir(), reason="set TF_DSV41_MODEL to the checkpoint")
CAP = 5600
PROMPTS = {2: (5, 5000), 3: (30, 127, 600), 4: (14, 62, 128, 1010)}     # each lane's prompt length
RANDOM = 20                             # random layouts beside every total of rows
WIDE = 24                               # rows of the widest shared forward, four lanes of six
STARTS = [5, 30, 62, 125, 600, 1010]
PACK_CAP, PACK_PROMPTS = 1024, (40, 400, 129, 5)
KEYED = Sampling(seed=7, temperature=1.0, top_k=20, top_p=0.95)


def _ids(seed: int, n: int, vocab: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(2, vocab, (n,), generator=g).tolist()


def _u8(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(-1).contiguous().view(torch.uint8)


def _bits(x: torch.Tensor) -> str:
    return hashlib.sha256(_u8(x.detach()).cpu().numpy().tobytes()).hexdigest()


def _layout(g: random.Random, S: int, T: int) -> list[tuple[int, int]]:
    """Live lanes in ascending order with their rows, 1 to MAX_ROWS each, T in all."""

    n = g.randint(-(-T // MAX_ROWS), min(S, T))
    rows = [1] * n
    for _ in range(T - n):
        rows[g.choice([i for i, R in enumerate(rows) if R < MAX_ROWS])] += 1
    return list(zip(sorted(g.sample(range(S), n)), rows))


def _layouts(S: int, seed: int, randoms: int) -> list[list[tuple[int, int]]]:
    """A layout for every T from 1 to MAX_ROWS * S, then ``randoms`` of random T."""

    g = random.Random(seed)
    return [_layout(random.Random(100 * S + T), S, T) for T in range(1, MAX_ROWS * S + 1)] + [
        _layout(g, S, g.randint(1, MAX_ROWS * S)) for _ in range(randoms)]


def _lanes(w, ref, S: int, cap: int, reader) -> tuple[Lanes, Buffers, list[D.Engine]]:
    """S lane Engines on the views of one stack, sharing the prompt buffers and the shared forward's."""

    cfg, dev = w.cfg, w.device
    lanes = Lanes(cfg, S, cap, dev)
    pbuf = Buffers(cfg, w.world, min(2048, cap), cap, prefill=True, device=dev)
    mbuf = Buffers(cfg, w.world, MAX_ROWS * S, cap, device=dev, lanes=S)
    return lanes, mbuf, [D.Engine(w, cap, hasher=ref.hasher, reader=reader, st=lanes.view(k), pbuf=pbuf, dbuf=mbuf,
                                  dwork=dspark.Work(cfg, w.world, dev), ahead=Ahead(cfg, w.world)) for k in range(S)]


def _stage_rows(e: D.Engine, b: Buffers, s0: int, tokens: list[int]) -> None:
    """Lane e's window's Engram rows into ``b`` from row s0, its first row read ahead through the lane's block."""

    w = e.w
    if not w.engram:
        return
    e.fetch(0, e.st.history, tokens[0])
    ready, R = e._ready(tokens), len(tokens)
    assert ready == 1, "the row read ahead is the window's first"
    back = w.cfg.engram_max_ngram_size - 1
    ids = np.empty((0, *b.eraw.shape[1:3]), dtype=np.int64)
    if ready < R:
        before = (list(e.st.history) + tokens[:ready])[-back:]
        ids = rank_columns(e.hasher.ids(np.asarray(before, dtype=np.int64), np.asarray(tokens[ready:],
                                                                                       dtype=np.int64)),
                           w.rank, w.world)
    engram.stage_rows(ids, e.reader, e.ahead.eraw_host[0], e.ahead.eraw_done[0], b.eraw[s0:s0 + R],
                      scales=w.engram_scales, idx_host=e.ahead.eidx_host[0], idx=b.eidx[s0:s0 + R], first=ready)


def _same_row(a: Buffers, ra: int, b: Buffers, rb: int, layers: int, what: str) -> None:
    for name, x, y in (("logits", a.logits[ra], b.logits[rb]), ("window KV", a.kvw[:layers, ra], b.kvw[:layers, rb]),
                       ("taps", a.taps[ra], b.taps[rb]), ("pooled rows", a.cmp[:, ra], b.cmp[:, rb]),
                       ("entry position", a.epos[ra], b.epos[rb])):
        assert torch.equal(_u8(x), _u8(y)), f"{what}: {name}"


def _same_state(st, want, what: str, open_entry: bool = True) -> None:
    assert st.pos == want.pos == int(st.pos_dev[0]) and st.history == want.history, what
    views = zip(st.row_views(st.pos), want.row_views(want.pos))
    if not open_entry:          # each cache's last entry is open at pos, and a warm's forwards may have written it
        views = ((x[:-1], y[:-1]) for x, y in views)
    for name, x, y in (("rings", st.rings, want.rings), ("tail", st.tail, want.tail),
                       ("tail flags", st.tail_valid, want.tail_valid), *(("cache rows", x, y) for x, y in views)):
        assert torch.equal(_u8(x), _u8(y)), f"{what}: {name}"


def _sweep(w, ref, S: int, prompts, layouts, seed: int, cap: int, reader, together: bool = False) -> int:
    """Lanes and legacy twins prefilled alike; per layout one shared forward, each row checked against its lane's legacy
    window, then each live lane committed and absorbed at a random keep (``together``: the lanes' kept rows in one
    pass) and checked -> rows checked."""

    layers, vocab = w.cfg.num_hidden_layers, w.cfg.vocab_size
    lanes, mbuf, engines = _lanes(w, ref, S, cap, reader)
    legacy = [D.Engine(w, cap, hasher=ref.hasher, reader=reader) for _ in range(S)]
    batch = proposals.Batch(w.cfg, w.world, S, w.device, absorb=True) if together else None
    g, pending, checked = random.Random(seed), [], 0
    for k, n in enumerate(prompts):
        prompt = _ids(seed + k, n, vocab)
        first = P.prefill(engines[k], prompt, None)
        assert P.prefill(legacy[k], prompt, None) == first, f"lane {k}: first token"
        _same_state(engines[k].st, legacy[k].st, f"lane {k} after its {n}-token prompt")
        pending.append(first)
    for i, layout in enumerate(layouts):
        windows = [(k, engines[k].st.pos, [pending[k], *(g.randrange(2, vocab) for _ in range(R - 1))])
                   for k, R in layout]
        T, seg0 = stage_tables(mbuf, windows), []
        for k, _, tokens in windows:
            seg0.append(sum(len(t) for _, _, t in windows[:len(seg0)]))
            _stage_rows(engines[k], mbuf, seg0[-1], tokens)
        F.compute(w, lanes, mbuf, T, prompt=False, head_rows=T)
        torch.cuda.synchronize()
        for (k, pos, tokens), s0 in zip(windows, seg0):
            legacy[k].forward(tokens)
            for r in range(len(tokens)):
                _same_row(mbuf, s0 + r, legacy[k].dbuf, r, layers, f"layout {i} {layout}: lane {k} row {r} at {pos + r}")
                checked += 1
        kept = []
        for (k, _, tokens), s0 in zip(windows, seg0):
            R, keep = len(tokens), g.randint(1, len(tokens))
            for e, row0 in ((engines[k], s0), (legacy[k], 0)):
                F.commit(w, e.st, e.dbuf, R, keep, row0=row0)
                if w.dspark is not None and (batch is None or e is legacy[k]):
                    dspark.absorb(e, e.dbuf, keep, prompt=False, first=row0)
            kept.append((k, s0, keep, engines[k].st.pos - keep))
            pending[k] = g.randrange(2, vocab)
        if batch is not None and w.dspark is not None:
            proposals.absorb(w, lanes, mbuf, batch, kept)
        for k, _, keep, _ in kept:
            _same_state(engines[k].st, legacy[k].st, f"layout {i} {layout}: lane {k} kept {keep}")
    for k in range(S):
        _same_state(engines[k].st, legacy[k].st, f"lane {k} at the end")
    assert lanes.pos_dev.tolist() == [e.st.pos for e in engines]
    return checked


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=CAP)


@pytest.mark.parametrize("S", sorted(PROMPTS))
def test_shared_forward_rows_equal_each_lane_alone(tiny, ref, S):
    layouts = _layouts(S, S, RANDOM)
    assert {sum(R for _, R in x) for x in layouts} == set(range(1, MAX_ROWS * S + 1))
    rows = _sweep(tiny, ref, S, PROMPTS[S], layouts, 17 * S, CAP, ref.reader)
    assert rows == sum(R for x in layouts for _, R in x)


@pytest.mark.parametrize("S", sorted(PROMPTS))
def test_lanes_absorbing_together_equal_each_lane_alone(tiny, ref, S):
    """Every live lane's kept rows into its stage rings in one pass: each lane's rings equal its legacy twin's."""

    layouts = _layouts(S, 3 * S, RANDOM // 2)
    assert tiny.dspark is not None
    rows = _sweep(tiny, ref, S, PROMPTS[S], layouts, 19 * S, CAP, ref.reader, together=True)
    assert rows == sum(R for x in layouts for _, R in x)


def test_lane_tables_and_engines_refuse_what_they_cannot_run(tiny, ref):
    lanes, mbuf, engines = _lanes(tiny, ref, 2, CAP, ref.reader)
    for windows in ([], [(1, 0, [3]), (0, 0, [3])], [(0, 0, [3]), (0, 0, [3])], [(2, 0, [3])],
                    [(0, 0, [3] * (MAX_ROWS + 1))], [(1, CAP - 2, [3] * 3)]):
        with pytest.raises(ValueError):
            stage_tables(mbuf, windows)
    assert stage_tables(mbuf, [(0, 0, [3] * MAX_ROWS), (1, CAP - 2, [3, 4])]) == MAX_ROWS + 2
    assert mbuf.segs[:2].tolist() == [[0, MAX_ROWS], [MAX_ROWS, 2]] and mbuf.nseg.tolist() == [2]
    assert mbuf.rpos[MAX_ROWS:MAX_ROWS + 2].tolist() == [CAP - 2, CAP - 1] and mbuf.seg0[MAX_ROWS + 1] == MAX_ROWS
    with pytest.raises(RuntimeError, match="shared forward"):
        engines[0].forward([3])
    with pytest.raises(ValueError, match="commit"):
        F.commit(tiny, engines[1].st, mbuf, 3, 1, row0=mbuf.rows - 2)
    with pytest.raises(ValueError, match="positions"):
        D.Engine(tiny, CAP - 1, hasher=ref.hasher, reader=ref.reader, st=lanes.view(0))


@pytest.mark.parametrize("start", STARTS)       # the 16/32-entry transitions, the ring's wrap
def test_a_24_row_window_row_equals_the_row_run_alone(tiny, ref, start):
    """Each row of a verify forward of up to 24 rows, every layer through the head, gets the bits of the same row run
    alone as a 1-row window after the rows before it were committed: logits, window KV and DSpark taps."""

    cfg = tiny.cfg
    e = D.Engine(tiny, CAP, hasher=ref.hasher, reader=ref.reader, dbuf=Buffers(cfg, 1, WIDE, CAP))
    tokens = [P.prefill(e, _ids(start, start, cfg.vocab_size), None)] + _ids(start + 1, WIDE - 1, cfg.vocab_size)

    def rows(window: list[int]) -> list[tuple[str, str, str]]:
        logits = e.forward(window)
        return [(_bits(logits[r]), _bits(e.dbuf.kvw[:cfg.num_hidden_layers, r]), _bits(e.dbuf.taps[r]))
                for r in range(len(window))]

    windows = [rows(tokens[:R]) for R in range(2, WIDE + 1)]      # a forward commits nothing
    alone: list[tuple[str, str, str]] = []
    for t in tokens:
        alone += rows([t])
        F.commit(e.w, e.st, e.dbuf, 1, 1)
    for got in windows:
        assert got == alone[:len(got)], f"a {len(got)}-row window at {start}"


def _memory() -> tuple[int, int, int]:
    """(memory_reserved, the device's free bytes, the host's MemAvailable bytes or 0 where /proc has none)."""

    avail = 0
    if Path("/proc/meminfo").is_file():
        fields = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines() if ":" in line)
        avail = int(fields.get("MemAvailable", "0 kB").split()[0]) * 1024
    return torch.cuda.memory_reserved(), torch.cuda.mem_get_info()[0], avail


def _warm(w, lg: LaneGraphs, S: int) -> None:
    """Warm the lane graphs, printing the capture seconds and memory before and after as information."""

    torch.cuda.synchronize()
    torch.cuda.empty_cache()            # as capture does on entry, so the deltas are the graphs'
    before, start = _memory(), time.perf_counter()
    count = lg.warm()
    seconds, after = time.perf_counter() - start, _memory()
    print(f"\nlane graphs at {S} lanes: {count} captured in {seconds:.1f} s; "
          + "; ".join(f"{name} {a} -> {b} ({b - a:+d})" for name, a, b in
                      zip(("memory_reserved", "mem_get_info free", "MemAvailable"), before, after)))
    assert count == len(lg.verify) + sum(map(len, lg.absorb)) + sum(map(len, lg.drafts))
    assert sorted(lg.verify) == list(range(1, MAX_ROWS * S + 1))
    if w.dspark is not None:
        assert count == 22 * S
        assert all(sorted(a) == list(GRAPH_ROWS) for a in lg.absorb) and all(len(d) == 2 * BLOCK for d in lg.drafts)


def _proposal(e: D.Engine, y: int, sampling, d: int, run=None) -> tuple[list[int], list[float], str]:
    e.dwork.host.zero_()                # nothing left from the last proposal
    e.dbuf.dlog.zero_()
    drafts, conf = dspark.propose(e, y, e.st.pos - 1, sampling, d, run=run)
    return drafts, conf, _bits(e.dbuf.dlog)


@pytest.mark.parametrize("S", (2, 4))
def test_lane_graphs_replay_the_eager_forward_absorb_and_proposals(tiny, ref, S):
    """verify[T] for every T and random layouts gives eager's logits, window KV and taps; each lane's absorb graph
    (forward row 0) gives eager's stage rings; each lane's chain graphs give eager's drafts, confidences and block
    logits, and a legacy engine's at the same state."""

    w = tiny
    layers, vocab = w.cfg.num_hidden_layers, w.cfg.vocab_size
    lanes, mbuf, engines = _lanes(w, ref, S, CAP, ref.reader)
    with pytest.raises(ValueError, match="lane graphs"):
        LaneGraphs(w, lanes, mbuf, engines[::-1])
    lg = LaneGraphs(w, lanes, mbuf, engines)
    _warm(w, lg, S)
    for e in engines:
        assert e.st.pos == 0 and not e.st.history and not e.st.tail_valid.any()
    assert not lanes.pos_dev.any()
    legacy = [D.Engine(w, CAP, hasher=ref.hasher, reader=ref.reader) for _ in range(S)]
    g, pending = random.Random(S), []
    for k, n in enumerate(PROMPTS[S]):
        prompt = _ids(31 * S + k, n, vocab)
        pending.append(P.prefill(engines[k], prompt, None))
        assert P.prefill(legacy[k], prompt, None) == pending[-1], f"lane {k}: first token"
    for k, e in enumerate(engines):                         # each lane alone, so its rows start at forward row 0
        for n in GRAPH_ROWS:
            tokens = [pending[k], *(g.randrange(2, vocab) for _ in range(g.randint(n, MAX_ROWS) - 1))]
            T = stage_tables(mbuf, [(k, e.st.pos, tokens)])
            _stage_rows(e, mbuf, 0, tokens)
            F.compute(w, lanes, mbuf, T, prompt=False, head_rows=T)
            legacy[k].forward(tokens)
            for x in (e, legacy[k]):
                F.commit(w, x.st, x.dbuf, T, n)
            saved = lanes.rings.clone()
            dspark.absorb(e, mbuf, n, prompt=False)
            want = _bits(lanes.rings)
            lanes.rings.copy_(saved)
            lg.absorb[k][n].replay()
            assert _bits(lanes.rings) == want, f"lane {k}: absorb[{n}] != eager"
            dspark.absorb(legacy[k], legacy[k].dbuf, n, prompt=False)
            _same_state(e.st, legacy[k].st, f"lane {k} kept {n} of {T}", open_entry=False)
            pending[k] = g.randrange(2, vocab)
        for keyed in (False, True):
            for d in range(1, BLOCK + 1):
                sampling = KEYED if keyed else None
                want = _proposal(legacy[k], pending[k], sampling, d)
                assert _proposal(e, pending[k], sampling, d) == want, f"lane {k}: eager proposal {d}, {keyed}"
                got = _proposal(e, pending[k], sampling, d, run=lambda _, d, keyed, k=k: lg.drafts[k][d, keyed].replay())
                assert got == want, f"lane {k}: drafts[{d}, {keyed}] != eager"
    for i, layout in enumerate(_layouts(S, S, RANDOM)):
        windows = [(k, engines[k].st.pos, [g.randrange(2, vocab) for _ in range(R)]) for k, R in layout]
        T, s0 = stage_tables(mbuf, windows), 0
        for k, _, tokens in windows:
            _stage_rows(engines[k], mbuf, s0, tokens)
            s0 += len(tokens)
        rows = (mbuf.logits[:T], mbuf.kvw[:layers, :T], mbuf.taps[:T])
        F.compute(w, lanes, mbuf, T, prompt=False, head_rows=T)
        want = [_bits(x) for x in rows]
        for x in rows:
            x.zero_()
        lg.verify[T].replay()
        assert [_bits(x) for x in rows] == want, f"layout {i} {layout}: verify[{T}] != eager"
        s0 = 0
        for k, _, tokens in windows:                       # lanes move on to random positions
            F.commit(w, engines[k].st, mbuf, len(tokens), g.randint(1, len(tokens)), row0=s0)
            s0 += len(tokens)


# -- the pack -----------------------------------------------------------------------------------------------------

@needs_model
def test_pack_shared_forward_rows_equal_each_lane_alone_on_two_ranks():
    cfg = Config.read(MODEL, 8)
    rw = RefWeights(MODEL, 8)
    S, comms = len(PACK_PROMPTS), pair()
    ws = [loader.load(MODEL, cfg, r, 2, comms[r], capacity=PACK_CAP) for r in range(2)]

    def rank(r):
        def go(_):
            return _sweep(ws[r], rw, S, PACK_PROMPTS, _layouts(S, 3, 4), 5, PACK_CAP, Reader(rw.reader.layout))
        return go

    got = run_pair(rank(0), rank(1), comms)
    assert got[0] == got[1] > 0
