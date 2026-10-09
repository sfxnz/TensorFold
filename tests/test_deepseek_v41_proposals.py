"""DeepSeek-V4.1's proposals of several lanes in one block, on the host: the scratch's shapes and row tables, the
geometry counting its bytes, the slot table's lanes, tokens and knobs, and every lane keeping what its own proposal
keeps from the same drafts and confidences."""

from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_LANES, dspark, geometry, layout, proposals, sample
from tensorfold.families.deepseek_v41.cuda import multi as M
from tensorfold.families.deepseek_v41.cuda.buffers import _device_bytes

CFG = Config.read(Path(__file__).parent / "fixtures" / "deepseek_v41")
META = torch.device("meta")
KEYED = (Sampling(seed=7, temperature=0.7, top_k=0, top_p=0.9, min_p=0.05),
         Sampling(seed=(1 << 64) - 1, temperature=1.0, top_k=20, top_p=1.0))


class _Event:
    def record(self) -> None:
        pass

    def synchronize(self) -> None:
        pass


def _host_batch(slots: int = MAX_LANES) -> proposals.Batch:
    k = proposals.Batch(CFG, 2, slots, "cpu")
    k.staged, k.landed = _Event(), [_Event() for _ in range(BLOCK)]
    return k


def test_scratch_shapes_row_tables_and_geometry():
    assert CFG.dspark_block_size == BLOCK
    for slots in range(2, MAX_LANES + 1):
        k, n = proposals.Batch(CFG, 2, slots, META), BLOCK * slots
        assert k.xd.shape == (n, CFG.hc_mult, CFG.hidden_size) and k.bkv.shape == (n, CFG.head_dim)
        assert k.dlog.shape == (n, CFG.vocab_size // 2) and k.dplan.rows == n
        assert k.chain.shape == (BLOCK + 1, slots) and k.me.shape == (BLOCK, slots, CFG.dspark_markov_rank)
        assert k.host.shape == (BLOCK, 2, slots) and k.got.numel() == 2 * slots * k.width
        added = (geometry.dsv41_geometry(CFG, 2, lanes=slots, batched=True).bytes_at(4096)
                 - geometry.dsv41_geometry(CFG, 2, lanes=slots).bytes_at(4096))
        assert added == _device_bytes(k, META) > 0
    k = _host_batch()
    rows = range(BLOCK * MAX_LANES)
    assert k.slot.tolist() == [r // BLOCK for r in rows] and k.step.tolist() == [r % BLOCK for r in rows]
    assert k.lists.tolist() == [[r // BLOCK * BLOCK + t for t in range(BLOCK)] for r in rows]    # its slot's rows
    assert k.counts.tolist() == [BLOCK] * len(rows) and k.ids.tolist() == [CFG.dspark_noise_token_id] * len(rows)


def test_the_slot_table_holds_lanes_tokens_and_each_keyed_slots_knobs():
    k = _host_batch()
    assert proposals.set_slots(k, [(3, 11, None), (0, 12, KEYED[0]), (2, 13, KEYED[1])]) == 2
    t = k.table.numpy()
    assert t[:3, :2].tolist() == [[3, 11], [0, 12], [2, 13]] and not t[0, 2:].any() and not t[3].any()
    draws = sample.Draws(CFG.vocab_size, 2, "cpu")
    for row, s in ((1, KEYED[0]), (2, KEYED[1])):
        assert draws.set(s)
        assert t[row, 2:].tolist() == draws.knobs.tolist(), "the knobs a lane's own Draws writes"
    with pytest.raises(ValueError, match="greedy slots first"):
        proposals.set_slots(k, [(0, 1, KEYED[0]), (1, 2, None)])


def _alone(drafts: list[int], conf: list[float], y: int, sampling, d: int, threshold) -> tuple:
    """``dspark.propose`` of one lane whose chain lands ``drafts`` and ``conf`` -> its result and landings."""

    work = dspark.Work(CFG, 2, "cpu")
    work.landed = [_Event() for _ in range(BLOCK)]
    e = SimpleNamespace(w=SimpleNamespace(cfg=CFG), st=SimpleNamespace(pos=101), dwork=work)

    def run(e, d, keyed):
        work.host[0, :d] = torch.tensor(drafts[:d], dtype=torch.int32)
        work.host[1, :d] = torch.tensor(conf[:d], dtype=torch.float32).view(torch.int32)

    seen = []
    out = dspark.propose(e, y, 100, sampling, d, threshold, run=run, landed=lambda i, t: seen.append((i, t)))
    return out, seen


@pytest.mark.parametrize("threshold", [None, 0.3])
def test_each_lane_keeps_what_its_own_proposal_keeps(threshold):
    rng = random.Random(5)
    wanted = [(2, 41, KEYED[0], 5), (0, 42, None, 3), (3, 43, None, 1), (1, 44, KEYED[1], 4)]
    drafts = {lane: [rng.randrange(2, CFG.vocab_size) for _ in range(BLOCK)] for lane, *_ in wanted}
    conf = {lane: [rng.uniform(-1.0, 4.0) for _ in range(BLOCK)] for lane, *_ in wanted}
    k, runs, seen = _host_batch(), [], []

    def run(L: int, keyed: int, d: int) -> None:
        runs.append((L, keyed, d))
        for j, lane in enumerate(k.table[:L, 0].tolist()):           # the slot table's lanes, greedy ones first
            k.host[:d, 0, j] = torch.tensor(drafts[lane][:d], dtype=torch.int32)
            k.host[:d, 1, j] = torch.tensor(conf[lane][:d], dtype=torch.float32).view(torch.int32)

    got = proposals.propose(k, wanted, threshold, run=run, landed=lambda x, i, t: seen.append((x, i, t)))
    assert runs == [(4, 2, 5)] and k.table[:4, 0].tolist() == [0, 3, 2, 1]
    for x, (lane, y, s, d) in enumerate(wanted):
        want, landings = _alone(drafts[lane], conf[lane], y, s, d, threshold)
        assert got[x] == want, lane
        assert [(i, t) for z, i, t in seen if z == x] == landings, lane
    assert all(len(dr) for dr, _ in got)


def test_a_proposal_takes_two_to_slots_lanes_of_one_to_block_drafts():
    k = _host_batch(2)
    for wanted in ([(0, 1, None, 3)], [(0, 1, None, 3)] * 3, [(0, 1, None, 0), (1, 1, None, 2)],
                   [(0, 1, None, BLOCK + 1), (1, 1, None, 2)]):
        with pytest.raises(ValueError, match="propose"):
            proposals.propose(k, wanted, None, run=lambda *a: None)


def test_the_setup_and_absorb_knobs_make_one_setting(monkeypatch):
    for name in (proposals.ENV, proposals.SETUP_ENV, proposals.ABSORB_ENV):
        monkeypatch.delenv(name, raising=False)
    assert proposals.mode() == 0 and proposals.setup() == "early" and not proposals.absorbing()
    monkeypatch.setenv(proposals.ABSORB_ENV, "1")
    assert proposals.mode() == 0 and not proposals.absorbing(), "the absorb rides on the block"
    monkeypatch.setenv(proposals.ENV, "1")
    seen = set()
    for i, when in enumerate(proposals.SETUPS):
        monkeypatch.setenv(proposals.SETUP_ENV, when)
        for absorb in ("0", "1"):
            monkeypatch.setenv(proposals.ABSORB_ENV, absorb)
            assert proposals.setup() == when and proposals.mode() == 1 + 2 * i + 8 * (absorb == "1")
            seen.add(proposals.mode())
    assert len(seen) == 2 * len(proposals.SETUPS), "the ranks tell every choice apart"
    monkeypatch.setenv(proposals.SETUP_ENV, "soon")
    with pytest.raises(ValueError, match=proposals.SETUP_ENV):
        proposals.mode()


def test_the_absorb_tables_come_with_the_absorb_only():
    for slots in (2, MAX_LANES):
        plain, k = proposals.Batch(CFG, 2, slots, META), proposals.Batch(CFG, 2, slots, META, absorb=True)
        assert not plain.absorbs and not hasattr(plain, "kept") and k.absorbs
        assert k.kept.shape == (3, proposals.MAX_ROWS * slots) and k.dest.shape == (proposals.MAX_ROWS * slots,)

        added = (geometry.dsv41_geometry(CFG, 2, lanes=slots, batched=True, absorb=True).bytes_at(4096)
                 - geometry.dsv41_geometry(CFG, 2, lanes=slots, batched=True).bytes_at(4096))
        assert added == _device_bytes(k, META) - _device_bytes(plain, META) > 0


def test_kept_rows_map_to_their_forward_rows_ring_rows_and_positions():
    k = proposals.Batch(CFG, 2, MAX_LANES, "cpu", absorb=True)
    k.kept_staged = _Event()
    roles, window, lane = 7, 128, 7 * 128            # a lane's rings: roles x window rows
    kept = [(1, 0, 2, 126), (3, 6, 1, 40), (0, 9, 3, 255)]      # (lane, forward row, rows, position), wrapping
    assert proposals.stage_kept(k, roles, window, kept, proposals.MAX_ROWS * MAX_LANES) == (6, 12)
    rows, ring, pos = k.kept[:, :6].tolist()
    assert rows == [0, 1, 6, 9, 10, 11] and pos == [126, 127, 40, 255, 256, 257]
    assert ring == [lane + 126, lane + 127, 3 * lane + 40, 127, 0, 1], "dspark.absorb's slot: position % window"
    for bad in ([], [(0, 0, 0, 5)], [(0, 20, 5, 5)], [(0, -1, 2, 5)]):
        with pytest.raises(ValueError, match="absorb"):
            proposals.stage_kept(k, roles, window, bad, proposals.MAX_ROWS * MAX_LANES)


def test_a_deferred_block_is_built_at_the_first_round_with_two_drafting_lanes():
    built, alone = [], []

    def engine(lane: int, sparse: bool = False) -> SimpleNamespace:
        e = SimpleNamespace(st=SimpleNamespace(pos=50), dwork=SimpleNamespace(sparse=sparse))
        e.propose = lambda y, p, s, d, c: (alone.append(lane), ([7] * d, [0.0] * d))[1]
        return e

    def stream(lane: int, draft: bool = True) -> SimpleNamespace:
        return SimpleNamespace(lane=lane, out=[5], count=100, draft=draft, sampling=None)

    dec = SimpleNamespace(drafts=3, confidence=None, batch=None, defer=True, defer_absorb=True,
                          _drafting=lambda s: s.draft, build_batch=lambda absorb, lane: built.append((absorb, lane)))
    M.LaneDecoder._propose(dec, [stream(0)], [engine(0)])
    M.LaneDecoder._propose(dec, [stream(0), stream(2, draft=False)], [engine(0), engine(2)])
    M.LaneDecoder._propose(dec, [stream(1), stream(3)], [engine(1, sparse=True), engine(3)])
    assert not built and alone == [0, 0, 1, 3], "one drafting lane, or a sparse one, proposes alone"
    M.LaneDecoder._propose(dec, [stream(1), stream(3)], [engine(1), engine(3)])
    assert built == [(True, 1)], "built once two lanes draft, its graphs on the first one's rows"


def test_the_layout_dump_names_device_tensors_only(monkeypatch):
    obj = SimpleNamespace(a=torch.empty(4, device=META), host=torch.zeros(3), n=5,
                          d={"x": torch.empty((2, 2), dtype=torch.bfloat16, device=META)},
                          rows=[torch.empty(1, device=META), "y"])
    got = layout.tensors({"g": obj, "none": None})
    assert sorted(got) == ["g.a", "g.d[x]", "g.rows[0]"] and got["g.a"][1] == 16 and got["g.d[x]"][1] == 8
    monkeypatch.delenv(layout.ENV, raising=False)
    assert layout.dump("start", None) is None, "nothing without the variable"
