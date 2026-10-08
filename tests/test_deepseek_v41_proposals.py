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
from tensorfold.families.deepseek_v41.cuda import BLOCK, MAX_LANES, dspark, geometry, proposals, sample
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
