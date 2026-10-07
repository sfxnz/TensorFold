"""Golden bits of the solo engine on the tiny checkpoint: tokens, each kept row's logits sha256 and the draft stats of
greedy and keyed, serial and drafted, long and resumed requests, keyed by the torch, triton and device that made them.

``TF_DSV41_GOLDEN_RECORD=1`` writes them into the fixture; without a matching key the test skips and names it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
triton = pytest.importorskip("triton")
pytest.importorskip("safetensors")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

import dsv41_tiny
from dsv41_ref_weights import RefWeights
from test_dsv41_decode import CAP, CONF, GRID, KEYED, LONG, PROMPTS, REPLY, SEEDS, _engine, _ids, _logit_rows

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import BLOCK, loader, snapshot
from tensorfold.families.deepseek_v41.cuda import prefill as P

tiny_dir = dsv41_tiny.tiny_dir          # the session fixture
GOLDEN = Path(__file__).parents[1] / "fixtures" / "deepseek_v41" / "golden_tiny.json"
RECORD = os.environ.get("TF_DSV41_GOLDEN_RECORD") == "1"
POLICIES = {"serial": None, "d3": (3, None), "conf": (BLOCK, CONF)}
RESUMED = (37, 128, 129, 300)           # odd and even, either side of the ring's wrap
RESUME_PROMPT = 301


def _key() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"torch {torch.__version__}, triton {triton.__version__}, sm {major}.{minor}"


def _cases() -> dict[str, tuple]:
    """name -> (prompt seed, prompt length, sampling, policy, resume point)."""

    out = {}
    for i, (t, k, p) in enumerate(GRID):
        seed = SEEDS[i % len(SEEDS)]
        for name, policy in POLICIES.items():
            out[f"grid t{t} k{k} p{p} seed{seed} {name}"] = (seed, PROMPTS[seed], Sampling(seed, t, k, p), policy, None)
    for name, policy in POLICIES.items():
        out[f"long keyed {name}"] = (12, LONG, KEYED, policy, None)
        out[f"long greedy {name}"] = (12, LONG, None, policy, None)
    out["resume fresh"] = (8, RESUME_PROMPT, KEYED, (BLOCK, CONF), None)
    for K in RESUMED:
        out[f"resume at {K}"] = (8, RESUME_PROMPT, KEYED, (BLOCK, CONF), K)
    return out


CASES = _cases()


def _load() -> dict:
    return json.loads(GOLDEN.read_text()) if GOLDEN.is_file() else {}


def _save(key: str, name: str, got: dict) -> None:
    data = _load()
    data.setdefault(key, {})[name] = got
    GOLDEN.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")


@pytest.fixture(scope="module")
def ref(tiny_dir):
    return RefWeights(tiny_dir)


@pytest.fixture(scope="module")
def tiny(tiny_dir):
    return loader.load(tiny_dir, Config.read(tiny_dir), 0, 1, None, capacity=LONG + 2 * REPLY)


@pytest.fixture(scope="module")
def engines(tiny, ref):
    return {CAP: _engine(tiny, ref, graphs=True), LONG: _engine(tiny, ref, capacity=LONG + 2 * REPLY, graphs=True)}


def _resume(e, prompt: list[int], K: int) -> snapshot.Snapshot:
    """The snapshot kept at K, its rows saved, then overwritten by another request and loaded back."""

    kept: list[snapshot.Snapshot] = []
    P.prefill(e, prompt[:K + 1], KEYED, keep_at=K, keep=kept.append)
    snap = kept.pop()
    arena = torch.empty((snapshot.row_bytes(e, snap),), dtype=torch.uint8, device="cuda")
    snapshot.save_rows(e, snap, arena)
    _logit_rows(e, _ids(K, 700, e.w.cfg.vocab_size), REPLY, KEYED, (BLOCK, None))
    snapshot.load_rows(e, snap)
    snap.rows = None
    return snap


def _run(engines, name: str) -> dict:
    seed, n, sampling, policy, K = CASES[name]
    e = engines[LONG if n == LONG else CAP]
    prompt = _ids(seed, n, e.w.cfg.vocab_size)
    kw = {} if K is None else {"resume": _resume(e, prompt, K)}
    res, rows = _logit_rows(e, prompt, REPLY, sampling, policy, **kw)
    return {"tokens": res.tokens, "rows": rows, "drafted": res.drafted, "accepted": res.accepted,
            "rounds": res.rounds}


@pytest.mark.parametrize("name", CASES)
def test_solo_bits_equal_the_golden_record(engines, name):
    key = _key()
    if RECORD:
        _save(key, name, _run(engines, name))
        return
    want = _load().get(key, {}).get(name)
    if want is None:
        pytest.skip(f"no golden bits for {key!r}, {name!r}: record them with TF_DSV41_GOLDEN_RECORD=1")
    got = _run(engines, name)
    assert got["tokens"] == want["tokens"], name
    assert got["rows"] == want["rows"], f"{name}: a kept row's logits differ"
    assert (got["drafted"], got["accepted"], got["rounds"]) == (want["drafted"], want["accepted"], want["rounds"])


def test_the_record_drafts_and_resumes_the_serial_bits():
    """Each recorded drafted reply has its serial reply's tokens and kept rows' logits; a resume is the fresh record."""

    data = _load()
    if not data:
        pytest.skip(f"no golden record at {GOLDEN.name}")
    for key, cases in data.items():
        assert set(cases) == set(CASES), key
        for name, got in cases.items():
            base = name.rsplit(" ", 1)[0] + " serial" if name.startswith(("grid", "long")) else "resume fresh"
            want = cases[base]
            assert len(got["tokens"]) == REPLY, (key, name)
            assert got["tokens"] == want["tokens"] and got["rows"] == want["rows"][:len(got["rows"])], (key, name)
            if name.startswith("resume"):
                assert got == want, f"{key}, {name}: the resume drafts as the fresh request"
