"""The lane decoder's two-rank ops on the host: an admission round-trips through the link (its prompt as base64 int32
bytes and their sha256), the state digest changes with every field of ``shape``, and building ``shape`` reads entry
lengths and cached digests, never ids."""

from __future__ import annotations

import copy
import threading
import time
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.deepseek_v41.cuda import multi_tp as T
from tensorfold.families.deepseek_v41.cuda.cache import ids_digest

SAMPLING = Sampling(seed=2**61 + 3, temperature=0.7, top_k=20, top_p=0.95, min_p=0.05)


class _Store:
    """The TCP store's set / wait / get / delete_key, in memory."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}
        self.cv = threading.Condition()

    def set(self, key, value) -> None:
        with self.cv:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.cv.notify_all()

    def wait(self, keys, timeout=None) -> None:
        with self.cv:
            if not self.cv.wait_for(lambda: all(k in self.data for k in keys), timeout=10):
                raise TimeoutError(keys)

    def get(self, key) -> bytes:
        return self.data[key]

    def delete_key(self, key) -> None:
        with self.cv:
            self.data.pop(key, None)


def test_an_admission_round_trips_through_the_link():
    prompt = [0, 1, 129279, 2**31 - 1, 7] * 300
    text, sha = T.pack_prompt(prompt)
    assert T.unpack_prompt(text) == (prompt, sha) and sha == ids_digest(prompt)
    s = Stream(prompt, 77, SAMPLING, draft=False, stop_eos=False, background=True)
    s.digest = sha
    plan = {"lane": 2, "cached": 41, "rows_in": "arena", "rows": 512}
    store = _Store()
    sender, receiver = T.Link(store), T.Link(store)
    sender.send(T.admit_op(5, s, text, plan))
    sender.send(["round", [3], {"fill": None, "spans": 0, "decode": True}])
    assert all(k.startswith("tf_dsv41/lanes/") for k in store.data), "the family's own store namespace"
    sid, got, told = T.read_admit(receiver.receive())
    assert (sid, told) == (5, plan)
    assert (got.prompt, got.count, got.draft, got.stop_eos, got.background) == (prompt, 77, False, False, True)
    assert got.sampling == SAMPLING and got.digest == got.sent == sha
    assert receiver.receive() == ["round", [3], {"fill": None, "spans": 0, "decode": True}]
    assert not store.data and T.Link.KEY == "tf_dsv41/lanes/socket"


def _dec(n: int = 1000) -> SimpleNamespace:
    """A lane decoder's host fields: two lanes, a decoding stream, a filling one, two kept entries."""

    def stream(sid, lane, out, fill):
        s = Stream([5] * 9, 30, SAMPLING if sid else None)
        s.sid, s.lane, s.out, s.fill, s.span_rows = sid, lane, out, fill, 512 if fill else None
        return s

    ids = list(range(n))
    entries = [SimpleNamespace(snap=SimpleNamespace(ids=ids), digest="ab" * 32, lane=None),
               SimpleNamespace(snap=SimpleNamespace(ids=ids[:5]), digest="cd" * 32, lane=1)]
    dec = SimpleNamespace(
        next_id=4, engines=[SimpleNamespace(st=SimpleNamespace(pos=40, history=[1, 2])),
                            SimpleNamespace(st=SimpleNamespace(pos=0, history=[]))],
        streams={2: stream(2, 0, [8, 9], None)}, filling=[stream(3, 1, [], SimpleNamespace(i=1))],
        kept=SimpleNamespace(cache=entries, entries=12), lanes=SimpleNamespace(slots=4, capacity=1024),
        drafts=5, confidence=0.15, eos=(1,), pbuf=SimpleNamespace(rows=2048), batch=None)
    dec.free = [2, 3]
    dec._free = lambda: list(dec.free)
    return dec


CHANGES = {
    "next id": lambda d: setattr(d, "next_id", 5),
    "a lane's pos": lambda d: setattr(d.engines[1].st, "pos", 1),
    "a lane's history": lambda d: d.engines[0].st.history.append(3),
    "stream sid": lambda d: setattr(d.streams[2], "sid", 7),
    "stream lane": lambda d: setattr(d.streams[2], "lane", 3),
    "len(out)": lambda d: d.streams[2].out.insert(0, 9),
    "last token": lambda d: d.streams[2].out.__setitem__(-1, 10),
    "count": lambda d: setattr(d.streams[2], "count", 31),
    "done": lambda d: setattr(d.streams[2], "done", True),
    "draft": lambda d: setattr(d.streams[2], "draft", False),
    "stop_eos": lambda d: setattr(d.streams[2], "stop_eos", False),
    "background": lambda d: setattr(d.filling[0], "background", True),
    "greedy to keyed": lambda d: setattr(d.streams[2], "sampling", Sampling(seed=0)),
    "the seed": lambda d: setattr(d.filling[0], "sampling", Sampling(seed=2**61 + 4, temperature=0.7, top_k=20,
                                                                     top_p=0.95, min_p=0.05)),
    "top_p by an ulp": lambda d: setattr(d.filling[0], "sampling", Sampling(seed=2**61 + 3, temperature=0.7, top_k=20,
                                                                           top_p=0.9500000000000001, min_p=0.05)),
    "fill span index": lambda d: setattr(d.filling[0].fill, "i", 2),
    "fill rows": lambda d: setattr(d.filling[0], "span_rows", 2048),
    "a filling stream decodes": lambda d: (d.streams.update({3: d.filling.pop()}), setattr(d.streams[3], "fill", None)),
    "entry length": lambda d: setattr(d.kept.cache[1].snap, "ids", list(range(6))),
    "entry digest": lambda d: setattr(d.kept.cache[0], "digest", "ef" * 32),
    "entry saved": lambda d: setattr(d.kept.cache[1], "lane", None),
    "entry lane": lambda d: setattr(d.kept.cache[1], "lane", 0),
    "entry order": lambda d: d.kept.cache.reverse(),
    "entry dropped": lambda d: d.kept.cache.pop(),
    "free lanes": lambda d: setattr(d, "free", [2]),
    "lanes": lambda d: setattr(d.lanes, "slots", 3),
    "capacity": lambda d: setattr(d.lanes, "capacity", 2048),
    "drafts": lambda d: setattr(d, "drafts", 3),
    "confidence": lambda d: setattr(d, "confidence", None),
    "eos": lambda d: setattr(d, "eos", (1, 2)),
    "prompt rows": lambda d: setattr(d.pbuf, "rows", 1024),
    "kept entries allowed": lambda d: setattr(d.kept, "entries", 8),
    "no kept": lambda d: setattr(d, "kept", None),
    "proposals in one block": lambda d: setattr(d, "batch", object()),
}


@pytest.mark.parametrize("field", CHANGES)
def test_the_digest_changes_with_every_shape_field(field):
    dec = _dec()
    before = T.digest(T.shape(dec))
    assert T.digest(T.shape(copy.deepcopy(dec))) == before and len(before) == 2
    CHANGES[field](dec)
    assert T.digest(T.shape(dec)) != before, field


def test_shape_time_does_not_grow_with_an_entrys_length():
    """H4: entries carry their ids' digest, so 1M ids cost what 1k ids cost."""

    def best(dec) -> float:
        times = []
        for _ in range(200):
            t = time.perf_counter()
            T.digest(T.shape(dec))
            times.append(time.perf_counter() - t)
        return min(times)

    short, long = _dec(1000), _dec(1_000_000)
    best(short), best(long)
    assert best(long) < 3 * best(short) + 20e-6
