"""DeepSeek-V4.1-Flash on two ranks: rank 0 serves requests and rank 1 mirrors each one in ``follow``."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import DEFAULT_SHARE, MAX_LANES, protocol

NO_DIGEST = 0           # the agreement's Engram digest for a checkpoint without Engram layers


class DeepSeekV41Engine:
    """One rank; ``policy`` is (drafts a round, confidence stop or None); ``lanes`` above 1 serves that many requests
    together, rounds taking ``share`` of each prompt span's time; ``comm``/``layers``/``graphs`` for tests."""

    tp = 2
    supports_logprobs = False
    vision = None

    def __init__(self, model_dir: str | Path, *, rank: int, master: str, port: int,
                 policy: tuple[int, float | None], context: int | None = None, context_explicit: bool | None = None,
                 serial_only: bool = False, lanes: int = 1, share: float = DEFAULT_SHARE, comm=None,
                 layers: int | None = None, graphs: bool = True) -> None:
        import torch

        from tensorfold.cuda.capacity import admit
        from tensorfold.cuda.comm import open_comm

        from ..config import Config
        from . import MAX_ROWS, PREFILL_ROWS, RING, l2warm, loader, split
        from .cache import Kept, entries_wanted
        from .decode import Engine
        from .geometry import cache_wanted, dsv41_geometry, kept_bytes
        from .proposals import absorbing, mode
        from .proposals import enabled as batched
        from .sample import greedy_on_device

        if rank not in (0, 1):
            raise ValueError(f"rank {rank}: {type(self).__name__} runs on ranks 0 and 1")
        if not 1 <= lanes <= MAX_LANES:
            raise ValueError(f"{lanes} lanes: {type(self).__name__} serves 1 to {MAX_LANES}")
        drafts, confidence = policy
        model_dir = Path(model_dir)
        torch.cuda.set_device(0)
        self.rank, self.policy, self.serial_only = rank, (int(drafts), confidence), bool(serial_only)
        self.slots, self.share, self.master = int(lanes), float(share), master
        self.concurrent, self.scheduler, self.decoder = self.slots > 1, None, None
        self.comm = comm if comm is not None else open_comm(rank, 2, master, port)
        self.comm.barrier()
        self.bell = protocol.Bell(getattr(self.comm, "store", None))
        cfg = Config.read(model_dir, layers)
        dspark = self.policy[0] > 0 and not self.serial_only
        explicit = context is not None if context_explicit is None else bool(context_explicit)
        try:
            plan = admit(model_dir, context, explicit, torch,
                         lambda text: dsv41_geometry(cfg, 2, lanes=self.slots, batched=batched() and dspark,
                                                     absorb=absorbing() and dspark),
                         split.rank_estimate(cfg, rank, dspark), rank=rank, world=2, gather=self._gather_ints)
        except ValueError as exc:               # the window is per lane: name the flag that multiplies it
            if self.slots == 1 or "cannot fit" not in str(exc):
                raise
            raise ValueError(f"{exc} With --parallel {self.slots}, each of the {self.slots} lanes holds a whole "
                             "window: lower --parallel or --context.") from None
        self.capacity_plan, self.limit = plan, plan["context_window"]
        capacity = plan["cache_slots"]
        hasher = reader = None
        failure, digest, wanted, entries, warm, batching = None, NO_DIGEST, 0, 0, None, 0
        try:                                    # a failure here reaches the gather, so both ranks name it
            batching = mode()
            extra = self.slots if self.concurrent else 0        # room for each lane's prompt beside the others'
            wanted, entries = kept_bytes(plan, cache_wanted()), entries_wanted(lanes=extra)
            warm = l2warm.wanted()
            hasher, reader = self._engram(model_dir, cfg)
            digest = reader.layout.digest() if reader is not None else NO_DIGEST
        except Exception as exc:                # noqa: BLE001 - raised below on both ranks
            failure = f"{type(exc).__name__}: {exc}"
        prefill_rows = min(PREFILL_ROWS, capacity)
        mine = protocol.settings(start_error=failure is not None, dspark=dspark, capacity=capacity,
                                 prefill_rows=prefill_rows, max_rows=MAX_ROWS, ring=RING, policy=self.policy,
                                 layers=cfg.num_hidden_layers, world=2, engram_digest=digest, lanes=self.slots,
                                 decode_share=self.share, batched_drafts=batching,
                                 greedy_device=greedy_on_device(), cache_bytes=wanted,
                                 cache_entries=entries)
        both = self._gather_ints(mine)
        if failure is not None or both[1 - rank][0]:
            raise ValueError("the TF_DSV41_* variables or the Engram tables could not be read on " +
                             (f"this rank: {failure}" if failure else f"rank {1 - rank}; see its log"))
        cache_bytes, entries = protocol.agree(both)
        plan["kept_bytes"] = cache_bytes
        for key in ("serving_peak_bytes_estimate", "total_bytes_estimate"):
            plan[key] = plan[key] + cache_bytes
        if rank == 0 and cache_bytes < cache_wanted():
            print(f"[tensorfold] other conversations' prompts are kept in {cache_bytes / 2**30:.1f} GiB, what the "
                  f"{self.limit}-token window leaves", flush=True)
        self.w = loader.load(model_dir, cfg, rank, 2, self.comm, dspark=dspark, capacity=capacity)
        if warm is not None:
            self.w.warm = l2warm.Warm(self.w, *warm)
        self.comm.ready("loading")              # a peer stuck loading is named, not waited on in the all-gather
        self.comm.barrier()
        if self.concurrent:
            self._lanes(capacity, prefill_rows, graphs, hasher, reader, cache_bytes, entries)
            return
        self.e = Engine(self.w, capacity, prefill_rows, graphs=graphs, hasher=hasher, reader=reader)
        self.kept = Kept(self.e, cache_bytes, entries)    # allocated now: a request allocates no snapshot memory
        self.eos = self.e.eos
        self._warm()

    def _lanes(self, capacity: int, prefill_rows: int, graphs: bool, hasher, reader, cache_bytes: int,
               entries: int) -> None:
        """The lane decoder over ``slots`` lanes, every lane on the one Engram reader; warmed, then rank 0's
        scheduler and link."""

        from tensorfold.cuda.nvfp4.linear import FUSED_ROWS
        from tensorfold.cuda.scheduler import Scheduler

        from . import MAX_ROWS, layout
        from .buffers import Buffers
        from .cache import Kept
        from .decode import Engine
        from .graphs import LaneGraphs
        from .lanes import Ahead, Lanes
        from .multi import LaneDecoder
        from .multi_tp import Link
        from .proposals import Batch, absorbing, enabled, setup
        from .sample import greedy_on_device

        w, S = self.w, self.slots
        cfg, dev = w.cfg, w.device
        rows = cfg.o_groups // w.world * MAX_ROWS * S
        if rows >= FUSED_ROWS:                  # the stacked wo_a rows must stay on the row-invariant tiles
            raise ValueError(f"--parallel {S}: {rows} stacked output-projection rows reach TF_QMMF_FUSED_ROWS "
                             f"{FUSED_ROWS}, where its tiles change with the row count")
        self.lanes = Lanes(cfg, S, capacity, dev)
        self.mbuf = Buffers(cfg, w.world, MAX_ROWS * S, capacity, device=dev, lanes=S, greedy=greedy_on_device())
        self.pbuf = Buffers(cfg, w.world, prefill_rows, capacity, prefill=True, device=dev)
        self.engines = [Engine(w, capacity, hasher=hasher, reader=reader, st=self.lanes.view(k), pbuf=self.pbuf,
                               dbuf=self.mbuf, ahead=Ahead(cfg, w.world)) for k in range(S)]
        self.kept = Kept(self.engines, cache_bytes, entries)
        self.eos = (cfg.eos_token_id,)
        when = setup() if enabled() and w.dspark is not None else None
        batch = Batch(cfg, w.world, S, dev, absorb=absorbing()) if when == "early" else None
        lane_graphs = LaneGraphs(w, self.lanes, self.mbuf, self.engines, batch) if graphs else None
        if lane_graphs is not None:
            lane_graphs.warm()
        self.decoder = LaneDecoder(w, self.lanes, self.engines, self.mbuf, self.pbuf, self.policy, self.eos,
                                   lane_graphs, self.share, kept=self.kept, batch=batch)
        self._warm_lanes()
        if when == "late":
            self.decoder.build_batch(absorbing())
        elif when == "lazy":
            self.decoder.defer, self.decoder.defer_absorb = True, absorbing()
        if self.rank == 0:
            self.decoder.link = Link(getattr(self.comm, "store", None), rank=0, host=self.master or None)
            self.scheduler = Scheduler(self.decoder, max_streams=S)
        layout.dump("start", self.decoder)

    def _warm(self) -> None:
        """Run a prompt chunk and decode rounds once and reset, so serving reserves no more device memory."""

        from tensorfold.engine.exact_sampling import Sampling

        from . import MAX_ROWS
        from .decode import dspark_decode, serial_decode
        from .prefill import prefill

        e, (drafts, confidence) = self.e, self.policy
        guards = [(b.exl3, vars(b.exl3).get("guard_left")) for b in (e.pbuf, e.dbuf)]
        sampling = Sampling(seed=0, temperature=1.0, top_k=0)
        first = prefill(e, [self.w.cfg.bos_token_id] * min(e.pbuf.rows, e.st.capacity - 3 * MAX_ROWS), sampling)
        if drafts == 0 or self.w.dspark is None:
            serial_decode(e, first, 2 * MAX_ROWS, sampling, stop_eos=False)
        else:
            dspark_decode(e, first, 2 * MAX_ROWS, sampling, drafts=drafts, confidence=confidence, stop_eos=False)
        e.reset()
        for s, left in guards:
            if left is None:
                vars(s).pop("guard_left", None)
            else:
                s.guard_left = left

    def _warm_lanes(self) -> None:
        """Serve requests once on both ranks and reset: a full span and one cut at its keep point, greedy, keyed and
        top_k 0 rounds with every lane drafting, kept rows copied between lanes and loaded from the arena."""

        from tensorfold.cuda.streams import Stream
        from tensorfold.engine.exact_sampling import Sampling

        from . import MAX_ROWS
        from .multi_fill import FillPlan

        dec, kept, S = self.decoder, self.kept, self.slots
        guards = [(b.exl3, vars(b.exl3).get("guard_left")) for b in (dec.pbuf, dec.mbuf)]
        modes = (None, Sampling(seed=0, temperature=1.0, top_k=20, top_p=0.95), Sampling(seed=0, temperature=1.0,
                                                                                         top_k=0))
        cfg, plan = self.w.cfg, dec.plan
        dec.plan = FillPlan(0.0)                # whole prompts first: the ranks plan alike without a link

        def prompt(k: int, n: int) -> list[int]:
            return [cfg.bos_token_id] + [(cfg.bos_token_id + 1 + k) % cfg.vocab_size] * (n - 1)

        def run(prompts: list[list[int]], first: int) -> None:
            waiting = [Stream(p, 2 * MAX_ROWS, modes[(first + j) % len(modes)], stop_eos=False)
                       for j, p in enumerate(prompts)]
            streams = list(waiting)
            while waiting or dec.live():
                while waiting and dec.live() < S:
                    dec.admit(waiting.pop(0))
                dec.finish(dec.round())
            failed = next((s.error for s in streams if s.error is not None), None)
            if failed is not None:
                raise failed

        long = min(dec.pbuf.rows + MAX_ROWS + 1, dec.lanes.capacity - 3 * MAX_ROWS)
        lanes = [prompt(k, long if k == 0 else 2 * MAX_ROWS + k) for k in range(S)]
        run(lanes, 0)       # lane 0's prompt: a full span, then one its keep point cuts
        # lane 1 resumed in place, then from another lane (copied), a new prompt, and lane 0's from the arena
        run([lanes[1], lanes[1] + [cfg.bos_token_id] * MAX_ROWS, prompt(S, 3 * MAX_ROWS), lanes[0]], 1)
        dec.drop()
        for x in list(kept.cache):
            kept._drop(x)
        for e in dec.engines:
            e.reset()
        dec.plan, dec.next_id, dec.used = plan, 0, [-1] * S
        for s, left in guards:
            if left is None:
                vars(s).pop("guard_left", None)
            else:
                s.guard_left = left

    @staticmethod
    def _engram(model_dir: Path, cfg):
        """(hasher, reader) of the checkpoint's Engram tables, each byte range checked against its file."""

        if not cfg.engram_layer_ids:
            return None, None
        from ..engram_hash import TOKEN_MAP_SHA256, TOKEN_MAP_SIZE, Hasher, TokenMap
        from ..engram_table import Layout, Reader
        from .engram import read_pool

        size = cfg.engram_compressed_vocab_size
        token_map = TokenMap.build(model_dir / "tokenizer.json", size, TOKEN_MAP_SHA256 if size == TOKEN_MAP_SIZE
                                   else None)
        hasher = Hasher(cfg, token_map)
        return hasher, Reader(Layout.read(model_dir, cfg.engram_layer_ids, hasher.primes.tolist()), pool=read_pool())

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        return protocol.gather_ints(self.comm, values)

    def _share(self, values: list[int] | None) -> list[int]:
        return protocol.share(self.comm, self.rank, values)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens: Callable[[list[int]], Any],
                 draft: bool = True, stop_eos: bool = True, background: bool = False) -> dict[str, Any]:
        """Serve one request on rank 0 while rank 1 mirrors it; ``draft`` False is the serial reference; with lanes,
        ``on_tokens`` returning True ends it and ``background`` goes last, yielding its lane to a waiting request."""

        prompt = [int(t) for t in prompt]
        if not prompt:
            raise ValueError("an empty prompt")
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        vocab = self.w.cfg.vocab_size
        if min(prompt) < 0 or max(prompt) >= vocab:
            raise ValueError(f"prompt token ids must lie in [0, {vocab})")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        drafting = bool(draft) and not self.serial_only
        if self.scheduler is not None:
            stats = self.scheduler.submit(prompt, max_tokens, sampling, drafting, on_tokens, stop_eos=stop_eos,
                                          background=background)
            serial = not drafting or self.policy[0] == 0 or self.w.dspark is None
            return {**stats, "policy": _policy(self.policy, serial)}
        hit = self.kept.lookup(prompt) if drafting else None
        header = protocol.encode(max_tokens, stop_eos, drafting, len(hit.ids) if hit is not None else 0, sampling,
                                 self.policy)
        self.bell.ring()                        # wakes rank 1, which idles on the store, not in the all-gather
        self._share(header)
        self._share(prompt)
        return self._run(prompt, protocol.decode(header), hit, on_tokens)

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, until rank 0's ``shutdown``."""

        if self.decoder is not None:
            from .multi_tp import Link

            self.decoder.follow(Link(getattr(self.comm, "store", None), rank=1, host=self.master or None))
            return
        while True:
            self.bell.wait()
            header = self._share(None)
            if header == protocol.STOP:
                return
            prompt = self._share(None)
            req = protocol.decode(header)
            hit = self.kept.find(prompt, req.cached) if req.cached else None
            self._run(prompt, req, hit, lambda tokens: None)

    def shutdown(self) -> None:
        """Rank 0: tell rank 1 to leave ``follow``."""

        if self.rank == 0 and self.decoder is not None:
            self.decoder.link.send(["stop"])
        elif self.rank == 0:
            self.bell.ring()
            self._share(protocol.STOP)

    def close(self) -> None:
        """Rank 0: stop the scheduler's worker once its requests have ended."""

        if self.scheduler is not None:
            self.scheduler.close()
            self.scheduler = None

    def _run(self, prompt: list[int], req: protocol.Request, hit, on_tokens) -> dict[str, Any]:
        """One request on this rank: snapshots taken over, the prompt prefilled from ``hit``, then decoded."""

        from .decode import dspark_decode, serial_decode
        from .prefill import prefill

        e, kept = self.e, self.kept
        drafts, confidence = req.policy
        t0 = time.perf_counter()
        keep_at = max(1, len(prompt) - 1) if req.draft else None
        space = None
        if keep_at is not None and (hit is None or len(hit.ids) != keep_at):
            space = kept.reserve(hit)
            keep_at = keep_at if space is not None else None
        try:
            kept.take_over(hit)
            first = prefill(e, prompt, req.sampling, resume=hit, keep_at=keep_at,
                            keep=kept.remember if keep_at is not None else None, space=space)
        finally:
            kept.settle()
        kept.live = list(prompt)
        stats: dict[str, Any] = {"prefill_s": time.perf_counter() - t0, "cached": req.cached}
        serial = not req.draft or drafts == 0 or self.w.dspark is None
        if serial:
            res = serial_decode(e, first, req.max_tokens, req.sampling, stop_eos=req.stop_eos, on_tokens=on_tokens)
        else:
            res = dspark_decode(e, first, req.max_tokens, req.sampling, drafts=drafts, confidence=confidence,
                                stop_eos=req.stop_eos, on_tokens=on_tokens)
        kept.live = list(prompt) + res.tokens[:e.st.pos - len(prompt)]
        stats.update(res.stats())
        stats.update(min_rows=1 + min(res.depths, default=0), drafts=req.draft, policy=_policy(req.policy, serial),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        return stats


def _policy(policy: tuple[int, float | None], serial: bool) -> str:
    """The stats' name of a request's drafting: "0" serial, "D" drafts a round, "cD:P" stopping by confidence."""

    drafts, confidence = policy
    return "0" if serial else f"{drafts}" if confidence is None else f"c{drafts}:{confidence:g}"
