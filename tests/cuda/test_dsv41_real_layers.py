"""Real weights against the fp32 reference port (memory class M, ``TF_DSV41_MODEL``): blocks fed TF's own stream,
DSpark's stage MoE, and a reduced model through TF's prompt path.

Blocks 0, 1, 2, 3, 8, 14, 20, 21, 24, 25, 37 and 39 each take the bf16 stream and FFN pre that TF's forward gives
them, on one rank, at contexts 100, 1,100 and 17,000 (the prompt rows that end each context, then decode windows
at even and odd positions). The reference builds its caches from every row and runs the whole block on those rows,
in mirror mode (rel-L2 4 x 2^-8, cosine 0.9999) and fp32 mode (rel-L2 3% over the rows it routes as TF does; the
rows it reroutes are counted). Both attend TF's index lists. TF's routed sets, index lists and candidate blocks must
equal an fp64 selection on TF's own inputs except at near-ties (§7.7). TF runs one layer at a time with only that
layer's weights on the device, so the 40 blocks fit one GPU.

Each DSpark stage's MoE runs on the stage input of the reference's own DSpark pass from TF's taps (whole stages
against the reference are ``test_dsv41_dspark.py``'s). The reduced model's logits must follow the mirror reference
(top-1 on 99% of rows, mean KL 1e-3, rel-L2 5%); the fp32 reference's distance to it is reported beside TF's.

The tokens are DeepSeek's MIT reference sources shipped in the checkpoint (``CORPUS``): paragraphs in the order a
seeded ``random.Random`` shuffles them, tokenized after BOS. ``TF_DSV41_REPORT=<path>`` writes every figure as JSON.
"""

from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
pytest.importorskip("safetensors")
pytest.importorskip("tokenizers")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)
MODEL = os.environ.get("TF_DSV41_MODEL", "")
if not MODEL or not Path(MODEL).is_dir():
    pytest.skip("set TF_DSV41_MODEL to the checkpoint", allow_module_level=True)

import torch.nn.functional as tnf
from dsv41_ref_weights import RefWeights, exl3_weight
from dsv41_reference import Mode, State, gate, hc_post, hc_pre, rms_norm
from safetensors import safe_open
from tokenizers import Tokenizer

from tensorfold.families.deepseek_v41.config import Config
from tensorfold.families.deepseek_v41.cuda import MAX_ROWS, PREFILL_ROWS, buffers, engram, loader, moe, score
from tensorfold.families.deepseek_v41.cuda import forward as F
from tensorfold.families.deepseek_v41.cuda import rope as tf_rope
from tensorfold.families.deepseek_v41.cuda.weights import Weights
from tensorfold.families.glm5_next.cuda import glue

LAYERS = (0, 1, 2, 3, 8, 14, 20, 21, 24, 25, 37, 39)
CORPUS = ("inference/*.py", "inference/README.md", "encoding/*.py", "encoding/README.md")
SEED = 0
CONTEXTS = (100, 1_100, 17_000)
DECODE = ((5, False), (1, False), (1, False), (2, False))   # from an even start: rows at even and odd positions
PLAN = ((100, True), *DECODE, (991, True), *DECODE, *((PREFILL_ROWS, True),) * 7,
        (CONTEXTS[2] - CONTEXTS[1] - 9 - 7 * PREFILL_ROWS, True), *DECODE)
ROWS = sum(rows for rows, _ in PLAN)
CAP = 17_024
TAIL = 256                              # compared prompt rows that end each context
BLOCK = PREFILL_ROWS                    # rows the reference writes caches for at once
MIRROR, FP32 = Mode("mirror", world=1), Mode("fp32", world=1)     # one rank: row sums over the whole input dim
REL_MIRROR, COS, REL_FP32 = 4 * 2**-8, 0.9999, 0.03      # T2, one block
TIE = 4                                 # a near-tie: the fp64 margin below TIE x the fp32 bound
DSPARK_AT = (99, 104, 1_099, 1_104, 16_999, 17_004)       # p after each context's prompt and its first window
REDUCED, REDUCED_ROWS = 8, 256          # reduced model: layers 0-7, norm, head over 256 positions
TOP1, KL, REL_LOGITS = 0.99, 1e-3, 0.05                  # T2, reduced model, against the mirror reference


class Ref(RefWeights):
    """``RefWeights`` reading EXL3 experts through open shards into pinned buffers; ``forget`` drops a layer's dense
    tensors and closes the shards (an open shard keeps every page it was read from)."""

    RING = 6                            # buffers: two experts' w1, w2, w3, so the expert in use is never overwritten

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, experts=0, **kwargs)
        self._files: dict[Path, object] = {}
        self._pinned: list[torch.Tensor] = []
        self._turn = 0

    def _read(self, name: str) -> torch.Tensor:
        path = self.pack.path(name)
        if path not in self._files:
            self._files[path] = safe_open(str(path), framework="pt", device="cpu")
        return self._files[path].get_tensor(name)

    def exl3(self, name: str) -> torch.Tensor:
        """W [K, N] fp32 decoded on the device as ``RefWeights.exl3``, returned in a pinned host buffer."""

        codebook = next((cb for cb in ("mcg", "mul1") if self.pack.has(f"{name}.{cb}")), "3inst")
        w = exl3_weight(*(self._read(f"{name}.{p}").to(self.device) for p in ("trellis", "suh", "svh")), codebook)
        if not self._pinned or self._pinned[0].numel() < w.numel():
            self._pinned = [torch.empty(w.numel(), dtype=torch.float32, pin_memory=True) for _ in range(self.RING)]
        out = self._pinned[self._turn].narrow(0, 0, w.numel()).view(w.shape)
        self._turn = (self._turn + 1) % self.RING
        return out.copy_(w)

    def forget(self) -> None:
        self._dense.clear()
        self._files.clear()


def corpus(cfg: Config, n: int) -> list[int]:
    """BOS, then CORPUS's paragraphs in seeded shuffles (as many passes as ``n`` tokens take), cut at ``n``."""

    root = Path(MODEL)
    files = sorted({p for pattern in CORPUS for p in root.glob(pattern)})
    paragraphs = [p.strip() for f in files for p in f.read_text(encoding="utf-8").split("\n\n") if p.strip()]
    tok, rng, ids = Tokenizer.from_file(str(root / "tokenizer.json")), random.Random(SEED), [cfg.bos_token_id]
    while len(ids) < n:
        order = paragraphs[:]
        rng.shuffle(order)
        for p in order:
            ids += tok.encode(p + "\n\n", add_special_tokens=False).ids
            if len(ids) >= n:
                break
    return ids[:n]


def _forwards() -> list[tuple[int, int, bool]]:
    out, at = [], 0
    for rows, prompt in PLAN:
        out.append((at, rows, prompt))
        at += rows
    return out


def _groups() -> list[tuple[int, int, str]]:
    """Compared rows by kind, in order: the prompt rows that end each context, then its decode windows together."""

    out: list[tuple[int, int, str]] = []
    for a, n, prompt in _forwards():
        lo = (a + n - min(TAIL, n) if a + n in CONTEXTS else a + n) if prompt else a
        kind = "prompt" if prompt else "decode"
        if lo == a + n:
            continue
        out = out[:-1] + [(out[-1][0], a + n, kind)] if out and out[-1][1:] == (lo, kind) else out + [(lo, a + n, kind)]
    return out


def _compared() -> list[tuple[int, int]]:
    """The compared row ranges with adjacent ones merged, the reference's calls."""

    spans: list[tuple[int, int]] = []
    for lo, hi, _ in _groups():
        spans = spans[:-1] + [(spans[-1][0], hi)] if spans and spans[-1][1] == lo else spans + [(lo, hi)]
    return spans


def _segments(total: int) -> list[tuple[int, int, bool]]:
    """(begin, end, compared) over all rows in order, cache-only stretches cut into BLOCK rows."""

    out, at = [], 0
    for lo, hi in _compared() + [(total, total)]:
        out += [(a, min(a + BLOCK, lo), False) for a in range(at, lo, BLOCK)]
        out += [(lo, hi, True)] if hi > lo else []
        at = hi
    return out


def _close(got: torch.Tensor, want: torch.Tensor) -> dict:
    g, x = got.double().flatten(), want.double().flatten()
    return {"rel": float((g - x).norm() / x.norm()), "cos": float(g @ x / (g.norm() * x.norm()))}


def _routing(cfg: Config, xf: torch.Tensor, gate: torch.Tensor, bias: torch.Tensor, pick: torch.Tensor) -> dict:
    """TF's routed sets against an fp64 router on TF's input (N3, M:809-827): rows that differ, those of them not at
    a near-tie, and the least k-th to (k+1)-th margin over its tolerance (TIE x the fp32 error bound)."""

    x, g, k = xf.double(), gate.double(), cfg.num_experts_per_tok
    z = x @ g.T / cfg.gate_temp
    dz = 8 * math.sqrt(x.shape[1]) * 2.0**-24 * (x.abs() @ g.abs().T) / cfg.gate_temp
    f = tnf.softplus(z).sqrt()
    s = f + bias.double()
    ds = torch.sigmoid(z) / (2 * f) * dz + 2.0**-22 * (f + bias.double().abs())
    order = torch.sort(s, dim=-1, descending=True, stable=True).indices
    kth, nxt = order[:, k - 1:k], order[:, k:k + 1]
    margin = (s.gather(1, kth) - s.gather(1, nxt))[:, 0]
    tol = TIE * (ds.gather(1, kth) + ds.gather(1, nxt))[:, 0]
    differ = (order[:, :k].sort(-1).values != pick.long().sort(-1).values).any(-1)
    return {"rows": len(x), "differ": int(differ.sum()), "not_ties": int((differ & (margin >= tol)).sum()),
            "least_margin": float((margin / tol).min())}


class Stream:
    """TF's forward over PLAN one layer at a time on one rank, the next layer's weights read while one runs."""

    def __init__(self, cfg: Config, tokens: list[int], hasher, reader) -> None:
        dev = torch.device("cuda", torch.cuda.current_device())
        self.cfg, self.tokens, self.hasher, self.reader, self.dev = cfg, tokens, hasher, reader, dev
        self.pack = loader._Pack(MODEL, 0, 1, dev)
        self.build = loader._Build(self.pack, cfg)
        tables = {kind: tf_rope.tables(cfg, kind, CAP, dev) for kind in tf_rope.KINDS}
        self.w = Weights(cfg, 0, 1, None, dev, 0, self.build.t("embed.weight"), [], None, None, None, {}, tables)
        self.st = buffers.State(cfg, CAP, dev)
        self.pbuf = buffers.Buffers(cfg, 1, PREFILL_ROWS, CAP, prefill=True, device=dev)
        self.dbuf = buffers.Buffers(cfg, 1, MAX_ROWS, CAP, device=dev)
        n, S, D = ROWS, cfg.hc_mult, cfg.hidden_size
        self.X, self.Y = (torch.empty((n, S, D), dtype=torch.bfloat16, device=dev) for _ in range(2))
        self.pre, self.pre_out = (torch.empty((n, S), dtype=torch.float32, device=dev) for _ in range(2))
        self.taps = torch.zeros((n, len(cfg.dspark_target_layer_ids), D), dtype=torch.bfloat16, device=dev)
        self.lists: dict[int, tuple] = {}       # forward -> the last index layer's lists, counts
        self.cand: dict[int, tuple] = {}        # forward -> the candidate source's blocks, counts
        self.picked: dict[int, list[torch.Tensor]] = {}     # index layer -> its list of each compared row

    def layer(self, L: int) -> dict:
        """Block L over every forward: Y, pre_out take its output; returns its compared rows' FFN input and picks."""

        cfg, w, st = self.cfg, self.w, self.st
        lw, eg = self.build.layer(L)
        if L + 1 < cfg.num_hidden_layers:
            self.pack.queue(self.pack.names(f"layers.{L + 1}."))
        w.layers, w.engram = [lw], ({L: eg} if eg is not None else {})
        role, spans, seen = lw.role, _compared(), {"xf": [], "pick": []}
        own_lists = role.ratio and role.idx_src == L
        index = {k: {"picked": 0, "differ": 0, "not_ties": 0, "worst": 0.0} for k in ("lists", "blocks")}
        st.reset()
        for f, (a, n, prompt) in enumerate(_forwards()):
            b = self.pbuf if prompt else self.dbuf
            R = F.stage(w, st, b, self.tokens[a:a + n], self.hasher, self.reader)
            if L == 0:
                glue.embed(b.ids[:R], w.embed, cfg.hidden_size, cfg.hc_mult, self.X[a:a + n])
                self.pre[a:a + n].zero_()
                self.pre[a:a + n, 0].fill_(1.0)
            b.X[:R].copy_(self.X[a:a + n])
            b.pre_in[:R].copy_(self.pre[a:a + n])
            e = engram.exchange(engram.dequant_rows(b.eraw[:R], b.eloc[:R]), w.comm, b.egat, b.eng) if w.engram \
                else None
            if role.ratio and not own_lists:
                b.lists[:R].copy_(self.lists[f][0])
                b.list_n[:R].copy_(self.lists[f][1])
            if role.uses_candidates:
                b.cand[:R].copy_(self.cand[f][0])
                b.cand_n[:R].copy_(self.cand[f][1])
            F.layer(lw, w, st, b, R, e, prompt)
            if own_lists:
                self.lists[f] = (b.lists[:R].clone(), b.list_n[:R].clone())
            if role.candidate_source:
                self.cand[f] = (b.cand[:R].clone(), b.cand_n[:R].clone())
            if role.tap is not None:
                k = min(R, b.taps.shape[0])
                self.taps[a + R - k:a + R, role.tap].copy_(b.taps[:k, role.tap])
            for lo, hi in spans:
                lo, hi = max(lo, a), min(hi, a + n)
                if lo < hi:
                    seen["xf"].append(b.xn[lo - a:hi - a].cpu())
                    seen["pick"].append(b.pick[lo - a:hi - a].cpu())
                    if own_lists:
                        _index(cfg, role, b, st.index_k[role.kv_src], lo - a, hi - a, lo, index)
                        counts = b.list_n[lo - a:hi - a].tolist()
                        rows = b.lists[lo - a:hi - a].cpu().long()
                        self.picked.setdefault(L, []).extend(rows[i, :c] for i, c in enumerate(counts))
            self.Y[a:a + n].copy_(b.X[:R])
            self.pre_out[a:a + n].copy_(b.pre_in[:R])
            F.commit(w, st, b, R, R)
        seen = {k: torch.cat(v) for k, v in seen.items()}
        seen["routing"] = _routing(cfg, seen["xf"], lw.moe.gate.cpu(), lw.moe.bias.cpu(), seen["pick"])
        seen["index"] = index
        w.layers, w.engram = [], {}
        del lw, eg
        return seen

    def advance(self) -> None:
        self.X, self.Y, self.pre, self.pre_out = self.Y, self.X, self.pre_out, self.pre

    def close(self) -> None:
        self.pack.close()


def _ties(out: dict, s: torch.Tensor, tol: torch.Tensor, k: torch.Tensor, given: torch.Tensor) -> None:
    """Adds to ``out`` how many of the masks ``given`` [rows, n] differ from fp64 top-k of ``s`` (ties to the lower
    index), those not at a near-tie (|s - k-th| >= the two tolerances), and the worst |s - k-th| over its tolerance."""

    ranked = torch.sort(s, dim=-1, descending=True, stable=True)
    mine = torch.arange(s.shape[1], device=s.device) < k[:, None]
    mine = torch.zeros_like(given).scatter_(1, ranked.indices, mine)
    at = (k - 1).clamp_min(0)[:, None]
    kth = torch.where(k[:, None] > 0, ranked.values.gather(1, at), -math.inf)
    lim = tol + tol.gather(1, ranked.indices.gather(1, at))
    ratio = ((s - kth).abs() / lim).nan_to_num(math.inf)[mine ^ given]
    out["picked"] += int(k.sum())
    out["differ"] += len(ratio)
    out["not_ties"] += int((ratio >= 1).sum())
    out["worst"] = max(out["worst"], float(ratio.max()) if len(ratio) else 0.0)


def _index(cfg: Config, role, b, keys: torch.Tensor, i: int, j: int, first: int, out: dict) -> None:
    """TF's lists of forward rows i..j (at ``first`` on), and the candidate source's blocks, against an fp64
    selection on TF's own index Q, weights and keys (M:556-610): every difference must be a near-tie (§7.7)."""

    H, D, G = cfg.index_n_heads, cfg.index_head_dim, cfg.candidate_block_size
    bound = TIE * 8 * math.sqrt(H * D) * 2.0**-24          # T1 fp32 bound, the scale summed below
    visible = (torch.arange(first, first + j - i, device=keys.device) + 1) // role.ratio
    K = keys[:int(visible.max())].double()
    t = torch.arange(len(K), device=K.device)
    for r in range(i, j, 8):
        n = min(8, j - r)
        q, w, v = b.qI[r:r + n].double(), b.wI[r:r + n].double(), visible[r - i:r - i + n, None]
        s = (torch.einsum("nhd,td->nht", q, K).relu() * w[..., None]).sum(1).masked_fill(t >= v, -math.inf)
        tol = bound * torch.einsum("nhd,td->nt", q.abs() * w.abs()[..., None], K.abs())
        if role.candidate_source or role.uses_candidates:
            blocks = torch.zeros((n, -(-len(K) // G)), dtype=torch.bool, device=K.device)
            for x, c in enumerate(b.cand_n[r:r + n].tolist()):
                blocks[x, b.cand[r + x, :c].long()] = True
        if role.candidate_source:
            bs = tnf.pad(s, (0, -len(K) % G), value=-math.inf).unflatten(-1, (-1, G)).amax(-1)
            bs = bs.masked_fill(torch.arange(bs.shape[1], device=K.device) == (v - 1) // G, math.inf)
            bt = tnf.pad(tol, (0, -len(K) % G)).unflatten(-1, (-1, G)).amax(-1)
            k = (bs > -math.inf).sum(-1).clamp_max(cfg.candidate_topk_blocks)
            _ties(out["blocks"], bs, bt, k, blocks)
        if role.uses_candidates:
            s = s.masked_fill(~blocks.repeat_interleave(G, -1)[:, :len(K)], -math.inf)
        given = torch.zeros_like(s, dtype=torch.bool)
        for x, c in enumerate(b.list_n[r:r + n].tolist()):
            given[x, b.lists[r + x, :c].long()] = True
        _ties(out["lists"], s, tol, (s > -math.inf).sum(-1).clamp_max(cfg.index_topk), given)


def ref_block(rw: Ref, L: int, mode: Mode, state: State, stream: Stream, eng_ids) -> tuple[torch.Tensor, torch.Tensor]:
    """Block L of the reference (M:968-994) on TF's input stream: caches from every row, the whole block on the
    compared rows, their MoE in one call -> (their output stream fp32 [rows, hc, D], their routed experts)."""

    c, role, p, eps = rw.cfg, rw.cfg.roles[L], f"layers.{L}", rw.cfg.rms_norm_eps
    ref = rw.reference(mode, state)
    given: list[torch.Tensor] = []
    ref.indexer = lambda *_: given
    at, mids = 0, []
    for a, b, full in _segments(ROWS):
        X, pre = stream.X[a:b].float().cpu(), stream.pre[a:b].cpu()
        if role.engram:
            X = ref.engram(L, X, torch.as_tensor(eng_ids[a:b, c.engram_layer_ids.index(L)]))
        xa = rms_norm(hc_pre(X, pre, mode), ref.W(f"{p}.attn_norm.weight"), eps, mode)
        if not full:
            ref._commit(L, ref._kv(p, xa, ref.cis(role.rope, torch.arange(a, b))), a)
            if role.ratio and role.kv_src == L:
                ref.compress(L, xa, a)
            continue
        if role.ratio:
            given[:] = stream.picked[role.idx_src][at:at + b - a]
            state.lists = list(given)
        at += b - a
        a_pre, a_post, a_comb = ref._mix(f"{p}.hc_attn", X)
        X = hc_post(ref.attention(L, xa, a), X, a_post, a_comb, mode)
        _, f_post, f_comb = ref._mix(f"{p}.hc_ffn", X)
        xf = rms_norm(hc_pre(X, a_pre, mode), ref.W(f"{p}.ffn_norm.weight"), eps, mode)
        mids.append((X, xf, f_post, f_comb))
    X, xf, post, comb = (torch.cat(t) for t in zip(*mids))
    _, picked = gate(xf, ref.W(f"{p}.ffn.gate.weight"), ref.W(f"{p}.ffn.gate.bias"), c.num_experts_per_tok, c)
    return hc_post(ref.moe(L, xf), X, post, comb, mode), picked


def _record_moe(ref) -> list[tuple]:
    """(layer, input, output) of every MoE call ``ref`` makes from now on."""

    seen, own = [], ref.moe

    def moe_(layer: int, x: torch.Tensor) -> torch.Tensor:
        seen.append((layer, x, own(layer, x)))
        return seen[-1][2]

    ref.moe = moe_
    return seen


def _dspark(stream: Stream, rw: Ref) -> list[dict]:
    """Each stage's MoE (TF's Experts4, router and shared expert) on the reference's own stage input, DSpark run in
    mirror mode from TF's taps at each p of DSPARK_AT: TF and the fp32 reference against the mirror reference."""

    cfg, runs = stream.cfg, []
    for p in DSPARK_AT:
        n = min(p + 1, cfg.sliding_window)
        taps = stream.taps[p + 1 - n:p + 1].flatten(1).float().cpu()
        ref = rw.reference(MIRROR, State(pos=p + 1))
        seen = _record_moe(ref)
        ref.absorb(taps, p + 1 - n)
        ref.propose(stream.tokens[p + 1])
        runs.append((p, seen))
    out = []
    for s in range(cfg.num_nextn_predict_layers):
        stage, L = stream.build.stage(s), cfg.num_hidden_layers + s
        fp32 = rw.reference(FP32)
        for p, seen in runs:
            (x, want), = [(x, y) for layer, x, y in seen if layer == L]
            got = moe.dspark_moe(cfg, stage.moe, x.to(torch.bfloat16).to(stream.dev), stream.dbuf).cpu()
            out.append({"stage": s, "p": p, "tf": _close(got, want), "fp32": _close(fp32.moe(L, x), want)})
        del stage
    return out


def _figures(cfg: Config, L: int, stream: Stream, want: dict, seen: dict) -> dict:
    """Block L's figures per compared group: TF against each mode and the rows each mode routes to other experts
    than TF. Against fp32 mode ``rel`` covers the rows it routes as TF does (its FFN input moves further than T1, so
    a near-tie it crosses is no discrete-rule failure); ``all_rows`` and the mirror reference's distance are
    reported."""

    groups, at = [], 0
    tf_sets = seen["pick"].long().sort(-1).values
    for lo, hi, kind in _groups():
        got, g = stream.Y[lo:hi].cpu(), {"rows": [lo, hi], "kind": kind}
        for name, mode in (("mirror", MIRROR), ("fp32", FP32)):
            out, routed = (t[at:at + hi - lo] for t in want[mode])
            same = (routed.sort(-1).values == tf_sets[at:at + hi - lo]).all(-1)
            g[name] = {**_close(got, out), "rerouted": int((~same).sum())}
        g["fp32"] = {**g["fp32"], "all_rows": g["fp32"]["rel"], **_close(got[same], out[same])}
        g["fp32"]["mirror_ref"] = _close(want[MIRROR][0][at:at + hi - lo], want[FP32][0][at:at + hi - lo])["rel"]
        groups.append(g)
        at += hi - lo
        print(f"layer {L} ({cfg.roles[L].mode}) {kind} rows {lo}-{hi - 1}: rel-L2 {g['mirror']['rel']:.3g} mirror "
              f"(cosine {g['mirror']['cos']:.6f}), {g['fp32']['rel']:.3g} fp32 where it routes as TF ("
              f"{g['fp32']['rerouted']} rows rerouted; all rows {g['fp32']['all_rows']:.3g}, the mirror reference "
              f"{g['fp32']['mirror_ref']:.3g})")
    print(f"layer {L}: TF routing against fp64 {seen['routing']}; TF index selection against fp64 {seen['index']}")
    return {"role": cfg.roles[L].mode, "groups": groups, "routing": seen["routing"], "index": seen["index"]}


@pytest.fixture(scope="module")
def report():
    out = {"corpus": list(CORPUS), "seed": SEED, "contexts": list(CONTEXTS), "plan": [list(p) for p in PLAN],
           "compared": [list(g) for g in _groups()], "layers": {}, "dspark": [], "reduced": {}}
    yield out
    path = os.environ.get("TF_DSV41_REPORT")
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(out, indent=1) + "\n")
        print(f"report: {path}")


@pytest.fixture(scope="module")
def layers(report):
    """Every checked block's figures and DSpark's, from one pass of TF over the 40 layers."""

    cfg = Config.read(MODEL)
    rw = Ref(MODEL)
    tokens = corpus(cfg, ROWS + 1)      # one more: the token after the last row, DSpark's y there
    eng_ids = rw.hasher.ids([], tokens[:ROWS])
    stream = Stream(cfg, tokens, rw.hasher, rw.reader)
    states = {MIRROR: State(), FP32: State()}
    try:
        for L in range(cfg.num_hidden_layers):
            seen = stream.layer(L)
            if L in LAYERS:
                want = {mode: ref_block(rw, L, mode, states[mode], stream, eng_ids) for mode in (MIRROR, FP32)}
                rw.forget()
                report["layers"][L] = _figures(cfg, L, stream, want, seen)
            stream.advance()
        stream.X = stream.Y = stream.pre = stream.pre_out = stream.pbuf = None
        report["dspark"] = _dspark(stream, rw)
        for d in report["dspark"]:
            print(f"DSpark stage {d['stage']} MoE at p={d['p']}: TF rel-L2 {d['tf']['rel']:.3g} (cosine "
                  f"{d['tf']['cos']:.6f}), fp32 reference {d['fp32']['rel']:.3g}")
    finally:
        stream.close()
    return report


@pytest.mark.parametrize("L", LAYERS)
def test_block_follows_the_reference(layers, L):
    out = layers["layers"][L]
    for g in out["groups"]:
        m, f = g["mirror"], g["fp32"]
        assert m["rel"] <= REL_MIRROR and m["cos"] >= COS, f"layer {L} rows {g['rows']}: mirror {m}"
        assert f["rel"] <= REL_FP32, f"layer {L} rows {g['rows']}: fp32 {f}"
    assert out["routing"]["not_ties"] == 0, f"layer {L}: routed sets differ away from near-ties {out['routing']}"
    for kind, d in out["index"].items():
        assert d["not_ties"] == 0, f"layer {L}: index {kind} differ away from near-ties {d}"


def test_dspark_stage_moe_follows_the_reference(layers):
    assert len(layers["dspark"]) == 3 * len(DSPARK_AT)
    for d in layers["dspark"]:
        assert d["tf"]["rel"] <= REL_MIRROR and d["tf"]["cos"] >= COS, f"DSpark {d}"
        assert d["fp32"]["rel"] <= REL_FP32, f"DSpark {d}"


def _logits(got: torch.Tensor, want: torch.Tensor) -> dict:
    """Top-1 agreement, mean KL(want || got) and rel-L2 of two [rows, V] logits."""

    p = torch.softmax(want, -1)
    return {"top1": float((got.argmax(-1) == want.argmax(-1)).double().mean()),
            "kl": float((p * (torch.log_softmax(want, -1) - torch.log_softmax(got, -1))).sum(-1).mean()),
            "rel": float((got - want).norm() / want.norm())}


def test_reduced_model_follows_the_reference(report):
    cfg = Config.read(MODEL, REDUCED)
    rw = Ref(MODEL, REDUCED)
    ids = corpus(cfg, REDUCED_ROWS)
    w = loader.load(MODEL, cfg, 0, 1, None, dspark=False, capacity=REDUCED_ROWS)
    e = SimpleNamespace(w=w, st=buffers.State(cfg, REDUCED_ROWS, w.device), hasher=rw.hasher, reader=rw.reader,
                        pbuf=buffers.Buffers(cfg, 1, REDUCED_ROWS, REDUCED_ROWS, prefill=True, device=w.device))
    got = score.prompt_logits(e, ids).double()
    del e, w
    torch.cuda.empty_cache()
    mirror, fp32 = (rw.model(ids, mode, all_logits=True)[0].double() for mode in (MIRROR, FP32))
    out = {"layers": REDUCED, "positions": REDUCED_ROWS, "tf": _logits(got, mirror), "fp32": _logits(fp32, mirror)}
    report["reduced"] = out
    print(f"reduced model, {REDUCED} layers over {REDUCED_ROWS} positions, against the mirror reference: TF {out['tf']}"
          f", the fp32 reference {out['fp32']}")
    tf = out["tf"]
    assert tf["top1"] >= TOP1 and tf["kl"] <= KL and tf["rel"] <= REL_LOGITS, out
