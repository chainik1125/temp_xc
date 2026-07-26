"""TXC-win sweep — train the panel ONCE, score many candidate tasks against it.

The key economy: **dictionary training is label-free.** So one trained panel per
(architecture, window size) can be scored against dozens of candidate labels for
the price of a linear probe each. That turns "which task does TXC win?" from one
experiment per task into one experiment per architecture.

Pipeline
    1. cache   — run the subject model over a committed token stream, keep one
                 layer's residual activations as a float16 memmap.
    2. train   — for each (arch, T): train a dictionary on windows from that cache.
                 Label-free, so this is done once and reused by every task.
    3. score   — for each candidate label: build per-tile rows, fit a linear probe
                 on the tile's code, report held-out skill. Cheap.
    4. rank    — report, per task, the best window architecture minus the best
                 per-token-decoded baseline. That difference IS the paper's claim.

Candidate tasks are the committed exact-label packs in
`task_hunt/labels/*.npz` — trailing rates, switch clocks, turn clocks, repetition
lag. They are chosen because they are **state-tracking / accumulation** latents: a
running quantity the model must carry, not a relation it resolves once. The
relational exploration showed relations get linearised per position within one
attention layer; accumulations are the class that can stay distributed.

STATUS: this is a **triage** harness, not a canonical result. It trains in-process
rather than through `temp_bench.core.runner`, so nothing here touches the
leaderboard. Whatever wins triage gets re-run through the canonical runner with
seeds before it is called a result.

Run:
  .venv/bin/python -m experiments.explorations.txcwin.sweep cache --model gpt2
  .venv/bin/python -m experiments.explorations.txcwin.sweep run --model gpt2 --steps 400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
LABELS = HERE.parent / "task_hunt" / "labels"
RESULTS = HERE / "results"
CACHE_DIR = Path("/workspace/txcwin_caches")

# ── the candidate tasks ─────────────────────────────────────────────────
# (task name, npz stem, label field, kind, plain-language description)
# kind: "reg" = scalar regression (Pearson r), "cls" = binary (rank AUC)
TASKS = [
    ("switch_clock", "interleave_fineweb", "tss", "reg",
     "tokens since the text switched to the other source document"),
    ("source_id", "interleave_fineweb", "source", "cls",
     "which of two interleaved documents the current token is from"),
    ("turn_clock", "dialevel_dailydialog", "tst", "reg",
     "tokens since the last speaker change"),
    ("turn_level", "dialevel_dailydialog", "tlevel", "reg",
     "trailing mean turn length — rapid-fire vs long-form exchange"),
    ("novelty_rate", "novelty_fineweb", "nov_rate", "reg",
     "trailing rate of first-occurrence word types"),
    ("novelty_resid", "novelty_fineweb", "nov_resid", "reg",
     "novelty rate with the document-position trend removed"),
    ("list_density", "punctint_fineweb", "lam_list", "reg",
     "trailing density of list/enumeration markers"),
    ("question_rate", "punctint_fineweb", "lam_q", "reg",
     "trailing rate of question sentences"),
]

# ── the panel (paper hyperparameters, scaled by expansion factor) ───────
def _registry_path(name: str) -> str:
    """Class path straight from configs/archs.yaml — the repo's own registry, so a
    renamed class cannot silently drift out of this sweep (it did once: the sweep
    hard-coded `tsae:TSAE` while the class is `TSAEPaper`)."""
    import yaml
    reg = yaml.safe_load(
        (Path(__file__).resolve().parents[3] / "configs" / "archs.yaml").read_text())
    return reg["archs"][name]["class_path"]


_PANEL_SPEC = [
    ("batchtopk_sae", 1, "per-token"),
    # T-SAE is implemented as a PER-TOKEN module in this repo (its __init__ raises
    # if T != 1): the contrastive term couples positions during training, but the
    # encoder/decoder see one position. So it lives in the T=1 group, which is also
    # how the paper describes it.
    ("tsae", 1, "per-token + contrastive coupling"),
    ("stacked_batchtopk", None, "independent dict per position"),
    ("txc_batchtopk_pre", None, "window, additive"),
    ("txc_batchtopk_post", None, "window, position-mixing (the paper's TXC)"),
]
PANEL = [(n, _registry_path(n), fx, b) for n, fx, b in _PANEL_SPEC]
PER_TOKEN_ARCHS = {"batchtopk_sae", "tsae"}   # read at one position by convention


def _import(path: str):
    mod, cls = path.split(":")
    import importlib
    return getattr(importlib.import_module(mod), cls)


# ── 1. cache ────────────────────────────────────────────────────────────
def stream_sha(stem: str, model_name: str) -> str:
    import hashlib
    npz = np.load(LABELS / f"{stem}_{_npz_key(stem, model_name)}.npz")
    return hashlib.sha256(npz["token_ids"].tobytes()).hexdigest()[:12]


def build_cache(model_name: str, stem: str, layer: int, batch_tokens: int = 8192):
    """Run the subject model over a committed token stream; keep one layer.

    Keyed by the SHA of the token stream, so label packs that share a stream
    (novelty / punctint / replag all use the same pinned fineweb sample) share
    one cache instead of paying for it three times.
    """
    from transformers import AutoModelForCausalLM
    tag = f"{stream_sha(stem, model_name)}_{model_name.split('/')[-1]}_L{layer}"
    out = CACHE_DIR / f"{tag}.f16"
    meta_p = CACHE_DIR / f"{tag}.json"
    if meta_p.exists():
        print(f"[cache] {tag} exists")
        return tag
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    key = _npz_key(stem, model_name)
    npz = np.load(LABELS / f"{stem}_{key}.npz")
    ids = npz["token_ids"].astype(np.int64)
    doc_off = npz["doc_off"].astype(np.int64)

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda")
    model.eval()
    d = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    assert 0 <= layer <= n_layers, f"layer {layer} > {n_layers}"

    mm = np.memmap(out, dtype=np.float16, mode="w+", shape=(len(ids), d))
    t0 = time.time()
    # feed document by document (documents are independent contexts), chunked so
    # no forward pass exceeds the context window
    ctx = min(1024, getattr(model.config, "max_position_embeddings", 1024))
    written = 0
    for i in range(len(doc_off) - 1):
        s, e = int(doc_off[i]), int(doc_off[i + 1])
        for c0 in range(s, e, ctx):
            c1 = min(c0 + ctx, e)
            chunk = torch.tensor(ids[c0:c1], dtype=torch.long, device="cuda")[None]
            with torch.no_grad():
                hs = model(chunk, output_hidden_states=True).hidden_states[layer]
            mm[c0:c1] = hs[0].to(torch.float16).cpu().numpy()
            written += c1 - c0
    mm.flush()
    meta = {"tag": tag, "model": model_name, "stem": stem, "layer": layer,
            "n_tokens": int(len(ids)), "d": int(d), "written": int(written),
            "seconds": round(time.time() - t0, 1),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
    meta_p.write_text(json.dumps(meta, indent=1))
    print(f"[cache] {meta}")
    del model
    torch.cuda.empty_cache()
    return tag


def _npz_key(stem: str, model_name: str) -> str:
    m = model_name.lower()
    if "gpt2" in m:
        return "gpt2"
    if "gemma" in m:
        return "gemma2"
    return "llama31"


def load_cache(tag: str, in_ram: bool = True):
    """Load a cache. `in_ram=True` copies it into memory: training draws random
    windows, and random reads from a memmap were the dominant cost (about a
    minute per cell). The box has ~1.6 TB free RAM and the caches are ~1 GB each.
    """
    meta = json.loads((CACHE_DIR / f"{tag}.json").read_text())
    mm = np.memmap(CACHE_DIR / f"{tag}.f16", dtype=np.float16, mode="r",
                   shape=(meta["n_tokens"], meta["d"]))
    if in_ram:
        mm = np.ascontiguousarray(mm)
    return mm, meta


# ── 2. train (label-free) ───────────────────────────────────────────────
# Training-input contracts, established by smoke.py (run it after any arch change):
#   batchtopk_sae : train_step wants (B, d_in)
#   tsae          : constructs at T=1 but train_step wants (B, seq>=2, d_in) — it
#                   samples a consecutive pair internally for its contrastive term
#   window archs  : train_step wants (B, T, d_in)
# At ENCODE time every arch accepts (B, T, d_in) and is read as ONE code vector
# per row (TXC returns (B,1,d_sae); the others (B,T,d_sae) -> last position).
TRAIN_2D = {"batchtopk_sae"}
TSAE_TRAIN_SEQ = 8


def _loss_of(out):
    """Architectures return a dict, a (loss, info) tuple, or a bare tensor."""
    if isinstance(out, dict):
        return out.get("loss", next(iter(out.values())))
    if isinstance(out, (tuple, list)):
        return out[0]
    return out


def train_one(mm, meta, arch_name: str, path: str, T: int, d_sae: int,
              k_pos: int, steps: int, batch: int, seed: int,
              lr: float = 3e-4) -> torch.nn.Module:
    torch.manual_seed(seed)
    cls = _import(path)
    kw = dict(d_in=meta["d"], d_sae=d_sae, k_pos=k_pos)
    if arch_name != "batchtopk_sae" or T > 1:
        kw["T"] = T
    arch = cls(**kw).cuda()
    train_T = (1 if arch_name in TRAIN_2D
               else TSAE_TRAIN_SEQ if arch_name == "tsae" else T)
    opt = torch.optim.Adam(arch.parameters(), lr=lr)
    n = meta["n_tokens"]
    rng = np.random.default_rng(seed)
    for step in range(steps):
        starts = rng.integers(0, n - train_T - 1, batch)
        idx = starts[:, None] + np.arange(train_T)[None, :]
        x = torch.from_numpy(np.ascontiguousarray(mm[idx.reshape(-1)])
                             ).view(batch, train_T, meta["d"]).cuda().float()
        if arch_name in TRAIN_2D:
            x = x[:, -1, :]
        arch.pre_step()
        out = arch.train_step(x)
        loss = _loss_of(out)
        opt.zero_grad()
        loss.backward()
        opt.step()
        arch.post_step()
    arch.eval()
    return arch


@torch.no_grad()
def encode_rows(arch, mm, meta, starts: np.ndarray, T: int,
                chunk: int = 512) -> tuple[np.ndarray, float]:
    """Codes for windows beginning at `starts` (label read at the LAST position)."""
    codes, l0 = [], []
    for c0 in range(0, len(starts), chunk):
        st = starts[c0:c0 + chunk]
        idx = st[:, None] + np.arange(T)[None, :]
        x = torch.from_numpy(np.ascontiguousarray(mm[idx.reshape(-1)])
                             ).view(len(st), T, meta["d"]).cuda().float()
        z = arch.encode(x)
        if z.dim() == 3:                     # per-position codes -> last position
            z = z[:, -1, :]
        codes.append(z.float().cpu().numpy())
        l0.append((z != 0).float().sum(-1).mean().item())
    return np.concatenate(codes), float(np.mean(l0))


def calibrate_k(mm, meta, arch_name, path, T, d_sae, target, mode="nnz",
                probe_steps=40, batch=128, seed=0, tol=0.12, max_iter=7):
    """Find the nominal k that gives this architecture the intended code budget.

    Necessary because nominal k buys wildly different things per architecture. At
    T=8, nominal k=20 yields a code with 20 non-zeros for the per-token SAE and
    for TXC-post, 19.8 for Stacked, but 114 for TXC-pre (it keeps k per position
    and then sums). Measured, not assumed.

    mode="nnz"      match the number of features the probe sees (equal probe budget)
    mode="pertoken" match atoms per token of the stream (equal reconstruction cost)
    """
    lo, hi = 1, max(4, int(target * max(T, 1) * 4))
    best = (None, None)
    for _ in range(max_iter):
        k = max(1, (lo + hi) // 2)
        arch = train_one(mm, meta, arch_name, path, T, d_sae, k, probe_steps,
                         batch, seed)
        starts = np.arange(T, T + 256) + 0
        _, nnz = encode_rows(arch, mm, meta, starts, T)
        del arch
        torch.cuda.empty_cache()
        got = nnz if mode == "nnz" else nnz / max(T, 1)
        best = (k, got)
        if abs(got - target) <= tol * target:
            break
        if got > target:
            hi = max(1, k - 1)
        else:
            lo = k + 1
        if lo > hi:
            break
    return best


# ── 3. score ────────────────────────────────────────────────────────────
def _ridge_r(Xtr, ytr, Xte, yte, lam: float = 1.0) -> float:
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    A = (Xtr - mu) / sd
    B = (Xte - mu) / sd
    ym = ytr.mean()
    G = A.T @ A + lam * np.eye(A.shape[1], dtype=np.float32)
    w = np.linalg.solve(G, A.T @ (ytr - ym))
    p = B @ w + ym
    if p.std() < 1e-9 or yte.std() < 1e-9:
        return 0.0
    return float(np.corrcoef(p, yte)[0, 1])


def _auc(scores, y) -> float:
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = float(y.sum()), float((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return 0.5
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _fit_predict(codes, y, tr, te, kind):
    if kind == "reg":
        mu, sd = codes[tr].mean(0), codes[tr].std(0) + 1e-6
        A, B = (codes[tr] - mu) / sd, (codes[te] - mu) / sd
        ym = y[tr].mean()
        G = A.T @ A + np.eye(A.shape[1], dtype=np.float32)
        w = np.linalg.solve(G, A.T @ (y[tr] - ym))
        return B @ w + ym
    mu, sd = codes[tr].mean(0), codes[tr].std(0) + 1e-6
    A, B = (codes[tr] - mu) / sd, (codes[te] - mu) / sd
    yt = y[tr].astype(np.float32)
    G = A.T @ A + np.eye(A.shape[1], dtype=np.float32)
    w = np.linalg.solve(G, A.T @ (yt - yt.mean()))
    return B @ w


def _skill(pred, yte, kind) -> float:
    if kind == "reg":
        if pred.std() < 1e-9 or yte.std() < 1e-9:
            return 0.0
        return float(np.corrcoef(pred, yte)[0, 1])
    a = _auc(pred, yte)
    return max(a, 1 - a)


def score_task(codes: np.ndarray, y: np.ndarray, docs: np.ndarray,
               kind: str, split_seed: int = 7, n_boot: int = 300) -> dict:
    """Held-out skill of a LINEAR probe on the code, with a bootstrap CI and
    degeneracy diagnostics. Split is BY DOCUMENT so no document is shared."""
    uniq = np.unique(docs)
    rng = np.random.default_rng(split_seed)
    rng.shuffle(uniq)
    tr_docs = set(uniq[:max(1, int(0.8 * len(uniq)))].tolist())
    tr = np.array([d in tr_docs for d in docs])
    te = ~tr
    diag = {"n_train": int(tr.sum()), "n_test": int(te.sum()),
            "label_std": float(y.std()), "code_nnz": float((codes != 0).sum(1).mean()),
            "dead_frac": float((codes != 0).any(0).mean())}
    if tr.sum() < 50 or te.sum() < 50 or y[te].std() < 1e-6:
        return {"skill": float("nan"), "ci_lo": float("nan"),
                "ci_hi": float("nan"), "degenerate": "too few rows or flat label",
                **diag}
    if diag["code_nnz"] < 0.5:
        return {"skill": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "degenerate": "dictionary is dead (no active latents)", **diag}
    pred = _fit_predict(codes, y, tr, te, kind)
    sk = _skill(pred, y[te], kind)
    # bootstrap the TEST rows (the probe is fixed; this is CI on the estimate)
    yte = y[te]
    br = np.random.default_rng(split_seed + 1)
    vals = []
    for _ in range(n_boot):
        i = br.integers(0, len(yte), len(yte))
        if kind == "cls" and (yte[i].sum() in (0, len(i))):
            continue
        vals.append(_skill(pred[i], yte[i], kind))
    lo, hi = (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) \
        if vals else (sk, sk)
    deg = None
    if pred.std() < 1e-9:
        deg = "probe predicts a constant"
    return {"skill": float(sk), "ci_lo": lo, "ci_hi": hi, "degenerate": deg,
            "pred_std": float(pred.std()), **diag}


def task_rows(stem: str, field: str, model_name: str, T: int,
              max_rows: int = 6000, seed: int = 11):
    """Window starts, labels, and document ids for one candidate task."""
    npz = np.load(LABELS / f"{stem}_{_npz_key(stem, model_name)}.npz")
    y_all = npz[field].astype(np.float32)
    doc_off = npz["doc_off"].astype(np.int64)
    docs = np.zeros(len(y_all), dtype=np.int32)
    for i in range(len(doc_off) - 1):
        docs[doc_off[i]:doc_off[i + 1]] = i
    ok = np.isfinite(y_all)
    # a window must sit inside one document
    starts_ok = np.ones(len(y_all), dtype=bool)
    for i in range(len(doc_off) - 1):
        s = int(doc_off[i])
        starts_ok[s:s + T] = False
    elig = np.flatnonzero(ok & starts_ok)
    elig = elig[elig + 1 < len(y_all)]
    rng = np.random.default_rng(seed)
    if len(elig) > max_rows:
        elig = np.sort(rng.choice(elig, max_rows, replace=False))
    return elig - (T - 1), y_all[elig], docs[elig]


# ── 4. driver ───────────────────────────────────────────────────────────
def run(model_name: str, layer: int, Ts: list[int], steps: int, d_sae: int,
        k_pos: int, batch: int, seed: int, tag_out: str, max_rows: int):
    RESULTS.mkdir(parents=True, exist_ok=True)
    stems = sorted({t[1] for t in TASKS})
    caches = {}
    for stem in stems:
        p = LABELS / f"{stem}_{_npz_key(stem, model_name)}.npz"
        if not p.exists():
            print(f"[skip] no label pack for {stem} / {model_name}")
            continue
        caches[stem] = load_cache(build_cache(model_name, stem, layer))

    out = {"meta": {"model": model_name, "layer": layer, "steps": steps,
                    "d_sae": d_sae, "k_pos": k_pos, "batch": batch,
                    "seed": seed, "Ts": Ts, "max_rows": max_rows,
                    "triage_only": True}, "cells": []}
    for arch_name, path, fixedT, blurb in PANEL:
        for T in ([1] if fixedT == 1 else Ts):
            t0 = time.time()
            for stem, (mm, meta) in caches.items():
                # nominal k buys different budgets per arch -> calibrate it so the
                # probe sees a comparable number of features (measured, not assumed)
                k, got_nnz = calibrate_k(mm, meta, arch_name, path, T, d_sae,
                                         k_pos, "nnz")
                # trained AND untrained (random init) -- the untrained control is
                # the floor that separates "the architecture LEARNED it" from
                # "the architecture had access to it at init".
                for trained in (True, False):
                    arch = train_one(mm, meta, arch_name, path, T, d_sae, k,
                                     steps if trained else 0, batch, seed)
                    for name, s_, field, kind, desc in TASKS:
                        if s_ != stem:
                            continue
                        starts, y, docs = task_rows(stem, field, model_name, T,
                                                    max_rows)
                        codes, l0 = encode_rows(arch, mm, meta, starts, T)
                        r = score_task(codes, y, docs, kind)
                        out["cells"].append({
                            "arch": arch_name, "family": blurb, "T": T,
                            "task": name, "kind": kind, "trained": trained,
                            "l0": round(l0, 2), "rows": int(len(starts)),
                            "seconds": round(time.time() - t0, 1),
                            "nominal_k": k, "calibrated_nnz": round(got_nnz, 2),
                            **r})
                        flag = "" if not r.get("degenerate") else \
                            f"  DEGENERATE: {r['degenerate']}"
                        print(f"  {arch_name:20s} T={T:<2} "
                              f"{'trained' if trained else 'init   '} "
                              f"{name:14s} skill={r['skill']:+.3f} "
                              f"[{r['ci_lo']:+.3f},{r['ci_hi']:+.3f}] "
                              f"l0={l0:.1f}{flag}", flush=True)
                    del arch
                    torch.cuda.empty_cache()
    p = RESULTS / f"sweep_{tag_out}.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"[sweep] wrote {p} ({len(out['cells'])} cells)")
    rank(out)


def rank(out: dict):
    """Per task: best window arch minus best per-token-decoded baseline.

    Only TRAINED cells count, and a win must also beat its own untrained control
    -- otherwise the architecture merely had access to the label at random init.
    """
    cells = [c for c in out["cells"] if c.get("trained", True)
             and not c.get("degenerate")]
    init = {(c["arch"], c["T"], c["task"]): c["skill"]
            for c in out["cells"] if not c.get("trained", True)}
    tasks = sorted({c["task"] for c in cells})
    print("\n=== ranking: window advantage per task "
          "(best window arch − best per-token baseline) ===")
    rows = []
    for task in tasks:
        cs = [c for c in cells if c["task"] == task]
        base = max((c["skill"] for c in cs if c["arch"] in PER_TOKEN_ARCHS),
                   default=float("nan"))
        wins = [c for c in cs if c["arch"] not in PER_TOKEN_ARCHS and c["T"] >= 2]
        if not wins:
            continue
        best = max(wins, key=lambda c: c["skill"])
        post = [c for c in cs if c["arch"] == "txc_batchtopk_post" and c["T"] >= 2]
        bpost = max(post, key=lambda c: c["skill"]) if post else None
        rows.append((best["skill"] - base, task, best, base, bpost))
    for adv, task, best, base, bpost in sorted(rows, reverse=True):
        pm = (f"post {bpost['skill']:+.3f} @T{bpost['T']}"
              if bpost else "post n/a")
        i0 = init.get((best["arch"], best["T"], task), float("nan"))
        learn = best["skill"] - i0
        print(f"  {task:14s} adv {adv:+.3f}  best={best['arch']}@T{best['T']} "
              f"{best['skill']:+.3f} [{best['ci_lo']:+.3f},{best['ci_hi']:+.3f}]"
              f"  per-token {base:+.3f}  {pm}  over-init {learn:+.3f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cache")
    c.add_argument("--model", required=True)
    c.add_argument("--layer", type=int, default=6)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--layer", type=int, default=6)
    r.add_argument("--t-ladder", default="1,2,4,8")
    r.add_argument("--steps", type=int, default=400)
    r.add_argument("--d-sae", type=int, default=2048)
    r.add_argument("--k-pos", type=int, default=20)
    r.add_argument("--batch", type=int, default=256)
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--max-rows", type=int, default=6000)
    r.add_argument("--tag", default="triage")
    a = ap.parse_args()
    if a.cmd == "cache":
        for stem in sorted({t[1] for t in TASKS}):
            if (LABELS / f"{stem}_{_npz_key(stem, a.model)}.npz").exists():
                build_cache(a.model, stem, a.layer)
    else:
        Ts = [int(x) for x in a.t_ladder.split(",") if int(x) >= 1]
        run(a.model, a.layer, [t for t in Ts if t >= 2], a.steps, a.d_sae,
            a.k_pos, a.batch, a.seed, a.tag, a.max_rows)


if __name__ == "__main__":
    main()
