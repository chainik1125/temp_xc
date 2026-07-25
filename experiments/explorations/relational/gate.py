"""Track-A gate — raw-activation ceilings on a balanced-marginal equality label.

No dictionaries are trained here. This is the § 8 discriminability gate from
`synthetic/README.md`, run on real activations: if a per-token probe reads the
label as well as a window probe, the latent is ambient/converted and no
architecture grid can separate on it (task_hunt round-1 rule: per-token-first
triage is kill authority).

Arms per cell (model x layer x T), all on IDENTICAL rows so every difference is
attributable to the representation:

  per_token      linear probe on h[p]                      (one position)
  window_flat    linear probe on h[p-T+1..p] flattened     (order preserved)
  window_mean    linear probe on mean_t h[p-T+1..p]        (order destroyed)
  window_shuf    linear probe on a per-row permutation of the window

  g       = window_flat - per_token     (does a window help at all?)
  g_order = window_flat - window_mean   (does ORDER carry any of it?)

Controls the round-1 record demands:
  * permutation null (labels shuffled, N_NULL draws) -> sigma_null, so
    "beats per-token" has a scale; read every gap against 3*sigma_null.
  * bootstrap CI on every AUC (2.5/97.5 over BOOT resamples of the test rows).
  * IN/OUT stratification: constituent A sits at a known token distance, so the
    SAME label at the SAME probe position is scored separately for rows whose
    window reaches A and rows whose window does not. A window advantage present
    only in the IN stratum is cross-position binding; one present in both is an
    artifact. This is immune to the shuffle-gap-grows-with-T confound that bit
    task_hunt candidate 2 (RECORD § 2).

Probe recipe is problib's, verbatim (EPOCHS/LR/WD/standardization). `_fit_scores`
mirrors it locally only because the frozen helper returns summary stats and
bootstrap CIs need per-row test scores; `--crosscheck` asserts the two agree to
1e-6 before any result is written.

Run:
  .venv/bin/python -m experiments.explorations.relational.gate \
      --model Qwen/Qwen3-1.7B --tasks agreement,contradiction --tag pilot
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.explorations.conversion_depth.problib import (
    EPOCHS, LR, WD, MLP_HIDDEN, DEVICE, fit_probe, rank_auc,
)

HERE = Path(__file__).resolve().parent
LABELS = HERE / "labels"
RESULTS = HERE / "results"

T_LADDER = [2, 4, 8, 16, 32, 64]
BOOT = 1000
SPLIT_SEED = 7
SHUF_SEED = 23
NULL_SEED = 99
N_NULL = 8
MIN_ROWS = 80


# ── probe (problib recipe, returning test scores) ────────────────────────
def _fit_scores(Xtr, ytr, Xte, yte, seed: int = 0, hidden: int = 0) -> dict:
    torch.manual_seed(seed)
    ftr = torch.from_numpy(Xtr).to(DEVICE).float()
    fte = torch.from_numpy(Xte).to(DEVICE).float()
    Ytr = torch.from_numpy(ytr).to(DEVICE).long()
    mu = ftr.mean(0, keepdim=True)
    sd = ftr.std(0, keepdim=True).clamp(min=1e-6)
    ftr = (ftr - mu) / sd
    fte = (fte - mu) / sd
    if hidden:
        probe = nn.Sequential(nn.Linear(ftr.shape[1], hidden), nn.ReLU(),
                              nn.Linear(hidden, 2)).to(DEVICE)
    else:
        probe = nn.Linear(ftr.shape[1], 2).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        loss = F.cross_entropy(probe(ftr), Ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        lo_tr = probe(ftr)
        lo_te = probe(fte)
        s_te = (lo_te[:, 1] - lo_te[:, 0]).cpu().numpy()
        acc_tr = (lo_tr.argmax(-1) == Ytr).float().mean().item()
    peak = torch.cuda.max_memory_allocated() / 1e9 if DEVICE.type == "cuda" else 0.0
    del ftr, fte, lo_tr, lo_te
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"scores": s_te, "acc_train": acc_tr, "peak_vram_gb": peak}


def auc_ci(scores: np.ndarray, y: np.ndarray, n_boot: int = BOOT,
           seed: int = 11) -> dict:
    """Bootstrap CI on rank-AUC over test rows. CIs are mandatory here."""
    rng = np.random.default_rng(seed)
    point = rank_auc(scores, y)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() in (0, n):
            continue
        vals.append(rank_auc(scores[idx], yy))
    if not vals:
        return {"value": float(point), "ci_lo": float(point),
                "ci_hi": float(point), "n": int(n)}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"value": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "n": int(n)}


# ── activation extraction ───────────────────────────────────────────────
def _span_end_token(offsets, char_end: int) -> int | None:
    for i, (s, e) in enumerate(offsets):
        if s == e:
            continue
        if s <= char_end < e:
            return i
    return None


def _assert_offsets_usable(tok) -> None:
    """Offsets must be monotone, non-overlapping, and recover the exact span.

    Replaces the `is_fast` check, which lies for the slow Llama tokenizer.
    """
    probe = "They reported that the keys to the cabinets is rusted."
    enc = tok(probe, return_offsets_mapping=True, add_special_tokens=True)
    offs = [(s, e) for s, e in enc["offset_mapping"] if s != e]
    prev_end = -1
    for s, e in offs:
        assert s >= prev_end, f"overlapping offsets {offs[:8]} — slow tokenizer"
        prev_end = e
    target = "the keys"
    c = probe.find(target) + len(target) - 1
    hit = [i for i, (s, e) in enumerate(offs) if s <= c < e]
    assert hit, f"span lookup failed on offsets {offs[:8]}"
    got = probe[offs[hit[0]][0]:offs[hit[0]][1]].strip()
    assert got == "keys", f"span lookup recovered {got!r}, expected 'keys'"


def build_cache(model_name: str, task: str, layers: list[int], t_max: int,
                batch_size: int = 24) -> dict:
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              PreTrainedTokenizerFast)

    data = json.loads((LABELS / f"{task}_stimuli.json").read_text())
    items = data["items"]
    # RECORD § 4 note 5: AutoTokenizer can return the SLOW Llama tokenizer while
    # reporting is_fast=True, and its offsets are silently unusable (overlapping
    # spans). `is_fast` is therefore NOT a valid guard — validate the offsets.
    try:
        tok = PreTrainedTokenizerFast.from_pretrained(model_name)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    _assert_offsets_usable(tok)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    rows, kept = [], []
    for it in items:
        enc = tok(it["text"], return_offsets_mapping=True, add_special_tokens=True)
        offs = enc["offset_mapping"]
        p = _span_end_token(offs, it["b_char_end"])   # generator-recorded offsets
        v = _span_end_token(offs, it["a_char_end"])
        if p is None or v is None or p <= v:
            continue
        rows.append((enc["input_ids"], p, v))
        kept.append(it)
    assert len(rows) > 0.9 * len(items), f"only {len(rows)}/{len(items)} rows usable"

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float16, device_map="cuda")
    model.eval()
    n_layers = model.config.num_hidden_layers
    layers = sorted({l for l in layers if 0 <= l <= n_layers})
    d = model.config.hidden_size

    win = {l: np.zeros((len(rows), t_max, d), dtype=np.float16) for l in layers}
    dist = np.zeros(len(rows), dtype=np.int32)
    peak, t0 = 0.0, time.time()
    for b0 in range(0, len(rows), batch_size):
        chunk = rows[b0:b0 + batch_size]
        maxlen = max(len(c[0]) for c in chunk)
        ids = torch.full((len(chunk), maxlen), tok.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), maxlen), dtype=torch.long)
        for i, (seq, _, _) in enumerate(chunk):
            ids[i, :len(seq)] = torch.tensor(seq)
            mask[i, :len(seq)] = 1
        with torch.no_grad():
            out = model(ids.to("cuda"), attention_mask=mask.to("cuda"),
                        output_hidden_states=True)
        for l in layers:
            hs = out.hidden_states[l]
            for i, (_, p, v) in enumerate(chunk):
                lo = max(0, p - t_max + 1)
                seg = hs[i, lo:p + 1].to(torch.float16).cpu().numpy()
                win[l][b0 + i, t_max - seg.shape[0]:] = seg
        for i, (_, p, v) in enumerate(chunk):
            dist[b0 + i] = p - v
        peak = max(peak, torch.cuda.max_memory_allocated() / 1e9)
        del out
    del model
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    finite = dist[dist < 10_000]
    meta = {"model": model_name, "task": task, "n_rows": len(rows), "d": d,
            "t_max": t_max, "layers": layers, "n_layers": n_layers,
            "cache_peak_vram_gb": round(peak, 3),
            "cache_seconds": round(time.time() - t0, 1),
            "dist_median": float(np.median(finite)),
            "dist_min": int(finite.min()), "dist_max": int(finite.max()),
            "dist_p90": float(np.percentile(finite, 90))}
    return {"win": win, "y": np.array([it["label"] for it in kept], dtype=np.int64),
            "group": np.array([it["group"] for it in kept], dtype=np.int64),
            "dist": dist, "meta": meta}


# ── splits + cells ──────────────────────────────────────────────────────
def split_by_group(grp: np.ndarray, seed: int = SPLIT_SEED, frac: float = 0.8):
    """Split BY TEMPLATE GROUP — no template appears in both halves."""
    rng = np.random.default_rng(seed)
    gs = np.unique(grp)
    rng.shuffle(gs)
    tr_g = set(gs[:max(1, int(round(frac * len(gs))))].tolist())
    tr = np.array([g in tr_g for g in grp])
    return tr, ~tr


def run_cell(cache: dict, layer: int, T: int, stratum: str = "all",
             row_cap: int | None = None, n_null: int = N_NULL) -> dict:
    win, y, grp, dist = (cache["win"][layer], cache["y"], cache["group"],
                         cache["dist"])
    t_max = cache["meta"]["t_max"]

    keep = np.ones(len(y), dtype=bool)
    if stratum == "in":
        keep = dist < T
    elif stratum == "out":
        keep = dist >= T
    if keep.sum() < MIN_ROWS or len(np.unique(y[keep])) < 2:
        return {"skipped": f"{stratum}: {int(keep.sum())} rows < {MIN_ROWS}"}
    idx = np.flatnonzero(keep)
    if row_cap and len(idx) > row_cap:
        idx = np.random.default_rng(5).choice(idx, row_cap, replace=False)

    W = win[idx][:, t_max - T:, :].astype(np.float32)
    yy, gg = y[idx], grp[idx]
    tr, te = split_by_group(gg)
    if tr.sum() < 40 or te.sum() < 40:
        return {"skipped": "split too small"}

    rng = np.random.default_rng(SHUF_SEED)
    Wsh = np.empty_like(W)
    for i in range(len(W)):
        Wsh[i] = W[i][rng.permutation(T)]

    # LINEAR arms are the ADDITIVE ceiling: a linear probe on the flattened
    # window is itself additive over per-position features, so it bounds what
    # every additive dictionary (per-token SAE, T-SAE, Stacked, TXC-pre) can
    # linearly expose. The MLP arms are the NONLINEAR ceiling. The gap between
    # them is the regime-3 headroom only a position-mixing code can capture.
    flat = W.reshape(len(W), -1)
    arms = {
        "per_token": (W[:, -1, :], 0),
        "window_flat": (flat, 0),
        "window_mean": (W.mean(1), 0),
        "window_shuf": (Wsh.reshape(len(Wsh), -1), 0),
        "per_token_mlp": (W[:, -1, :], MLP_HIDDEN),
        "window_mlp": (flat, MLP_HIDDEN),
    }
    out = {"layer": layer, "T": T, "stratum": stratum,
           "rows": int(len(idx)), "n_train": int(tr.sum()),
           "n_test": int(te.sum()), "pos_rate": float(yy.mean()),
           "dist_median": float(np.median(dist[idx]))}
    peak = 0.0
    for name, (X, hid) in arms.items():
        r = _fit_scores(X[tr], yy[tr], X[te], yy[te], hidden=hid)
        out[name] = auc_ci(r["scores"], yy[te])
        out[name]["acc_train"] = round(r["acc_train"], 4)
        peak = max(peak, r["peak_vram_gb"])

    # permutation null on the flatten arm: how big a gap does noise produce?
    nulls = []
    for k in range(n_null):
        rp = np.random.default_rng(NULL_SEED + k)
        yperm = rp.permutation(yy)
        r = _fit_scores(flat[tr], yperm[tr], flat[te], yperm[te], seed=k)
        nulls.append(rank_auc(r["scores"], yperm[te]))
    out["null_mean"] = float(np.mean(nulls))
    out["sigma_null"] = float(np.std(nulls))
    out["three_sigma"] = float(3 * np.std(nulls))

    out["g"] = round(out["window_flat"]["value"] - out["per_token"]["value"], 4)
    out["g_order"] = round(out["window_flat"]["value"] - out["window_mean"]["value"], 4)
    out["g_shuf"] = round(out["window_flat"]["value"] - out["window_shuf"]["value"], 4)
    add_ceiling = max(out["window_flat"]["value"], out["per_token"]["value"])
    out["additive_ceiling"] = round(add_ceiling, 4)
    out["nonlinear_residual"] = round(out["window_mlp"]["value"] - add_ceiling, 4)
    out["mlp_token_residual"] = round(out["window_mlp"]["value"]
                                      - out["per_token_mlp"]["value"], 4)
    out["clears_3sigma"] = bool(out["g"] > out["three_sigma"])
    out["regime3_headroom"] = bool(out["nonlinear_residual"] > out["three_sigma"])
    out["peak_vram_gb"] = round(peak, 3)
    return out


# ── driver ──────────────────────────────────────────────────────────────
def crosscheck(cache: dict, layer: int) -> dict:
    """Assert the local probe reproduces the frozen problib probe."""
    win, y, grp = cache["win"][layer], cache["y"], cache["group"]
    t_max = cache["meta"]["t_max"]
    X = win[:, -1, :].astype(np.float32)
    tr, te = split_by_group(grp)
    mine = _fit_scores(X[tr], y[tr], X[te], y[te], seed=0)
    a_mine = rank_auc(mine["scores"], y[te])
    theirs = fit_probe(torch.from_numpy(X[tr]), torch.from_numpy(y[tr]),
                       torch.from_numpy(X[te]), torch.from_numpy(y[te]), 2, seed=0)
    return {"local_auc": float(a_mine), "problib_auc": float(theirs["auc"]),
            "abs_diff": float(abs(a_mine - theirs["auc"]))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default="agreement,contradiction")
    ap.add_argument("--layers", default="")
    ap.add_argument("--t-ladder", default=",".join(map(str, T_LADDER)))
    ap.add_argument("--t-max", type=int, default=64)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--n-null", type=int, default=N_NULL)
    args = ap.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)
    ts = args.t_ladder and [int(x) for x in args.t_ladder.split(",")]
    for task in args.tasks.split(","):
        t0 = time.time()
        # depth sweep: quarter/half/three-quarter/final residual layers
        if args.layers:
            layers = [int(x) for x in args.layers.split(",")]
        else:
            layers = None
        print(f"[gate] caching {task} on {args.model} ...", flush=True)
        if layers is None:
            from transformers import AutoConfig
            n = AutoConfig.from_pretrained(args.model).num_hidden_layers
            layers = sorted({max(1, n // 4), n // 2, (3 * n) // 4, n})
        cache = build_cache(args.model, task, layers, args.t_max,
                            batch_size=args.batch_size)
        print(f"[gate] {cache['meta']}", flush=True)

        cc = crosscheck(cache, layers[len(layers) // 2])
        print(f"[gate] probe crosscheck vs problib: {cc}", flush=True)
        assert cc["abs_diff"] < 1e-6, f"probe recipe diverged: {cc}"

        cells, ooms = [], []
        for layer in cache["meta"]["layers"]:
            for T in ts:
                for stratum in ("all", "in", "out"):
                    key = f"L{layer}/T{T}/{stratum}"
                    try:
                        c = run_cell(cache, layer, T, stratum, n_null=args.n_null)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        pk = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                        try:
                            c = run_cell(cache, layer, T, stratum, row_cap=800, n_null=args.n_null)
                            ooms.append({"cell": key, "peak_vram_gb": pk,
                                         "action": "retried at row_cap=800, succeeded"})
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            c = {"oom_skipped": True}
                            ooms.append({"cell": key, "peak_vram_gb": pk,
                                         "action": "oom_skipped (disclosed)"})
                    if "skipped" not in c and "oom_skipped" not in c:
                        print(f"  {key:16s} tok {c['per_token']['value']:.3f} "
                              f"flat {c['window_flat']['value']:.3f} "
                              f"mean {c['window_mean']['value']:.3f} "
                              f"wMLP {c['window_mlp']['value']:.3f} "
                              f"g {c['g']:+.3f} nlr {c['nonlinear_residual']:+.3f} "
                              f"(3s {c['three_sigma']:.3f})"
                              f"{' R3' if c['regime3_headroom'] else ''}",
                              flush=True)
                    cells.append({"key": key, **c})
        payload = {"meta": {**cache["meta"], "tag": args.tag,
                            "crosscheck": cc, "t_ladder": ts,
                            "wall_seconds": round(time.time() - t0, 1),
                            "boot": BOOT, "n_null": args.n_null},
                   "oom_events": ooms, "cells": cells}
        out = RESULTS / f"gate_{task}_{args.tag}.json"
        out.write_text(json.dumps(payload, indent=1))
        print(f"[gate] wrote {out} ({len(cells)} cells, "
              f"{len(ooms)} oom events, {payload['meta']['wall_seconds']}s)")


if __name__ == "__main__":
    main()
