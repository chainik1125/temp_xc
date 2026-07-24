"""Task-hunt candidate 1 — the Stage-1 screen (frozen protocol, card.md).

Per (model ∈ {base, distill} × layer ∈ {hs13 primary, hs11 confirmatory}
× T ∈ {2, 4, 8, 16, 32}): per-token linear, window-flatten linear,
window-MEAN linear (g_agg/g_order), within-window-SHUFFLED linear
(per-row permutation, seed 23), on the frozen `problib` stack; MLP
presence checks at T = 16 only; permutation null (seed 99) on the
per-token + flatten pair. Primary target = top-vs-bottom tercile of λ̂
(train-set cuts); secondaries per card.md (regression r, λ̂_hist,
position-only floor).

Row recipe (frozen): eligibility = map_ok ∧ in_think ∧ p ≥ 32 ∧
λ̂ finite; split BY TRACE 80/20 (rng(7), the conversion_depth split);
per-trace row cap 120 (rng 13 train / 14 test), class-stratified.
Identical rows for every layer, model, and T — every difference is
attributable to the representation.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.lambda_intensity.screen
Appends cells to results/lambda_screen.json (idempotent per cell).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import (
    EPOCHS, LR, WD, fit_probe, rank_auc,
)

STREAM_DIR = Path("/workspace/conv_depth_caches/ward_stream")
CACHE_ROOT = Path("/workspace/conv_depth_caches")
LABEL_DIR = Path("/workspace/task_hunt_labels/lambda_intensity")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "lambda_screen.json"

TS = [2, 4, 8, 16, 32]
P_MIN = 32
LAYERS = [13, 11]              # hs13 = resid_post L12 primary; hs11 = L10
MODELS = ["base", "distill"]
SPLIT_SEED = 7
CAP_SEED_TR, CAP_SEED_TE = 13, 14
SHUFFLE_SEED = 23
NULL_SEED = 99
ROWS_PER_TRACE = 120
MLP_T = 16


def build_rows():
    """Frozen row recipe -> per target {train/test: (rows, y, lam, pos)}."""
    map_ok = np.load(STREAM_DIR / "map_ok.npy")
    in_think = np.load(STREAM_DIR / "in_think.npy")
    trace_idx = np.load(STREAM_DIR / "trace_idx.npy")
    lam = np.load(LABEL_DIR / "lam_hat.npy")
    lam_h = np.load(LABEL_DIR / "lam_hist.npy")
    spos = np.load(LABEL_DIR / "sent_pos.npy")

    elig = map_ok & in_think & np.isfinite(lam)
    elig[:, :P_MIN] = False

    traces = np.unique(trace_idx)
    perm = np.random.default_rng(SPLIT_SEED).permutation(len(traces))
    test_traces = set(traces[perm[:len(traces) // 5]].tolist())

    out = {}
    for tgt, lab in [("lam_hat", lam), ("lam_hist", lam_h)]:
        # tercile cuts from ALL eligible TRAIN rows (uncapped)
        tw = np.where(np.isin(trace_idx,
                              [t for t in traces
                               if int(t) not in test_traces]))[0]
        vals = lab[tw][elig[tw]]
        lo, hi = np.percentile(vals, [33.3, 66.7])
        splits = {}
        for split, is_train in [("train", True), ("test", False)]:
            rng = np.random.default_rng(CAP_SEED_TR if is_train
                                        else CAP_SEED_TE)
            rows, ys = [], []
            for ti in traces:
                if (int(ti) in test_traces) == is_train:
                    continue
                widx = np.where(trace_idx == ti)[0]
                e = elig[widx]
                lv = lab[widx]
                cls = np.full(lv.shape, -1, dtype=np.int64)
                cls[e & (lv <= lo)] = 0
                cls[e & (lv >= hi)] = 1
                cand_w, cand_p = np.where(cls >= 0)
                if cand_w.size == 0:
                    continue
                take = rng.permutation(cand_w.size)[:ROWS_PER_TRACE]
                for j in take:
                    rows.append((int(widx[cand_w[j]]), int(cand_p[j])))
                    ys.append(int(cls[cand_w[j], cand_p[j]]))
            rows = np.array(rows, dtype=np.int64)
            ys = np.array(ys, dtype=np.int64)
            lamv = lab[rows[:, 0], rows[:, 1]].astype(np.float32)
            posv = spos[rows[:, 0], rows[:, 1]].astype(np.float32)
            splits[split] = (rows, ys, lamv, posv)
            print(f"[rows] {tgt}/{split}: {len(ys)} rows "
                  f"({int(ys.sum())} pos) cuts=({lo:.4f},{hi:.4f})",
                  flush=True)
        out[tgt] = {"splits": splits, "cuts": (float(lo), float(hi))}
    return out


def gather_windows(acts_t, rows, T):
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    n = len(rows)
    d = acts_t.shape[-1]
    X = torch.empty((n, T, d), dtype=acts_t.dtype)
    for j in range(T):
        X[:, j] = acts_t[w, p - (T - 1) + j]
    return X


def fit_reg(Xtr, ytr, Xte, yte, seed=0):
    """Linear regression head, frozen hyperparameters; test Pearson r."""
    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ftr, fte = Xtr.to(dev).float(), Xte.to(dev).float()
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True).clamp(min=1e-6)
    ftr, fte = (ftr - mu) / sd, (fte - mu) / sd
    ytr_t = torch.from_numpy(ytr).to(dev).float()
    head = torch.nn.Linear(ftr.shape[1], 1).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        loss = torch.nn.functional.mse_loss(head(ftr).squeeze(-1), ytr_t)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        pred = head(fte).squeeze(-1).cpu().numpy()
    r = float(np.corrcoef(pred, yte)[0, 1])
    del ftr, fte
    if dev == "cuda":
        torch.cuda.empty_cache()
    return {"pearson_r": r}


def pos_floor(splits):
    """Position-only tercile AUC (scalar logistic = rank on pos)."""
    (_, ytr, _, ptr), (_, yte, _, pte) = splits["train"], splits["test"]
    # scalar monotone feature: AUC is rank_auc of pos (sign-corrected)
    auc = rank_auc(pte.astype(np.float64), yte)
    return {"auc": float(max(auc, 1 - auc))}


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    targets = build_rows()
    done["meta"] = {"protocol": "card.md (frozen)", "Ts": TS,
                    "layers": LAYERS, "p_min": P_MIN,
                    "rows_per_trace": ROWS_PER_TRACE,
                    "cuts": {t: targets[t]["cuts"] for t in targets}}
    for tgt in targets:
        done.setdefault("floors", {})[tgt] = pos_floor(targets[tgt]["splits"])
        print(f"[floor] {tgt}: pos-only AUC "
              f"{done['floors'][tgt]['auc']:.3f}", flush=True)

    for tag in MODELS:
        cache = CACHE_ROOT / tag
        if not (cache / "meta.json").exists():
            print(f"[skip] {tag}: cache not ready", flush=True)
            continue
        for hs in LAYERS:
            acts_t = torch.from_numpy(np.ascontiguousarray(
                np.load(cache / f"hs{hs}.npy", mmap_mode="r")))
            for tgt, tdata in targets.items():
                sp = tdata["splits"]
                (rtr, ytr, ltr, _), (rte, yte, lte, _) = sp["train"], sp["test"]
                ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
                key0 = f"{tag}/hs{hs}/{tgt}"

                # per-token (T-independent)
                if f"{key0}/tok" not in done["cells"]:
                    t0 = time.time()
                    Xtr = gather_windows(acts_t, rtr, 1).reshape(len(rtr), -1)
                    Xte = gather_windows(acts_t, rte, 1).reshape(len(rte), -1)
                    cell = {"linear": fit_probe(Xtr, ytr_t, Xte, yte_t, 2,
                                                class_weight=True)}
                    g = torch.Generator().manual_seed(NULL_SEED)
                    cell["null"] = fit_probe(
                        Xtr, ytr_t[torch.randperm(len(ytr_t), generator=g)],
                        Xte, yte_t[torch.randperm(len(yte_t), generator=g)],
                        2, class_weight=True)
                    cell["mlp"] = fit_probe(Xtr, ytr_t, Xte, yte_t, 2,
                                            hidden=512, class_weight=True)
                    cell["reg"] = fit_reg(Xtr, ltr, Xte, lte)
                    done["cells"][f"{key0}/tok"] = cell
                    print(f"[{key0}/tok] auc={cell['linear']['auc']:.3f} "
                          f"r={cell['reg']['pearson_r']:.3f} "
                          f"null={cell['null']['auc']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))

                for T in TS:
                    key = f"{key0}/T{T}"
                    if key in done["cells"]:
                        continue
                    t0 = time.time()
                    Wtr = gather_windows(acts_t, rtr, T)
                    Wte = gather_windows(acts_t, rte, T)
                    n_tr, n_te = len(rtr), len(rte)
                    cell = {}
                    cell["flat"] = fit_probe(
                        Wtr.reshape(n_tr, -1), ytr_t,
                        Wte.reshape(n_te, -1), yte_t, 2, class_weight=True)
                    cell["mean"] = fit_probe(
                        Wtr.float().mean(1).half(), ytr_t,
                        Wte.float().mean(1).half(), yte_t, 2,
                        class_weight=True)
                    gs = torch.Generator().manual_seed(SHUFFLE_SEED)
                    Str = torch.stack([w[torch.randperm(T, generator=gs)]
                                       for w in Wtr])
                    Ste = torch.stack([w[torch.randperm(T, generator=gs)]
                                       for w in Wte])
                    cell["shuf"] = fit_probe(
                        Str.reshape(n_tr, -1), ytr_t,
                        Ste.reshape(n_te, -1), yte_t, 2, class_weight=True)
                    g = torch.Generator().manual_seed(NULL_SEED)
                    cell["null_flat"] = fit_probe(
                        Wtr.reshape(n_tr, -1),
                        ytr_t[torch.randperm(n_tr, generator=g)],
                        Wte.reshape(n_te, -1),
                        yte_t[torch.randperm(n_te, generator=g)],
                        2, class_weight=True)
                    cell["reg_flat"] = fit_reg(Wtr.reshape(n_tr, -1), ltr,
                                               Wte.reshape(n_te, -1), lte)
                    if T == MLP_T:
                        cell["mlp_flat"] = fit_probe(
                            Wtr.reshape(n_tr, -1), ytr_t,
                            Wte.reshape(n_te, -1), yte_t, 2,
                            hidden=512, class_weight=True)
                    tok_auc = done["cells"][f"{key0}/tok"]["linear"]["auc"]
                    cell["g"] = cell["flat"]["auc"] - tok_auc
                    cell["g_agg"] = cell["mean"]["auc"] - tok_auc
                    cell["g_order"] = cell["flat"]["auc"] - cell["mean"]["auc"]
                    cell["shuffle_gap"] = (cell["flat"]["auc"]
                                           - cell["shuf"]["auc"])
                    done["cells"][key] = cell
                    print(f"[{key}] flat={cell['flat']['auc']:.3f} "
                          f"mean={cell['mean']['auc']:.3f} "
                          f"shuf={cell['shuf']['auc']:.3f} "
                          f"g={cell['g']:+.3f} g_ord={cell['g_order']:+.3f} "
                          f"r={cell['reg_flat']['pearson_r']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))
                    del Wtr, Wte, Str, Ste
            del acts_t
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
