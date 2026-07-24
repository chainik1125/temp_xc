"""Task-hunt candidate 2 — proof-operation run structure Stage-1 screen.

Frozen protocol: `card.md`. Reads runpod-b's committed labels
(`labels/proofops.npz`, balanced manifests) — no duplicate label build.

Per (model ∈ {base, distill} × layer ∈ {hs13 primary, hs11 confirmatory}
× target ∈ {tir PRIMARY, boundary, op ANCHOR} × T ∈ {8, 16, 32, 64}):
per-token linear, window-flatten linear, window-MEAN linear
(g_agg/g_order), within-window-SHUFFLED linear on the frozen `problib`
stack; MLP-512 presence at T = 32; permutation null (seed 99).
Multiclass targets report **macro one-vs-rest AUC** over the frozen
probe's logits (the binary rank-AUC generalization; balanced manifests
make macro-OvR the natural ceiling statistic).

Rows: manifest rows with p >= 64, split by the npz's own `trace_split`,
capped 12,000 train / 3,000 test (rng 13/14, class-stratified) — the
disclosed memory constraint (T=64 flatten = 262,144 dims).

Run:  .venv/bin/python -m experiments.explorations.task_hunt.proofops.screen
Appends cells to results/proofops_screen.json (idempotent per cell).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import (
    DEVICE, EPOCHS, LR, MLP_HIDDEN, WD, rank_auc, _standardize,
)

CACHE_ROOT = Path("/workspace/conv_depth_caches")
LABELS_NPZ = (Path(__file__).resolve().parents[1] / "labels"
              / "proofops.npz")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "proofops_screen.json"

TS = [8, 16, 32, 64]
P_MIN = 64
LAYERS = [13, 11]
MODELS = ["base", "distill"]
TARGETS = {"tir": 3, "boundary": 2, "op": 5}
CAP_TR, CAP_TE = 12000, 3000
CAP_SEED_TR, CAP_SEED_TE = 13, 14
SHUFFLE_SEED = 23
NULL_SEED = 99
MLP_T = 32


def macro_ovr_auc(logits: np.ndarray, y: np.ndarray, n_classes: int) -> float:
    """Macro one-vs-rest rank AUC from the probe's class logits."""
    aucs = []
    for c in range(n_classes):
        yb = (y == c).astype(np.int64)
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        aucs.append(rank_auc(logits[:, c], yb))
    return float(np.mean(aucs))


def fit(Xtr, ytr, Xte, yte, n_classes, hidden=0, seed=0):
    """Frozen problib probe, returning macro-OvR AUC + accuracy."""
    torch.manual_seed(seed)
    ftr, fte = Xtr.to(DEVICE).float(), Xte.to(DEVICE).float()
    ytr_t, yte_t = ytr.to(DEVICE).long(), yte.to(DEVICE).long()
    ftr, fte = _standardize(ftr, fte)
    D = ftr.shape[1]
    if hidden:
        probe = torch.nn.Sequential(torch.nn.Linear(D, hidden),
                                    torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, n_classes))
    else:
        probe = torch.nn.Linear(D, n_classes)
    probe = probe.to(DEVICE)
    cnt = torch.bincount(ytr_t, minlength=n_classes).float().clamp(min=1)
    w = (cnt.sum() / (n_classes * cnt)).to(DEVICE)
    opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        loss = torch.nn.functional.cross_entropy(probe(ftr), ytr_t, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        lg = probe(fte)
        acc = (lg.argmax(-1) == yte_t).float().mean().item()
        auc = macro_ovr_auc(lg.cpu().numpy(), yte.numpy(), n_classes)
    del ftr, fte
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return {"auc_macro_ovr": auc, "acc_test": acc,
            "n_train": int(ytr.numel()), "n_test": int(yte.numel())}


def build_rows():
    z = np.load(LABELS_NPZ)
    split = z["trace_split"]          # 1 = test trace
    trace_idx = z["trace_idx"]
    out = {}
    for tgt in TARGETS:
        doc = z[f"man_{tgt}_doc"]
        pos = z[f"man_{tgt}_pos"]
        cls = z[f"man_{tgt}_cls"]
        keep = pos >= P_MIN
        doc, pos, cls = doc[keep], pos[keep], cls[keep]
        is_test = split[trace_idx[doc]] == 1
        splits = {}
        for name, mask, cap, seed in [("train", ~is_test, CAP_TR, CAP_SEED_TR),
                                      ("test", is_test, CAP_TE, CAP_SEED_TE)]:
            d, p, c = doc[mask], pos[mask], cls[mask]
            rng = np.random.default_rng(seed)
            # class-stratified cap
            per = max(1, cap // TARGETS[tgt])
            sel = []
            for k in range(TARGETS[tgt]):
                idx = np.where(c == k)[0]
                sel.append(rng.permutation(idx)[:per])
            sel = np.concatenate(sel)
            rows = np.stack([d[sel], p[sel]], axis=1).astype(np.int64)
            splits[name] = (rows, c[sel].astype(np.int64))
            print(f"[rows] {tgt}/{name}: {len(sel)} rows "
                  f"({np.bincount(c[sel], minlength=TARGETS[tgt])})",
                  flush=True)
        out[tgt] = splits
    return out


def gather(acts_t, rows, T):
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    X = torch.empty((len(rows), T, acts_t.shape[-1]), dtype=acts_t.dtype)
    for j in range(T):
        X[:, j] = acts_t[w, p - (T - 1) + j]
    return X


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    rows = build_rows()
    done["meta"] = {"protocol": "card.md (frozen)", "Ts": TS,
                    "layers": LAYERS, "p_min": P_MIN,
                    "cap": [CAP_TR, CAP_TE],
                    "labels": "labels/proofops.npz (runpod-b)"}
    for tag in MODELS:
        cache = CACHE_ROOT / tag
        if not (cache / "meta.json").exists():
            print(f"[skip] {tag}: cache not ready", flush=True)
            continue
        for hs in LAYERS:
            acts_t = torch.from_numpy(np.ascontiguousarray(
                np.load(cache / f"hs{hs}.npy", mmap_mode="r")))
            for tgt, nc in TARGETS.items():
                (rtr, ytr), (rte, yte) = rows[tgt]["train"], rows[tgt]["test"]
                ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
                key0 = f"{tag}/hs{hs}/{tgt}"
                if f"{key0}/tok" not in done["cells"]:
                    t0 = time.time()
                    Xtr = gather(acts_t, rtr, 1).reshape(len(rtr), -1)
                    Xte = gather(acts_t, rte, 1).reshape(len(rte), -1)
                    cell = {"linear": fit(Xtr, ytr_t, Xte, yte_t, nc)}
                    g = torch.Generator().manual_seed(NULL_SEED)
                    cell["null"] = fit(
                        Xtr, ytr_t[torch.randperm(len(ytr_t), generator=g)],
                        Xte, yte_t[torch.randperm(len(yte_t), generator=g)],
                        nc)
                    done["cells"][f"{key0}/tok"] = cell
                    print(f"[{key0}/tok] auc={cell['linear']['auc_macro_ovr']:.3f} "
                          f"null={cell['null']['auc_macro_ovr']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))
                for T in TS:
                    key = f"{key0}/T{T}"
                    if key in done["cells"]:
                        continue
                    t0 = time.time()
                    Wtr, Wte = gather(acts_t, rtr, T), gather(acts_t, rte, T)
                    n_tr, n_te = len(rtr), len(rte)
                    cell = {}
                    cell["flat"] = fit(Wtr.reshape(n_tr, -1), ytr_t,
                                       Wte.reshape(n_te, -1), yte_t, nc)
                    cell["mean"] = fit(Wtr.float().mean(1).half(), ytr_t,
                                       Wte.float().mean(1).half(), yte_t, nc)
                    gs = torch.Generator().manual_seed(SHUFFLE_SEED)
                    Str = torch.stack([x[torch.randperm(T, generator=gs)]
                                       for x in Wtr])
                    Ste = torch.stack([x[torch.randperm(T, generator=gs)]
                                       for x in Wte])
                    cell["shuf"] = fit(Str.reshape(n_tr, -1), ytr_t,
                                       Ste.reshape(n_te, -1), yte_t, nc)
                    g = torch.Generator().manual_seed(NULL_SEED)
                    cell["null_flat"] = fit(
                        Wtr.reshape(n_tr, -1),
                        ytr_t[torch.randperm(n_tr, generator=g)],
                        Wte.reshape(n_te, -1),
                        yte_t[torch.randperm(n_te, generator=g)], nc)
                    if T == MLP_T:
                        cell["mlp_flat"] = fit(Wtr.reshape(n_tr, -1), ytr_t,
                                               Wte.reshape(n_te, -1), yte_t,
                                               nc, hidden=MLP_HIDDEN)
                    tok_auc = done["cells"][f"{key0}/tok"]["linear"]["auc_macro_ovr"]
                    cell["g"] = cell["flat"]["auc_macro_ovr"] - tok_auc
                    cell["g_agg"] = cell["mean"]["auc_macro_ovr"] - tok_auc
                    cell["g_order"] = (cell["flat"]["auc_macro_ovr"]
                                       - cell["mean"]["auc_macro_ovr"])
                    cell["shuffle_gap"] = (cell["flat"]["auc_macro_ovr"]
                                           - cell["shuf"]["auc_macro_ovr"])
                    done["cells"][key] = cell
                    print(f"[{key}] flat={cell['flat']['auc_macro_ovr']:.3f} "
                          f"mean={cell['mean']['auc_macro_ovr']:.3f} "
                          f"shuf={cell['shuf']['auc_macro_ovr']:.3f} "
                          f"g={cell['g']:+.3f} g_ord={cell['g_order']:+.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))
                    del Wtr, Wte, Str, Ste
            del acts_t
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
