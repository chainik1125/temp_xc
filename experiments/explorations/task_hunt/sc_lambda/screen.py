"""Factory candidate B1 (λ̂_sc) — the Stage-1 screen (frozen: CARD.md).

Consumes runpod-b's screen-ready bundle **unmodified**
(`../labels/sc_lambda.npz`): its manifests, binning, trace split and
marker-token masking are used exactly as shipped. Per (model ∈ {base,
distill} × layer ∈ {hs13 primary, hs11 confirmatory} × T ∈ {2,4,8,16,32}):
per-token, window-flatten, window-MEAN, within-window-SHUFFLED linear
probes on the frozen `problib` stack, plus a permutation null (seed 99)
on the per-token/flatten pair, and the whole stack repeated on the
bundle's NULL labels (within-trace event shuffle) as the ambient-rate
control.

**Per-token-first triage** (binding hunt convention): the per-token arm
for a cell is computed and flushed to disk BEFORE any window arm of that
cell starts, so the ordering is auditable in the results file.

PRIMARY target = top vs bottom bin (`man_cls ∈ {0,2}`). Rows are
identical for every model, layer and T, so every difference is
attributable to the representation.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.sc_lambda.screen
Appends cells to results/sc_screen.json (idempotent per cell).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe

CACHE_ROOT = Path("/workspace/conv_depth_caches")
BUNDLE = (Path(__file__).resolve().parents[1] / "labels" / "sc_lambda.npz")
STATS = (Path(__file__).resolve().parents[1] / "labels"
         / "sc_lambda_stats.json")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "sc_screen.json"

TS = [2, 4, 8, 16, 32]
LAYERS = [13, 11]              # hs13 = resid_post L12 primary; hs11 = L10
MODELS = ["base", "distill"]
SHUFFLE_SEED = 23
NULL_SEED = 99
ARMS = ("real", "null")        # bundle manifests: man_* and man_null_*


def load_rows():
    """The bundle's manifests, as shipped → per-arm train/test rows.

    PRIMARY = top vs bottom bin: keep `man_cls in {0,2}`, map 0→0, 2→1.
    Split by the bundle's own `trace_split` (1 = test).
    """
    z = np.load(BUNDLE)
    ti, ts = z["trace_idx"], z["trace_split"]
    out = {}
    for arm in ARMS:
        pre = "man_" if arm == "real" else "man_null_"
        doc, pos, cls = z[pre + "doc"], z[pre + "pos"], z[pre + "cls"]
        keep = np.isin(cls, (0, 2))
        doc, pos, cls = doc[keep], pos[keep], cls[keep]
        y = (cls == 2).astype(np.int64)
        is_test = ts[ti[doc]] == 1
        splits = {}
        for name, m in (("train", ~is_test), ("test", is_test)):
            rows = np.stack([doc[m], pos[m]], axis=1).astype(np.int64)
            splits[name] = (rows, y[m])
            print(f"[rows] {arm}/{name}: {m.sum()} rows "
                  f"({int(y[m].sum())} pos)", flush=True)
        out[arm] = splits
    return out


def gather_windows(acts_t, rows, T):
    """Right-edge length-T window ending at each manifest position."""
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    X = torch.empty((len(rows), T, acts_t.shape[-1]), dtype=acts_t.dtype)
    for j in range(T):
        X[:, j] = acts_t[w, p - (T - 1) + j]
    return X


def main():
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    arms = load_rows()
    stats = json.loads(STATS.read_text())
    done["meta"] = {
        "protocol": "sc_lambda/CARD.md (frozen)", "Ts": TS, "layers": LAYERS,
        "primary": "top-vs-bottom bin (man_cls in {0,2})",
        "bundle_triage": stats["triage"],
        "visible_evidence_auc": stats["visible_evidence_auc"],
        "corr_with_ward_lam_hist": stats["corr_lam_sc_ward_lam_hist"],
    }
    OUT.write_text(json.dumps(done, indent=1))

    for tag in MODELS:
        cache = CACHE_ROOT / tag
        if not (cache / "meta.json").exists():
            print(f"[skip] {tag}: cache not ready", flush=True)
            continue
        for hs in LAYERS:
            acts_t = torch.from_numpy(np.ascontiguousarray(
                np.load(cache / f"hs{hs}.npy", mmap_mode="r")))
            for arm, splits in arms.items():
                (rtr, ytr), (rte, yte) = splits["train"], splits["test"]
                ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)
                key0 = f"{tag}/hs{hs}/{arm}"

                # ── per-token FIRST (triage convention), flushed before windows
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
                    done["cells"][f"{key0}/tok"] = cell
                    print(f"[{key0}/tok] auc={cell['linear']['auc']:.3f} "
                          f"null={cell['null']['auc']:.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))
                    del Xtr, Xte

                tok_auc = done["cells"][f"{key0}/tok"]["linear"]["auc"]
                for T in TS:
                    key = f"{key0}/T{T}"
                    if key in done["cells"]:
                        continue
                    t0 = time.time()
                    Wtr, Wte = (gather_windows(acts_t, rtr, T),
                                gather_windows(acts_t, rte, T))
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
                          f"({time.time()-t0:.0f}s)", flush=True)
                    OUT.write_text(json.dumps(done, indent=1))
                    del Wtr, Wte, Str, Ste
            del acts_t
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
