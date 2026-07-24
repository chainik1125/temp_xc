"""Generic Stage-1 screen for the candidate-factory Ward-grid bundles.

The factory ships every Ward-grid bundle in one format (`labels/*.npz`):
manifests `man_<target>_{doc,pos,cls}` plus a within-trace-shuffled null
arm `man_<target>_null_{doc,pos,cls}`, a `trace_split` (1 = test) and a
`trace_idx`. `sc_lambda` is the degenerate case where the target prefix
is empty (`man_doc` / `man_null_doc`).

This module is the SAME frozen protocol `sc_lambda/screen.py` ran,
parameterised by (bundle, target) so each candidate's screen is a card
+ one call rather than a copied script. Protocol (frozen per candidate
card): models {base, distill} × layers {hs13 primary, hs11 confirmatory}
× T ∈ {2,4,8,16,32}; arms per cell = per-token, window-flatten,
window-MEAN, within-window-SHUFFLE (seed 23); permutation null (seed 99)
on the per-token/flatten pair → σ_null. Rows are identical across model,
layer and T, so every difference is attributable to the representation.

**Per-token-first triage** (binding hunt convention) is implemented
literally: a cell's per-token arm is computed and flushed to disk BEFORE
any of its window arms start, so the ordering is auditable in the
results file.

**The window-MEAN arm is the load-bearing control**, not a nicety: it
carries the SAME d_in dimensionality as the per-token probe, so if
g_agg ≈ g the window gain cannot be probe capacity (the artifact
RECORD § 3c found in the Stage-2 panel). Report g_agg beside g always.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.factory_screen \
        <bundle> <target> [out_dir]
e.g.  … factory_screen oprate ver experiments/explorations/task_hunt/oprate
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe

CACHE_ROOT = Path("/workspace/conv_depth_caches")
LABEL_DIR = Path(__file__).resolve().parent / "labels"

TS = [2, 4, 8, 16, 32]
LAYERS = [13, 11]              # hs13 = resid_post L12 primary; hs11 = L10
MODELS = ["base", "distill"]
SHUFFLE_SEED = 23
NULL_SEED = 99


def load_rows(bundle: str, target: str):
    """Bundle manifests, as shipped → {arm: {split: (rows, y)}}.

    PRIMARY = top vs bottom class (`cls in {0,2}` → 0/1); split by the
    bundle's own `trace_split`. `target=""` selects the unprefixed
    `man_*` manifests (sc_lambda's layout).
    """
    z = np.load(LABEL_DIR / f"{bundle}.npz")
    ti, ts = z["trace_idx"], z["trace_split"]
    pre = f"man_{target}_" if target else "man_"
    npre = f"man_{target}_null_" if target else "man_null_"
    out = {}
    for arm, p in (("real", pre), ("null", npre)):
        doc, pos, cls = z[p + "doc"], z[p + "pos"], z[p + "cls"]
        keep = np.isin(cls, (0, 2))
        doc, pos, cls = doc[keep], pos[keep], cls[keep]
        y = (cls == 2).astype(np.int64)
        is_test = ts[ti[doc]] == 1
        splits = {}
        for name, m in (("train", ~is_test), ("test", is_test)):
            splits[name] = (np.stack([doc[m], pos[m]], 1).astype(np.int64), y[m])
            print(f"[rows] {arm}/{name}: {int(m.sum())} rows "
                  f"({int(y[m].sum())} pos)", flush=True)
        out[arm] = splits
    return out


def gather_windows(acts_t, rows, T):
    """Right-edge length-T window ending at each manifest position."""
    w, p = torch.from_numpy(rows[:, 0]), torch.from_numpy(rows[:, 1])
    X = torch.empty((len(rows), T, acts_t.shape[-1]), dtype=acts_t.dtype)
    for j in range(T):
        X[:, j] = acts_t[w, p - (T - 1) + j]
    return X


def run(bundle: str, target: str, out_dir: Path):
    out_path = out_dir / "results" / f"{bundle}_{target or 'main'}_screen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_path.read_text()) if out_path.exists() else {"cells": {}}
    arms = load_rows(bundle, target)
    stats = json.loads((LABEL_DIR / f"{bundle}_stats.json").read_text())
    done["meta"] = {"bundle": bundle, "target": target, "Ts": TS,
                    "layers": LAYERS, "primary": "top-vs-bottom class",
                    "bundle_stats_labels": stats.get("labels"),
                    "bundle_triage": stats.get("triage")}
    out_path.write_text(json.dumps(done, indent=1))

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

                if f"{key0}/tok" not in done["cells"]:          # per-token FIRST
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
                    out_path.write_text(json.dumps(done, indent=1))
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
                    cell["flat"] = fit_probe(Wtr.reshape(n_tr, -1), ytr_t,
                                             Wte.reshape(n_te, -1), yte_t, 2,
                                             class_weight=True)
                    cell["mean"] = fit_probe(Wtr.float().mean(1).half(), ytr_t,
                                             Wte.float().mean(1).half(), yte_t,
                                             2, class_weight=True)
                    gs = torch.Generator().manual_seed(SHUFFLE_SEED)
                    Str = torch.stack([w[torch.randperm(T, generator=gs)]
                                       for w in Wtr])
                    Ste = torch.stack([w[torch.randperm(T, generator=gs)]
                                       for w in Wte])
                    cell["shuf"] = fit_probe(Str.reshape(n_tr, -1), ytr_t,
                                             Ste.reshape(n_te, -1), yte_t, 2,
                                             class_weight=True)
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
                    cell["shuffle_gap"] = cell["flat"]["auc"] - cell["shuf"]["auc"]
                    done["cells"][key] = cell
                    print(f"[{key}] flat={cell['flat']['auc']:.3f} "
                          f"mean={cell['mean']['auc']:.3f} "
                          f"g={cell['g']:+.3f} g_agg={cell['g_agg']:+.3f} "
                          f"g_ord={cell['g_order']:+.3f} "
                          f"({time.time()-t0:.0f}s)", flush=True)
                    out_path.write_text(json.dumps(done, indent=1))
                    del Wtr, Wte, Str, Ste
            del acts_t
    print(f"-> {out_path}", flush=True)


def main():
    bundle = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else ""
    target = "" if target in ("-", "main") else target
    out_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else \
        Path(__file__).resolve().parent / bundle
    run(bundle, target, out_dir)


if __name__ == "__main__":
    main()
