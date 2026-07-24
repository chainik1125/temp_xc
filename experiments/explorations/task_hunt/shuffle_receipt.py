"""The order-sensitivity receipt for the EXISTING backtracking case study.

`briefings/task-hunt.md` § "Also wanted": on the Ward caches, per-token
vs window vs SHUFFLED-window raw ceilings at L10, T = 16 — the
within-window shuffle control the paper's § 5.2 task never had.

Reuses the conversion-depth probe rows VERBATIM
(`conversion_depth/results/probe_rows.npz`, the frozen § 2 recipe:
ant_kw D+ 8–13, bt_freq negatives, by-trace 80/20) and the frozen
`problib` stack, so the per-token / window-flatten numbers reproduce
RECORD § 3 exactly and the only new quantity is the shuffled arm.
Adds the window-MEAN arm so the gap splits into g_agg / g_order.

Shuffle: an independent permutation of the T positions PER ROW
(seed 23), destroying within-window order while preserving the exact
multiset of position-activations — the § 3 order-0 control applied to
raw activations.

Layer L10 = hs11 (the paper's layer) primary; hs13 (L12) reported
alongside since RECORD § 3 found the residual g(ℓ) plateau flat.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.shuffle_receipt
Writes results/shuffle_receipt.json + figs/shuffle_receipt.{png,pdf}.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe

CACHE_ROOT = Path("/workspace/conv_depth_caches")
ROWS_NPZ = (Path(__file__).resolve().parents[2] / "explorations"
            / "conversion_depth" / "results" / "probe_rows.npz")
HERE = Path(__file__).resolve().parent
OUT = HERE / "results" / "shuffle_receipt.json"
FIGS = HERE / "figs"

T = 16
LAYERS = [11, 13]            # hs11 = resid_post L10 (the paper's layer)
MODELS = ["base", "distill"]
TARGETS = ["ant_kw", "ant_bts", "is_bt"]
SHUFFLE_SEED = 23
NULL_SEED = 99


def gather(acts_t, rows, T):
    w = torch.from_numpy(rows[:, 0])
    p = torch.from_numpy(rows[:, 1])
    X = torch.empty((len(rows), T, acts_t.shape[-1]), dtype=acts_t.dtype)
    for j in range(T):
        X[:, j] = acts_t[w, p - (T - 1) + j]
    return X


def main():
    z = np.load(ROWS_NPZ)
    done = json.loads(OUT.read_text()) if OUT.exists() else {"cells": {}}
    done["meta"] = {"T": T, "rows": str(ROWS_NPZ.name),
                    "recipe": "conversion_depth RECORD § 2 (frozen)",
                    "shuffle": "per-row permutation of T positions, seed 23"}
    for tag in MODELS:
        cache = CACHE_ROOT / tag
        if not (cache / "meta.json").exists():
            continue
        for hs in LAYERS:
            acts_t = torch.from_numpy(np.ascontiguousarray(
                np.load(cache / f"hs{hs}.npy", mmap_mode="r")))
            for tgt in TARGETS:
                key = f"{tag}/hs{hs}/{tgt}"
                if key in done["cells"]:
                    continue
                t0 = time.time()
                rtr, rte = z[f"{tgt}_train_rows"], z[f"{tgt}_test_rows"]
                ytr = torch.from_numpy(z[f"{tgt}_train_y"])
                yte = torch.from_numpy(z[f"{tgt}_test_y"])
                Wtr, Wte = gather(acts_t, rtr, T), gather(acts_t, rte, T)
                n_tr, n_te = len(rtr), len(rte)
                cell = {}
                cell["tok"] = fit_probe(Wtr[:, -1], ytr, Wte[:, -1], yte, 2,
                                        class_weight=True)
                cell["flat"] = fit_probe(Wtr.reshape(n_tr, -1), ytr,
                                         Wte.reshape(n_te, -1), yte, 2,
                                         class_weight=True)
                cell["mean"] = fit_probe(Wtr.float().mean(1).half(), ytr,
                                         Wte.float().mean(1).half(), yte, 2,
                                         class_weight=True)
                g = torch.Generator().manual_seed(SHUFFLE_SEED)
                Str = torch.stack([x[torch.randperm(T, generator=g)]
                                   for x in Wtr])
                Ste = torch.stack([x[torch.randperm(T, generator=g)]
                                   for x in Wte])
                cell["shuf"] = fit_probe(Str.reshape(n_tr, -1), ytr,
                                         Ste.reshape(n_te, -1), yte, 2,
                                         class_weight=True)
                gn = torch.Generator().manual_seed(NULL_SEED)
                cell["null_flat"] = fit_probe(
                    Wtr.reshape(n_tr, -1),
                    ytr[torch.randperm(n_tr, generator=gn)],
                    Wte.reshape(n_te, -1),
                    yte[torch.randperm(n_te, generator=gn)], 2,
                    class_weight=True)
                cell["g"] = cell["flat"]["auc"] - cell["tok"]["auc"]
                cell["g_agg"] = cell["mean"]["auc"] - cell["tok"]["auc"]
                cell["g_order"] = cell["flat"]["auc"] - cell["mean"]["auc"]
                cell["shuffle_gap"] = cell["flat"]["auc"] - cell["shuf"]["auc"]
                done["cells"][key] = cell
                print(f"[{key}] tok={cell['tok']['auc']:.3f} "
                      f"flat={cell['flat']['auc']:.3f} "
                      f"mean={cell['mean']['auc']:.3f} "
                      f"shuf={cell['shuf']['auc']:.3f} "
                      f"shuffle_gap={cell['shuffle_gap']:+.3f} "
                      f"({time.time()-t0:.0f}s)", flush=True)
                OUT.write_text(json.dumps(done, indent=1))
                del Wtr, Wte, Str, Ste
            del acts_t

    nulls = [abs(v["null_flat"]["auc"] - 0.5) for v in done["cells"].values()]
    done["sigma_null"] = float(np.std(nulls))
    done["3sigma_null"] = 3 * done["sigma_null"]
    OUT.write_text(json.dumps(done, indent=1))
    _plot(done)
    print(f"-> {OUT}")


def _plot(done):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIGS.mkdir(exist_ok=True)
    cells = done["cells"]
    keys = [k for k in cells if "/hs11/" in k]
    if not keys:
        return
    labels = [k.replace("/hs11/", " L10 ") for k in keys]
    arms = [("tok", "per-token", "#7f7f7f"), ("mean", "window MEAN", "#2ca02c"),
            ("shuf", "window SHUFFLED", "#ff7f0e"),
            ("flat", "window (ordered)", "#1f77b4")]
    x = np.arange(len(keys))
    w = 0.2
    fig, ax = plt.subplots(figsize=(1.9 * len(keys) + 3, 4.4))
    for i, (arm, lab, col) in enumerate(arms):
        ax.bar(x + (i - 1.5) * w, [cells[k][arm]["auc"] for k in keys],
               w, label=lab, color=col)
    ax.axhline(0.5, color="k", lw=0.8, ls=":")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8, rotation=15)
    ax.set_ylabel("test AUC"); ax.set_ylim(0.45, 1.0)
    ax.set_title(f"Backtracking case study — order-sensitivity receipt "
                 f"(T={T}, raw activations)", fontsize=11)
    ax.legend(fontsize=8); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"shuffle_receipt.{ext}",
                    dpi=140 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] -> {FIGS}/shuffle_receipt.*")


if __name__ == "__main__":
    main()
