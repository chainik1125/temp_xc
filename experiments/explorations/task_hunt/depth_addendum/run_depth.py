"""Early-layer addendum — g_order(ℓ) for lag4 + g_agg(ℓ) for slope8.

Executes `PREDICTIONS.md` (frozen in the same commit, BEFORE any cell).
Zero new data: cached activations, the frozen round-1 manifests, the
frozen `conversion_depth.problib` stack (no retuning), and the screens'
own gather/shuffle conventions (`replag.screen`, `confidence.screen`).
Screen-layer cells are RE-RUN on identical rows rather than copied, so
each model's depth curve is internally paired and the overlap cells
double as a reproduction check against the committed screen JSONs.

Readouts per cell (all acc_test, 3- or 4-class, matched rows):
  replag arm  : tok, win(T), mean(T), shuf(T) at T ∈ {4, 8}
                → g_order(ℓ, T) = win − mean; shuffle drop = win − shuf
  slope8 arm  : tok, mean64 → g_agg(ℓ) = mean64 − tok

Incremental/resumable (one JSON, saved after every cell). Off-grid,
off-leaderboard: this is a raw-activation diagnostic, not a panel.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.depth_addendum.run_depth
"""

from __future__ import annotations

import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from experiments.explorations.conversion_depth.problib import fit_probe
from experiments.explorations.task_hunt.confidence.screen import (
    build_rows,
    gather_tok as conf_gather_tok,
)
from experiments.explorations.task_hunt.replag.build_labels import (
    CACHE_ROOT as REPLAG_CACHE,
    LABELS_DIR,
    MODELS,
)
from experiments.explorations.task_hunt.replag.cache_acts import HS_CAPTURE
from experiments.explorations.task_hunt.replag.screen import (
    SHUF_SEED,
    gather_tok,
    gather_win,
    shuffle_context,
    summarize,
    win_mean,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = RES / "depth.json"

WARD_CACHE = Path("/workspace/conv_depth_caches")
WARD_HS = [0] + list(range(1, 32, 2))          # the 17 capture points
REPLAG_TS = (4, 8)
SLOPE_T = 64
NULL_SEED = 99


def _load(path):
    if OUT.exists():
        return json.loads(OUT.read_text())
    return {"meta": {"predictions": "PREDICTIONS.md (frozen pre-run)",
                     "replag_ts": list(REPLAG_TS), "slope_t": SLOPE_T,
                     "ward_hs": WARD_HS,
                     "replag_hs": {k: sorted(v) for k, v in
                                   HS_CAPTURE.items()}},
            "cells": {}}


def main() -> None:
    RES.mkdir(exist_ok=True)
    done = _load(OUT)
    cells = done["cells"]

    def save():
        OUT.write_text(json.dumps(done, indent=1))

    def run(cell_key, fn):
        if cell_key in cells:
            return
        t0 = time.time()
        cells[cell_key] = fn()
        cells[cell_key]["wall_s"] = round(time.time() - t0, 1)
        print(f"[depth {cell_key}] "
              + " ".join(f"{k}={v:.3f}" for k, v in cells[cell_key].items()
                         if isinstance(v, float) and k != "wall_s"),
              flush=True)
        save()

    # ---------------- replag arm: lag4 across cached depths ----------------
    for key in ("gpt2", "gemma2_2b", "llama31_8b"):
        stats = json.loads(
            (LABELS_DIR / f"replag_{key}_stats.json").read_text())
        if not (stats["tasks"]["lag4/train"]["ok"]
                and stats["tasks"]["lag4/test"]["ok"]):
            print(f"[depth {key}/lag4] SKIP (insufficient matched rows)")
            continue
        z = np.load(LABELS_DIR / f"replag_{key}_manifests.npz")
        rtr, ytr = z["lag4_train_rows"], z["lag4_train_y"]
        rte, yte = z["lag4_test_rows"], z["lag4_test_y"]
        ytr_t = torch.from_numpy(ytr.astype(np.int64))
        yte_t = torch.from_numpy(yte.astype(np.int64))
        for hs in sorted(HS_CAPTURE[key]):
            acts = torch.from_numpy(np.ascontiguousarray(
                np.load(REPLAG_CACHE / key / f"hs{hs}.npy", mmap_mode="r")))
            tag = f"replag/{key}/hs{hs}"
            Xtr, Xte = gather_tok(acts, rtr), gather_tok(acts, rte)
            run(f"{tag}/tok_linear", lambda: summarize(
                fit_probe(Xtr, ytr_t, Xte, yte_t, 4), 4))
            # pre-registered null pair at the new-layer calibration cell
            if key == "gpt2" and hs == 4:
                rngn = np.random.default_rng(NULL_SEED)
                yn = torch.from_numpy(rngn.permutation(ytr))
                run(f"{tag}/T4/null_win_linear", lambda: summarize(
                    fit_probe(gather_win(acts, rtr, 4).reshape(len(rtr), -1),
                              yn,
                              gather_win(acts, rte, 4).reshape(len(rte), -1),
                              yte_t, 4), 4))
            for T in REPLAG_TS:
                Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
                run(f"{tag}/T{T}/win_linear", lambda: summarize(
                    fit_probe(Wtr.reshape(len(rtr), -1), ytr_t,
                              Wte.reshape(len(rte), -1), yte_t, 4), 4))
                run(f"{tag}/T{T}/win_mean_linear", lambda: summarize(
                    fit_probe(win_mean(Wtr), ytr_t, win_mean(Wte), yte_t,
                              4), 4))
                srng = np.random.default_rng(
                    SHUF_SEED
                    + zlib.crc32(f"lag4/T{T}".encode()) % 2 ** 16)
                Str = shuffle_context(Wtr, srng).reshape(len(rtr), -1)
                Ste = shuffle_context(Wte, srng).reshape(len(rte), -1)
                run(f"{tag}/T{T}/win_shuf_linear", lambda: summarize(
                    fit_probe(Str, ytr_t, Ste, yte_t, 4), 4))
                del Wtr, Wte
            del acts, Xtr, Xte

    # ---------------- slope8 arm: g_agg across the 17 Ward depths ----------
    manifests, mstats = build_rows()
    rtr, ytr = manifests[("slope8", "train")]
    rte, yte = manifests[("slope8", "test")]
    ytr_t = torch.from_numpy(ytr)
    yte_t = torch.from_numpy(yte)
    done["meta"]["slope8_rows"] = {
        "train": mstats["slope8/train"], "test": mstats["slope8/test"]}
    save()

    def mean_win(acts, rows, T):
        """Trailing-T window mean without materializing (n, T, d)."""
        w = torch.from_numpy(rows[:, 0])
        p = torch.from_numpy(rows[:, 1])
        acc = torch.zeros((len(rows), acts.shape[-1]), dtype=torch.float32)
        for j in range(T):
            acc += acts[w, p - (T - 1) + j].float()
        return (acc / T).to(torch.float16)

    for reader in ("distill", "base"):
        for hs in WARD_HS:
            acts = torch.from_numpy(np.ascontiguousarray(
                np.load(WARD_CACHE / reader / f"hs{hs}.npy", mmap_mode="r")))
            tag = f"slope8/{reader}/hs{hs}"
            Xtr, Xte = conf_gather_tok(acts, rtr), conf_gather_tok(acts, rte)
            run(f"{tag}/tok_linear", lambda: summarize(
                fit_probe(Xtr, ytr_t, Xte, yte_t, 3), 3))
            run(f"{tag}/T{SLOPE_T}/win_mean_linear", lambda: summarize(
                fit_probe(mean_win(acts, rtr, SLOPE_T), ytr_t,
                          mean_win(acts, rte, SLOPE_T), yte_t, 3), 3))
            if reader == "distill" and hs == 1:
                rngn = np.random.default_rng(NULL_SEED)
                yn = torch.from_numpy(rngn.permutation(ytr))
                run(f"{tag}/T{SLOPE_T}/null_win_mean_linear", lambda:
                    summarize(fit_probe(mean_win(acts, rtr, SLOPE_T), yn,
                                        mean_win(acts, rte, SLOPE_T),
                                        yte_t, 3), 4))
            del acts, Xtr, Xte

    print(f"[depth] DONE — {len(cells)} cells -> {OUT}")


if __name__ == "__main__":
    main()
