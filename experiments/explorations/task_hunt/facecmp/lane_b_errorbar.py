"""LANE B — give the screen an error bar. Does NOT move the bar.

The hub's finding (`0e99dea91`): every screen cell is a single-seed point
estimate compared against a hardcoded `gain >= +0.05`, with no seed, CI,
bootstrap or std anywhere, and per-example predictions unsaved so paired
SEs cannot be recovered retroactively. **Confirmed at source:**
`problib.fit_probe(..., seed: int = 0)` and its return dict carries
`acc_train/acc_test/per_class/n_train/n_test` — **no predictions.** So the
whole record ran at seed 0 and the paired SE is genuinely unrecoverable.

## Two variances, and the record has measured neither

  SAMPLING  finite test set. Estimable in closed form, and the hub did:
            SE(acc) ~ 0.0074 at n=4497, p~0.43.
  TRAINING  probe init + optimisation noise across seeds. **Not estimable
            in closed form, never measured, and it does not shrink with
            n_test.** If it dominates, every closed-form sigma in the
            hub's table is an underestimate of the real uncertainty.

This measures BOTH, and reports the gain's total uncertainty.

## One correction to the hub's sigma table, offered before the numbers

Its shortfall column uses an illustrative pairing rho=0.5. Rho is a free
parameter there and the conclusion moves with it:

    rho   0.0    0.3    0.5    0.7    0.85   0.95
    sigma 0.38   0.46   0.54   0.70   0.99   1.71   <- evalage gemma, 0.0040 short

The window arm and the tok arm predict the SAME labels on the SAME rows
from related features, so their errors should be strongly correlated —
rho is plausibly high, which makes evalage's shortfall **larger** than
0.54 sigma, not smaller. **That cuts against the rescue case**, and it is
why this script measures rho from the actual predictions instead of
assuming it.

## Method

For each cell, refit at N seeds keeping per-example predictions, then:
  - seed spread: mean/std/min/max of acc across seeds (TRAINING variance)
  - paired bootstrap over test examples on the seed-averaged gain
    (SAMPLING variance, correctly paired -- no rho assumption)
  - the empirical pairing rho, so the hub's table can be re-read

`_fit_with_preds` REPLICATES `fit_probe` and is **asserted equal to it**
on acc_test for the same seed before any result is used. If the assert
fires, the replication has drifted and the numbers are void.

**Nothing here changes the +0.05 bar.** Per the brief: if the CI work
argues the bar should move, that is the hub's and Han's call, and a
threshold that moves after seeing the data is not a threshold.

Run: PYTHONPATH=. python -m experiments.explorations.task_hunt.facecmp.lane_b_errorbar
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.explorations.conversion_depth.problib import (
    DEVICE,
    EPOCHS,
    LR,
    WD,
    _standardize,
    fit_probe,
)

SEEDS = [0, 1, 2, 42, 7]
N_BOOT = 2000
BOOT_SEED = 20260728
CHANCE = 1.0 / 3.0


def _fit_with_preds(ftr, ytr, fte, yte, n_classes, hidden=0, seed=0):
    """Byte-for-byte the same procedure as problib.fit_probe, but also
    returns per-example test predictions. Verified against fit_probe."""
    torch.manual_seed(seed)
    ftr = ftr.to(DEVICE).float()
    fte = fte.to(DEVICE).float()
    ytr = ytr.to(DEVICE).long()
    yte = yte.to(DEVICE).long()
    ftr, fte = _standardize(ftr, fte)
    D = ftr.shape[1]
    probe = (nn.Sequential(nn.Linear(D, hidden), nn.ReLU(),
                           nn.Linear(hidden, n_classes)).to(DEVICE)
             if hidden else nn.Linear(D, n_classes).to(DEVICE))
    opt = torch.optim.Adam(probe.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        loss = F.cross_entropy(probe(ftr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = probe(fte).argmax(-1)
        acc = (pred == yte).float().mean().item()
    return acc, (pred == yte).cpu().numpy().astype(np.float64)


def paired_bootstrap(corr_a, corr_b, n_boot=N_BOOT, seed=BOOT_SEED):
    """CI on (mean(a) - mean(b)) resampling TEST EXAMPLES, keeping the
    pairing. Makes no assumption about rho."""
    rng = np.random.default_rng(seed)
    n = len(corr_a)
    d = corr_a - corr_b
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = d[idx].mean(axis=1)
    return {"gain": float(d.mean()),
            "se_boot": float(boots.std(ddof=1)),
            "ci95": [float(np.percentile(boots, 2.5)),
                     float(np.percentile(boots, 97.5))],
            "p_gain_below_0.05": float((boots < 0.05).mean())}


def main():
    import experiments.explorations.task_hunt.facecmp.arm_test as at
    import experiments.explorations.task_hunt.facecmp.face_battery as fb

    root = os.environ.get("FACECMP_CACHE_ROOT")
    if root:
        at.CACHE_ROOT = Path(root)
    key = os.environ.get("FACECMP_MODEL", "gemma2_2b")
    tag = os.environ.get("FACECMP_TAG", "gemma2_512")
    Ts = [int(x) for x in os.environ.get("FACECMP_TS", "16,64").split(",")]

    at.FACE, at.H = f"LB_{tag}", 64
    at.rate_face = fb.f_age
    manifests, mstats, fl = at.build_rows(key)
    from experiments.explorations.task_hunt.replag.cache_acts import SCREEN_HS
    hs = SCREEN_HS[key]
    acts = torch.from_numpy(np.ascontiguousarray(
        np.load(at.CACHE_ROOT / key / f"hs{hs}.npy", mmap_mode="r")))
    F_ = at.FACE
    rtr, ytr = manifests[(F_, "train")]
    rte, yte = manifests[(F_, "test")]
    ytr_t, yte_t = torch.from_numpy(ytr), torch.from_numpy(yte)

    from experiments.explorations.task_hunt.replag.screen import gather_tok, gather_win
    from experiments.explorations.task_hunt.dialevel.capacity_check import anchor_ctxmean

    feats = {"tok": (gather_tok(acts, rtr), gather_tok(acts, rte))}
    for T in Ts:
        Wtr, Wte = gather_win(acts, rtr, T), gather_win(acts, rte, T)
        feats[f"T{T}"] = (anchor_ctxmean(Wtr), anchor_ctxmean(Wte))
        del Wtr, Wte

    # ---- replication guard: my fit must equal problib's -----------------
    a_ref = fit_probe(feats["tok"][0], ytr_t, feats["tok"][1], yte_t, 3,
                      hidden=512)["acc_test"]
    a_mine, _ = _fit_with_preds(feats["tok"][0], ytr_t, feats["tok"][1],
                                yte_t, 3, hidden=512, seed=0)
    assert abs(a_ref - a_mine) < 1e-9, (
        f"REPLICATION DRIFT: fit_probe {a_ref} vs mine {a_mine} — results void")
    print(f"replication guard OK (fit_probe {a_ref:.6f} == mine {a_mine:.6f})")

    out = {"model": key, "tag": tag, "seeds": SEEDS, "n_boot": N_BOOT,
           "n_test": int(len(yte)), "cells": {}}
    corr = {}
    for name, (Xtr, Xte) in feats.items():
        accs, cs = [], []
        for s in SEEDS:
            a, c = _fit_with_preds(Xtr, ytr_t, Xte, yte_t, 3, hidden=512, seed=s)
            accs.append(a)
            cs.append(c)
        corr[name] = np.mean(cs, axis=0)
        out["cells"][name] = {
            "acc_by_seed": accs, "mean": float(np.mean(accs)),
            "std_across_seeds": float(np.std(accs, ddof=1)),
            "min": float(min(accs)), "max": float(max(accs)),
            "range": float(max(accs) - min(accs))}
        print(f"  {name:<6} mean={np.mean(accs):.4f} "
              f"SD_seed={np.std(accs, ddof=1):.4f} range={max(accs)-min(accs):.4f}")

    print(f"\n{'cell':<8}{'gain':>9}{'SE_boot':>10}{'95% CI':>20}"
          f"{'SD_seed(gain)':>15}{'rho':>7}{'P(gain<.05)':>13}")
    print("-" * 84)
    for T in Ts:
        n = f"T{T}"
        bs = paired_bootstrap(corr[n], corr["tok"])
        gseeds = [out["cells"][n]["acc_by_seed"][i]
                  - out["cells"]["tok"]["acc_by_seed"][i]
                  for i in range(len(SEEDS))]
        rho = float(np.corrcoef(corr[n], corr["tok"])[0, 1])
        bs.update({"sd_seed_gain": float(np.std(gseeds, ddof=1)),
                   "gain_by_seed": gseeds, "pairing_rho": rho})
        out["cells"][n]["gain_stats"] = bs
        print(f"{n:<8}{bs['gain']:>+9.4f}{bs['se_boot']:>10.4f}"
              f"  [{bs['ci95'][0]:+.4f},{bs['ci95'][1]:+.4f}]"
              f"{bs['sd_seed_gain']:>15.4f}{rho:>7.3f}"
              f"{bs['p_gain_below_0.05']:>13.3f}")

    d = Path(__file__).resolve().parent / "results" / "lane_b"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"errorbar_{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {d / f'errorbar_{tag}.json'}")
    print("NOTE: the +0.05 bar is NOT changed here. Any argument to move it "
          "goes to the hub/Han with evidence.")


if __name__ == "__main__":
    main()
