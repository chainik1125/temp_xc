"""tsae-arm top-up bounds — the R4/R5 update numbers, recomputed from canon.

Mirrors `../receipts_check.py` R4/R5 machinery EXACTLY (same `_lb_cells`
selection from `results/leaderboard.jsonl`, same t/Welch arithmetic) with
the tsae seed population extended by the top-up seeds {3,4,5}. Prints and
writes `results/topup_bounds_tsae.json`:

- per-cell t-CIs at the updated n (pre/T4, pre/T8 at n=6; tsae/T1 at
  round-1 n=3, new-only n=3, pooled n=6);
- b's criterion (one-sided 95% LB > 0 on the pre-vs-tsae T8 margin),
  evaluated three ways: PAIRED on all shared seeds, WELCH pre(6) vs
  tsae(pooled 6), and — because the pooling carries a disclosed
  cross-cache caveat (LOG 2026-07-26 mac-a) — WELCH pre(6) vs tsae
  new-seeds-ONLY (3), reported separately, never silently pooled.

Run:  .venv/bin/python -m \
  experiments.explorations.task_hunt.lambda_intensity.topup_bounds_tsae
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
DS = "ward_real_lambda_base_l12"
ROUND1 = (1, 2, 42)
NEW = (3, 4, 5)


def _lb_cells(arch, T, kinds=("trained",), k_pos=8, seeds=None):
    vals = {}
    with LEADERBOARD.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != DS or r["arch"] != arch:
                continue
            tc = r["training_cfg"]
            ov = tc["arch_hparams_override"]
            if ov["T"] != T or ov.get("k_pos") != k_pos:
                continue
            kind = "untrained" if tc["n_steps"] == 0 else "trained"
            if kind not in kinds:
                continue
            if seeds is not None and r["seed"] not in seeds:
                continue
            vals[r["seed"]] = r["metrics"]["lambda_recovery"]
    return vals


def _tci(v):
    v = np.asarray(v, float)
    n = len(v)
    se = v.std(ddof=1) / np.sqrt(n)
    lo, hi = stats.t.interval(0.95, n - 1, loc=v.mean(), scale=se)
    return {"n": n, "mean": float(v.mean()), "t_ci95": [float(lo), float(hi)],
            "sd": float(v.std(ddof=1))}


def _one_sided_lb95(v):
    v = np.asarray(v, float)
    n = len(v)
    return float(v.mean() - stats.t.ppf(0.95, n - 1) * v.std(ddof=1) / np.sqrt(n))


def _welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1)
                    + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    diff = float(a.mean() - b.mean())
    return {"diff": diff, "lb95_one_sided": diff - float(stats.t.ppf(0.95, df)) * float(se),
            "p_one_sided": float(stats.t.sf(diff / se, df)), "df": float(df),
            "n_a": len(a), "n_b": len(b)}


def main():
    pre8 = _lb_cells("txc_batchtopk_pre", 8, seeds=set(ROUND1) | set(NEW))
    pre4 = _lb_cells("txc_batchtopk_pre", 4, seeds=set(ROUND1) | set(NEW))
    ts_r1 = _lb_cells("tsae", 1, seeds=set(ROUND1))
    ts_new = _lb_cells("tsae", 1, seeds=set(NEW))
    ts_all = {**ts_r1, **ts_new}

    out = {
        "per_seed": {"pre_T8": pre8, "pre_T4": pre4,
                     "tsae_round1": ts_r1, "tsae_new": ts_new},
        "cells": {
            "pre_T4_n6": _tci(list(pre4.values())),
            "pre_T8_n6": _tci(list(pre8.values())),
            "tsae_T1_round1_n3": _tci(list(ts_r1.values())),
            "tsae_T1_new_n3": _tci(list(ts_new.values())) if len(ts_new) >= 2 else None,
            "tsae_T1_pooled_n6": _tci(list(ts_all.values())) if len(ts_all) >= 2 else None,
        },
    }

    # b's criterion: one-sided 95% LB > 0 on the pre-vs-tsae T8 margin.
    verdicts = {}
    shared_all = sorted(set(pre8) & set(ts_all), key=str)
    if len(shared_all) >= 2:
        d = np.array([pre8[s] for s in shared_all]) - \
            np.array([ts_all[s] for s in shared_all])
        verdicts["paired_pooled"] = {
            "seeds": shared_all, "diff": float(d.mean()),
            "lb95_one_sided": _one_sided_lb95(d),
            "all_seeds_positive": bool((d > 0).all())}
    if len(ts_all) >= 2:
        verdicts["welch_pre6_vs_tsae_pooled"] = _welch(
            list(pre8.values()), list(ts_all.values()))
    if len(ts_new) >= 2:
        verdicts["welch_pre6_vs_tsae_new_only"] = _welch(
            list(pre8.values()), list(ts_new.values()))
        shared_new = sorted(set(pre8) & set(ts_new))
        d_new = np.array([pre8[s] for s in shared_new]) - \
            np.array([ts_new[s] for s in shared_new])
        verdicts["paired_new_only"] = {
            "seeds": shared_new, "diff": float(d_new.mean()),
            "lb95_one_sided": _one_sided_lb95(d_new)}
    # POST-HOC robustness (chosen AFTER seeing the data, labeled as such):
    # new seeds 3 and 4 realized l0/token = 3.59 / 3.12, well UNDER the
    # round-1 realized band (6.52–7.20; s5 = 7.08 is in-band). An
    # under-spent tsae comparator plausibly INFLATES the pre−tsae margin,
    # so the exclusion goes AGAINST the headline: recompute with the two
    # under-band cells dropped (tsae = seeds {1,2,42,5}).
    inband = {s: v for s, v in ts_all.items() if s not in (3, 4)}
    if len(inband) >= 2:
        verdicts["welch_pre6_vs_tsae_excl_underband_POSTHOC"] = _welch(
            list(pre8.values()), list(inband.values()))
        sh = sorted(set(pre8) & set(inband), key=str)
        d_ib = np.array([pre8[s] for s in sh]) - np.array([inband[s] for s in sh])
        verdicts["paired_excl_underband_POSTHOC"] = {
            "seeds": sh, "diff": float(d_ib.mean()),
            "lb95_one_sided": _one_sided_lb95(d_ib)}

    for k, v in verdicts.items():
        lb = v["lb95_one_sided"]
        v["criterion_bounded"] = bool(lb > 0)
    out["verdicts"] = verdicts
    out["realized_l0_note"] = (
        "round-1 tsae realized l0/token 6.52–7.20; new seeds: s3=3.59, "
        "s4=3.12 (UNDER band), s5=7.08 (in-band). Under-band cells "
        "disclosed as residual mismatches; POSTHOC variants above drop them.")

    dst = HERE / "results" / "topup_bounds_tsae.json"
    dst.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\n-> {dst}")


if __name__ == "__main__":
    main()
