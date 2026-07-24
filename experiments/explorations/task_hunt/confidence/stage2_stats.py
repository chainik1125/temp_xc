"""Variance receipts for the hedging-LEVEL Stage-2 panel (card § 8).

The card's KEEP rule is variance-aware by construction ("beyond the
combined seed spread", "mean ± sd over 3 seeds", n = 3 — the λ̂ review's
binding note 2). This script derives every statistic the record is
allowed to quote, from the panel rows only, using the SHARED small-n
lib (`support_stats.stats_lib`, runpod-b): exact within-seed trend
permutation for the T-rise, paired t CIs + exact sign-flip for
cross-arch margins, and the trained−untrained margins.

Emits `results/stage2_stats.json` + a printed summary. Off-leaderboard
(a reading of committed rows, not a new result).

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.confidence.stage2_stats
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.explorations.task_hunt.support_stats.stats_lib import (
    seeds_for_bound,
    sign_flip_p,
    t_ci95,
    within_seed_trend,
)

HERE = Path(__file__).resolve().parent
DS = "ward_real_slope8_distill_l14"
SEEDS = (1, 2, 42)
METRIC = "lambda_recovery"
RISE_TS = (2, 4, 8)          # the pre-registered rise range (card P5)
WINDOW_ARCHS = ("txc_batchtopk_pre", "txc_batchtopk_post",
                "stacked_batchtopk")
TOKEN_ARCHS = ("batchtopk_sae", "tsae")


def main() -> None:
    rows = json.loads(
        (HERE / "results" / f"stage2_{DS}.json").read_text())
    trained: dict = defaultdict(dict)      # (arch, T) -> {seed: r}
    untrained: dict = defaultdict(dict)
    for r in rows:
        if not r.get("ok"):
            continue
        d = untrained if r.get("kind") == "untrained" else trained
        d[(r["arch"], r["T"])][r["seed"]] = r["metrics"].get(METRIC)

    def vec(arch, T, d=trained):
        return [d[(arch, T)][s] for s in SEEDS] if all(
            s in d.get((arch, T), {}) for s in SEEDS) else None

    out: dict = {"meta": {"ds": DS, "seeds": list(SEEDS), "metric": METRIC,
                          "rise_ts": list(RISE_TS),
                          "n_cells_ok": sum(1 for r in rows if r.get("ok")),
                          "off_leaderboard": True}}

    # 1) T-rise per window arch (exact within-seed permutation over RISE_TS)
    out["t_rise"] = {}
    for arch in WINDOW_ARCHS:
        vals = [vec(arch, T) for T in RISE_TS]
        if any(v is None for v in vals):
            continue
        m = np.array(vals).T                      # (n_seeds, n_T)
        slope_sum, per_seed, p, n_perms = within_seed_trend(m, RISE_TS)
        out["t_rise"][arch] = {
            "Ts": list(RISE_TS), "per_seed_slopes": per_seed,
            "slope_sum": slope_sum, "p_exact": p, "n_perms": n_perms,
            "cell_means": {f"T{T}": float(np.mean(v))
                           for T, v in zip(RISE_TS, vals)}}

    # 2) trained − untrained margins (paired by seed)
    out["margins_trained_minus_untrained"] = {}
    for (arch, T) in sorted(trained):
        tv, uv = vec(arch, T), vec(arch, T, untrained)
        if tv is None or uv is None:
            continue
        d = np.array(tv) - np.array(uv)
        mean, lo, hi = t_ci95(d)
        out["margins_trained_minus_untrained"][f"{arch}/T{T}"] = {
            "mean": mean, "ci95": [lo, hi], "per_seed": d.tolist()}

    # 3) cross-arch margins vs each token arch (paired by seed + sign-flip)
    out["window_minus_token"] = {}
    for arch in WINDOW_ARCHS:
        for T in sorted({t for (a, t) in trained if a == arch}):
            wv = vec(arch, T)
            if wv is None:
                continue
            for tok in TOKEN_ARCHS:
                bv = vec(tok, 1)
                if bv is None:
                    continue
                d = np.array(wv) - np.array(bv)
                mean, lo, hi = t_ci95(d)
                p, npat = sign_flip_p(d)
                need = seeds_for_bound(float(d.mean()),
                                       float(d.std(ddof=1)))
                out["window_minus_token"][f"{arch}/T{T} - {tok}"] = {
                    "mean": mean, "ci95": [lo, hi],
                    "sign_flip_p": p, "n_patterns": npat,
                    "all_seeds_positive": bool((d > 0).all()),
                    "bounded_at_n3": bool(lo > 0),
                    "seeds_needed_to_bound": need,
                    "per_seed": d.tolist()}

    dst = HERE / "results" / "stage2_stats.json"
    dst.write_text(json.dumps(out, indent=2))

    print("=== T-rise (exact within-seed permutation, T = 2/4/8) ===")
    for a, v in out["t_rise"].items():
        print(f"  {a:22s} slopes {['%+.3f' % s for s in v['per_seed_slopes']]}"
              f"  p={v['p_exact']:.4f}  means "
              + " ".join(f"{k}={x:.3f}" for k, x in v["cell_means"].items()))
    print("=== window − token margins (paired, n = 3) ===")
    for k, v in out["window_minus_token"].items():
        flag = "BOUNDED" if v["bounded_at_n3"] else "not bounded"
        print(f"  {k:46s} {v['mean']:+.3f} CI [{v['ci95'][0]:+.3f}, "
              f"{v['ci95'][1]:+.3f}] {flag}  signflip p={v['sign_flip_p']:.3f}")
    print("wrote", dst)


if __name__ == "__main__":
    main()
