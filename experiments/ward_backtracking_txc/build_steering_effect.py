"""Baseline-corrected steering effect analysis.

The mag=0 row is the resampling-noise baseline (steering hook is a no-op
at magnitude 0). At our cohort it independently rescues 7/31 truly-wrong
questions and regresses 1/30 correct questions for ALL six architectures
identically — that's pure cut-and-continue noise, not a steering effect.

This script produces TWO baseline-corrected views:

1. **Δ-rescues** = net_rescues(mag) − net_rescues(mag=0). The "additional
   rescues caused by actually applying steering, on top of the noise floor."

2. **Steering-induced flips** — paired per (arch, question_id) outcomes:
   for each question, did its post-steering correctness differ from its
   mag=0 outcome? McNemar-paired test on this 2×2 isolates the steering
   effect from the baseline.

Outputs:
  - steering_effect.parquet — long-form rows
    (arch, magnitude, question_id, before, after_baseline, after_steered,
     steering_changed_outcome, change_direction)
  - steering_effect_summary.csv — per (arch, magnitude) net Δ + paired counts
  - steering_mcnemar_paired.csv — paired McNemar at per-arch best magnitude
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.steering_effect")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--flip-matrix", type=Path, required=True,
                   help="path to flip_matrix.parquet built from b3 sweep")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory")
    p.add_argument("--baseline-mag", type=float, default=0.0,
                   help="magnitude that defines the no-steering baseline (default 0.0)")
    args = p.parse_args(argv)

    df = pd.read_parquet(args.flip_matrix)
    log.info("[in] %d rows; archs=%s; mags=%s",
             len(df), sorted(df["arch"].unique()),
             sorted(df["magnitude"].unique()))

    # Pull the baseline mag=0 row per (arch, question)
    base = df[df["magnitude"] == args.baseline_mag][
        ["arch", "question_id", "before_correct", "after_correct"]
    ].rename(columns={"after_correct": "after_baseline"})
    log.info("[baseline] %d (arch, question) baseline rows at mag=%.1f",
             len(base), args.baseline_mag)

    # Join to all magnitudes
    merged = df.merge(
        base[["arch", "question_id", "after_baseline"]],
        on=["arch", "question_id"],
        how="inner",
    )
    merged = merged.rename(columns={"after_correct": "after_steered"})
    merged["steering_changed_outcome"] = (
        merged["after_baseline"] != merged["after_steered"]
    )
    # Direction: "+" = baseline-incorrect → steered-correct (extra rescue),
    # "-" = baseline-correct → steered-incorrect (steering broke it),
    # "0" = no change.
    def direction(row):
        if row["after_baseline"] and not row["after_steered"]:
            return "broke"
        if not row["after_baseline"] and row["after_steered"]:
            return "extra_rescue"
        return "no_change"
    merged["change_direction"] = merged.apply(direction, axis=1)

    args.out.mkdir(parents=True, exist_ok=True)
    parquet_path = args.out / "steering_effect.parquet"
    merged.to_parquet(parquet_path, compression="snappy")
    log.info("[saved] %s (%d rows)", parquet_path, len(merged))

    # Per (arch, mag) summary: net Δ rescues = extra_rescue − broke
    rows = []
    for (arch, mag), g in merged.groupby(["arch", "magnitude"]):
        n_extra_rescue = int((g["change_direction"] == "extra_rescue").sum())
        n_broke = int((g["change_direction"] == "broke").sum())
        n_no_change = int((g["change_direction"] == "no_change").sum())
        delta_net = n_extra_rescue - n_broke
        rows.append({
            "arch": arch, "magnitude": mag,
            "n_extra_rescue": n_extra_rescue,
            "n_broke_by_steering": n_broke,
            "n_no_change_vs_baseline": n_no_change,
            "delta_net_vs_baseline": delta_net,
            "n_total": len(g),
        })
    summary = pd.DataFrame(rows).sort_values(["arch", "magnitude"])
    summary_path = args.out / "steering_effect_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("[saved] %s", summary_path)

    # Per-arch best magnitude by Δnet vs baseline
    bests = {}
    for arch, g in summary.groupby("arch"):
        best = g.loc[g["delta_net_vs_baseline"].idxmax()]
        bests[arch] = float(best["magnitude"])

    # Paired McNemar at peak: cell counts on the 2×2 of
    # (baseline_correct, steered_correct) per question. The discordant
    # cells are extra_rescue (n_ic_paired) and broke (n_ci_paired).
    mcnemar_rows = []
    for arch, peak_mag in bests.items():
        sub = merged[(merged["arch"] == arch) & (merged["magnitude"] == peak_mag)]
        n_extra = int((sub["change_direction"] == "extra_rescue").sum())
        n_broke = int((sub["change_direction"] == "broke").sum())
        n_disc = n_extra + n_broke
        # McNemar with continuity correction
        chi2_cc = (abs(n_extra - n_broke) - 1) ** 2 / n_disc if n_disc > 0 else 0.0
        # Exact two-sided binomial on the discordant cells
        if n_disc > 0:
            bt = binomtest(min(n_extra, n_broke), n=n_disc, p=0.5,
                           alternative="two-sided")
            p_exact = float(bt.pvalue)
        else:
            p_exact = 1.0
        mcnemar_rows.append({
            "arch": arch, "magnitude": peak_mag,
            "n_extra_rescue": n_extra,
            "n_broke_by_steering": n_broke,
            "n_disc": n_disc,
            "delta_net_vs_baseline": n_extra - n_broke,
            "mcnemar_chi2_cc": float(chi2_cc),
            "mcnemar_p_2sided_exact": p_exact,
        })
    mc = pd.DataFrame(mcnemar_rows).sort_values("arch")
    mc_path = args.out / "steering_mcnemar_paired.csv"
    mc.to_csv(mc_path, index=False)
    log.info("[saved] %s", mc_path)
    for _, r in mc.iterrows():
        log.info("  arch=%-10s peak_mag=%+5.1f extra=%2d broke=%2d delta=%+d chi2=%.2f p=%.4f",
                 r["arch"], r["magnitude"], r["n_extra_rescue"],
                 r["n_broke_by_steering"], r["delta_net_vs_baseline"],
                 r["mcnemar_chi2_cc"], r["mcnemar_p_2sided_exact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
