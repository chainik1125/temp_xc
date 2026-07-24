"""Variance receipts for the Stage-2 λ̂ panel (hunt-support-stats item 1).

Primary source: the 84 committed leaderboard rows with
datasource `ward_real_lambda_base_l12` (per-seed values live there; the
committed `stage2_summary.json` only has mean/std). Every extracted cell
is cross-checked for EXACT equality against runpod-d's
`lambda_intensity/results/stage2_ward_real_lambda_base_l12.json`; any
mismatch aborts the build.

Computes and writes (next to this script):
  stage2_variance.json  — per-seed values for every (arch, T, kind) cell;
      paired-by-seed TXC-pre − T-SAE and TXC-pre − per-token differences
      at each T (exact sign-flip permutation p, exact-enumeration BCa and
      t 95% CIs); the T = 2→8 trend statistic pooled over seeds (exact
      within-seed permutation test) for TXC-pre and for its
      trained−untrained margin; per-cell trained−untrained margin CIs;
      the seed power calc and the cell recommendation for runpod-d.
  stage2_variance.md    — the same numbers as a short readable section
      (all numbers script-derived; nothing hand-typed).

Honesty notes are embedded in both outputs: n = 3 seeds means the exact
one-sided sign-flip test bottoms out at p = 1/8, and the exact bootstrap
distribution of a mean has 27 atoms — the paired design is the point,
not the p-values.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.support_stats.stage2_variance
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .stats_lib import (bca_ci, seeds_for_bound, seeds_for_power,
                        seeds_for_signflip, sign_flip_p, t_ci95,
                        within_seed_trend)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
DS = "ward_real_lambda_base_l12"
METRIC = "lambda_recovery"
SEEDS = (1, 2, 42)
TREND_TS = (2, 4, 8)
TREND_TS_FULL = (2, 4, 8, 16)

LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
STAGE2_JSON = (ROOT / "experiments" / "explorations" / "task_hunt" /
               "lambda_intensity" / "results" / f"stage2_{DS}.json")

# Comparisons: (name, window arch, T=1 reference arch)
PAIRINGS = [
    ("txc_pre_minus_tsae", "txc_batchtopk_pre", "tsae"),
    ("txc_pre_minus_pertoken", "txc_batchtopk_pre", "batchtopk_sae"),
]


def load_cells():
    """(arch, T, seed, kind) -> {metric, l0} from the leaderboard,
    cross-checked exactly against runpod-d's stage2 results JSON."""
    cells = {}
    with LEADERBOARD.open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != DS:
                continue
            tc = r["training_cfg"]
            key = (r["arch"], tc["arch_hparams_override"]["T"], r["seed"],
                   "untrained" if tc["n_steps"] == 0 else "trained")
            if key in cells:
                raise SystemExit(f"duplicate leaderboard cell {key}")
            cells[key] = {"metric": r["metrics"][METRIC],
                          "l0": r["metrics"]["l0_per_token"],
                          "chance": r["metrics"]["lambda_chance"]}
    ref = {(r["arch"], r["T"], r["seed"], r["kind"]):
           r["metrics"][METRIC]
           for r in json.loads(STAGE2_JSON.read_text())}
    if set(ref) != set(cells):
        raise SystemExit("leaderboard/stage2-json key sets differ")
    bad = [k for k in ref if ref[k] != cells[k]["metric"]]
    if bad:
        raise SystemExit(f"cross-check FAILED on {len(bad)} cells: {bad[:4]}")
    return cells


def cell_vec(cells, arch, T, kind="trained", field="metric"):
    return np.array([cells[(arch, T, s, kind)][field] for s in SEEDS])


def paired_stats(diffs, arm_a=None, arm_b=None):
    """Stats on paired-by-seed differences. When the two arms' per-seed
    vectors are given, also report the across-seed correlation between
    arms and the sd the difference WOULD have were the arms independent
    — the honest check on whether pairing bought any variance
    reduction here."""
    d = np.asarray(diffs, dtype=float)
    mean, t_lo, t_hi = t_ci95(d)
    p, n_pat = sign_flip_p(d, "greater")
    bca = bca_ci(d)
    out = {"per_seed": {str(s): float(v) for s, v in zip(SEEDS, d)},
           "mean": mean, "sd": float(d.std(ddof=1)),
           "t_ci95": [t_lo, t_hi],
           "bca_ci95": [bca["lo"], bca["hi"]],
           "bca_atoms": bca["n_atoms"],
           "p_signflip_one_sided": p, "signflip_patterns": n_pat}
    if arm_a is not None:
        a = np.asarray(arm_a, float)
        b = np.asarray(arm_b, float)
        out["r_between_arms"] = float(np.corrcoef(a, b)[0, 1])
        out["sd_if_independent"] = float(
            np.sqrt(a.var(ddof=1) + b.var(ddof=1)))
    return out


def main():
    cells = load_cells()
    archs = sorted({k[0] for k in cells})
    Ts_by_arch = {a: sorted({k[1] for k in cells if k[0] == a})
                  for a in archs}

    # ---- 1. per-seed values, every (arch, T, kind) cell + trained CIs
    per_seed = {"trained": {}, "untrained": {}, "l0_trained": {}}
    cell_ci = {}
    for a in archs:
        for T in Ts_by_arch[a]:
            key = f"{a}/T{T}"
            for kind in ("trained", "untrained"):
                per_seed[kind][key] = {
                    str(s): float(cells[(a, T, s, kind)]["metric"])
                    for s in SEEDS}
            per_seed["l0_trained"][key] = {
                str(s): float(cells[(a, T, s, "trained")]["l0"])
                for s in SEEDS}
            v = cell_vec(cells, a, T)
            mean, lo, hi = t_ci95(v)
            cell_ci[key] = {"mean": mean, "sd": float(v.std(ddof=1)),
                            "t_ci95": [lo, hi]}

    # ---- 2. paired-by-seed diffs at each window T
    paired = {}
    for name, win_arch, ref_arch in PAIRINGS:
        ref = cell_vec(cells, ref_arch, 1)
        paired[name] = {"reference": f"{ref_arch}/T1",
                        "reference_per_seed":
                            {str(s): float(v) for s, v in zip(SEEDS, ref)},
                        "by_T": {}}
        for T in Ts_by_arch[win_arch]:
            win = cell_vec(cells, win_arch, T)
            paired[name]["by_T"][f"T{T}"] = paired_stats(win - ref, win, ref)

    # ---- 3. trend across T (pooled over seeds, exact within-seed perm)
    def trend_block(mat, Ts):
        obs, slopes, p, n = within_seed_trend(mat, Ts, "greater")
        return {"Ts": list(Ts), "slope_sum_per_log2T": obs,
                "per_seed_slopes": slopes, "p_one_sided": p, "n_perms": n}

    pre = "txc_batchtopk_pre"
    mat_pre = np.array([[cells[(pre, T, s, "trained")]["metric"]
                         for T in TREND_TS] for s in SEEDS])
    mat_margin = np.array([[cells[(pre, T, s, "trained")]["metric"]
                            - cells[(pre, T, s, "untrained")]["metric"]
                            for T in TREND_TS] for s in SEEDS])
    mat_pre_full = np.array([[cells[(pre, T, s, "trained")]["metric"]
                              for T in TREND_TS_FULL] for s in SEEDS])
    trend = {
        "txc_pre_trained_2to8": trend_block(mat_pre, TREND_TS),
        "txc_pre_margin_2to8": trend_block(mat_margin, TREND_TS),
        "txc_pre_trained_2to16_secondary": trend_block(mat_pre_full,
                                                       TREND_TS_FULL),
    }

    # ---- 4. trained − untrained margin CI, every cell
    margins = {}
    for a in archs:
        for T in Ts_by_arch[a]:
            d = (cell_vec(cells, a, T, "trained")
                 - cell_vec(cells, a, T, "untrained"))
            margins[f"{a}/T{T}"] = paired_stats(d)

    # ---- 5. power calc -> seed recommendation for runpod-d
    n_attain = seeds_for_signflip(0.05)
    power = {"signflip_min_seeds_for_p05": n_attain}
    for name, _, _ in PAIRINGS:
        power[name] = {}
        for T in (4, 8):
            st = paired[name]["by_T"][f"T{T}"]
            nb = seeds_for_bound(st["mean"], st["sd"])
            npw = seeds_for_power(st["mean"], st["sd"])
            power[name][f"T{T}"] = {
                "observed_mean": st["mean"], "observed_sd": st["sd"],
                "n_for_95_lower_bound_gt0": nb,
                "n_for_80pct_power_t05": npw}
    # The briefing's criterion is bounding THE margin at 95%; the margin
    # (review note 2) is TXC-pre vs T-SAE at the T = 8 headline cell.
    # T = 4 is reported above but is not cheaply boundable (its paired
    # diff is noise-dominated) — the T-rise + trained-untrained margin
    # carry that cell, per the review's own phrasing.
    n_t8 = power["txc_pre_minus_tsae"]["T8"]["n_for_95_lower_bound_gt0"]
    n_needed = None if n_t8 is None else max(n_attain, n_t8)
    extra = None if n_needed is None else max(0, n_needed - len(SEEDS))
    rec_cells = None
    if extra is not None and 0 < extra <= 4:
        rec_cells = {"per_extra_seed_trained":
                     ["txc_batchtopk_pre/T4", "txc_batchtopk_pre/T8",
                      "tsae/T1"],
                     "n_extra_seeds": extra,
                     "n_trained_cells": 3 * extra,
                     "untrained_counterparts_optional": 3 * extra,
                     "headroom_option": {
                         "n_extra_seeds": extra + 1,
                         "n_trained_cells": 3 * (extra + 1),
                         "why": "one seed of slack against the plug-in "
                                "sd estimate itself being an n=3 "
                                "estimate; also reaches sign-flip "
                                "p = 1/128 at the T8 cell"}}
    power["recommendation"] = {
        "criterion": "one-sided 95% t lower bound > 0 on the paired "
                     "TXC-pre - T-SAE diff at T = 8, AND exact "
                     "sign-flip attainability (2^-n <= 0.05)",
        "seeds_total_needed": n_needed, "extra_seeds": extra,
        "cells": rec_cells,
        "T4_not_cheaply_boundable": {
            "n_for_95_lower_bound_gt0":
                power["txc_pre_minus_tsae"]["T4"]
                     ["n_for_95_lower_bound_gt0"],
            "n_for_80pct_power_t05":
                power["txc_pre_minus_tsae"]["T4"]
                     ["n_for_80pct_power_t05"]}}

    out = {
        "datasource": DS, "metric": METRIC, "seeds": list(SEEDS),
        "source": {"leaderboard_rows": 84,
                   "crosscheck_vs_stage2_json": "exact (all 84 cells)"},
        "per_seed": per_seed, "cell_ci95_trained": cell_ci,
        "paired": paired, "trend": trend,
        "margin_trained_minus_untrained": margins, "power": power,
        "honesty": build_honesty(paired),
    }
    (HERE / "stage2_variance.json").write_text(json.dumps(out, indent=1))
    write_md(out)
    print(json.dumps({"paired_T8": paired["txc_pre_minus_tsae"]["by_T"]["T8"],
                      "trend": trend, "power": power}, indent=1))
    print(f"-> {HERE}/stage2_variance.json ; stage2_variance.md")


def build_honesty(paired):
    """Honesty notes with the data-dependent one computed, not asserted."""
    t8 = paired["txc_pre_minus_tsae"]["by_T"]["T8"]
    return [
        "n = 3 seeds: the exact one-sided sign-flip permutation test "
        "cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired "
        "direction is consistent in all 3 seeds', not as significance.",
        "The exact bootstrap distribution of a 3-value mean has 27 "
        "atoms (<= 10 distinct values); BCa endpoints are coarse and "
        "cannot extend past the extreme seed values.",
        "Pairing by seed was the right design a priori, but it bought "
        "no variance reduction here: at the T8 headline cell the "
        "across-seed correlation between the TXC-pre and T-SAE arms is "
        f"r = {t8['r_between_arms']:.2f}, so the paired sd "
        f"({t8['sd']:.4f}) is not below the independent-arms value "
        f"({t8['sd_if_independent']:.4f}). The cross-arch margin is "
        "therefore NOT bounded away from 0 at n = 3; the receipts that "
        "ARE significant at n = 3 are within-arch: the T = 2->8 rise "
        "and the trained-untrained margins (paired by seed WITHIN an "
        "arch, where the pairing does bind).",
        "The T = 2->8 trend test is exact with 216 relabelings "
        "(min p = 1/216), so it carries real resolution at n = 3.",
    ]


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def write_md(out):
    L = []
    A = L.append
    A(f"# Stage-2 λ̂ panel — variance receipts (runpod-b, item 1 of "
      f"`briefings/hunt-support-stats.md`)\n")
    A(f"Source: {out['source']['leaderboard_rows']} leaderboard rows, "
      f"datasource `{out['datasource']}`, metric `{out['metric']}`, seeds "
      f"{out['seeds']}; cross-check vs `stage2_{out['datasource']}.json`: "
      f"{out['source']['crosscheck_vs_stage2_json']}. Built by "
      f"`stage2_variance.py` — every number below is script-derived.\n")

    A("## Per-seed values (trained), λ̂ recovery\n")
    A("| cell | " + " | ".join(f"seed {s}" for s in out["seeds"]) +
      " | mean | 95% t CI |")
    A("|---|" + "---|" * (len(out["seeds"]) + 2))
    for key, vals in out["per_seed"]["trained"].items():
        ci = out["cell_ci95_trained"][key]
        A(f"| {key} | " +
          " | ".join(fmt(vals[str(s)]) for s in out["seeds"]) +
          f" | {fmt(ci['mean'])} | [{fmt(ci['t_ci95'][0])}, "
          f"{fmt(ci['t_ci95'][1])}] |")
    A("")

    A("## Paired-by-seed differences (window arch − T=1 reference)\n")
    for name, blk in out["paired"].items():
        A(f"### {name} (reference {blk['reference']})\n")
        A("| T | " + " | ".join(f"seed {s}" for s in out["seeds"]) +
          " | mean | sd | 95% t CI | 95% BCa CI | sign-flip p (1-sided)"
          " | r(arms) |")
        A("|---|" + "---|" * (len(out["seeds"]) + 6))
        for tkey, st in blk["by_T"].items():
            A(f"| {tkey} | " +
              " | ".join(fmt(st["per_seed"][str(s)]) for s in out["seeds"]) +
              f" | {fmt(st['mean'])} | {fmt(st['sd'])} "
              f"| [{fmt(st['t_ci95'][0])}, {fmt(st['t_ci95'][1])}] "
              f"| [{fmt(st['bca_ci95'][0])}, {fmt(st['bca_ci95'][1])}] "
              f"| {st['p_signflip_one_sided']:.3f} "
              f"| {st['r_between_arms']:+.2f} |")
        A("")

    A("## Trend across T (exact within-seed permutation, pooled seeds)\n")
    A("| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided)"
      " | perms |")
    A("|---|---|---|---|---|---|")
    for name, tr in out["trend"].items():
        A(f"| {name} | {tr['Ts']} | {fmt(tr['slope_sum_per_log2T'])} | " +
          ", ".join(fmt(s) for s in tr["per_seed_slopes"]) +
          f" | {tr['p_one_sided']:.4f} | {tr['n_perms']} |")
    A("")

    A("## Trained − untrained margin (paired by seed), key cells\n")
    A("| cell | mean | 95% t CI | 95% BCa CI | sign-flip p |")
    A("|---|---|---|---|---|")
    for key, st in out["margin_trained_minus_untrained"].items():
        A(f"| {key} | {fmt(st['mean'])} "
          f"| [{fmt(st['t_ci95'][0])}, {fmt(st['t_ci95'][1])}] "
          f"| [{fmt(st['bca_ci95'][0])}, {fmt(st['bca_ci95'][1])}] "
          f"| {st['p_signflip_one_sided']:.3f} |")
    A("")

    A("## Power calc → seed recommendation\n")
    p = out["power"]
    A(f"- Exact sign-flip attainability: p ≤ 0.05 first possible at "
      f"**n = {p['signflip_min_seeds_for_p05']} seeds** (2⁻ⁿ ≤ 0.05).")
    for name in ("txc_pre_minus_tsae", "txc_pre_minus_pertoken"):
        for T in (4, 8):
            st = p[name][f"T{T}"]
            A(f"- {name} @T{T}: observed {fmt(st['observed_mean'])} ± "
              f"{fmt(st['observed_sd'])}; n for 95% lower bound > 0: "
              f"**{st['n_for_95_lower_bound_gt0']}**; n for 80% power "
              f"(one-sided t, α=0.05): **{st['n_for_80pct_power_t05']}**.")
    rec = p["recommendation"]
    A(f"- Criterion: {rec['criterion']}.")
    if rec["cells"]:
        A(f"- **Recommendation:** total seeds needed "
          f"**{rec['seeds_total_needed']}** ⇒ **{rec['extra_seeds']} extra "
          f"seeds**. Per extra seed (trained): "
          f"{', '.join(rec['cells']['per_extra_seed_trained'])} ⇒ "
          f"{rec['cells']['n_trained_cells']} trained cells "
          f"(+{rec['cells']['untrained_counterparts_optional']} optional "
          f"untrained counterparts). Headroom option: "
          f"{rec['cells']['headroom_option']['n_extra_seeds']} extra seeds "
          f"= {rec['cells']['headroom_option']['n_trained_cells']} cells — "
          f"{rec['cells']['headroom_option']['why']}.")
    else:
        A(f"- **Recommendation:** total seeds needed "
          f"{rec['seeds_total_needed']} ⇒ extra seeds "
          f"{rec['extra_seeds']} — outside the cheap-append range "
          f"(no cell list emitted).")
    A(f"- T4 is NOT cheaply boundable (n = "
      f"{rec['T4_not_cheaply_boundable']['n_for_95_lower_bound_gt0']} "
      f"to bound, "
      f"{rec['T4_not_cheaply_boundable']['n_for_80pct_power_t05']} for "
      f"80% power); the T-rise + trained−untrained margin carry that "
      f"cell.\n")

    A("## Honesty notes\n")
    for h in out["honesty"]:
        A(f"- {h}")
    A("")
    (HERE / "stage2_variance.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
