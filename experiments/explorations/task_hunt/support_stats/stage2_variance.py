"""Variance receipts for a Stage-2 λ̂ panel (hunt-support-stats item 1;
pre-flighted for the oprate / fineweb panels by panel-support-audit item 1).

Primary source: the committed leaderboard rows for the panel datasource
(per-seed values live there; a panel's `stage2_summary.json` only has
mean/std). Every extracted cell is cross-checked for EXACT equality
against the panel run's own results JSON; any mismatch aborts the build.

Computes and writes (next to this script, or under ``--out-dir``):
  <out-prefix>.json  — per-seed values for every (arch, T, kind) cell;
      paired-by-seed TXC-pre − T-SAE and TXC-pre − per-token differences
      at each T (exact sign-flip permutation p, exact-enumeration BCa and
      t 95% CIs); the T = 2→8 trend statistic pooled over seeds (exact
      within-seed permutation test) for TXC-pre and for its
      trained−untrained margin; per-cell trained−untrained margin CIs;
      the seed power calc and the cell recommendation.
  <out-prefix>.md    — the same numbers as a short readable section
      (all numbers script-derived; nothing hand-typed).

Honesty notes are embedded in both outputs: n = 3 seeds means the exact
one-sided sign-flip test bottoms out at p = 1/8, and the exact bootstrap
distribution of a mean has 27 atoms — the paired design is the point,
not the p-values.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.support_stats.stage2_variance

**Probe-agnostic by construction** (`briefings/probe-adequacy.md` item 3):
every input is a CLI parameter with defaults that reproduce the committed
v1 receipts byte-identically. The exact invocation for each Stage-2 panel
lives in `PANEL_RECIPES.md` next to this script — run that, don't improvise.

Row-selection policy (all explicit, all defaulted to the committed λ̂
panel's semantics):

- ``--probe`` picks the metric columns (v1: `lambda_recovery`;
  v2: `lambda_recovery_v2`). Two row layouts exist on the leaderboard:
  the λ̂ panel's SPLIT layout (v1 rows carry no `lambda_probe_v2`
  eval_cfg flag; v2 rows are separate re-run eval_keys) and the
  oprate / fineweb panels' PAIRED layout (every row is flagged and
  carries BOTH column sets). ``--row-layout auto`` (default) resolves
  v1 to unflagged rows when any exist, else to the flagged rows' v1
  columns; v2 always reads flagged rows. The resolved layout is
  recorded in the output's `source` block.
- ``--k-pos`` pins the panel's frozen per-token budget (8).
  ``--post-k-rule times-T`` accepts ``--post-archs`` rows at
  k_pos = (--k-pos)·T instead — the code-rate convention the new panels
  run from the start (`card_stage2_postmatched.md` § 2). The default
  ``fixed`` reproduces the λ̂ panel, where the post-matched amendment
  rows (k_pos = 8·T, same datasource) are excluded by design — without
  this filter the leaderboard aborts the build on 24 duplicate cells.
- ``--seeds`` pins the panel seed population (default 1,2,42). The λ̂
  seed top-up added trained-only rows at seeds 3,4,5 (pre/T4, pre/T8)
  AFTER the v1 receipts were committed; without this filter the exact
  cross-check aborts on today's leaderboard.
- Populations with a partial T ladder (the fineweb replication cells
  exist at only two T values) degrade honestly: the cells are reported,
  the trend is skipped with the reason stated, and the power section
  keys on the T values that exist. A trend from two points is never
  emitted.
"""

from __future__ import annotations

import argparse
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
K_POS = 8                     # the panel's frozen per-token budget
SEEDS = (1, 2, 42)
TREND_TS = (2, 4, 8)

LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"


def stage2_json_default(ds: str) -> Path:
    return (ROOT / "experiments" / "explorations" / "task_hunt" /
            "lambda_intensity" / "results" / f"stage2_{ds}.json")

# Comparisons: (name, window arch, T=1 reference arch)
PAIRINGS = [
    ("txc_pre_minus_tsae", "txc_batchtopk_pre", "tsae"),
    ("txc_pre_minus_pertoken", "txc_batchtopk_pre", "batchtopk_sae"),
]
PRE = "txc_batchtopk_pre"


def load_cells(ds: str, metric: str, k_pos: int, probe: str,
               crosscheck_json: Path, *, seeds=SEEDS,
               leaderboard: Path = LEADERBOARD, row_layout: str = "auto",
               post_k_rule: str = "fixed",
               post_archs=("txc_batchtopk_post",)):
    """(arch, T, seed, kind) -> {metric, l0, chance} from the leaderboard,
    cross-checked exactly against the panel run's results JSON.

    Returns ``(cells, meta)`` where ``meta`` records the resolved row
    layout and any failed (ok=false) crosscheck rows skipped. Row filter:
    datasource, seed population, the per-arch k_pos rule, and the probe
    generation / row layout (see module docstring).
    """
    chance_key = metric.replace("recovery", "chance")
    post = set(post_archs)
    flagged, unflagged = {}, {}
    with Path(leaderboard).open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != ds:
                continue
            if r["seed"] not in seeds:
                continue
            tc = r["training_cfg"]
            ov = tc["arch_hparams_override"]
            want_k = (k_pos * ov["T"]
                      if (post_k_rule == "times-T" and r["arch"] in post)
                      else k_pos)
            if ov.get("k_pos") != want_k:
                continue
            pop = (flagged if r.get("eval_cfg", {}).get("lambda_probe_v2")
                   else unflagged)
            key = (r["arch"], ov["T"], r["seed"],
                   "untrained" if tc["n_steps"] == 0 else "trained")
            if key in pop:
                raise SystemExit(f"duplicate leaderboard cell {key}")
            pop[key] = r

    if probe == "v2":
        rows, layout = flagged, "flagged"
    elif row_layout == "split":
        rows, layout = unflagged, "split"
    elif row_layout == "paired":
        rows, layout = flagged, "paired"
    else:                     # auto
        rows, layout = ((unflagged, "split") if unflagged
                        else (flagged, "paired"))
    if not rows:
        raise SystemExit(
            f"0 leaderboard rows selected (ds={ds}, probe={probe}, "
            f"k_pos={k_pos}, post_k_rule={post_k_rule}, "
            f"row_layout={row_layout}) — wrong --ds / --k-pos, or a "
            f"paired-layout panel (every row flagged) read with "
            f"--row-layout split?")

    cells = {}
    for key, r in rows.items():
        m = r["metrics"]
        if metric not in m or chance_key not in m:
            raise SystemExit(
                f"row {key} lacks metric column '{metric}' / "
                f"'{chance_key}' — probe/population mismatch "
                f"(row_layout resolved: {layout})")
        cells[key] = {"metric": m[metric], "l0": m["l0_per_token"],
                      "chance": m[chance_key]}

    missing = []
    for a, T in sorted({(k[0], k[1]) for k in cells}):
        for s in seeds:
            for kind in ("trained", "untrained"):
                if (a, T, s, kind) not in cells:
                    missing.append((a, T, s, kind))
    if missing:
        raise SystemExit(
            f"incomplete panel population: {len(missing)} missing cells "
            f"(e.g. {missing[:6]}) — partial panel, or v1 rows split "
            f"across layouts? (row_layout resolved: {layout})")

    raw = json.loads(Path(crosscheck_json).read_text())
    n_failed = sum(1 for r in raw if not r.get("ok", True))
    if n_failed:
        print(f"[warn] crosscheck json: skipping {n_failed} failed "
              f"(ok=false) rows — they have no leaderboard counterpart")
    try:
        ref = {(r["arch"], r["T"], r["seed"], r["kind"]):
               r["metrics"][metric]
               for r in raw if r.get("ok", True) and r["seed"] in seeds}
    except KeyError as e:
        raise SystemExit(f"crosscheck json rows lack key {e} — wrong "
                         f"--crosscheck-json for this probe/panel?")
    if set(ref) != set(cells):
        only_json = sorted(set(ref) - set(cells), key=str)
        only_lb = sorted(set(cells) - set(ref), key=str)
        hint = ""
        if only_json and all(k[0] in post for k in only_json):
            hint = (" HINT: every json-only cell is a post-arch cell — "
                    "this panel runs post at k_pos = k·T; pass "
                    "--post-k-rule times-T.")
        raise SystemExit(
            f"leaderboard/stage2-json key sets differ: {len(only_json)} "
            f"only in json (e.g. {only_json[:3]}), {len(only_lb)} only "
            f"on leaderboard (e.g. {only_lb[:3]}).{hint}")
    bad = [k for k in ref if ref[k] != cells[k]["metric"]]
    if bad:
        raise SystemExit(f"cross-check FAILED on {len(bad)} cells: {bad[:4]}")
    return cells, {"row_layout": layout,
                   "crosscheck_failed_rows_skipped": n_failed}


def cell_vec(cells, arch, T, kind="trained", field="metric", seeds=SEEDS):
    return np.array([cells[(arch, T, s, kind)][field] for s in seeds])


def paired_stats(diffs, arm_a=None, arm_b=None, seeds=SEEDS):
    """Stats on paired-by-seed differences. When the two arms' per-seed
    vectors are given, also report the across-seed correlation between
    arms and the sd the difference WOULD have were the arms independent
    — the honest check on whether pairing bought any variance
    reduction here."""
    d = np.asarray(diffs, dtype=float)
    mean, t_lo, t_hi = t_ci95(d)
    p, n_pat = sign_flip_p(d, "greater")
    bca = bca_ci(d)
    out = {"per_seed": {str(s): float(v) for s, v in zip(seeds, d)},
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


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Variance receipts for a Stage-2 λ̂ panel; defaults "
                    "reproduce the committed v1 receipts byte-identically. "
                    "Per-panel invocations: see PANEL_RECIPES.md.")
    ap.add_argument("--ds", default=DS, help="panel datasource registry key")
    ap.add_argument("--probe", choices=("v1", "v2"), default="v1",
                    help="which λ-probe generation to report "
                         "(picks the metric columns; see --row-layout)")
    ap.add_argument("--metric", default=None,
                    help="headline metric column (default: lambda_recovery "
                         "for --probe v1, lambda_recovery_v2 for v2)")
    ap.add_argument("--k-pos", type=int, default=K_POS,
                    help="panel per-token budget row filter")
    ap.add_argument("--post-k-rule", choices=("fixed", "times-T"),
                    default="fixed",
                    help="'times-T': accept --post-archs rows at "
                         "k_pos = (--k-pos)·T — the code-rate convention "
                         "the oprate/fineweb panels run from the start. "
                         "Default 'fixed' = the λ̂ panel (post at --k-pos; "
                         "8·T amendment rows excluded).")
    ap.add_argument("--post-archs", default="txc_batchtopk_post",
                    help="comma list of archs the times-T rule applies to")
    ap.add_argument("--row-layout", choices=("auto", "split", "paired"),
                    default="auto",
                    help="v1 row selection: 'split' = unflagged rows "
                         "(λ̂ panel), 'paired' = flagged rows carrying "
                         "both column sets (oprate/fineweb panels), "
                         "'auto' = split when unflagged rows exist, else "
                         "paired. v2 always reads flagged rows.")
    ap.add_argument("--seeds", default="1,2,42",
                    help="comma list: the panel seed population (top-up / "
                         "stray extra-seed rows are excluded from the "
                         "receipts and the exact cross-check)")
    ap.add_argument("--leaderboard", type=Path, default=LEADERBOARD,
                    help="leaderboard JSONL (default: the canonical one)")
    ap.add_argument("--crosscheck-json", type=Path, default=None,
                    help="panel results JSON for the exact cross-check "
                         "(default: lambda_intensity/results/stage2_<ds>"
                         ".json — the oprate/fineweb panels write theirs "
                         "under their own results/, pass it explicitly)")
    ap.add_argument("--out-prefix", default="stage2_variance",
                    help="output basename — a re-based run writes new "
                         "files next to the committed v1 receipts, never "
                         "over them")
    ap.add_argument("--out-dir", type=Path, default=HERE,
                    help="output directory (default: next to this script)")
    args = ap.parse_args(argv)
    if args.metric is None:
        args.metric = METRIC if args.probe == "v1" else METRIC + "_v2"
    if args.crosscheck_json is None:
        args.crosscheck_json = stage2_json_default(args.ds)
    args.seeds = tuple(int(s) for s in str(args.seeds).split(","))
    args.post_archs = tuple(a for a in args.post_archs.split(",") if a)
    return args


def main(argv=None):
    args = parse_args(argv)
    cells, meta = load_cells(
        args.ds, args.metric, args.k_pos, args.probe, args.crosscheck_json,
        seeds=args.seeds, leaderboard=args.leaderboard,
        row_layout=args.row_layout, post_k_rule=args.post_k_rule,
        post_archs=args.post_archs)
    seeds = args.seeds
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
                    for s in seeds}
            per_seed["l0_trained"][key] = {
                str(s): float(cells[(a, T, s, "trained")]["l0"])
                for s in seeds}
            v = cell_vec(cells, a, T, seeds=seeds)
            mean, lo, hi = t_ci95(v)
            cell_ci[key] = {"mean": mean, "sd": float(v.std(ddof=1)),
                            "t_ci95": [lo, hi]}

    # ---- 2. paired-by-seed diffs at each window T
    paired = {}
    for name, win_arch, ref_arch in PAIRINGS:
        if (win_arch not in archs or ref_arch not in archs
                or 1 not in Ts_by_arch.get(ref_arch, [])):
            paired[name] = {"skipped": f"arm missing from population "
                                       f"(need {win_arch} + {ref_arch}/T1)"}
            continue
        ref = cell_vec(cells, ref_arch, 1, seeds=seeds)
        paired[name] = {"reference": f"{ref_arch}/T1",
                        "reference_per_seed":
                            {str(s): float(v) for s, v in zip(seeds, ref)},
                        "by_T": {}}
        for T in Ts_by_arch[win_arch]:
            win = cell_vec(cells, win_arch, T, seeds=seeds)
            paired[name]["by_T"][f"T{T}"] = paired_stats(
                win - ref, win, ref, seeds=seeds)

    # ---- 3. trend across T (pooled over seeds, exact within-seed perm)
    def trend_block(mat, Ts):
        obs, slopes, p, n = within_seed_trend(mat, Ts, "greater")
        return {"Ts": list(Ts), "slope_sum_per_log2T": obs,
                "per_seed_slopes": slopes, "p_one_sided": p, "n_perms": n}

    trend = {}
    pre_ts = tuple(Ts_by_arch.get(PRE, ()))
    if all(t in pre_ts for t in TREND_TS):
        mat_pre = np.array([[cells[(PRE, T, s, "trained")]["metric"]
                             for T in TREND_TS] for s in seeds])
        mat_margin = np.array([[cells[(PRE, T, s, "trained")]["metric"]
                                - cells[(PRE, T, s, "untrained")]["metric"]
                                for T in TREND_TS] for s in seeds])
        trend["txc_pre_trained_2to8"] = trend_block(mat_pre, TREND_TS)
        trend["txc_pre_margin_2to8"] = trend_block(mat_margin, TREND_TS)
    else:
        trend["txc_pre_trend"] = {
            "skipped": (f"frozen 2->8 trend undefined: {PRE} present at "
                        f"T={list(pre_ts)} only — a trend statistic over "
                        f"{len(pre_ts)} T value(s) has no within-seed "
                        f"permutation resolution; the cells themselves are "
                        f"reported in the per-seed / paired sections")}
    if len(pre_ts) >= 3 and pre_ts != TREND_TS:
        mat_full = np.array([[cells[(PRE, T, s, "trained")]["metric"]
                              for T in pre_ts] for s in seeds])
        sec_key = f"txc_pre_trained_{pre_ts[0]}to{pre_ts[-1]}_secondary"
        try:
            trend[sec_key] = trend_block(mat_full, pre_ts)
        except ValueError as e:
            # 5-T ladders (T32 panels) exceed the exact-enumeration cap
            # ((5!)^3 relabelings); degrade honestly — the frozen 2->8
            # primary above is the pre-registered statistic.
            trend[sec_key] = {
                "skipped": (f"secondary full-ladder trend over "
                            f"T={list(pre_ts)} not computed: {e} — the "
                            f"frozen 2->8 primary carries the trend "
                            f"receipt; per-cell values are all reported")}

    # ---- 4. trained − untrained margin CI, every cell
    margins = {}
    for a in archs:
        for T in Ts_by_arch[a]:
            d = (cell_vec(cells, a, T, "trained", seeds=seeds)
                 - cell_vec(cells, a, T, "untrained", seeds=seeds))
            margins[f"{a}/T{T}"] = paired_stats(d, seeds=seeds)

    # ---- 5. power calc -> seed recommendation
    n_attain = seeds_for_signflip(0.05)
    power = {"signflip_min_seeds_for_p05": n_attain}
    for name, _, _ in PAIRINGS:
        by_T = paired.get(name, {}).get("by_T")
        if not by_T:
            continue
        power[name] = {}
        for T in (4, 8):
            st = by_T.get(f"T{T}")
            if st is None:
                continue
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
    by_T_main = paired.get("txc_pre_minus_tsae", {}).get("by_T") or {}
    avail_T = sorted(int(k[1:]) for k in by_T_main)
    if by_T_main:
        head_T = 8 if 8 in avail_T else avail_T[-1]
        st = by_T_main[f"T{head_T}"]
        n_head = seeds_for_bound(st["mean"], st["sd"])
        n_needed = None if n_head is None else max(n_attain, n_head)
        extra = None if n_needed is None else max(0, n_needed - len(seeds))
        rec_cells = None
        if extra is not None and 0 < extra <= 4:
            per_seed_cells = [f"{PRE}/T{t}" for t in (4, 8)
                              if t in Ts_by_arch.get(PRE, [])] + ["tsae/T1"]
            rec_cells = {"per_extra_seed_trained": per_seed_cells,
                         "n_extra_seeds": extra,
                         "n_trained_cells": len(per_seed_cells) * extra,
                         "untrained_counterparts_optional":
                             len(per_seed_cells) * extra,
                         "headroom_option": {
                             "n_extra_seeds": extra + 1,
                             "n_trained_cells":
                                 len(per_seed_cells) * (extra + 1),
                             "why": "one seed of slack against the plug-in "
                                    "sd estimate itself being an n=3 "
                                    "estimate; also reaches sign-flip "
                                    "p = 1/128 at the T8 cell"}}
        power["recommendation"] = {
            "criterion": f"one-sided 95% t lower bound > 0 on the paired "
                         f"TXC-pre - T-SAE diff at T = {head_T}, AND exact "
                         f"sign-flip attainability (2^-n <= 0.05)",
            "seeds_total_needed": n_needed, "extra_seeds": extra,
            "cells": rec_cells}
        if head_T != 8:
            power["recommendation"]["headline_T_note"] = (
                f"T8 absent from this population; criterion keyed on the "
                f"largest available window T = {head_T}")
        t4 = power.get("txc_pre_minus_tsae", {}).get("T4")
        if t4 is not None:
            power["recommendation"]["T4_not_cheaply_boundable"] = {
                "n_for_95_lower_bound_gt0": t4["n_for_95_lower_bound_gt0"],
                "n_for_80pct_power_t05": t4["n_for_80pct_power_t05"]}
    else:
        power["recommendation"] = {
            "skipped": "txc_pre_minus_tsae pairing absent from this "
                       "population — no seed recommendation"}

    legacy = (args.ds == DS and args.probe == "v1"
              and args.k_pos == K_POS and seeds == SEEDS
              and meta["row_layout"] == "split"
              and args.post_k_rule == "fixed")
    source = {"leaderboard_rows": len(cells),
              "crosscheck_vs_stage2_json": f"exact (all {len(cells)} cells)"}
    if not legacy:
        source.update({"probe": args.probe, "row_layout": meta["row_layout"],
                       "k_pos": args.k_pos,
                       "post_k_rule": args.post_k_rule,
                       "seed_population": list(seeds)})
        if meta["crosscheck_failed_rows_skipped"]:
            source["crosscheck_failed_rows_skipped"] = \
                meta["crosscheck_failed_rows_skipped"]
    out = {
        "datasource": args.ds, "metric": args.metric, "seeds": list(seeds),
        "source": source,
        "per_seed": per_seed, "cell_ci95_trained": cell_ci,
        "paired": paired, "trend": trend,
        "margin_trained_minus_untrained": margins, "power": power,
        "honesty": build_honesty(paired, trend),
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{args.out_prefix}.json").write_text(json.dumps(out, indent=1))
    write_md(out, args.out_prefix, args.crosscheck_json.name, out_dir)
    head_key = (f"paired_T{8 if 8 in avail_T else (avail_T[-1] if avail_T else 0)}")
    summary = {"trend": trend, "power": power}
    if by_T_main:
        summary = {head_key: by_T_main[head_key.replace("paired_", "")],
                   **summary}
    print(json.dumps(summary, indent=1))
    print(f"-> {out_dir}/{args.out_prefix}.json ; {args.out_prefix}.md")


def build_honesty(paired, trend):
    """Honesty notes with the data-dependent one computed, not asserted."""
    notes = [
        "n = 3 seeds: the exact one-sided sign-flip permutation test "
        "cannot report p < 1/8 = 0.125; treat p = 0.125 as 'the paired "
        "direction is consistent in all 3 seeds', not as significance.",
        "The exact bootstrap distribution of a 3-value mean has 27 "
        "atoms (<= 10 distinct values); BCa endpoints are coarse and "
        "cannot extend past the extreme seed values.",
    ]
    by_T = paired.get("txc_pre_minus_tsae", {}).get("by_T") or {}
    if by_T:
        avail = sorted(int(k[1:]) for k in by_T)
        h = 8 if 8 in avail else avail[-1]
        t8 = by_T[f"T{h}"]
        notes.append(
            "Pairing by seed was the right design a priori, but it bought "
            "no variance reduction here: at the T" + str(h) +
            " headline cell the "
            "across-seed correlation between the TXC-pre and T-SAE arms is "
            f"r = {t8['r_between_arms']:.2f}, so the paired sd "
            f"({t8['sd']:.4f}) is not below the independent-arms value "
            f"({t8['sd_if_independent']:.4f}). The cross-arch margin is "
            "therefore NOT bounded away from 0 at n = 3; the receipts that "
            "ARE significant at n = 3 are within-arch: the T = 2->8 rise "
            "and the trained-untrained margins (paired by seed WITHIN an "
            "arch, where the pairing does bind).")
    if "txc_pre_trained_2to8" in trend:
        notes.append(
            "The T = 2->8 trend test is exact with 216 relabelings "
            "(min p = 1/216), so it carries real resolution at n = 3.")
    return notes


def fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def write_md(out, out_prefix="stage2_variance",
             crosscheck_name=None, out_dir=HERE):
    if crosscheck_name is None:
        crosscheck_name = f"stage2_{out['datasource']}.json"
    probe = out["source"].get("probe")   # present only for non-legacy runs
    L = []
    A = L.append
    if probe is None:
        A(f"# Stage-2 λ̂ panel — variance receipts (runpod-b, item 1 of "
          f"`briefings/hunt-support-stats.md`)\n")
    else:
        A(f"# Stage-2 panel — variance receipts (`{out['datasource']}`, "
          f"probe {probe})\n")
    A(f"Source: {out['source']['leaderboard_rows']} leaderboard rows, "
      f"datasource `{out['datasource']}`, metric `{out['metric']}`, seeds "
      f"{out['seeds']}; cross-check vs `{crosscheck_name}`: "
      f"{out['source']['crosscheck_vs_stage2_json']}. Built by "
      f"`stage2_variance.py` — every number below is script-derived.\n")
    if probe is not None:
        A(f"Selection: probe {probe}, row layout "
          f"{out['source']['row_layout']}, k_pos {out['source']['k_pos']} "
          f"(post rule {out['source']['post_k_rule']}).\n")

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
        if "skipped" in blk:
            A(f"### {name}\n")
            A(f"_Skipped: {blk['skipped']}._\n")
            continue
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
    runs = {k: v for k, v in out["trend"].items() if "Ts" in v}
    skips = {k: v for k, v in out["trend"].items() if "Ts" not in v}
    if runs:
        A("| test | Ts | Σ slopes (per log₂T) | per-seed slopes | p (1-sided)"
          " | perms |")
        A("|---|---|---|---|---|---|")
        for name, tr in runs.items():
            A(f"| {name} | {tr['Ts']} | {fmt(tr['slope_sum_per_log2T'])} | " +
              ", ".join(fmt(s) for s in tr["per_seed_slopes"]) +
              f" | {tr['p_one_sided']:.4f} | {tr['n_perms']} |")
        A("")
    for name, tr in skips.items():
        A(f"- **{name}**: {tr['skipped']}")
    if skips:
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
            st = p.get(name, {}).get(f"T{T}")
            if st is None:
                continue
            A(f"- {name} @T{T}: observed {fmt(st['observed_mean'])} ± "
              f"{fmt(st['observed_sd'])}; n for 95% lower bound > 0: "
              f"**{st['n_for_95_lower_bound_gt0']}**; n for 80% power "
              f"(one-sided t, α=0.05): **{st['n_for_80pct_power_t05']}**.")
    rec = p["recommendation"]
    if "skipped" in rec:
        A(f"- **Recommendation skipped:** {rec['skipped']}.")
    else:
        A(f"- Criterion: {rec['criterion']}.")
        if rec.get("headline_T_note"):
            A(f"- Note: {rec['headline_T_note']}.")
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
        t4 = rec.get("T4_not_cheaply_boundable")
        if t4 is not None:
            A(f"- T4 is NOT cheaply boundable (n = "
              f"{t4['n_for_95_lower_bound_gt0']} "
              f"to bound, "
              f"{t4['n_for_80pct_power_t05']} for "
              f"80% power); the T-rise + trained−untrained margin carry "
              f"that cell.\n")
        else:
            A("")

    A("## Honesty notes\n")
    for h in out["honesty"]:
        A(f"- {h}")
    A("")
    (Path(out_dir) / f"{out_prefix}.md").write_text("\n".join(L))


if __name__ == "__main__":
    main()
