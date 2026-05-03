"""Flip-matrix extraction across (arch, hookpoint, magnitude, question_id).

Reads:
  - phase1_unsteered.json: list[dict] with `unique_id` and `unsteered_correct`.
    Only one phase 1 file exists (it's per-question, not per-arch).
  - <arch_dir>/phase2_rescue.json: list[dict] with `unique_id`, `magnitude`,
    `rescued_correct`. One file per arch run.

Emits:
  - flip_matrix.parquet: long-form rows (arch, hookpoint, cell_id, feature_id,
    feature_mode, magnitude, question_id, before_correct, after_correct,
    before_label, after_label, transition).
  - mcnemar_table.csv: McNemar test results per arch at the per-arch best
    rescue magnitude (max n_ic - n_ci).

Usage:
  python -m experiments.ward_backtracking_txc.build_flip_matrix \
      --variant cut25 \
      --phase1 results/ward_backtracking_txc/b3_math500/phase1_unsteered.json \
      --runs <run1.json> <run2.json> ...

Each run JSON should be co-located with a meta.json describing arch / cell.
For now, also accepts --runs-from-summary <root_dir> which discovers per-cell
subdirs of the form `b3_math500_cut25/<cell_id>__<feature>__<mode>/...`.
"""
from __future__ import annotations
import argparse
import json
import logging
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.flip_matrix")


_CELL_RE = re.compile(r"^(?P<arch>[a-z0-9_]+?)__(?P<hp>[a-z0-9_]+_L\d+)__k(?P<k>\d+)__s(?P<seed>\d+)$")


def parse_cell(cell_id: str) -> dict:
    """Parse `<arch>__<hp>__k<k>__s<seed>` into components.

    `arch` may include underscores (`topk_sae`, `stacked_sae`, `tsae_paper`).
    """
    m = _CELL_RE.match(cell_id)
    if not m:
        raise ValueError(f"unparseable cell_id: {cell_id}")
    return {
        "arch": m.group("arch"),
        "hookpoint": m.group("hp"),
        "k_per_position": int(m.group("k")),
        "seed": int(m.group("seed")),
    }


def transition_label(before: bool, after: bool) -> str:
    return ("cc" if before and after else
            "ci" if before and not after else
            "ic" if not before and after else
            "ii")


def load_phase1(path: Path) -> dict[str, bool]:
    """Return {question_id: unsteered_correct}."""
    rows = json.loads(path.read_text())
    out = {}
    for r in rows:
        qid = r["unique_id"]
        # `unsteered_correct` is the canonical truth. Some legacy rows may
        # have `unsteered_answer is None` (truncated); we keep them as
        # incorrect. b3_variants drops them from the steered cohort, so
        # they won't appear in phase2 anyway.
        out[qid] = bool(r["unsteered_correct"])
    return out


def load_run(rescue_path: Path, meta_path: Path) -> tuple[dict, list[dict]]:
    """Load one phase2 run + its meta.

    meta.json schema (we write it ourselves alongside the rescue.json):
      {"cell_id": "...", "feature_id": int, "feature_mode": "pos0"|"union"}
    """
    meta = json.loads(meta_path.read_text())
    rescue = json.loads(rescue_path.read_text())
    return meta, rescue


def build_long_df(phase1: dict[str, bool], runs: list[tuple[dict, list[dict]]]) -> pd.DataFrame:
    rows = []
    for meta, rescue in runs:
        cell_id = meta["cell_id"]
        cell_parts = parse_cell(cell_id)
        for r in rescue:
            qid = r["unique_id"]
            # Prefer the new `before_correct` field (b3_variants post-2026-05-02
            # writes it directly). Fall back to phase1 lookup for legacy runs.
            if "before_correct" in r:
                before = bool(r["before_correct"])
            elif qid in phase1:
                before = phase1[qid]
            else:
                log.warning("[skip] qid=%s in rescue, no before_correct or phase1 entry", qid)
                continue
            after = bool(r["rescued_correct"])
            rows.append({
                **cell_parts,
                "cell_id": cell_id,
                "feature_id": int(meta["feature_id"]),
                "feature_mode": meta["feature_mode"],
                "magnitude": float(r["magnitude"]),
                "question_id": qid,
                "before_correct": before,
                "after_correct": after,
                "transition": transition_label(before, after),
            })
    df = pd.DataFrame(rows)
    if df.empty:
        log.warning("[empty] no rows produced")
    return df


def confusion_at_mag(df: pd.DataFrame, arch: str, magnitude: float) -> dict:
    sub = df[(df["arch"] == arch) & (df["magnitude"] == magnitude)]
    counts = sub["transition"].value_counts().to_dict()
    return {
        "arch": arch,
        "magnitude": magnitude,
        "n_total": len(sub),
        "n_cc": int(counts.get("cc", 0)),
        "n_ci": int(counts.get("ci", 0)),
        "n_ic": int(counts.get("ic", 0)),
        "n_ii": int(counts.get("ii", 0)),
    }


def mcnemar_chi2(n_ic: int, n_ci: int) -> tuple[float, float]:
    """Mid-p McNemar: (chi2_continuity, two-sided p-value).

    Uses Edwards' continuity correction:  chi2 = (|n_ic - n_ci| - 1)^2 / (n_ic + n_ci).
    For (n_ic + n_ci) < 25, recommend exact binomial; we compute both and
    return the continuity-corrected one as primary.
    """
    n = n_ic + n_ci
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(n_ic - n_ci) - 1) ** 2 / n if n > 0 else 0.0
    # Two-sided exact binomial p-value (more reliable for small n)
    from scipy.stats import binomtest
    bt = binomtest(min(n_ic, n_ci), n=n, p=0.5, alternative="two-sided")
    return float(chi2), float(bt.pvalue)


def best_magnitude_per_arch(df: pd.DataFrame) -> dict[str, float]:
    """For each arch, the magnitude maximizing (n_ic - n_ci)."""
    agg = df.groupby(["arch", "magnitude"]).apply(
        lambda g: (g["transition"] == "ic").sum() - (g["transition"] == "ci").sum(),
        include_groups=False,
    ).rename("net").reset_index()
    return {arch: g.loc[g["net"].idxmax(), "magnitude"] for arch, g in agg.groupby("arch")}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--phase1", type=Path, required=True,
                   help="phase1_unsteered.json (single canonical file)")
    p.add_argument("--runs", type=Path, nargs="+", default=[],
                   help="one or more <run_dir> each containing phase2_rescue.json + meta.json")
    p.add_argument("--legacy-run", type=Path, default=None,
                   help="legacy single-arch directory with phase2_rescue.json (no meta.json). "
                        "Combined with --legacy-cell <cell_id> --legacy-feature-id <id>.")
    p.add_argument("--legacy-cell", type=str, default=None)
    p.add_argument("--legacy-feature-id", type=int, default=None)
    p.add_argument("--legacy-feature-mode", type=str, default="pos0")
    p.add_argument("--out", type=Path, required=True,
                   help="output directory; flip_matrix.parquet + mcnemar_table.csv go here")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    phase1 = load_phase1(args.phase1)
    log.info("[phase1] %d unique questions", len(phase1))

    loaded: list[tuple[dict, list[dict]]] = []
    for run_dir in args.runs:
        rescue_path = run_dir / "phase2_rescue.json"
        meta_path = run_dir / "meta.json"
        meta, rescue = load_run(rescue_path, meta_path)
        log.info("[run] %s: %d rows (cell=%s, f=%s, mode=%s)",
                 run_dir.name, len(rescue), meta["cell_id"], meta["feature_id"], meta["feature_mode"])
        loaded.append((meta, rescue))

    if args.legacy_run is not None:
        if not (args.legacy_cell and args.legacy_feature_id is not None):
            raise SystemExit("--legacy-run requires --legacy-cell and --legacy-feature-id")
        rescue = json.loads((args.legacy_run / "phase2_rescue.json").read_text())
        meta = {"cell_id": args.legacy_cell,
                "feature_id": args.legacy_feature_id,
                "feature_mode": args.legacy_feature_mode}
        log.info("[legacy] %s: %d rows (cell=%s)", args.legacy_run, len(rescue), meta["cell_id"])
        loaded.append((meta, rescue))

    df = build_long_df(phase1, loaded)
    log.info("[df] %d rows, archs=%s, mags=%s",
             len(df), sorted(df["arch"].unique()) if len(df) else [],
             sorted(df["magnitude"].unique()) if len(df) else [])

    parquet_path = args.out / "flip_matrix.parquet"
    df.to_parquet(parquet_path, compression="snappy")
    log.info("[saved] %s", parquet_path)

    # McNemar table per arch at per-arch best magnitude
    if len(df):
        bests = best_magnitude_per_arch(df)
        rows = []
        for arch, mag in bests.items():
            conf = confusion_at_mag(df, arch, mag)
            chi2, p = mcnemar_chi2(conf["n_ic"], conf["n_ci"])
            rows.append({**conf, "mcnemar_chi2_cc": chi2, "mcnemar_p_2sided_exact": p})
        mc = pd.DataFrame(rows).sort_values("arch").reset_index(drop=True)
        mc_path = args.out / "mcnemar_table.csv"
        mc.to_csv(mc_path, index=False)
        log.info("[saved] %s", mc_path)
        for _, r in mc.iterrows():
            log.info("  arch=%s mag=%+5.1f n_ic=%d n_ci=%d chi2=%.3f p=%.4f",
                     r["arch"], r["magnitude"], r["n_ic"], r["n_ci"],
                     r["mcnemar_chi2_cc"], r["mcnemar_p_2sided_exact"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
