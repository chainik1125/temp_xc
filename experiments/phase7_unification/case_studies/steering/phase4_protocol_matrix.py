"""Phase 4 (W) — comprehensive protocol matrix at multiple coh thresholds.

Aggregates all V3/V5/V6/right-edge/per-position grades for OBLITERATION
and Cell C T=3, computes mean-curve cliff at coh thresholds in {1.5,
1.75, 2.0, 2.25} and AUC over coh ranges, against the T-SAE k=20 anchor
(mean-curve sd42+sd1).

Run: TQDM_DISABLE=1 .venv/bin/python -m \\
    experiments.phase7_unification.case_studies.steering.phase4_protocol_matrix

Outputs JSON to plots/phase4_protocol_matrix.json
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")


def aggregate(prefix: str, arch: str):
    grades_path = BASE / prefix / arch / "grades.jsonl"
    gens_path = BASE / prefix / arch / "generations.jsonl"
    if not grades_path.exists() or not gens_path.exists():
        return None
    s2sn: dict[tuple[str, float], float] = {}
    for line in gens_path.open():
        r = json.loads(line)
        if "s_norm" in r:
            s2sn[(r["concept_id"], r["strength"])] = r["s_norm"]
    by: dict[float, dict[str, list[float]]] = defaultdict(lambda: {"s": [], "c": []})
    for line in grades_path.open():
        r = json.loads(line)
        if r.get("success_grade") is None or r.get("coherence_grade") is None:
            continue
        sn = s2sn.get((r["concept_id"], r["strength"]))
        if sn is None:
            continue
        by[sn]["s"].append(r["success_grade"])
        by[sn]["c"].append(r["coherence_grade"])
    return [(s, sum(v["s"]) / len(v["s"]), sum(v["c"]) / len(v["c"]))
            for s, v in sorted(by.items())]


def mean_curve(curves):
    if not curves:
        return None
    s_set = set()
    for c in curves:
        s_set |= set(x[0] for x in c)
    out = []
    for s in sorted(s_set):
        items = [x for c in curves for x in c if x[0] == s]
        if items:
            out.append((s,
                        sum(x[1] for x in items) / len(items),
                        sum(x[2] for x in items) / len(items)))
    return out


def cliff(c, thresh):
    if not c:
        return 0.0
    ok = [(s, ms, mc) for s, ms, mc in c if mc >= thresh]
    return max(ok, key=lambda x: x[1])[1] if ok else 0.0


def auc_over_coh(curve, coh_lo, coh_hi):
    if not curve:
        return 0.0
    pts = sorted(curve, key=lambda x: x[2])
    cohs = [p[2] for p in pts]
    succs = [p[1] for p in pts]
    if max(cohs) < coh_lo:
        return 0.0
    coh_grid = np.linspace(coh_lo, coh_hi, 100)
    succ_interp = np.interp(coh_grid, cohs, succs, left=0, right=0)
    return float(np.trapezoid(succ_interp, coh_grid)) / (coh_hi - coh_lo)


CELLS = [
    ("OBLIT (T=2 H8) right-edge", "txc_h8_t2_kpos20_shifts2", "steering_paper_normalised"),
    ("OBLIT (T=2 H8) per-position", "txc_h8_t2_kpos20_shifts2", "steering_paper_window_perposition"),
    ("OBLIT (T=2 H8) V3 dec-additive", "txc_h8_t2_kpos20_shifts2", "steering_paper_window_dec_additive"),
    ("OBLIT (T=2 H8) V5 left-edge", "txc_h8_t2_kpos20_shifts2", "steering_paper_window_left_edge"),
    ("OBLIT (T=2 H8) V6 dec-broadcast", "txc_h8_t2_kpos20_shifts2", "steering_paper_window_dec_broadcast"),
    ("cell C T=3 V1 local", "txc_bare_antidead_t3_kpos20", "steering_paper_window_local"),
    ("cell C T=3 V2 anchored", "txc_bare_antidead_t3_kpos20", "steering_paper_window_anchored"),
    ("cell C T=3 V3 dec-additive", "txc_bare_antidead_t3_kpos20", "steering_paper_window_dec_additive"),
    ("cell C T=3 V4 tiled", "txc_bare_antidead_t3_kpos20", "steering_paper_window_tiled"),
    ("cell C T=3 right-edge", "txc_bare_antidead_t3_kpos20", "steering_paper_normalised"),
    ("cell C T=3 per-position", "txc_bare_antidead_t3_kpos20", "steering_paper_window_perposition"),
]

THRESHOLDS = [1.0, 1.5, 1.75, 2.0, 2.25, 2.5]
AUC_RANGES = [(1.0, 3.0), (1.5, 3.0), (1.75, 3.0), (2.0, 3.0)]


def main():
    # T-SAE anchor
    tsae_42 = aggregate("steering_paper_normalised", "tsae_paper_k20")
    tsae_1 = aggregate("steering_paper_normalised_seed1", "tsae_paper_k20")
    tsae_anchor = mean_curve([tsae_42, tsae_1])

    anchor_cliffs = {t: cliff(tsae_anchor, t) for t in THRESHOLDS}
    anchor_aucs = {f"{lo}-{hi}": auc_over_coh(tsae_anchor, lo, hi) for lo, hi in AUC_RANGES}

    print(f"=== T-SAE k=20 anchor (mean-curve, sd42+sd1) ===")
    for t, v in anchor_cliffs.items():
        print(f"  cliff @ {t}: {v:.3f}")
    for r, v in anchor_aucs.items():
        print(f"  AUC ({r}): {v:.3f}")
    print()

    results = {"anchor_cliffs": anchor_cliffs, "anchor_aucs": anchor_aucs, "cells": []}

    print(f"{'cell + protocol':50s} n  ", end="")
    for t in THRESHOLDS:
        print(f"@{t:>4} ", end="")
    print(f"  ", end="")
    for lo, hi in AUC_RANGES:
        print(f"AUC{lo}-{hi}  ", end="")
    print()

    for name, arch, prefix in CELLS:
        sd1 = aggregate(f"{prefix}_seed1", arch)
        sd2 = aggregate(f"{prefix}_seed2", arch)
        sd42 = aggregate(prefix, arch)
        seeds = [c for c in [sd42, sd1, sd2] if c]
        if not seeds:
            continue
        n = len(seeds)
        mc = mean_curve(seeds)
        cliffs = {t: cliff(mc, t) for t in THRESHOLDS}
        aucs = {f"{lo}-{hi}": auc_over_coh(mc, lo, hi) for lo, hi in AUC_RANGES}

        results["cells"].append({
            "name": name, "arch": arch, "protocol_dir": prefix, "n_seeds": n,
            "cliffs": cliffs, "aucs": aucs,
            "deltas_cliff": {t: cliffs[t] - anchor_cliffs[t] for t in THRESHOLDS},
            "deltas_auc": {r: aucs[r] - anchor_aucs[r] for r in anchor_aucs},
        })

        print(f"  {name:48s} {n:1d}  " + "  ".join(f"{cliffs[t]:>5.3f}" for t in THRESHOLDS), end="")
        print(f"  " + "  ".join(f"{aucs[f'{lo}-{hi}']:>7.3f}" for lo, hi in AUC_RANGES))

    out_path = BASE / "plots" / "phase4_protocol_matrix.json"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {out_path}")


if __name__ == "__main__":
    main()
