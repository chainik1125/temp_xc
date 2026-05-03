"""Multi-arch B3 cut25 sweep driver for the NeurIPS headline plot.

For each headline architecture, looks up the canonical (cell_id, feature_id,
feature_mode) from the per-arch picks below, then invokes b3_variants.py with
the densified magnitude grid from config.yaml. Saves outputs to per-cell
subdirs of `b3_math500_cut25/<cell_id>__f<id>_<mode>/` along with a
meta.json that build_flip_matrix.py reads.

Usage:
  # First, retrain TSAE at k=20 (and re-mine its features). Then:
  python -m experiments.ward_backtracking_txc.run_b3_multi_arch \
      --include-correct 50 \
      --gen-batch-size 36

The TSAE pick is RESOLVED from the new mining output, since the old TSAE
(k=160) features file no longer corresponds to the deployed model. Pass
--tsae-feature explicitly to override.
"""
from __future__ import annotations
import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.run_b3_multi_arch")


# Headline architecture picks. Keep this list short — these are the 4 lines
# in the headline plot. Each tuple is (label, cell_id, feature_id, feature_mode).
# - TXC: matches the canonical b3 default (winner of rank_global_sonnet).
# - TXC-H8: top-mean-diff feature from the H8 mining output.
# - SAE (TopK SAE): rank_global_sonnet top topk_sae cell + best B1 feature.
# - TSAE-paper: resolved at runtime from the freshly re-mined TSAE features.
HEADLINE_ARCHES = [
    {"label": "TXC",      "cell_id": "txc__resid_L10__k16__s42",       "feature_id": 14621, "feature_mode": "pos0"},
    {"label": "TXC-H8",   "cell_id": "txc_h8__resid_L10__k16__s42",    "feature_id":   344, "feature_mode": "pos0"},
    {"label": "SAE",      "cell_id": "topk_sae__ln1_L10__k64__s42",    "feature_id":  5263, "feature_mode": "pos0"},
    # TSAE-paper resolved at runtime; see resolve_tsae_pick().
    {"label": "TSAE-paper", "cell_id": "tsae__resid_L10__k32__s42",   "feature_id": None, "feature_mode": "pos0"},
]


def resolve_tsae_pick(features_dir: Path, cell_id: str, mode: str) -> int:
    """Pick top-mean-diff feature from the freshly mined TSAE npz."""
    fp = features_dir / f"{cell_id}.npz"
    if not fp.exists():
        raise SystemExit(f"TSAE features not mined yet: {fp}")
    z = np.load(fp, allow_pickle=True)
    feats = z["top_features"].tolist()
    if not feats:
        raise SystemExit(f"empty top_features in {fp}")
    return int(feats[0])


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--variant", default="cut25")
    p.add_argument("--gen-batch-size", type=int, default=36)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--include-correct", type=int, default=50,
                   help="N originally-correct questions to also steer (regression cohort).")
    p.add_argument("--correct-seed", type=int, default=42)
    p.add_argument("--archs", type=str, nargs="+", default=None,
                   help="restrict to these labels (TXC, TXC-H8, SAE, TSAE-paper). default: all 4.")
    p.add_argument("--tsae-feature", type=int, default=None,
                   help="override TSAE feature_id (otherwise auto-resolve from mined features).")
    p.add_argument("--out-root", type=Path,
                   default=Path("results/ward_backtracking_txc/b3_math500_cut25"))
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    mags = list(map(float, cfg["steering"]["magnitudes"]))
    log.info("[mags] %d magnitudes from config: %s", len(mags), mags)
    features_dir = Path(cfg["paths"]["features_dir"])

    # Resolve TSAE feature if needed
    plan = []
    for a in HEADLINE_ARCHES:
        if args.archs and a["label"] not in args.archs:
            continue
        entry = dict(a)
        if entry["label"] == "TSAE-paper" and entry["feature_id"] is None:
            entry["feature_id"] = (args.tsae_feature
                                    if args.tsae_feature is not None
                                    else resolve_tsae_pick(features_dir, entry["cell_id"],
                                                            entry["feature_mode"]))
            log.info("[tsae] resolved feature_id=%d for cell=%s",
                     entry["feature_id"], entry["cell_id"])
        plan.append(entry)

    args.out_root.mkdir(parents=True, exist_ok=True)
    rc_total = 0
    for entry in plan:
        cell, fid, mode = entry["cell_id"], entry["feature_id"], entry["feature_mode"]
        run_dir = args.out_root / f"{cell}__f{fid}_{mode}"
        run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = run_dir / "meta.json"
        meta_path.write_text(json.dumps({
            "label": entry["label"],
            "cell_id": cell, "feature_id": int(fid), "feature_mode": mode,
            "magnitudes": mags,
            "include_correct": int(args.include_correct),
        }, indent=2))

        cmd = [
            sys.executable, "-m", "experiments.ward_backtracking_txc.b3_variants",
            "--variant", args.variant,
            "--steering-cell", cell,
            "--feature-id", str(fid),
            "--feature-mode", mode,
            "--magnitudes", *map(str, mags),
            "--max-new-tokens", str(args.max_new_tokens),
            "--gen-batch-size", str(args.gen_batch_size),
            "--include-correct", str(args.include_correct),
            "--correct-seed", str(args.correct_seed),
            "--out", str(run_dir),
        ]
        log.info("=" * 70)
        log.info("[%s] %s f%s %s -> %s", entry["label"], cell, fid, mode, run_dir)
        log.info("[cmd] %s", " ".join(cmd))
        if args.dry_run:
            continue
        rc = subprocess.call(cmd)
        log.info("[%s] exit code=%d", entry["label"], rc)
        rc_total |= rc

    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
