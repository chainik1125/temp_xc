"""Per-arch steering-magnitude calibration via 95th-percentile feature activation.

Andre's idea, Dmitry-approved (May 02 meeting): cross-arch magnitudes are
unitless until normalized. For each (arch, steered feature), take the 95th
percentile of the feature's activation values over the eval-set firing
positions. Define "calibrated magnitude 1.0" per arch as 1 × that 95th-pctile.

Inputs: mined features npz at results/.../features/<cell_id>.npz, which
already contains `pos_act` (n_pos_sentences × |top_features|) and `neg_act`
(n_neg × |top_features|). Pool both, take 95th of nonzero values.

Output: calibration.json schema:
  {
    "<cell_id>__f<id>_<mode>": {
        "label": "TXC", "cell_id": "...", "feature_id": int,
        "feature_mode": "pos0"|"union",
        "p95_pos_only": float, "p95_neg_only": float, "p95_pooled": float,
        "n_pos": int, "n_neg": int, "n_zero_in_pool": int
    }, ...
  }

Usage:
  python -m experiments.ward_backtracking_txc.calibrate_magnitudes \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --out <out_root>/calibration.json
"""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.calibrate")


def p95_of_nonzero(arr: np.ndarray) -> tuple[float, int, int]:
    """Return (p95_of_abs_nonzero, n_total, n_zero).

    Note: takes 95th percentile of |values| filtered to nonzero. Originally
    we only included strictly-positive values, which works for TopK SAEs
    (codes are nonneg via ReLU+TopK) but breaks for TFA/TSAE — those
    architectures return signed reconstruction residuals (pred_codes +
    novel_codes), so most values are slightly negative even at the
    "top mean-diff" feature. For the calibration use case we want the
    natural *scale* of the activations, not their sign.
    """
    flat = np.asarray(arr).ravel().astype(np.float32)
    abs_flat = np.abs(flat)
    nonzero = abs_flat[abs_flat > 0]
    n_total, n_zero = int(flat.size), int((flat == 0).sum())
    if nonzero.size == 0:
        return 0.0, n_total, n_zero
    return float(np.percentile(nonzero, 95)), n_total, n_zero


def calibrate_one(features_dir: Path, cell_id: str, feature_id: int, mode: str) -> dict:
    npz_path = features_dir / f"{cell_id}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"missing features npz: {npz_path}")
    z = np.load(npz_path, allow_pickle=True)
    feats = z["top_features"].tolist()
    if feature_id not in feats:
        raise ValueError(f"feature {feature_id} not in {cell_id} top_features {feats[:8]}...")
    idx = feats.index(feature_id)
    pos_col = z["pos_act"][:, idx]
    neg_col = z["neg_act"][:, idx]
    p95_pos, n_pos, n_pos_zero = p95_of_nonzero(pos_col)
    p95_neg, n_neg, n_neg_zero = p95_of_nonzero(neg_col)
    pooled = np.concatenate([pos_col, neg_col])
    p95_pooled, n_pool, n_pool_zero = p95_of_nonzero(pooled)
    # L2-of-decoder calibration. The steering vector is normalized to
    # DoM-norm at b3 time (see b3_math500_rescue.py:normalize_to_dom_norm),
    # so the L2 of decoder_at_pos0 captures the per-feature direction
    # length BEFORE that normalization. This is what web-claude
    # recommended as the right alternative to the broken p95 calibration:
    # "consistent units of model-space distance per unit magnitude" rather
    # than "per-arch activation scale" which is incommensurable across
    # signed-residual archs (TFA / TSAE-paper) vs. TopK arches.
    decoder_key = "decoder_at_pos0" if mode == "pos0" else "decoder_union"
    if decoder_key not in z:
        l2_decoder_pos0 = float("nan")
    else:
        l2_decoder_pos0 = float(np.linalg.norm(z[decoder_key][idx]))
    # Always compute pos0 too for consistency
    l2_decoder_pos0_only = float(np.linalg.norm(z["decoder_at_pos0"][idx])) if "decoder_at_pos0" in z else float("nan")
    l2_decoder_union = float(np.linalg.norm(z["decoder_union"][idx])) if "decoder_union" in z else float("nan")
    return {
        "cell_id": cell_id,
        "feature_id": int(feature_id),
        "feature_mode": mode,
        "p95_pos_only": p95_pos,
        "p95_neg_only": p95_neg,
        "p95_pooled": p95_pooled,
        "l2_decoder_for_mode": l2_decoder_pos0,   # primary; matches the steered mode
        "l2_decoder_pos0": l2_decoder_pos0_only,
        "l2_decoder_union": l2_decoder_union,
        "n_pos": n_pos, "n_neg": n_neg, "n_zero_in_pool": n_pool_zero,
        "frac_zero_pooled": n_pool_zero / max(n_pool, 1),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--runs", type=Path, nargs="+", required=True,
                   help="run directories (each containing meta.json with cell_id, feature_id, feature_mode)")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    features_dir = Path(cfg["paths"]["features_dir"])

    out: dict[str, dict] = {}
    for run_dir in args.runs:
        meta = json.loads((run_dir / "meta.json").read_text())
        key = f"{meta['cell_id']}__f{meta['feature_id']}_{meta['feature_mode']}"
        try:
            entry = calibrate_one(features_dir, meta["cell_id"], int(meta["feature_id"]), meta["feature_mode"])
            entry["label"] = meta.get("label", "?")
            out[key] = entry
            log.info("[%s] cell=%s f=%d mode=%s  p95_pooled=%.4f  L2_decoder=%.4f",
                     entry["label"], entry["cell_id"], entry["feature_id"],
                     entry["feature_mode"], entry["p95_pooled"],
                     entry["l2_decoder_for_mode"])
        except (FileNotFoundError, ValueError) as e:
            log.warning("[skip] %s: %s", run_dir, e)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    log.info("[saved] %s (%d entries)", args.out, len(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
