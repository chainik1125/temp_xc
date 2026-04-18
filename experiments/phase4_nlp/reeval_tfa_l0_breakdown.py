"""Re-evaluate existing TFA-pos checkpoints to extract the l0_novel / l0_pred split.

Updates results JSONs in-place — adds `l0_components` field to each TFA row.
Does NOT re-train; uses the saved checkpoint state.

Usage:
    python scripts/reeval_tfa_l0_breakdown.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")

from src.architectures.tfa import TFASpec
from src.architectures._tfa_module import TemporalSAE
from src.data.nlp.loader import build_cached_activations_pipeline
from src.training.config import DataConfig
from src.eval.runner import evaluate_model


def reeval_one(ckpt_path: str, row: dict, device) -> dict:
    """Re-run evaluation for one TFA checkpoint, return l0_components dict."""
    # Reconstruct the spec (match the sweep's defaults for this layer)
    use_pos = row["arch"] == "TFA-pos"
    spec = TFASpec(use_pos_encoding=use_pos, bottleneck_factor=8)
    model = spec.create(
        d_in=row["d_in"], d_sae=row["d_sae"], k=row["k"], device=device,
    )
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    # Build the same pipeline used in the sweep (for eval_hidden)
    config = DataConfig(
        dataset_type="cached_activations",
        model_name=row["subject_model"],
        cached_dataset=row["cached_dataset"],
        cached_layer_key=row["layer_key"],
        shuffle_within_sequence=bool(row["shuffled"]),
        d_sae=row["d_sae"],
    )
    # TFA's _scale caches scaling_factor on first call. We need to prime it
    # with an unshuffled batch matching training distribution.
    pipeline = build_cached_activations_pipeline(config, device, window_sizes=None)

    # Prime scaling by calling gen_seq once (same distribution as training)
    with torch.no_grad():
        _ = spec._scale(pipeline.gen_seq(4))

    result = evaluate_model(
        spec, model, pipeline.eval_hidden, device,
        seq_len=pipeline.eval_hidden.shape[1],
    )
    return {
        "nmse": result.nmse,
        "l0": result.l0,
        "l0_components": result.l0_components,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    results_dirs = [
        "/home/elysium/temp_xc/results/nlp_sweep/gemma",
        "/home/elysium/temp_xc/results/nlp_sweep/deepseek",
    ]

    for results_dir in results_dirs:
        p = Path(results_dir)
        if not p.exists():
            print(f"  [skip] {results_dir} not found")
            continue

        for json_path in sorted(p.glob("results_*.json")):
            with open(json_path) as f:
                rows = json.load(f)

            any_updated = False
            for row in rows:
                if not row["arch"].startswith("TFA"):
                    continue
                if row.get("l0_components"):
                    continue  # already updated
                ckpt = row.get("checkpoint")
                if not ckpt or not os.path.exists(ckpt):
                    print(f"  [skip] missing checkpoint: {ckpt}")
                    continue

                print(f"  [reeval] {ckpt}")
                try:
                    out = reeval_one(ckpt, row, device)
                except Exception as e:
                    print(f"    ERROR: {e}")
                    continue

                # Sanity-check NMSE matches original (within 1%)
                old_nmse = row["nmse"]
                new_nmse = out["nmse"]
                if abs(new_nmse - old_nmse) / max(old_nmse, 1e-6) > 0.01:
                    print(
                        f"    WARNING: NMSE drift {old_nmse:.4f} -> {new_nmse:.4f}"
                    )

                row["l0_components"] = out["l0_components"]
                any_updated = True
                print(
                    f"    novel={out['l0_components']['l0_novel']:.1f} "
                    f"pred={out['l0_components']['l0_pred']:.1f}"
                )

            if any_updated:
                with open(json_path, "w") as f:
                    json.dump(rows, f, indent=2)
                print(f"  Saved {json_path}")


if __name__ == "__main__":
    main()
