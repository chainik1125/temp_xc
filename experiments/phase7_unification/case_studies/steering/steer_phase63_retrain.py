"""Wrapper to steer the phase63_track2_t20_retrain ckpt on its native model.

The Phase 6.3 retrain was trained on Gemma-2-2b-IT L13 (NOT base@L12 like
every other Phase 7 ckpt). This wrapper monkey-patches the steering
modules' SUBJECT_MODEL / ANCHOR_LAYER constants before calling the
existing select_for_arch + steer_for_arch functions so they target the
right subject model + layer for this one ckpt.

Outputs:
    results/case_studies/steering/<arch>/feature_selection.json
    results/case_studies/steering_paper/<arch>/{generations,grades}.jsonl

Usage:
    PHASE7_REPO=/root/temp_xc HF_HOME=/root/.cache/huggingface TQDM_DISABLE=1 \
    .venv/bin/python -m experiments.phase7_unification.case_studies.steering.steer_phase63_retrain
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")

ARCH_ID = "phase63_track2_t20_retrain"
SUBJECT_MODEL_OVERRIDE = "google/gemma-2-2b-it"
ANCHOR_LAYER_OVERRIDE = 13


def _patch(module_name: str) -> None:
    import importlib
    m = importlib.import_module(module_name)
    if hasattr(m, "SUBJECT_MODEL"):
        print(f"  patch {module_name}.SUBJECT_MODEL: {m.SUBJECT_MODEL!r} → {SUBJECT_MODEL_OVERRIDE!r}")
        m.SUBJECT_MODEL = SUBJECT_MODEL_OVERRIDE
    if hasattr(m, "ANCHOR_LAYER"):
        print(f"  patch {module_name}.ANCHOR_LAYER: {m.ANCHOR_LAYER} → {ANCHOR_LAYER_OVERRIDE}")
        m.ANCHOR_LAYER = ANCHOR_LAYER_OVERRIDE


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-feature-selection", action="store_true",
                    help="reuse existing feature_selection.json if present.")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    print(f"=== steering wrapper for {ARCH_ID} ===")
    print(f"    subject_model = {SUBJECT_MODEL_OVERRIDE}")
    print(f"    anchor_layer  = L{ANCHOR_LAYER_OVERRIDE}")
    print()

    # Patch every module that captures SUBJECT_MODEL / ANCHOR_LAYER at import.
    for mod in (
        "experiments.phase7_unification.case_studies.steering.select_features",
        "experiments.phase7_unification.case_studies.steering.intervene_paper_clamp_window",
    ):
        _patch(mod)

    # Now run the pipeline.
    if not args.skip_feature_selection:
        print(f"\n=== feature selection ===")
        from experiments.phase7_unification.case_studies.steering import select_features
        select_features.select_for_arch(ARCH_ID, batch_size=16)

    print(f"\n=== window-clamp intervention ===")
    from experiments.phase7_unification.case_studies.steering import intervene_paper_clamp_window
    intervene_paper_clamp_window.steer_for_arch(ARCH_ID, force=args.force)


if __name__ == "__main__":
    main()
