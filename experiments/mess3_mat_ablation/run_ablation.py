"""Entry point for the Mess3 Mat-TopK-SAE ablation.

Imports Dmitry's `sae_day.run_driver` and monkey-patches:
  1. `run_architecture` to recognize `family="matsae"` and route to our
     `evaluate_matsae_on_activations`.
  2. `run_architecture` (again) to capture
     `metrics["single_feature_probe"]["per_component_best_feature"]`,
     which Dmitry computes but discards before serialization.
  3. `run_one_cell` to inject the captured argmax feature IDs into the
     saved `results.json` and the in-memory payload (so `combined.json`
     also picks them up).

Everything else in the driver — transformer training, caching, probe
fitting — runs unchanged.

Usage:
    python run_ablation.py --config config_ablation.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evaluate_matsae import evaluate_matsae_on_activations  # noqa: E402
from sae_day import run_driver  # type: ignore[import-not-found]  # noqa: E402


# Per-cell scratchpad: arch name → per-component argmax feature IDs.
# Reset at the start of each cell by the run_one_cell wrapper.
_CAPTURED_FEAT_IDS: dict[str, list[int]] = {}


def _install_matsae_dispatch_and_capture() -> None:
    """Wrap `run_driver.run_architecture` for matsae dispatch + feature-ID capture."""
    original_dispatch = run_driver.run_architecture

    def patched_dispatch(arch_entry, **kwargs):
        if arch_entry.get("family") == "matsae":
            arch_cfg = run_driver._merge_kwargs(arch_entry["name"], arch_entry.get("kwargs", {}))
            run = evaluate_matsae_on_activations(
                seed=kwargs["seed"],
                arch_cfg=arch_cfg,
                train_acts=kwargs["train_acts"],
                eval_acts=kwargs["eval_acts"],
                eval_omega=kwargs["eval_omega"],
                d_model=kwargs["d_model"],
                device=kwargs["device"],
                checkpoint_path=kwargs.get("checkpoint_path"),
            )
        else:
            run = original_dispatch(arch_entry, **kwargs)

        # Capture per-component argmax feature IDs (always present in
        # Dmitry's summarize_single_feature_probe output, just not
        # otherwise serialized).
        m = run.get("metrics", {}).get("single_feature_probe", {})
        ids = m.get("per_component_best_feature")
        if ids is not None:
            _CAPTURED_FEAT_IDS[arch_entry["name"]] = [int(i) for i in ids]
        return run

    run_driver.run_architecture = patched_dispatch


def _install_run_one_cell_augmenter() -> None:
    """Wrap `run_driver.run_one_cell` to inject captured feature IDs into results.

    Mutates both:
      - the on-disk `cell_dir/results.json` (so re-runs with --skip-existing
        pick them up via the json read in main()).
      - the returned payload dict (so `combined.json` carries them).
    """
    original_run_one_cell = run_driver.run_one_cell

    def wrapped(*args, **kwargs):
        global _CAPTURED_FEAT_IDS
        _CAPTURED_FEAT_IDS = {}
        payload = original_run_one_cell(*args, **kwargs)
        cell_dir = kwargs.get("cell_dir")
        if cell_dir is None and len(args) > 2:
            cell_dir = args[2]
        for arch_name, feat_ids in _CAPTURED_FEAT_IDS.items():
            arch_dict = payload.get("architectures", {}).get(arch_name)
            if arch_dict is not None:
                arch_dict["per_component_best_feature"] = feat_ids
        if cell_dir is not None:
            results_json = Path(cell_dir) / "results.json"
            if results_json.exists():
                results_json.write_text(json.dumps(payload, indent=2))
        return payload

    run_driver.run_one_cell = wrapped


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True,
                   help="Path to config YAML (e.g. config_ablation.yaml).")
    args = p.parse_args()

    _install_matsae_dispatch_and_capture()
    _install_run_one_cell_augmenter()

    sys.argv = ["run_driver.py", "--config", str(args.config)]
    return run_driver.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
