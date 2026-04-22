"""Entry point for the Mess3 Mat-TopK-SAE ablation.

Imports Dmitry's `sae_day.run_driver` and monkey-patches its
`evaluate_one_arch` dispatch to recognize `family="matsae"`, routing to
our `evaluate_matsae_on_activations`. Everything else in his driver —
transformer training, caching, per-cell JSON emission, probe fitting —
runs unchanged.

Usage:
    python run_ablation.py --config config_ablation.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Resolve our evaluator + Dmitry's vendored driver via PYTHONPATH set in
# run_ablation.sh. Import here so any path error surfaces early.
from evaluate_matsae import evaluate_matsae_on_activations  # noqa: E402
from sae_day import run_driver  # type: ignore[import-not-found]  # noqa: E402


def _install_matsae_dispatch() -> None:
    """Wrap `run_driver.run_architecture` so family='matsae' dispatches here.

    Function name + signature match Dmitry's vendored run_driver.py at
    commit 16452d5: `run_architecture(arch_entry, *, seed, train_acts,
    train_acts_ml, eval_acts, eval_acts_ml, eval_omega, d_model, device,
    checkpoint_path=None)`.
    """
    original_dispatch = run_driver.run_architecture

    def patched_dispatch(arch_entry, **kwargs):
        if arch_entry.get("family") == "matsae":
            # Reuse _merge_kwargs so our config entry picks up the same
            # defaults layer as every other arch.
            arch_cfg = run_driver._merge_kwargs(arch_entry["name"], arch_entry.get("kwargs", {}))
            return evaluate_matsae_on_activations(
                seed=kwargs["seed"],
                arch_cfg=arch_cfg,
                train_acts=kwargs["train_acts"],
                eval_acts=kwargs["eval_acts"],
                eval_omega=kwargs["eval_omega"],
                d_model=kwargs["d_model"],
                device=kwargs["device"],
                checkpoint_path=kwargs.get("checkpoint_path"),
            )
        return original_dispatch(arch_entry, **kwargs)

    run_driver.run_architecture = patched_dispatch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, required=True,
                   help="Path to config YAML (e.g. config_ablation.yaml).")
    args = p.parse_args()

    _install_matsae_dispatch()

    # Delegate to Dmitry's driver main. It parses the YAML, loops cells,
    # trains/caches transformers, dispatches per arch, and writes per-cell
    # JSONs to `output_root`.
    sys.argv = ["run_driver.py", "--config", str(args.config)]
    return run_driver.main()


if __name__ == "__main__":
    sys.exit(main() or 0)
