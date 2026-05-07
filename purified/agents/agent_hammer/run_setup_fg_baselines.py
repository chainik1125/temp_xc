"""Setup F + G baseline backfill driver.

Fills the missing-baseline gaps tagged in agent_synth's handover:
- Setup F (coupled + obs noise σ ∈ {0.5, 1.0, 2.0}): missing
  ``tsae_paper``, ``tfa_pos``, ``stacked_sae`` T=2 + T=5.
- Setup G (hierarchical + obs noise σ ∈ {1.0, 2.0}): same gap.

Reuses ``make_train_fn`` / ``make_eval_fn`` from agent_synth's
``run_setup_f.py`` / ``run_setup_g.py`` (imports only — no edits).
The dispatch key is the datasource name prefix.

Fair-comparison knobs (matching agent_synth's existing F + G cells):
- n_steps=8000, batch_size=1024
- k_pos sweep = {1, 2, 3}
- 3 seeds = {1, 2, 42}
- d_sae=40 toy regime (override for tsae_paper which has d_sae=16384
  in YAML at component=c2)

Per-arch knobs:
- tfa_pos:    no T axis. arch_hparams_override = {k_pos}.
- stacked_sae T=2 / T=5: arch_hparams_override = {k_pos, T}.
- tsae_paper: train_window_size=2 (paper-faithful adjacent-pair).
              arch_hparams_override = {k_pos, d_sae=40}.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")

from temp_bench import runner
from temp_bench.schemas import TrainingConfig

# Import agent_synth's drivers (read-only — territory respected).
from experiments.c2_synthetic_coupled.run_setup_f import (
    make_train_fn as f_make_train,
    make_eval_fn  as f_make_eval,
)
from experiments.c2_hierarchical.run_setup_g import (
    make_train_fn as g_make_train,
    make_eval_fn  as g_make_eval,
)

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


def _is_setup_f(datasource: str) -> bool:
    return datasource.startswith("toy_coupled_obs_noise_")


def _is_setup_g(datasource: str) -> bool:
    return datasource.startswith("toy_hierarchical_") and "sigma" in datasource


def _make_fns(datasource: str):
    if _is_setup_f(datasource):
        return f_make_train(datasource), f_make_eval(datasource), "F"
    if _is_setup_g(datasource):
        return g_make_train(datasource), g_make_eval(datasource), "G"
    raise SystemExit(f"Unknown datasource '{datasource}' (not F or G).")


def _build_override(arch: str, k_pos: int, T: int | None) -> dict[str, Any]:
    """Per-arch override. d_sae=40 is the toy regime; ALL three baselines
    have d_sae=18432 in locked_archs.yaml at c2 (no per_component_hparams
    .c2 override), so we MUST set d_sae=40 explicitly or tfa_pos OOMs at
    67+ GB and the recovery metrics become incomparable to the d_sae=40
    Setup A / B / D / E cells."""
    if arch == "tsae_paper":
        return {"k_pos": int(k_pos), "d_sae": 40}
    if arch == "stacked_sae":
        if T is None:
            raise SystemExit("stacked_sae requires --T.")
        return {"k_pos": int(k_pos), "T": int(T), "d_sae": 40}
    if arch == "tfa_pos":
        # Per-token, no T axis. d_sae=40 critical (else tfa_pos OOMs).
        return {"k_pos": int(k_pos), "d_sae": 40}
    raise SystemExit(f"Unsupported arch '{arch}' (this driver only handles "
                     "tfa_pos, stacked_sae, tsae_paper).")


def _t_label(arch: str, T: int | None) -> str:
    if arch == "stacked_sae" and T is not None:
        return f"T={T}"
    return "default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["tfa_pos", "stacked_sae", "tsae_paper"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--T", type=int, default=None,
                    help="Required for stacked_sae; ignored for others.")
    ap.add_argument("--k-poses", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    train_fn, eval_fn, setup_letter = _make_fns(args.datasource)

    train_window_size = 2 if args.arch == "tsae_paper" else None

    for k_pos in args.k_poses:
        override = _build_override(args.arch, int(k_pos), args.T)
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
            train_window_size=train_window_size,
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": False,
            "_arch_hparams_override": override,
            "t_label": _t_label(args.arch, args.T),
            "n_steps_train": int(args.n_steps),
            "setup": setup_letter,
        }
        print(f"[fg_baselines] {args.arch:12s} T={args.T} k={k_pos:2d} "
              f"seed={args.seed} ds={args.datasource}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=args.arch,
            seed=int(args.seed),
            datasource_name=args.datasource,
            training_cfg=cfg,
            eval_cfg=eval_cfg,
            eval_protocol_version=EVAL_PROTOCOL_VERSION,
            train_fn=train_fn,
            eval_fn=eval_fn,
        )


if __name__ == "__main__":
    main()
