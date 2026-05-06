"""Coupled-features complexity sweep: gAUC vs n_parents.

Tests the hypothesis from part 3 of the synthetic-bench brief:

    "As HMM complexity grows (more hidden chains conflated into each
     emission), TXCDR's global-recovery advantage over a token-local
     SAE should grow, because per-token inversion of the coupling
     becomes more ill-posed and temporal context matters more."

Sweeps `n_parents in {1, 2, 3, 5, 7, 10}` (the complexity knob) crossed
with T x k x model x seed. n_parents=1 is the degenerate case where
emission directions equal hidden directions; n_parents=K = full
coupling.

Setup mirrors Dmitry's exp1c3:
    K=10 hidden chains, M=20 emission features, rho=0.7, pi=0.15.

For each cell, evaluates against BOTH emission-feature directions
(eAUC, "local") and aggregated hidden-feature directions (gAUC,
"global"). Headline plot is gAUC and ΔgAUC vs n_parents.

Writes results/coupled_complexity/sweep_results.json. Plot with
scripts/plot_fig_complexity.py.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from itertools import product

import torch

sys.path.insert(0, "src")

from temporal_bench.config import DataConfig, TrainConfig
from temporal_bench.data.pipeline import DataPipeline
from temporal_bench.metrics import evaluate
from temporal_bench.sweep import _create_model, _should_skip
from temporal_bench.train import train
from temporal_bench.utils import get_device, set_seed


# Fixed setup mirroring exp1c3.
K_HIDDEN = 10
M_EMISSION = 20
RHO = 0.7
PI = 0.15
DEFAULT_N_PARENTS = [1, 2, 3, 5, 7, 10]
DEFAULT_T_VALUES = [2, 5, 10]
DEFAULT_K_VALUES = [1, 3]
DEFAULT_MODELS = ["regular_sae", "regular_sae_kT", "txcdr"]


def run(
    models: list[str],
    n_parents_values: list[int],
    T_values: list[int],
    k_values: list[int],
    train_cfg: TrainConfig,
    base_data_cfg: DataConfig,
    output_dir: str,
    n_eval_sequences: int = 200,
    n_seeds: int = 1,
) -> list[dict]:
    device = get_device()

    combos = list(product(n_parents_values, models, T_values, k_values, range(n_seeds)))
    print(f"Coupled complexity sweep: {len(combos)} cells (n_seeds={n_seeds})")
    print(f"Device: {device}")
    print(
        f"Fixed: K={K_HIDDEN} hidden, M={M_EMISSION} emissions, "
        f"rho={RHO}, pi={PI}"
    )
    print(f"n_parents grid: {n_parents_values}")
    print(f"T grid: {T_values}, k grid: {k_values}")
    print(f"models: {models}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sweep_results.json")
    tmp_path = out_path + ".tmp"

    # One pipeline per n_parents value; the coupling matrix is part of the
    # generative model so it's regenerated for each n_parents (and reseeded
    # across seed_idx so multi-seed results sample independent couplings).
    results: list[dict] = []
    for n_parents in n_parents_values:
        for seed_idx in range(n_seeds):
            data_cfg = DataConfig(
                n_features=M_EMISSION,
                d_model=base_data_cfg.d_model,
                pi=PI,
                seed=base_data_cfg.seed + seed_idx,
                n_hidden=K_HIDDEN,
                n_parents=n_parents,
            )
            pipeline = DataPipeline(data_cfg, device=device)
            for model_name, T, k in product(models, T_values, k_values):
                if _should_skip(model_name, k, T, M_EMISSION):
                    continue
                seed = train_cfg.seed + seed_idx
                set_seed(seed)
                print(
                    f"\n--- {model_name} | n_parents={n_parents} | T={T} | "
                    f"k={k} | seed={seed} ---"
                )

                eval_x, eval_s, _eval_h = pipeline.eval_data_with_support(
                    n_sequences=n_eval_sequences, T=T, rho=RHO, seed=9999 + seed_idx
                )

                def data_fn(batch_size: int, _T=T) -> torch.Tensor:
                    return pipeline.sample_windows(batch_size, _T, rho=RHO)

                model = _create_model(
                    model_name,
                    d_in=base_data_cfg.d_model,
                    d_sae=M_EMISSION,
                    T=T,
                    k=k,
                    device=device,
                )
                train(
                    model=model,
                    data_fn=data_fn,
                    config=train_cfg,
                    eval_data=eval_x,
                    true_features=pipeline.true_features,
                    silent=False,
                )

                final = evaluate(
                    model,
                    eval_x,
                    pipeline.true_features,
                    eval_s=eval_s,
                    hidden_features=pipeline.hidden_features,
                )

                print(
                    f"  NMSE={final.nmse:.3f}  L0={final.l0:.1f}  "
                    f"eAUC={final.auc:.3f}  gAUC={final.auc_hidden:.3f}  "
                    f"d(loc/glo)={final.auc_decoder_local:.3f}/"
                    f"{final.auc_decoder_global:.3f}"
                )

                results.append(
                    {
                        "model": model_name,
                        "n_parents": n_parents,
                        "T": T,
                        "k": k,
                        "rho": RHO,
                        "K_hidden": K_HIDDEN,
                        "M_emission": M_EMISSION,
                        "seed": seed,
                        "nmse": final.nmse,
                        "l0": final.l0,
                        "auc": final.auc,
                        "r_at_90": final.r_at_90,
                        "r_at_80": final.r_at_80,
                        "mean_max_cos": final.mean_max_cos,
                        "auc_decoder_local": final.auc_decoder_local,
                        "auc_decoder_global": final.auc_decoder_global,
                        "auc_activation_local": final.auc_activation_local,
                        "auc_activation_global": final.auc_activation_global,
                        "auc_hidden": final.auc_hidden,
                        "r_at_90_hidden": final.r_at_90_hidden,
                        "r_at_80_hidden": final.r_at_80_hidden,
                        "mean_max_cos_hidden": final.mean_max_cos_hidden,
                    }
                )

                with open(tmp_path, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(tmp_path, out_path)

    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--n-parents", type=int, nargs="+", default=DEFAULT_N_PARENTS,
        help="Complexity axis: parents per emission.",
    )
    parser.add_argument("--T", type=int, nargs="+", default=DEFAULT_T_VALUES)
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--n-seeds", type=int, default=1)
    parser.add_argument(
        "--output-dir", type=str, default="results/coupled_complexity"
    )
    parser.add_argument("--n-eval-sequences", type=int, default=200)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
    )
    parser.add_argument("--d-model", type=int, default=80)
    args = parser.parse_args()

    base_data_cfg = DataConfig(
        n_features=M_EMISSION,
        d_model=args.d_model,
        pi=PI,
        seed=args.data_seed,
        n_hidden=K_HIDDEN,
        n_parents=DEFAULT_N_PARENTS[0],  # placeholder, overridden per cell
    )
    train_cfg = TrainConfig(
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=3e-4,
        grad_clip=1.0,
        eval_every=max(args.steps // 10, 1),
        seed=args.seed,
    )
    run(
        models=args.models,
        n_parents_values=args.n_parents,
        T_values=args.T,
        k_values=args.k,
        train_cfg=train_cfg,
        base_data_cfg=base_data_cfg,
        output_dir=args.output_dir,
        n_eval_sequences=args.n_eval_sequences,
        n_seeds=args.n_seeds,
    )


if __name__ == "__main__":
    main()
