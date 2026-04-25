"""Fig 8/9 grid: HMM noisy-emission denoising benchmark.

Setup from report §3.3:
    n_features=40, d_model=80, d_sae=80, pi=0.15
    Heterogeneous rho: 10 features each at rho in {0.1, 0.4, 0.7, 0.95}
    Stochastic emissions: p_A=0, p_B=0.625 (gamma ~= 0.59)
    T in {2, 3, 4, 5, 6, 8, 10, 12}, k in {1, 3, 5}

For each (model, T, k) cell, trains the model and computes:
  - Standard feature-recovery AUC/NMSE/L0 against decoder cosine.
  - Per-feature local/global Pearson correlations (best-match latent vs s, h).
  - Per-feature Ridge-probe R^2 from full latent vector to s (local) and h (global).

Writes `results/hmm_denoising/sweep_results.json` consumed by plot_fig8 /
plot_fig9. Per-feature lists are retained so plotting can do per-rho-group
breakdowns.
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
from temporal_bench.metrics import evaluate, evaluate_denoising
from temporal_bench.sweep import _create_model, _should_skip
from temporal_bench.train import train
from temporal_bench.utils import get_device, set_seed


# Heterogeneous-rho groups from the report.
RHO_GROUPS = [0.1, 0.4, 0.7, 0.95]
GROUP_SIZE = 10  # features per group
DEFAULT_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]
DEFAULT_K_VALUES = [1, 3, 5]
DEFAULT_MODELS = ["regular_sae", "stacked_sae", "txcdr", "regular_sae_kT"]


def build_rho_per_feature() -> list[float]:
    return [r for r in RHO_GROUPS for _ in range(GROUP_SIZE)]


def run(
    models: list[str],
    T_values: list[int],
    k_values: list[int],
    train_cfg: TrainConfig,
    data_cfg: DataConfig,
    output_dir: str,
    n_eval_sequences: int = 200,
) -> list[dict]:
    device = get_device()
    pipeline = DataPipeline(data_cfg, device=device)

    combos = list(product(models, T_values, k_values))
    print(f"HMM denoising sweep: {len(combos)} cells")
    print(f"Device: {device}")
    print(
        f"Data: n_features={data_cfg.n_features}, d_model={data_cfg.d_model}, "
        f"pi={data_cfg.pi}, p_A={data_cfg.p_A}, p_B={data_cfg.p_B}, "
        f"hetero_rho={data_cfg.rho_per_feature}"
    )

    # Incremental-save path: flushed after every completed cell.
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "sweep_results.json")
    tmp_path = out_path + ".tmp"

    results = []
    for model_name, T, k in combos:
        if _should_skip(model_name, k, T, data_cfg.n_features):
            continue
        set_seed(train_cfg.seed)
        print(f"\n--- {model_name} | T={T} | k={k} ---")

        eval_x, eval_s, eval_h = pipeline.eval_data_with_support(
            n_sequences=n_eval_sequences, T=T, rho=0.0, seed=9999
        )

        def data_fn(batch_size: int, _T=T) -> torch.Tensor:
            return pipeline.sample_windows(batch_size, _T, rho=0.0)

        model = _create_model(
            model_name,
            d_in=data_cfg.d_model,
            d_sae=data_cfg.n_features,  # d_sae == n_features per report §3.3
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

        final = evaluate(model, eval_x, pipeline.true_features)
        denoise = evaluate_denoising(
            model,
            eval_x,
            eval_s,
            eval_h,
            pipeline.true_features,
            feature_rho=data_cfg.rho_per_feature,
        )

        print(
            f"  AUC={final.auc:.3f}  L0={final.l0:.1f}  "
            f"corr_local={denoise.corr_local:.3f}  corr_global={denoise.corr_global:.3f}  "
            f"ratio(corr)={denoise.denoising_ratio_corr:.3f}  "
            f"ratio(R^2)={denoise.denoising_ratio_r2:.3f}"
        )

        results.append(
            {
                "model": model_name,
                "T": T,
                "k": k,
                "seed": train_cfg.seed,
                "nmse": final.nmse,
                "l0": final.l0,
                "auc": final.auc,
                "r_at_90": final.r_at_90,
                "r_at_80": final.r_at_80,
                "mean_max_cos": final.mean_max_cos,
                "corr_local": denoise.corr_local,
                "corr_global": denoise.corr_global,
                "r2_local": denoise.r2_local,
                "r2_global": denoise.r2_global,
                "denoising_ratio_corr": denoise.denoising_ratio_corr,
                "denoising_ratio_r2": denoise.denoising_ratio_r2,
                "corr_local_per_feature": denoise.corr_local_per_feature,
                "corr_global_per_feature": denoise.corr_global_per_feature,
                "r2_local_per_feature": denoise.r2_local_per_feature,
                "r2_global_per_feature": denoise.r2_global_per_feature,
                "feature_rho": denoise.feature_rho,
            }
        )

        # Atomic incremental save (write to temp, rename over the target).
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=2)
        os.replace(tmp_path, out_path)

    print(f"\nResults saved to {out_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=65_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--p-A", type=float, default=0.0)
    parser.add_argument("--p-B", type=float, default=0.625)
    parser.add_argument(
        "--output-dir", type=str, default="results/hmm_denoising"
    )
    parser.add_argument("--n-eval-sequences", type=int, default=200)
    parser.add_argument(
        "--models", nargs="+", default=DEFAULT_MODELS,
        help="Subset of models to run. Useful for adding a new arch to existing results.",
    )
    args = parser.parse_args()

    data_cfg = DataConfig(
        n_features=len(RHO_GROUPS) * GROUP_SIZE,  # 40
        d_model=80,
        pi=0.15,
        seed=args.data_seed,
        p_A=args.p_A,
        p_B=args.p_B,
        rho_per_feature=build_rho_per_feature(),
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
        T_values=DEFAULT_T_VALUES,
        k_values=DEFAULT_K_VALUES,
        train_cfg=train_cfg,
        data_cfg=data_cfg,
        output_dir=args.output_dir,
        n_eval_sequences=args.n_eval_sequences,
    )


if __name__ == "__main__":
    main()
