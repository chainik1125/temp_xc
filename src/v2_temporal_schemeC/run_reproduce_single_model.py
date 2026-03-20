"""Run a single model through Exp 1 (TopK) and Exp 2 (L1). For parallel execution.

Usage:
  python src/v2_temporal_schemeC/run_reproduce_single_model.py <model_name> <output_json>

Model names: SAE, TFA, TFA-shuf, TFA-pos, TFA-pos-shuf, TXCDR-T2, TXCDR-T5
"""

import json
import os
import sys
import time
from dataclasses import asdict

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.utils.seed import set_seed
from src.v2_temporal_schemeC.experiment.data_pipeline import DataConfig, build_data_pipeline
from src.v2_temporal_schemeC.experiment.model_specs import (
    SAEModelSpec, TFAModelSpec, TXCDRModelSpec, ModelEntry,
)
from src.v2_temporal_schemeC.experiment.eval_unified import evaluate_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CFG = DataConfig(
    num_features=20, hidden_dim=40, seq_len=64,
    pi=[0.5] * 20,
    rho=[0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4,
    dict_width=40, seed=42, eval_n_seq=2000,
)

TOPK_K_VALUES = [1, 3, 5, 8, 10, 15, 20]
L1_COEFFS_SAE = np.logspace(-2.3, 1.3, 15).tolist()
L1_COEFFS_TFA = np.logspace(-0.8, 1.8, 12).tolist()
L1_COEFFS_TXCDR = np.logspace(-1.5, 1.5, 12).tolist()

MODEL_DEFS = {
    "SAE": (SAEModelSpec(), "flat",
            {"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4},
            L1_COEFFS_SAE),
    "TFA": (TFAModelSpec(n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
            {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
            L1_COEFFS_TFA),
    "TFA-shuf": (TFAModelSpec(n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq_shuffled",
                 {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
                 None),  # no L1 for shuffled
    "TFA-pos": (TFAModelSpec(use_pos_encoding=True, n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq",
                {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
                L1_COEFFS_TFA),
    "TFA-pos-shuf": (TFAModelSpec(use_pos_encoding=True, n_heads=4, n_attn_layers=1, bottleneck_factor=1), "seq_shuffled",
                     {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
                     None),
    "TXCDR-T2": (TXCDRModelSpec(T=2), "window_2",
                 {"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4},
                 L1_COEFFS_TXCDR),
    "TXCDR-T5": (TXCDRModelSpec(T=5), "window_5",
                 {"total_steps": 80_000, "batch_size": 2048, "lr": 3e-4},
                 L1_COEFFS_TXCDR),
}


def main():
    model_name = sys.argv[1]
    output_path = sys.argv[2]

    if model_name not in MODEL_DEFS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODEL_DEFS.keys())}")
        sys.exit(1)

    spec, gen_key, train_params, l1_coeffs = MODEL_DEFS[model_name]
    print(f"=== {model_name} on {DEVICE} ===", flush=True)
    t_start = time.time()

    # Build pipeline (need window sizes for TXCDR)
    window_sizes = []
    if gen_key.startswith("window_"):
        window_sizes.append(int(gen_key.split("_")[1]))
    pipeline = build_data_pipeline(DATA_CFG, DEVICE, window_sizes=window_sizes or None)

    # Get generator
    if gen_key == "flat":
        gen_fn = pipeline.gen_flat
    elif gen_key == "seq":
        gen_fn = pipeline.gen_seq
    elif gen_key == "seq_shuffled":
        gen_fn = pipeline.gen_seq_shuffled
    elif gen_key.startswith("window_"):
        T = int(gen_key.split("_")[1])
        gen_fn = pipeline.gen_windows[T]

    # ── Experiment 1: TopK sweep ──────────────────────────────────
    print(f"\n--- TopK sweep ---", flush=True)
    topk_results = []
    for k in TOPK_K_VALUES:
        set_seed(DATA_CFG.seed)
        t0 = time.time()
        model = spec.create(d_in=40, d_sae=40, k=k, device=DEVICE)
        config = spec.make_train_config(
            total_steps=train_params["total_steps"],
            batch_size=train_params["batch_size"],
            lr=train_params["lr"],
            l1_coeff=0.0,
            log_every=train_params["total_steps"],
        )
        model, _ = spec.train(model, gen_fn, config, DEVICE)
        result = evaluate_model(spec, model, pipeline.eval_hidden, DEVICE,
                                true_features=pipeline.true_features, seq_len=64)
        topk_results.append(result.to_dict() | {"k": k})
        print(f"  k={k:>2}: NMSE={result.nmse:.6f} nL0={result.novel_l0:.1f} "
              f"tL0={result.total_l0:.1f} AUC={result.auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
        del model; torch.cuda.empty_cache()

    # ── Experiment 2: L1 sweep (if applicable) ────────────────────
    l1_results = []
    if l1_coeffs:
        print(f"\n--- L1 sweep ---", flush=True)
        for l1c in l1_coeffs:
            set_seed(DATA_CFG.seed)
            t0 = time.time()
            model = spec.create(d_in=40, d_sae=40, k=None, device=DEVICE)
            config = spec.make_train_config(
                total_steps=train_params["total_steps"],
                batch_size=train_params["batch_size"],
                lr=train_params["lr"],
                l1_coeff=l1c,
                log_every=train_params["total_steps"],
            )
            model, _ = spec.train(model, gen_fn, config, DEVICE)
            result = evaluate_model(spec, model, pipeline.eval_hidden, DEVICE,
                                    true_features=pipeline.true_features, seq_len=64)
            l1_results.append(result.to_dict() | {"l1": l1c})
            print(f"  l1={l1c:.4f}: NMSE={result.nmse:.6f} nL0={result.novel_l0:.2f} "
                  f"tL0={result.total_l0:.2f} AUC={result.auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
            del model; torch.cuda.empty_cache()

    elapsed = time.time() - t_start
    output = {"model": model_name, "topk": topk_results, "l1": l1_results, "elapsed_s": elapsed}
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nDone in {elapsed/60:.1f}m. Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
