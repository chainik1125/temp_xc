"""Full experiment suite on event-structured data (Setting 1).

Runs Experiments 1 (TopK sweep) and 2 (L1 Pareto) for all models on
event-structured data, analogous to the independent-feature results in
docs/han/research_logs/2026-03-04-v2-progress.md.

Event config: 4 events × 5 features = 20 features. Each event has its own
Markov chain (ρ ∈ {0.3, 0.5, 0.7, 0.9}). When an event is active, ALL 5
features in that group activate together. π_event = 0.5 → E[active events]=2,
E[L0] = 2×5 = 10 (same as independent setup).

For parallel execution per model:
  python run_event_experiments.py <model_name> <output_json>

Model names: SAE, TFA, TFA-shuf, TFA-pos, TFA-pos-shuf, TXCDR-T2, TXCDR-T5

Usage (all models):
  for m in SAE TFA TFA-shuf TFA-pos TFA-pos-shuf TXCDR-T2 TXCDR-T5; do
    TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=... python -u \\
      src/v2_temporal_schemeC/run_event_experiments.py $m results/event/$m.json &
  done
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
from src.v2_temporal_schemeC.experiment import (
    DataConfig, EventConfig, build_data_pipeline,
    SAEModelSpec, TFAModelSpec, TXCDRModelSpec, ModelEntry,
    evaluate_model,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Event config: 4 events × 5 features, same E[L0]=10 as independent
EVENT_DATA_CFG = DataConfig(
    num_features=20,
    hidden_dim=40,
    seq_len=64,
    pi=[0.5] * 20,  # ignored when event_config is set
    rho=[0.5] * 20,  # ignored when event_config is set
    dict_width=40,
    seed=42,
    eval_n_seq=2000,
    event_config=EventConfig(
        n_events=4,
        features_per_event=5,
        pi_events=[0.5, 0.5, 0.5, 0.5],
        rho_events=[0.3, 0.5, 0.7, 0.9],
    ),
)

TOPK_K_VALUES = [1, 3, 5, 8, 10, 15, 20]
L1_COEFFS_SAE = np.logspace(-2.3, 1.3, 15).tolist()
L1_COEFFS_TFA = np.logspace(-0.8, 1.8, 12).tolist()
L1_COEFFS_TXCDR = np.logspace(-1.5, 1.5, 12).tolist()

MODEL_DEFS = {
    "SAE": (SAEModelSpec(), "flat",
            {"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4},
            L1_COEFFS_SAE),
    "TFA": (TFAModelSpec(), "seq",
            {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
            L1_COEFFS_TFA),
    "TFA-shuf": (TFAModelSpec(), "seq_shuffled",
                 {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
                 None),
    "TFA-pos": (TFAModelSpec(use_pos_encoding=True), "seq",
                {"total_steps": 30_000, "batch_size": 64, "lr": 1e-3},
                L1_COEFFS_TFA),
    "TFA-pos-shuf": (TFAModelSpec(use_pos_encoding=True), "seq_shuffled",
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
    print(f"=== {model_name} (event data) on {DEVICE} ===", flush=True)
    t_start = time.time()

    # Build pipeline
    window_sizes = [int(gen_key.split("_")[1])] if gen_key.startswith("window_") else []
    pipeline = build_data_pipeline(EVENT_DATA_CFG, DEVICE, window_sizes=window_sizes or None)

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

    # ── Experiment 1: TopK sweep ──
    print(f"\n--- TopK sweep ---", flush=True)
    topk_results = []
    for k in TOPK_K_VALUES:
        set_seed(EVENT_DATA_CFG.seed)
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

    # ── Experiment 2: L1 sweep ──
    l1_results = []
    if l1_coeffs:
        print(f"\n--- L1 sweep ---", flush=True)
        for l1c in l1_coeffs:
            set_seed(EVENT_DATA_CFG.seed)
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
    output = {"model": model_name, "data": "event", "topk": topk_results,
              "l1": l1_results, "elapsed_s": elapsed}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nDone in {elapsed/60:.1f}m. Saved to {output_path}", flush=True)


if __name__ == "__main__":
    main()
