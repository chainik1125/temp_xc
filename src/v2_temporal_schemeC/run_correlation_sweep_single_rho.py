"""Run correlation sweep for a single ρ value. For parallel execution.

Usage:
  python src/v2_temporal_schemeC/run_correlation_sweep_single_rho.py <rho> <output_json>
"""

import json
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import torch

from src.v2_temporal_schemeC.experiment import (
    DataConfig, SAEModelSpec, TFAModelSpec, ModelEntry, run_topk_sweep,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K_VALUES = [3, 10]
NUM_FEATURES = 20
HIDDEN_DIM = 40

MODELS = [
    ModelEntry("SAE", SAEModelSpec(), "flat",
               training_overrides={"total_steps": 30_000, "batch_size": 4096, "lr": 3e-4}),
    ModelEntry("TFA", TFAModelSpec(), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-shuf", TFAModelSpec(), "seq_shuffled",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-pos", TFAModelSpec(use_pos_encoding=True), "seq",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
    ModelEntry("TFA-pos-shuf", TFAModelSpec(use_pos_encoding=True), "seq_shuffled",
               training_overrides={"total_steps": 30_000, "batch_size": 64, "lr": 1e-3}),
]


def main():
    rho = float(sys.argv[1])
    output_path = sys.argv[2]
    print(f"ρ={rho} on {DEVICE}", flush=True)
    t0 = time.time()

    cfg = DataConfig(
        num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM, seq_len=64,
        pi=[0.5] * NUM_FEATURES, rho=[rho] * NUM_FEATURES,
        dict_width=HIDDEN_DIM, seed=42, eval_n_seq=2000,
    )

    results = run_topk_sweep(models=MODELS, k_values=K_VALUES, data_config=cfg, device=DEVICE)

    out = {
        "rho": rho,
        "k_values": K_VALUES,
        "results": {
            name: [r.to_dict() | {"k": K_VALUES[i]} for i, r in enumerate(rs)]
            for name, rs in results.items()
        },
        "elapsed_s": time.time() - t0,
    }
    with open(output_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"Done in {(time.time()-t0)/60:.0f}m", flush=True)


if __name__ == "__main__":
    main()
