"""Full Exp 1 + Exp 2 run with model caching for all 11 models.

First run: trains all models and caches checkpoints.
Subsequent runs: loads cached models, re-evaluates only.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_full_cached.py
"""

import json
import os
import sys
import time
from dataclasses import asdict

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.pipeline.toy_models import (DataConfig, SAEModelSpec, TFAModelSpec, TXCDRModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec, ModelEntry, run_topk_sweep, run_l1_sweep, save_results)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_CFG = DataConfig(
    num_features=20, hidden_dim=40, seq_len=64,
    pi=[0.5] * 20,
    rho=[0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4,
    dict_width=40, seed=42, eval_n_seq=2000,
)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]
L1_COEFFS_SAE = np.logspace(-2.3, 1.3, 15).tolist()
L1_COEFFS_TFA = np.logspace(-0.8, 1.8, 12).tolist()
L1_COEFFS_TXCDR = np.logspace(-1.5, 1.5, 12).tolist()

BASE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE, "model_cache")
RESULTS_DIR = os.path.join(BASE, "results", "reproduction")

EXP1_MODELS = [
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
    ModelEntry("Stacked-T2", StackedSAEModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("Stacked-T5", StackedSAEModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDR-T2", TXCDRModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDR-T5", TXCDRModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDRv2-T2", TXCDRv2ModelSpec(T=2), "window_2",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
    ModelEntry("TXCDRv2-T5", TXCDRv2ModelSpec(T=5), "window_5",
               training_overrides={"total_steps": 30_000, "batch_size": 2048, "lr": 3e-4}),
]

# L1 sweep: exclude shuffled variants
EXP2_MODELS = [m for m in EXP1_MODELS if "shuf" not in m.name]
EXP2_L1 = {
    "SAE": L1_COEFFS_SAE,
    "TFA": L1_COEFFS_TFA,
    "TFA-pos": L1_COEFFS_TFA,
    "Stacked-T2": L1_COEFFS_TXCDR,
    "Stacked-T5": L1_COEFFS_TXCDR,
    "TXCDR-T2": L1_COEFFS_TXCDR,
    "TXCDR-T5": L1_COEFFS_TXCDR,
    "TXCDRv2-T2": L1_COEFFS_TXCDR,
    "TXCDRv2-T5": L1_COEFFS_TXCDR,
}


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Cache dir: {CACHE_DIR}", flush=True)
    t_start = time.time()

    # ── Experiment 1: TopK sweep ──
    print(f"\n{'='*70}\nEXPERIMENT 1: TopK sweep\n{'='*70}", flush=True)
    exp1 = run_topk_sweep(
        models=EXP1_MODELS, k_values=K_VALUES,
        data_config=DATA_CFG, device=DEVICE,
        cache_dir=CACHE_DIR,
    )
    for name, results in exp1.items():
        data = {"model": name, "topk": [r.to_dict() | {"k": K_VALUES[i]} for i, r in enumerate(results)], "l1": []}
        with open(os.path.join(RESULTS_DIR, f"{name}.json"), "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    # ── Experiment 2: L1 sweep ──
    print(f"\n{'='*70}\nEXPERIMENT 2: L1 Pareto\n{'='*70}", flush=True)
    exp2 = run_l1_sweep(
        models=EXP2_MODELS, l1_coeffs=EXP2_L1,
        data_config=DATA_CFG, device=DEVICE,
        cache_dir=CACHE_DIR,
    )
    # Merge L1 results into existing JSONs
    for name, results in exp2.items():
        path = os.path.join(RESULTS_DIR, f"{name}.json")
        with open(path) as f:
            data = json.load(f)
        l1_keys = list(EXP2_L1.get(name, []))
        data["l1"] = [r.to_dict() | {"l1": l1_keys[i]} for i, r in enumerate(results)]
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    elapsed = time.time() - t_start
    print(f"\nAll done in {elapsed/60:.0f}m", flush=True)
    print(f"Models cached in {CACHE_DIR}/", flush=True)
    print(f"Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
