"""Linear probe analysis for Experiment 1c2 (sparse heterogeneous-rho).

Uses cached models from model_cache/exp1c2/.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_exp1c2_linear_probe.py
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.utils.seed import set_seed
from src.data_generation.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data_generation.dataset import generate_dataset
from src.v2_temporal_schemeC.experiment import (
    TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec,
)
from src.v2_temporal_schemeC.experiment.denoising import (
    extract_latents_tfa, extract_latents_windowed, run_linear_probes,
)
from src.v2_temporal_schemeC.train_tfa import create_tfa
from src.v2_temporal_schemeC.temporal_crosscoder import TemporalCrosscoder
from src.v2_temporal_schemeC.stacked_sae import StackedSAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same parameters as run_experiment1c2.py
NUM_FEATURES = 40
HIDDEN_DIM = 80
DICT_WIDTH = 80
SEQ_LEN = 64
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500
PI = 0.15
P_A = 0.0
P_B = 0.625
RHO_GROUPS = [0.1, 0.4, 0.7, 0.95]
FEATURES_PER_GROUP = NUM_FEATURES // len(RHO_GROUPS)
PER_FEATURE_PI = [PI] * NUM_FEATURES
PER_FEATURE_RHO = []
for rho in RHO_GROUPS:
    PER_FEATURE_RHO.extend([rho] * FEATURES_PER_GROUP)

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]
TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]

BASE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c2")
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c2_sparse")

MODELS = [
    ("TFA-pos", "tfa", lambda: TFAModelSpec(use_pos_encoding=True), {}),
    ("Stacked-T2", "stacked", lambda: StackedSAEModelSpec(T=2), {"T": 2}),
    ("Stacked-T5", "stacked", lambda: StackedSAEModelSpec(T=5), {"T": 5}),
] + [
    (f"TXCDRv2-T{T}", "txcdrv2", lambda T=T: TXCDRv2ModelSpec(T=T), {"T": T})
    for T in TXCDR_T_VALUES
]


def generate_data():
    cfg = DataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=0.3, p=0.5),
        emission=EmissionConfig(p_A=P_A, p_B=P_B),
        features=FeatureConfig(k=NUM_FEATURES, d=HIDDEN_DIM),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + TRAIN_N_SEQ),
        seed=SEED,
        per_feature_pi=PER_FEATURE_PI,
        per_feature_rho=PER_FEATURE_RHO,
    )
    result = generate_dataset(cfg)
    x_all = result["x"]
    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf
    return (
        x_all[:EVAL_N_SEQ].to(DEVICE),
        result["support"][:EVAL_N_SEQ],
        result["hidden_states"][:EVAL_N_SEQ],
    )


def load_model(name, model_type, k, extra):
    path = os.path.join(CACHE_DIR, f"{name}_k{k}.pt")
    if not os.path.exists(path):
        return None
    if model_type == "tfa":
        model = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
            n_attn_layers=1, bottleneck_factor=1,
            use_pos_encoding=True, device=DEVICE,
        )
    elif model_type == "stacked":
        model = StackedSAE(HIDDEN_DIM, DICT_WIDTH, extra["T"], k=k).to(DEVICE)
    elif model_type == "txcdrv2":
        k_eff = k * extra["T"]
        if k_eff > DICT_WIDTH:
            return None
        model = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, extra["T"], k=k_eff).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Linear probe analysis for Experiment 1c2", flush=True)
    t_start = time.time()

    set_seed(SEED)
    eval_x, eval_support, eval_hidden = generate_data()
    print(f"  eval_x: {eval_x.shape}", flush=True)

    all_results = {}

    for name, model_type, spec_factory, extra in MODELS:
        print(f"\n{'='*50}\n{name}\n{'='*50}", flush=True)
        all_results[name] = []

        for k in K_VALUES:
            model = load_model(name, model_type, k, extra)
            if model is None:
                print(f"  k={k:>2}: SKIP", flush=True)
                continue

            t0 = time.time()
            if model_type == "tfa":
                z = extract_latents_tfa(model, eval_x, DICT_WIDTH)
            else:
                is_xc = model_type == "txcdrv2"
                z = extract_latents_windowed(
                    model, eval_x, extra["T"], DICT_WIDTH, SEQ_LEN, is_xc,
                )

            probe_result = run_linear_probes(
                z, eval_support, eval_hidden, NUM_FEATURES,
            )
            probe_result["k"] = k
            all_results[name].append(probe_result)

            print(f"  k={k:>2}: local_R²={probe_result['mean_local_r2']:.4f}  "
                  f"global_R²={probe_result['mean_global_r2']:.4f}  "
                  f"ratio={probe_result['ratio']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

            del model
            torch.cuda.empty_cache()

    save_path = os.path.join(RESULTS_DIR, "linear_probe_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {save_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"Done in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
