"""Sweep emission amplitude γ at fixed hidden persistence λ=0.3 (ρ=0.7).

Tests how observation noise affects temporal exploitation. γ=1 is our
standard setup (deterministic emissions). γ<1 means observations are
noisy indicators of the hidden state.

All configurations maintain μ=0.5 (same marginal sparsity).

Uses Aniket's data generation pipeline (src/data_generation/).

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_gamma_sweep.py
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from src.utils.plot import save_figure
from src.utils.seed import set_seed
from src.data_generation.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data_generation.dataset import generate_dataset
from src.data_generation.transition import hmm_autocorrelation_amplitude
from src.v2_temporal_schemeC.experiment import (
    SAEModelSpec, TFAModelSpec, ModelEntry, EvalResult, evaluate_model,
)
from src.v2_temporal_schemeC.experiment.model_specs import EvalOutput
from src.v2_temporal_schemeC.relu_sae import ReLUSAE, ReLUSAETrainingConfig, train_relu_sae
from src.v2_temporal_schemeC.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.v2_temporal_schemeC.feature_recovery import (
    feature_recovery_score, sae_decoder_directions, tfa_decoder_directions,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fixed parameters
LAM = 0.3        # hidden state mixing rate (ρ=0.7)
MU = 0.5         # target marginal sparsity
NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
DICT_WIDTH = 40
K_VALUES = [3, 10]
EVAL_N_SEQ = 2000
SEED = 42

# γ sweep: maintain μ=0.5 by adjusting (q, p_B)
# With p_A=0, μ = q*p_B, γ = (1-q)/q when μ=0.5
GAMMA_CONFIGS = [
    {"gamma_target": 1.00, "q": 0.500, "p_B": 1.000, "p_A": 0.0},
    {"gamma_target": 0.50, "q": 0.667, "p_B": 0.750, "p_A": 0.0},
    {"gamma_target": 0.25, "q": 0.800, "p_B": 0.625, "p_A": 0.0},
    {"gamma_target": 0.10, "q": 0.909, "p_B": 0.550, "p_A": 0.0},
    {"gamma_target": 0.05, "q": 0.952, "p_B": 0.525, "p_A": 0.0},
]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gamma_sweep")

# Models
MODELS_CFG = {
    "SAE": {"spec": SAEModelSpec(), "steps": 30_000, "batch": 4096, "lr": 3e-4},
    "TFA": {"spec": TFAModelSpec(), "steps": 30_000, "batch": 64, "lr": 1e-3},
    "TFA-shuf": {"spec": TFAModelSpec(), "steps": 30_000, "batch": 64, "lr": 1e-3},
    "TFA-pos": {"spec": TFAModelSpec(use_pos_encoding=True), "steps": 30_000, "batch": 64, "lr": 1e-3},
    "TFA-pos-shuf": {"spec": TFAModelSpec(use_pos_encoding=True), "steps": 30_000, "batch": 64, "lr": 1e-3},
}


def generate_data_for_gamma(gcfg):
    """Generate data using Aniket's pipeline for a given γ config."""
    cfg = DataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=LAM, p=gcfg["q"]),
        emission=EmissionConfig(p_A=gcfg["p_A"], p_B=gcfg["p_B"]),
        features=FeatureConfig(k=NUM_FEATURES, d=HIDDEN_DIM),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + 500),
        seed=SEED,
    )
    result = generate_dataset(cfg)

    # Compute actual gamma
    gamma = hmm_autocorrelation_amplitude(cfg.transition.matrix, gcfg["p_A"], gcfg["p_B"])

    # Split into train data (for generators) and eval data
    x_all = result["x"]  # (n_seq, T, d)
    features = result["features"]  # (k, d)

    # Scaling factor
    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf

    eval_hidden = x_all[:EVAL_N_SEQ].to(DEVICE)
    train_x = x_all[EVAL_N_SEQ:]

    return eval_hidden, train_x, features, sf, gamma


def make_flat_gen(train_x):
    flat = train_x.reshape(-1, HIDDEN_DIM).to(DEVICE)
    def gen(bs):
        idx = torch.randint(0, flat.shape[0], (bs,), device=DEVICE)
        return flat[idx]
    return gen


def make_seq_gen(train_x, shuffle=False):
    seqs = train_x.to(DEVICE)
    def gen(n):
        idx = torch.randint(0, seqs.shape[0], (n,), device=DEVICE)
        batch = seqs[idx]
        if shuffle:
            for i in range(n):
                batch[i] = batch[i, torch.randperm(SEQ_LEN, device=DEVICE)]
        return batch
    return gen


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"λ={LAM} (ρ={1-LAM}), μ={MU}, sweeping γ", flush=True)
    t_start = time.time()

    all_results = {}

    for gcfg in GAMMA_CONFIGS:
        gamma_target = gcfg["gamma_target"]
        print(f"\n{'='*60}", flush=True)
        print(f"γ = {gamma_target}", flush=True)
        print(f"{'='*60}", flush=True)

        set_seed(SEED)
        eval_hidden, train_x, features, sf, gamma_actual = generate_data_for_gamma(gcfg)
        true_feats = features  # (k, d)
        print(f"  actual γ={gamma_actual:.3f}, μ={eval_hidden.reshape(-1, HIDDEN_DIM).abs().gt(0.01).float().mean():.3f}", flush=True)

        gen_flat = make_flat_gen(train_x)
        gen_seq = make_seq_gen(train_x, shuffle=False)
        gen_seq_shuf = make_seq_gen(train_x, shuffle=True)

        gamma_results = {}

        for ki, k in enumerate(K_VALUES):
            print(f"\n  k={k}:", flush=True)

            for model_name, mcfg in MODELS_CFG.items():
                set_seed(SEED)
                t0 = time.time()
                spec = mcfg["spec"]

                if model_name == "SAE":
                    sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(DEVICE)
                    cfg = ReLUSAETrainingConfig(total_steps=mcfg["steps"], batch_size=mcfg["batch"],
                                                 lr=mcfg["lr"], log_every=mcfg["steps"])
                    sae, _ = train_relu_sae(sae, gen_flat, cfg, DEVICE)
                    r = evaluate_model(SAEModelSpec(), sae, eval_hidden, DEVICE,
                                       true_features=true_feats, seq_len=SEQ_LEN)
                else:
                    use_pos = "pos" in model_name
                    shuffle = "shuf" in model_name
                    tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                                     n_attn_layers=1, bottleneck_factor=1,
                                     use_pos_encoding=use_pos, device=DEVICE)
                    gfn = gen_seq_shuf if shuffle else gen_seq
                    cfg = TFATrainingConfig(total_steps=mcfg["steps"], batch_size=mcfg["batch"],
                                            lr=mcfg["lr"], log_every=mcfg["steps"])
                    tfa, _ = train_tfa(tfa, gfn, cfg, DEVICE)
                    r = evaluate_model(TFAModelSpec(use_pos_encoding=use_pos), tfa, eval_hidden,
                                       DEVICE, true_features=true_feats, seq_len=SEQ_LEN)

                if model_name not in gamma_results:
                    gamma_results[model_name] = []
                gamma_results[model_name].append(r.to_dict() | {"k": k})
                print(f"    {model_name:>15}: NMSE={r.nmse:.6f} AUC={r.auc:.4f} ({time.time()-t0:.0f}s)", flush=True)
                torch.cuda.empty_cache()

        all_results[str(gamma_target)] = gamma_results

    # Save
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump({"gamma_configs": GAMMA_CONFIGS, "lam": LAM, "mu": MU,
                    "k_values": K_VALUES, "results": all_results},
                  f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Plot
    gamma_vals = [g["gamma_target"] for g in GAMMA_CONFIGS]
    style = {
        "SAE": ("tab:blue", "o", "-"),
        "TFA": ("tab:orange", "s", "-"),
        "TFA-shuf": ("tab:red", "^", "--"),
        "TFA-pos": ("tab:brown", "X", "-"),
        "TFA-pos-shuf": ("tab:pink", "v", "--"),
    }

    for ki, k in enumerate(K_VALUES):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for name, (c, m, ls) in style.items():
            nmse = [all_results[str(g)][name][ki]["nmse"] for g in gamma_vals]
            auc = [all_results[str(g)][name][ki]["auc"] for g in gamma_vals]
            axes[0].plot(gamma_vals, nmse, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)
            axes[1].plot(gamma_vals, auc, marker=m, linestyle=ls, color=c, lw=2, ms=8, label=name)
        axes[0].set(xlabel="γ (emission amplitude)", ylabel="NMSE", title="NMSE vs γ")
        axes[1].set(xlabel="γ (emission amplitude)", ylabel="AUC", title="AUC vs γ")
        for ax in axes:
            ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        plt.suptitle(f"Emission amplitude sweep (k={k}, λ={LAM}, μ={MU})", fontsize=13)
        plt.tight_layout()
        save_figure(fig, os.path.join(RESULTS_DIR, f"gamma_sweep_k{k}.png"))
        plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.0f}m. Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
