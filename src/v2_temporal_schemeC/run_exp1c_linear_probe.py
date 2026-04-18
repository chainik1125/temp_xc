"""Linear probe analysis for Experiment 1c: noisy emissions.

For each model and k value, extract latent activations on eval data and
train Ridge regression probes to predict:
  - s_i (noisy observed support) from z  → "local" R²
  - h_i (true hidden state) from z       → "global" R²

Uses cached models from model_cache/exp1c/.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_exp1c_linear_probe.py
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
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.plotting.save_figure import save_figure
from src.utils.seed import set_seed
from src.data.toy.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data.toy.dataset import generate_dataset
from src.pipeline.toy_models import (TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec)
from src.training.train_tfa import create_tfa
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.stacked_sae import StackedSAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Same parameters as run_experiment1c_noisy.py
LAM = 0.3
MU = 0.5
NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
DICT_WIDTH = 40
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500
Q = 0.8
P_B = 0.625
P_A = 0.0

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

BASE = os.path.dirname(__file__)
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c")
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c_noisy")

# TXCDRv2 window sizes (must match run_experiment1c_noisy.py)
TXCDR_T_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Model configs: (name, type, spec_factory, extra_kwargs)
MODELS = [
    ("TFA-pos", "tfa", lambda: TFAModelSpec(use_pos_encoding=True), {}),
    ("Stacked-T2", "stacked", lambda: StackedSAEModelSpec(T=2), {"T": 2}),
    ("Stacked-T5", "stacked", lambda: StackedSAEModelSpec(T=5), {"T": 5}),
] + [
    (f"TXCDRv2-T{T}", "txcdrv2", lambda T=T: TXCDRv2ModelSpec(T=T), {"T": T})
    for T in TXCDR_T_VALUES
]


def generate_data():
    """Generate eval data with hidden states and support."""
    cfg = DataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=LAM, p=Q),
        emission=EmissionConfig(p_A=P_A, p_B=P_B),
        features=FeatureConfig(k=NUM_FEATURES, d=HIDDEN_DIM),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + TRAIN_N_SEQ),
        seed=SEED,
    )
    result = generate_dataset(cfg)

    x_all = result["x"]
    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf

    eval_x = x_all[:EVAL_N_SEQ].to(DEVICE)
    eval_support = result["support"][:EVAL_N_SEQ]      # (n_eval, k, T)
    eval_hidden = result["hidden_states"][:EVAL_N_SEQ]  # (n_eval, k, T)

    return eval_x, eval_support, eval_hidden


def load_model(name, model_type, k, extra):
    """Load a cached model."""
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


@torch.no_grad()
def extract_latents_tfa(model, eval_x):
    """Extract per-token latent activations from TFA. Returns (n_eval*T, d_sae)."""
    all_z = []
    for s in range(0, eval_x.shape[0], 256):
        x = eval_x[s:s + 256]
        _, inter = model(x)
        z = inter["novel_codes"] + inter["pred_codes"]  # (B, T, d_sae)
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0).reshape(-1, DICT_WIDTH).numpy()  # (n*T, d_sae)


@torch.no_grad()
def extract_latents_windowed(model, eval_x, T_win, is_crosscoder):
    """Extract per-position latent activations from windowed models.

    Averages z across overlapping windows for each position.
    Returns (n_eval*T, d_sae).
    """
    n_eval = eval_x.shape[0]
    z_sum = torch.zeros(n_eval, SEQ_LEN, DICT_WIDTH)
    counts = torch.zeros(n_eval, SEQ_LEN)

    for t_start in range(SEQ_LEN - T_win + 1):
        windows = eval_x[:, t_start:t_start + T_win, :]
        for s in range(0, n_eval, 256):
            w = windows[s:s + 256]
            bs = w.shape[0]
            if is_crosscoder:
                z = model.encode(w).cpu()  # (B, d_sae)
                for t_off in range(T_win):
                    z_sum[s:s + bs, t_start + t_off] += z
                    counts[s:s + bs, t_start + t_off] += 1
            else:
                _, _, z = model(w)  # (B, T_win, d_sae)
                z = z.cpu()
                for t_off in range(T_win):
                    z_sum[s:s + bs, t_start + t_off] += z[:, t_off]
                    counts[s:s + bs, t_start + t_off] += 1

    counts_exp = counts.unsqueeze(-1).clamp(min=1)
    z_avg = z_sum / counts_exp
    return z_avg.reshape(-1, DICT_WIDTH).numpy()  # (n*T, d_sae)


def run_linear_probes(z, support, hidden_states, alpha=1.0):
    """Train Ridge probes z→s_i and z→h_i for each feature.

    Args:
        z: (n_tokens, d_sae) latent activations
        support: (n_eval, k, T) observed binary support
        hidden_states: (n_eval, k, T) true hidden states
        alpha: Ridge regularization

    Returns:
        dict with per-feature and mean R² for local and global probes.
    """
    # Reshape targets to (n_tokens, k)
    sup = support.permute(0, 2, 1).reshape(-1, NUM_FEATURES).numpy()  # (n*T, k)
    hid = hidden_states.permute(0, 2, 1).reshape(-1, NUM_FEATURES).numpy()  # (n*T, k)

    n = z.shape[0]
    # Train/test split: 80/20
    split = int(0.8 * n)
    z_train, z_test = z[:split], z[split:]
    sup_train, sup_test = sup[:split], sup[split:]
    hid_train, hid_test = hid[:split], hid[split:]

    local_r2s = []
    global_r2s = []

    for feat_i in range(NUM_FEATURES):
        # Local probe: z → s_i
        probe_local = Ridge(alpha=alpha)
        probe_local.fit(z_train, sup_train[:, feat_i])
        pred_local = probe_local.predict(z_test)
        r2_local = r2_score(sup_test[:, feat_i], pred_local)
        local_r2s.append(r2_local)

        # Global probe: z → h_i
        probe_global = Ridge(alpha=alpha)
        probe_global.fit(z_train, hid_train[:, feat_i])
        pred_global = probe_global.predict(z_test)
        r2_global = r2_score(hid_test[:, feat_i], pred_global)
        global_r2s.append(r2_global)

    return {
        "local_r2": local_r2s,
        "global_r2": global_r2s,
        "mean_local_r2": float(np.mean(local_r2s)),
        "mean_global_r2": float(np.mean(global_r2s)),
        "ratio": float(np.mean(global_r2s)) / max(float(np.mean(local_r2s)), 1e-12),
    }


def main():
    print(f"Device: {DEVICE}", flush=True)
    print("Linear probe analysis for Experiment 1c", flush=True)
    t_start = time.time()

    # Generate data (same seed as experiment)
    print("Generating data...", flush=True)
    set_seed(SEED)
    eval_x, eval_support, eval_hidden = generate_data()
    print(f"  eval_x: {eval_x.shape}", flush=True)
    print(f"  support: {eval_support.shape}, hidden: {eval_hidden.shape}", flush=True)

    # Sanity check: what R² does the raw observation achieve?
    sup_flat = eval_support.permute(0, 2, 1).reshape(-1, NUM_FEATURES).numpy()
    hid_flat = eval_hidden.permute(0, 2, 1).reshape(-1, NUM_FEATURES).numpy()
    oracle_r2s = []
    for i in range(NUM_FEATURES):
        oracle_r2s.append(r2_score(hid_flat[:, i], sup_flat[:, i]))
    print(f"\n  Oracle R² (s→h, per feature): mean={np.mean(oracle_r2s):.4f}, "
          f"min={np.min(oracle_r2s):.4f}, max={np.max(oracle_r2s):.4f}", flush=True)

    all_results = {}

    for name, model_type, spec_factory, extra in MODELS:
        print(f"\n{'='*50}", flush=True)
        print(f"{name}", flush=True)
        print(f"{'='*50}", flush=True)
        all_results[name] = []

        for k in K_VALUES:
            model = load_model(name, model_type, k, extra)
            if model is None:
                print(f"  k={k:>2}: SKIP (no cache or k*T > d_sae)", flush=True)
                continue

            t0 = time.time()

            # Extract latents
            if model_type == "tfa":
                z = extract_latents_tfa(model, eval_x)
            else:
                is_xc = model_type == "txcdrv2"
                z = extract_latents_windowed(model, eval_x, extra["T"], is_xc)

            # Run probes
            probe_result = run_linear_probes(z, eval_support, eval_hidden)
            probe_result["k"] = k

            all_results[name].append(probe_result)

            print(f"  k={k:>2}: local_R²={probe_result['mean_local_r2']:.4f}  "
                  f"global_R²={probe_result['mean_global_r2']:.4f}  "
                  f"ratio={probe_result['ratio']:.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

            del model
            torch.cuda.empty_cache()

    # Save results
    save_path = os.path.join(RESULTS_DIR, "linear_probe_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {save_path}", flush=True)

    # ── Plots ──

    import matplotlib.cm as _cm
    _cmap = _cm.get_cmap("RdPu", len(TXCDR_T_VALUES) + 2)
    _markers = ["o", "s", "D", "^", "v", "<", "p", ">", "h", "H", "*"]
    STYLE = {
        "TFA-pos":    {"color": "#2ca02c", "marker": "X", "ls": "-"},
        "Stacked-T2": {"color": "#9467bd", "marker": "o", "ls": "-"},
        "Stacked-T5": {"color": "#9467bd", "marker": "^", "ls": "--"},
    }
    for _i, _T in enumerate(TXCDR_T_VALUES):
        STYLE[f"TXCDRv2-T{_T}"] = {
            "color": _cmap(_i + 2), "marker": _markers[_i % len(_markers)], "ls": "-",
        }

    # Plot 1: R² vs k (local and global)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for name, results in all_results.items():
        if not results:
            continue
        s = STYLE[name]
        ks = [r["k"] for r in results]
        local_r2 = [r["mean_local_r2"] for r in results]
        global_r2 = [r["mean_global_r2"] for r in results]
        ratios = [r["ratio"] for r in results]

        axes[0].plot(ks, local_r2, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)
        axes[1].plot(ks, global_r2, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)
        axes[2].plot(ks, ratios, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)

    axes[0].set(xlabel="k", ylabel=r"$R^2$", title=r"Local probe: $z \to s_i$ (noisy obs)")
    axes[1].set(xlabel="k", ylabel=r"$R^2$", title=r"Global probe: $z \to h_i$ (hidden state)")
    axes[2].set(xlabel="k", ylabel="Global / Local R² ratio",
                title="Denoising ratio (linear probe)")
    axes[2].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)

    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(r"Linear probe analysis ($\gamma = 0.25$)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_DIR, f"exp1c_linear_probe.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    # Plot 2: Global R² vs Local R² scatter
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, label="y=x")

    for name, results in all_results.items():
        if not results:
            continue
        s = STYLE[name]
        local_r2 = [r["mean_local_r2"] for r in results]
        global_r2 = [r["mean_global_r2"] for r in results]
        ax.scatter(local_r2, global_r2, color=s["color"], marker=s["marker"],
                   s=80, alpha=0.7, label=name, zorder=5)

    ax.set(xlabel=r"Local $R^2$ ($z \to s_i$)",
           ylabel=r"Global $R^2$ ($z \to h_i$)",
           title=r"Linear probe: global vs local feature recovery ($\gamma = 0.25$)")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(RESULTS_DIR, f"exp1c_probe_scatter.{ext}")
        if ext == "png":
            save_figure(fig, path)
        else:
            fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
