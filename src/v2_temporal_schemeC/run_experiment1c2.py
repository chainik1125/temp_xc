"""Experiment 1c2: Denoising in sparse regime with heterogeneous persistence.

Same denoising analysis as Experiment 1c but with a wildly different data setup:
  - 40 features (vs 20), d=80 (vs 40), d_sae=80 (vs 40)
  - Sparse: pi=0.15 for all features (vs pi=0.5)
  - Heterogeneous rho: 10 features each at {0.1, 0.4, 0.7, 0.95} (vs uniform 0.7)
  - Same emission noise: p_A=0, p_B=0.625

Tests whether TXCDRv2's denoising advantage generalizes beyond the dense,
uniform-persistence regime of 1c. The per-feature rho groups enable analysis
of whether denoising is selective (slow features denoise better) or uniform.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_experiment1c2.py
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as mcm
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.plot import save_figure
from src.utils.seed import set_seed
from src.data_generation.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data_generation.dataset import generate_dataset
from src.v2_temporal_schemeC.experiment import (
    TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec,
    evaluate_model,
)
from src.v2_temporal_schemeC.experiment.denoising import compute_global_recovery
from src.v2_temporal_schemeC.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.bench.architectures.crosscoder import TemporalCrosscoder
from src.v2_temporal_schemeC.temporal_crosscoder import (
    CrosscoderTrainingConfig, train_crosscoder,
)
from src.v2_temporal_schemeC.stacked_sae import (
    StackedSAE, StackedSAETrainingConfig, train_stacked_sae,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data parameters (different from 1c) ──
NUM_FEATURES = 40
HIDDEN_DIM = 80
DICT_WIDTH = 80
SEQ_LEN = 64
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500

# Sparse regime
PI = 0.15  # E[L0_hidden] = 40 * 0.15 = 6

# Heterogeneous persistence: 4 groups of 10 features
RHO_GROUPS = [0.1, 0.4, 0.7, 0.95]
FEATURES_PER_GROUP = NUM_FEATURES // len(RHO_GROUPS)  # 10
PER_FEATURE_PI = [PI] * NUM_FEATURES
PER_FEATURE_RHO = []
for rho in RHO_GROUPS:
    PER_FEATURE_RHO.extend([rho] * FEATURES_PER_GROUP)

# Same emission noise as 1c
P_A = 0.0
P_B = 0.625

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

# TXCDRv2 window sizes
TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c2_sparse")
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c2")

# ── Models ──
MODELS = [
    {
        "name": "TFA-pos",
        "type": "tfa",
        "spec": TFAModelSpec(use_pos_encoding=True),
        "use_pos": True,
        "shuffle": False,
        "steps": 30_000, "batch": 64, "lr": 1e-3,
    },
    {
        "name": "Stacked-T2",
        "type": "stacked",
        "T": 2,
        "spec": StackedSAEModelSpec(T=2),
        "steps": 30_000, "batch": 2048, "lr": 3e-4,
    },
    {
        "name": "Stacked-T5",
        "type": "stacked",
        "T": 5,
        "spec": StackedSAEModelSpec(T=5),
        "steps": 30_000, "batch": 2048, "lr": 3e-4,
    },
] + [
    {
        "name": f"TXCDRv2-T{T}",
        "type": "txcdrv2",
        "T": T,
        "spec": TXCDRv2ModelSpec(T=T),
        "steps": 30_000, "batch": 2048, "lr": 3e-4,
    }
    for T in TXCDR_T_VALUES
]


# ── Data generation ──

def generate_data():
    """Generate sparse heterogeneous-rho data."""
    cfg = DataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=0.3, p=0.5),  # ignored
        emission=EmissionConfig(p_A=P_A, p_B=P_B),
        features=FeatureConfig(k=NUM_FEATURES, d=HIDDEN_DIM),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + TRAIN_N_SEQ),
        seed=SEED,
        per_feature_pi=PER_FEATURE_PI,
        per_feature_rho=PER_FEATURE_RHO,
    )
    result = generate_dataset(cfg)

    x_all = result["x"]
    features = result["features"]
    support = result["support"]
    hidden_states = result["hidden_states"]

    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf
    print(f"  Scaling factor: {sf:.4f}", flush=True)

    eval_x = x_all[:EVAL_N_SEQ].to(DEVICE)
    train_x = x_all[EVAL_N_SEQ:]
    eval_support = support[:EVAL_N_SEQ]
    eval_hidden = hidden_states[:EVAL_N_SEQ]

    return eval_x, train_x, features, eval_support, eval_hidden, sf


# ── Generators ──

def make_seq_gen(train_x):
    seqs = train_x.to(DEVICE)
    def gen(n):
        idx = torch.randint(0, seqs.shape[0], (n,), device=DEVICE)
        return seqs[idx]
    return gen


def make_window_gen(train_x, T):
    seqs = train_x.to(DEVICE)
    n_win = SEQ_LEN - T + 1
    def gen(batch_size):
        n_seq = max(1, batch_size // n_win) + 1
        idx = torch.randint(0, seqs.shape[0], (n_seq,), device=DEVICE)
        batch = seqs[idx]
        windows = torch.cat([batch[:, t:t+T, :] for t in range(n_win)], dim=0)
        sel = torch.randperm(windows.shape[0], device=DEVICE)[:batch_size]
        return windows[sel]
    return gen


# ── Model caching ──

def _cache_path(name, k):
    return os.path.join(CACHE_DIR, f"{name}_k{k}.pt")


def _try_load(spec, name, k):
    path = _cache_path(name, k)
    if not os.path.exists(path):
        return None
    model = spec.create(d_in=HIDDEN_DIM, d_sae=DICT_WIDTH, k=k, device=DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def _save(model, name, k):
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), _cache_path(name, k))


# ── Training ──

def train_model(mcfg, k, gen_fns):
    name = mcfg["name"]
    spec = mcfg["spec"]

    cached = _try_load(spec, name, k)
    if cached is not None:
        return cached, spec, True

    set_seed(SEED)

    if mcfg["type"] == "tfa":
        model = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
            n_attn_layers=1, bottleneck_factor=1,
            use_pos_encoding=mcfg["use_pos"], device=DEVICE,
        )
        cfg = TFATrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"],
        )
        model, _ = train_tfa(model, gen_fns["seq"], cfg, DEVICE)

    elif mcfg["type"] == "stacked":
        T = mcfg["T"]
        model = StackedSAE(HIDDEN_DIM, DICT_WIDTH, T, k=k).to(DEVICE)
        cfg = StackedSAETrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"],
        )
        model, _ = train_stacked_sae(model, gen_fns[f"window_{T}"], cfg, DEVICE)

    elif mcfg["type"] == "txcdrv2":
        T = mcfg["T"]
        k_eff = k * T
        if k_eff > DICT_WIDTH:
            return None, spec, False
        model = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k=k_eff).to(DEVICE)
        cfg = CrosscoderTrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"],
        )
        model, _ = train_crosscoder(model, gen_fns[f"window_{T}"], cfg, DEVICE)

    else:
        raise ValueError(f"Unknown model type: {mcfg['type']}")

    _save(model, name, k)
    return model, spec, False


# ── Main ──

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Experiment 1c2: sparse heterogeneous-rho denoising", flush=True)
    print(f"  {NUM_FEATURES} features, d={HIDDEN_DIM}, d_sae={DICT_WIDTH}", flush=True)
    print(f"  pi={PI}, rho groups={RHO_GROUPS} ({FEATURES_PER_GROUP} each)", flush=True)
    print(f"  p_A={P_A}, p_B={P_B}", flush=True)
    t_start = time.time()

    # Generate data
    print("\nGenerating data...", flush=True)
    eval_x, train_x, features, eval_support, eval_hidden, sf = generate_data()
    print(f"  eval: {eval_x.shape}, train: {train_x.shape}", flush=True)

    # Verify per-group marginal rates and autocorrelations
    h = eval_hidden  # (n_eval, k, T)
    for gi, rho in enumerate(RHO_GROUPS):
        start = gi * FEATURES_PER_GROUP
        end = start + FEATURES_PER_GROUP
        pi_emp = h[:, start:end, :].mean().item()
        # Empirical lag-1 autocorr
        s0 = h[:, start:end, :-1].reshape(-1)
        s1 = h[:, start:end, 1:].reshape(-1)
        cov = ((s0 - s0.mean()) * (s1 - s1.mean())).mean()
        var = ((s0 - s0.mean())**2).mean()
        rho_emp = (cov / var).item() if var > 1e-8 else 0
        print(f"  rho={rho:.2f} group: pi_emp={pi_emp:.3f} rho_emp={rho_emp:.3f}", flush=True)

    # Build generators
    gen_fns = {"seq": make_seq_gen(train_x)}
    for T in sorted(set(m["T"] for m in MODELS if "T" in m)):
        gen_fns[f"window_{T}"] = make_window_gen(train_x, T)

    # Run sweep
    all_results = {}
    all_recovery = {}
    for mcfg in MODELS:
        all_results[mcfg["name"]] = []
        all_recovery[mcfg["name"]] = {}

    for k in K_VALUES:
        print(f"\n{'='*60}\nk = {k}\n{'='*60}", flush=True)

        for mcfg in MODELS:
            name = mcfg["name"]
            spec = mcfg["spec"]

            if mcfg["type"] == "txcdrv2" and k * mcfg["T"] > DICT_WIDTH:
                print(f"  {name:>15}: SKIPPED (k*T={k*mcfg['T']} > d_sae={DICT_WIDTH})", flush=True)
                continue

            t0 = time.time()
            model, spec, cached = train_model(mcfg, k, gen_fns)
            if model is None:
                print(f"  {name:>15}: SKIPPED", flush=True)
                continue

            r = evaluate_model(spec, model, eval_x, DEVICE,
                               true_features=features, seq_len=SEQ_LEN)

            rec = compute_global_recovery(
                spec, model, eval_x, features, eval_support, eval_hidden,
                num_features=NUM_FEATURES, seq_len=SEQ_LEN,
                dict_width=DICT_WIDTH, device=DEVICE,
            )

            result_dict = r.to_dict() | {"k": k} | {
                "mean_local_corr": rec["mean_local"],
                "mean_global_corr": rec["mean_global"],
                "local_corrs": rec["local_corrs"],
                "global_corrs": rec["global_corrs"],
            }
            all_results[name].append(result_dict)
            all_recovery[name][str(k)] = rec

            ratio = rec["mean_global"] / rec["mean_local"] if rec["mean_local"] > 0.01 else 0
            src = "cached" if cached else f"{time.time()-t0:.0f}s"
            print(f"  {name:>15}: NMSE={r.nmse:.6f} AUC={r.auc:.4f} "
                  f"ratio={ratio:.2f} ({src})", flush=True)

            del model
            torch.cuda.empty_cache()

    # Save results
    results = {
        "experiment": "1c2_sparse_heterogeneous",
        "num_features": NUM_FEATURES,
        "hidden_dim": HIDDEN_DIM,
        "dict_width": DICT_WIDTH,
        "pi": PI,
        "rho_groups": RHO_GROUPS,
        "features_per_group": FEATURES_PER_GROUP,
        "p_A": P_A,
        "p_B": P_B,
        "k_values": K_VALUES,
        "models": all_results,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)

    # ── Plotting ──
    print("\nPlotting...", flush=True)

    _cmap = mcm.get_cmap("RdPu", len(TXCDR_T_VALUES) + 2)
    _markers = ["o", "s", "D", "^", "v", "<", "p", ">", "h", "H", "*"]
    STYLE = {
        "TFA-pos":    {"color": "#2ca02c", "marker": "X", "ls": "-"},
        "Stacked-T2": {"color": "#9467bd", "marker": "o", "ls": "-"},
        "Stacked-T5": {"color": "#9467bd", "marker": "^", "ls": "--"},
    }
    for i, T in enumerate(TXCDR_T_VALUES):
        STYLE[f"TXCDRv2-T{T}"] = {
            "color": _cmap(i + 2), "marker": _markers[i % len(_markers)], "ls": "-",
        }

    SUPTITLE = r"Experiment 1c2: sparse regime ($\pi=0.15$, heterogeneous $\rho$)"

    # Plot 1: NMSE and AUC
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for name, results in all_results.items():
        if not results or name not in STYLE:
            continue
        s = STYLE[name]
        ks = [r["k"] for r in results]
        axes[0].plot(ks, [r["nmse"] for r in results], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=name)
        axes[1].plot(ks, [r["auc"] for r in results], marker=s["marker"],
                     ls=s["ls"], color=s["color"], lw=2, ms=7, label=name)
        axes[2].plot([r["nmse"] for r in results], [r["auc"] for r in results],
                     marker=s["marker"], ls=s["ls"], color=s["color"], lw=2, ms=7,
                     label=name)
    axes[0].set(xlabel="k", ylabel="NMSE", title="NMSE vs k", yscale="log")
    axes[1].set(xlabel="k", ylabel="AUC", title="Feature recovery AUC vs k")
    axes[2].set(xlabel="NMSE", ylabel="AUC", title="NMSE vs AUC", xscale="log")
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{SUPTITLE}: NMSE and AUC", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "exp1c2_topk_auc.png"))
    plt.close(fig)

    # Plot 2: Denoising ratio vs k (3 panels: local, global, ratio)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for name, results in all_results.items():
        if not results or name not in STYLE:
            continue
        s = STYLE[name]
        ks, locs, globs, rats = [], [], [], []
        for r in results:
            loc = r["mean_local_corr"]
            glob = r["mean_global_corr"]
            if loc > 0.01:
                ks.append(r["k"])
                locs.append(loc)
                globs.append(glob)
                rats.append(glob / loc)
        axes[0].plot(ks, locs, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)
        axes[1].plot(ks, globs, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)
        axes[2].plot(ks, rats, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=7, label=name)
    axes[0].set(xlabel="k", ylabel="Pearson corr", title=r"Local: corr($z_j$, $s_i$)")
    axes[1].set(xlabel="k", ylabel="Pearson corr", title=r"Global: corr($z_j$, $h_i$)")
    axes[2].set(xlabel="k", ylabel="Global / Local", title="Denoising ratio vs k")
    axes[2].axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{SUPTITLE}: single-latent correlation", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "exp1c2_denoising.png"))
    plt.close(fig)

    # Plot 3: Denoising ratio vs T at fixed k
    k_fixed = [1, 3, 5]
    fig, axes = plt.subplots(1, len(k_fixed), figsize=(6*len(k_fixed), 6), sharey=True)
    for ki, k in enumerate(k_fixed):
        ax = axes[ki]
        Ts, rats = [], []
        for T in TXCDR_T_VALUES:
            key = f"TXCDRv2-T{T}"
            if key not in all_results:
                continue
            for r in all_results[key]:
                if r["k"] == k and r["mean_local_corr"] > 0.01:
                    Ts.append(T)
                    rats.append(r["mean_global_corr"] / r["mean_local_corr"])
                    break
        if Ts:
            ax.plot(Ts, rats, "o-", color="#d62728", lw=2, ms=8)
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5, lw=1)
        ax.set(xlabel="Window size T", title=f"k = {k}")
        ax.set_xticks([t for t in TXCDR_T_VALUES if t <= 14])
        if ki == 0:
            ax.set_ylabel("Global / Local ratio")
        ax.grid(True, alpha=0.3)
    plt.suptitle(f"{SUPTITLE}: TXCDRv2 denoising vs window size", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "exp1c2_denoising_vs_T.png"))
    plt.close(fig)

    # Plot 4: Per-rho-group denoising analysis
    # For TXCDRv2-T5 and TFA-pos at k=3, show per-feature denoising grouped by rho
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax_i, (model_name, title) in enumerate([
        ("TXCDRv2-T5", "TXCDRv2 T=5"),
        ("TFA-pos", "TFA-pos"),
    ]):
        ax = axes[ax_i]
        k_target = 3
        if model_name not in all_results:
            continue
        for r in all_results[model_name]:
            if r["k"] == k_target and "local_corrs" in r:
                local_corrs = np.array(r["local_corrs"])
                global_corrs = np.array(r["global_corrs"])
                # Group by rho
                for gi, rho in enumerate(RHO_GROUPS):
                    start = gi * FEATURES_PER_GROUP
                    end = start + FEATURES_PER_GROUP
                    lc = local_corrs[start:end]
                    gc = global_corrs[start:end]
                    ratios = gc / np.maximum(lc, 1e-8)
                    x_pos = np.full(FEATURES_PER_GROUP, gi)
                    ax.scatter(x_pos + np.random.randn(FEATURES_PER_GROUP)*0.05,
                               ratios, alpha=0.6, s=40)
                    ax.bar(gi, ratios.mean(), alpha=0.3, width=0.6)
                break
        ax.set_xticks(range(len(RHO_GROUPS)))
        ax.set_xticklabels([f"ρ={r}" for r in RHO_GROUPS])
        ax.axhline(1.0, color="gray", ls="--", alpha=0.5)
        ax.set(ylabel="Global / Local ratio", title=f"{title} (k={k_target})")
        ax.grid(True, alpha=0.3, axis='y')
    plt.suptitle(f"{SUPTITLE}: per-ρ-group denoising at k=3", fontsize=13, y=1.02)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "exp1c2_per_rho_denoising.png"))
    plt.close(fig)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}m. Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
