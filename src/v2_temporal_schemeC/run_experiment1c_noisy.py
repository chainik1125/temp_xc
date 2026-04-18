"""Experiment 1c: TopK sweep with noisy emissions (γ=0.25).

Identical to Experiment 1 but with stochastic HMM emissions: features fire
only 62.5% of the time when the hidden state is ON (p_B=0.625, p_A=0).
Tests which models degrade gracefully under observation noise, and whether
temporal models can denoise (recover hidden state better than noisy obs).

6 models × 12 k values = 72 training runs.

Uses Aniket's data generation pipeline (src/data_generation/).

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_experiment1c_noisy.py
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
import torch.nn.functional as F

from src.utils.plot import save_figure
from src.utils.seed import set_seed
from src.data.toy.configs import (
    DataGenerationConfig, EmissionConfig, TransitionConfig,
    FeatureConfig, SequenceConfig,
)
from src.data.toy.dataset import generate_dataset
from src.pipeline.toy_models import (SAEModelSpec, TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec, ModelEntry)
from src.eval.toy_unified import evaluate_model
from src.eval.toy_unified import EvalResult
from src.architectures.relu_sae import ReLUSAE, ReLUSAETrainingConfig, train_relu_sae
from src.training.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.stacked_sae import StackedSAE
from src.training.train_crosscoder import (
    CrosscoderTrainingConfig, train_crosscoder,
)
from src.training.train_stacked_sae import (
    StackedSAETrainingConfig, train_stacked_sae,
)
from src.eval.denoising import compute_global_recovery
from src.eval.feature_recovery import feature_recovery_score, cos_sims

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Fixed parameters ──
LAM = 0.3           # hidden state mixing rate (ρ = 0.7)
MU = 0.5            # target marginal sparsity
NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
DICT_WIDTH = 40
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500   # extra sequences for training generators

# γ = 0.25: with p_A=0 and μ = q*p_B = 0.5, γ = (1-q)/q → q=0.8, p_B=0.625
Q = 0.8
P_B = 0.625
P_A = 0.0

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c_noisy")
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c")
EXP1_RESULTS_DIR = os.path.join(BASE, "results", "reproduction")

# ── Model configs ──
# TXCDRv2 window sizes to sweep (T=2,5 are the originals; others added to
# test whether denoising improves monotonically with window size)
TXCDR_T_VALUES = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

# Each entry: (name, type, kwargs, training_config)
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
    """Generate noisy-emission data using Aniket's pipeline."""
    cfg = DataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=LAM, p=Q),
        emission=EmissionConfig(p_A=P_A, p_B=P_B),
        features=FeatureConfig(k=NUM_FEATURES, d=HIDDEN_DIM),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + TRAIN_N_SEQ),
        seed=SEED,
    )
    result = generate_dataset(cfg)

    x_all = result["x"]               # (n_seq, T, d)
    features = result["features"]      # (k, d)
    support = result["support"]        # (n_seq, k, T)
    hidden_states = result["hidden_states"]  # (n_seq, k, T)

    # Compute scaling factor: sqrt(d) / mean(||x||)
    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf
    print(f"  Scaling factor: {sf:.4f}", flush=True)

    # Split eval / train
    eval_x = x_all[:EVAL_N_SEQ].to(DEVICE)
    train_x = x_all[EVAL_N_SEQ:]
    eval_support = support[:EVAL_N_SEQ]       # (eval, k, T)
    eval_hidden = hidden_states[:EVAL_N_SEQ]  # (eval, k, T)

    return eval_x, train_x, features, eval_support, eval_hidden, sf


# ── Generator factories ──

def make_seq_gen(train_x, shuffle=False):
    """Sequence generator for TFA: returns (n_seq, T, d)."""
    seqs = train_x.to(DEVICE)
    def gen(n):
        idx = torch.randint(0, seqs.shape[0], (n,), device=DEVICE)
        batch = seqs[idx]
        if shuffle:
            for i in range(n):
                batch[i] = batch[i, torch.randperm(SEQ_LEN, device=DEVICE)]
        return batch
    return gen


def make_window_gen(train_x, T):
    """Window generator for TXCDR/Stacked: returns (batch, T, d)."""
    seqs = train_x.to(DEVICE)
    n_windows_per_seq = SEQ_LEN - T + 1
    def gen(batch_size):
        # Sample sequences, extract all windows, subsample
        n_seq = max(1, batch_size // n_windows_per_seq) + 1
        idx = torch.randint(0, seqs.shape[0], (n_seq,), device=DEVICE)
        batch_seqs = seqs[idx]
        windows = []
        for t in range(n_windows_per_seq):
            windows.append(batch_seqs[:, t:t + T, :])
        all_w = torch.cat(windows, dim=0)
        sel = torch.randperm(all_w.shape[0], device=DEVICE)[:batch_size]
        return all_w[sel]
    return gen


# ── Model caching ──

def _cache_path(model_name, k):
    return os.path.join(CACHE_DIR, f"{model_name}_k{k}.pt")


def _try_load_cached(spec, model_name, k):
    path = _cache_path(model_name, k)
    if not os.path.exists(path):
        return None
    model = spec.create(d_in=HIDDEN_DIM, d_sae=DICT_WIDTH, k=k, device=DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def _save_cached(model, model_name, k):
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), _cache_path(model_name, k))


# ── Training ──

def train_model(mcfg, k, gen_fns):
    """Train a single model. Returns (model, spec)."""
    name = mcfg["name"]
    spec = mcfg["spec"]

    # Check cache
    cached = _try_load_cached(spec, name, k)
    if cached is not None:
        return cached, spec, True

    set_seed(SEED)

    if mcfg["type"] == "tfa":
        model = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
            n_attn_layers=1, bottleneck_factor=1,
            use_pos_encoding=mcfg["use_pos"], device=DEVICE,
        )
        gfn = gen_fns["seq_shuf"] if mcfg["shuffle"] else gen_fns["seq"]
        cfg = TFATrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"],
        )
        model, _ = train_tfa(model, gfn, cfg, DEVICE)

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
            return None, spec, False  # skip: k*T > d_sae
        model = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k=k_eff).to(DEVICE)
        cfg = CrosscoderTrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"],
        )
        model, _ = train_crosscoder(model, gen_fns[f"window_{T}"], cfg, DEVICE)

    else:
        raise ValueError(f"Unknown model type: {mcfg['type']}")

    _save_cached(model, name, k)
    return model, spec, False


# ── Load Experiment 1 results for overlay ──

def load_exp1_results():
    """Load γ=1 Experiment 1 results for comparison overlay."""
    exp1 = {}
    for name in ["TFA-pos", "Stacked-T2", "Stacked-T5",
                  "TXCDRv2-T2", "TXCDRv2-T5"]:
        path = os.path.join(EXP1_RESULTS_DIR, f"{name}.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            exp1[name] = data.get("topk", [])
    return exp1


# ── Plotting ──

import matplotlib.cm as _cm
_cmap = _cm.get_cmap("RdPu", len(TXCDR_T_VALUES) + 2)
_markers = ["o", "s", "D", "^", "v", "<", "p", ">", "h", "H", "*"]
STYLE = {
    "TFA-pos":    {"color": "tab:brown", "marker": "X", "ls": "-"},
    "Stacked-T2": {"color": "tab:green", "marker": "o", "ls": "-"},
    "Stacked-T5": {"color": "tab:green", "marker": "^", "ls": "--"},
}
for _i, _T in enumerate(TXCDR_T_VALUES):
    STYLE[f"TXCDRv2-T{_T}"] = {
        "color": _cmap(_i + 2), "marker": _markers[_i % len(_markers)], "ls": "-",
    }


def plot_nmse_auc(all_results, exp1_results):
    """Plot NMSE and AUC vs k, with γ=1 overlay."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, results in all_results.items():
        s = STYLE[name]
        ks = [r["k"] for r in results]
        nmse = [r["nmse"] for r in results]
        auc = [r["auc"] for r in results]
        axes[0].plot(ks, nmse, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=8, label=f"{name} (γ=0.25)")
        axes[1].plot(ks, auc, marker=s["marker"], ls=s["ls"], color=s["color"],
                     lw=2, ms=8, label=f"{name} (γ=0.25)")

    # Overlay γ=1
    for name, topk in exp1_results.items():
        if name not in STYLE:
            continue
        s = STYLE[name]
        ks = [r["k"] for r in topk]
        nmse = [r["nmse"] for r in topk]
        auc = [r["auc"] for r in topk]
        axes[0].plot(ks, nmse, marker=s["marker"], ls=":", color=s["color"],
                     lw=1, ms=5, alpha=0.5, label=f"{name} (γ=1)")
        axes[1].plot(ks, auc, marker=s["marker"], ls=":", color=s["color"],
                     lw=1, ms=5, alpha=0.5, label=f"{name} (γ=1)")

    axes[0].set(xlabel="k (TopK)", ylabel="NMSE",
                title="NMSE vs k (γ=0.25 vs γ=1)")
    axes[0].set_yscale("log")
    axes[1].set(xlabel="k (TopK)", ylabel="AUC",
                title="Feature Recovery AUC vs k")

    for ax in axes:
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.suptitle(f"Experiment 1c: Noisy emissions (γ=0.25, λ={LAM}, μ={MU})",
                 fontsize=13)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "nmse_auc_vs_k.png"))
    plt.close(fig)


def plot_global_recovery(all_recovery, k_subset=None):
    """Plot global vs local correlation for each model/k."""
    if k_subset is None:
        k_subset = [3, 5, 8, 10, 15]

    fig, axes = plt.subplots(1, len(k_subset), figsize=(4 * len(k_subset), 5),
                             sharey=True)
    if len(k_subset) == 1:
        axes = [axes]

    for ki, k in enumerate(k_subset):
        ax = axes[ki]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1)  # diagonal

        for name in all_recovery:
            if str(k) not in all_recovery[name]:
                continue
            rec = all_recovery[name][str(k)]
            s = STYLE[name]
            ax.scatter(rec["mean_local"], rec["mean_global"],
                       color=s["color"], marker=s["marker"], s=100,
                       label=name, zorder=5)

        ax.set_xlabel("Local corr (vs noisy obs)")
        if ki == 0:
            ax.set_ylabel("Global corr (vs hidden state)")
        ax.set_title(f"k = {k}")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)

    plt.suptitle("Global Feature Recovery: denoising test (γ=0.25)", fontsize=13)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "global_vs_local_recovery.png"))
    plt.close(fig)


def plot_denoising_ratio(all_recovery):
    """Plot global/local ratio vs k for each model."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name in all_recovery:
        s = STYLE[name]
        ks, ratios = [], []
        for k_str in sorted(all_recovery[name].keys(), key=int):
            rec = all_recovery[name][k_str]
            if rec["mean_local"] > 0.01:
                ks.append(int(k_str))
                ratios.append(rec["mean_global"] / rec["mean_local"])
        if ks:
            ax.plot(ks, ratios, marker=s["marker"], ls=s["ls"],
                    color=s["color"], lw=2, ms=8, label=name)

    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="No denoising")
    ax.set(xlabel="k (TopK)", ylabel="Global / Local correlation ratio",
           title="Denoising ratio vs k (>1 means model denoises)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    save_figure(fig, os.path.join(RESULTS_DIR, "denoising_ratio_vs_k.png"))
    plt.close(fig)


# ── Main ──

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Experiment 1c: γ=0.25, λ={LAM} (ρ={1-LAM}), μ={MU}", flush=True)
    print(f"q={Q}, p_B={P_B}, p_A={P_A}", flush=True)
    t_start = time.time()

    # Generate data
    print("\nGenerating data...", flush=True)
    eval_x, train_x, features, eval_support, eval_hidden, sf = generate_data()
    true_feats = features  # (k, d)
    print(f"  eval: {eval_x.shape}, train: {train_x.shape}", flush=True)

    # Verify marginal sparsity
    flat = eval_x.reshape(-1, HIDDEN_DIM)
    mu_obs = flat.abs().gt(0.01).float().mean().item()
    print(f"  observed marginal density: {mu_obs:.3f}", flush=True)

    # Build generators (one per unique T value used by any model)
    gen_fns = {"seq": make_seq_gen(train_x, shuffle=False)}
    for T in sorted(set(m["T"] for m in MODELS if "T" in m)):
        gen_fns[f"window_{T}"] = make_window_gen(train_x, T=T)

    # Load Experiment 1 results for overlay
    exp1_results = load_exp1_results()
    print(f"  Loaded γ=1 results for: {list(exp1_results.keys())}", flush=True)

    # Run sweep
    all_results = {}   # name → list of result dicts
    all_recovery = {}  # name → {k_str → recovery_dict}

    for mcfg in MODELS:
        name = mcfg["name"]
        all_results[name] = []
        all_recovery[name] = {}

    for k in K_VALUES:
        print(f"\n{'='*60}", flush=True)
        print(f"k = {k}", flush=True)
        print(f"{'='*60}", flush=True)

        for mcfg in MODELS:
            name = mcfg["name"]
            spec = mcfg["spec"]

            # Skip TXCDRv2 T=5 when k*5 > dict_width
            if mcfg["type"] == "txcdrv2" and mcfg["T"] == 5 and k * 5 > DICT_WIDTH:
                print(f"  {name:>15}: SKIPPED (k*T={k*5} > d_sae={DICT_WIDTH})", flush=True)
                continue

            t0 = time.time()
            model, spec, cached = train_model(mcfg, k, gen_fns)

            if model is None:
                print(f"  {name:>15}: SKIPPED", flush=True)
                continue

            # Standard eval (NMSE, AUC)
            r = evaluate_model(spec, model, eval_x, DEVICE,
                               true_features=true_feats, seq_len=SEQ_LEN)

            # Global feature recovery
            rec = compute_global_recovery(
                spec, model, eval_x, true_feats, eval_support, eval_hidden,
                num_features=NUM_FEATURES, seq_len=SEQ_LEN,
                dict_width=DICT_WIDTH, device=DEVICE,
            )

            result_dict = r.to_dict() | {"k": k} | {
                "mean_local_corr": rec["mean_local"],
                "mean_global_corr": rec["mean_global"],
                "denoising_frac": rec["denoising_frac"],
            }
            all_results[name].append(result_dict)
            all_recovery[name][str(k)] = rec

            src = "cached" if cached else f"{time.time()-t0:.0f}s"
            print(f"  {name:>15}: NMSE={r.nmse:.6f} AUC={r.auc:.4f} "
                  f"local={rec['mean_local']:.3f} global={rec['mean_global']:.3f} "
                  f"denoise={rec['denoising_frac']:.0%} ({src})", flush=True)

            del model
            torch.cuda.empty_cache()

    # Save results
    results_path = os.path.join(RESULTS_DIR, "results.json")
    with open(results_path, "w") as f:
        json.dump({
            "experiment": "1c_noisy_emissions",
            "gamma": 0.25,
            "lam": LAM,
            "mu": MU,
            "q": Q,
            "p_B": P_B,
            "k_values": K_VALUES,
            "models": all_results,
            "global_recovery": {
                name: {k: {kk: vv for kk, vv in v.items()
                           if kk not in ("local_corrs", "global_corrs")}
                       for k, v in recs.items()}
                for name, recs in all_recovery.items()
            },
        }, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Also save full per-feature recovery data separately
    recovery_path = os.path.join(RESULTS_DIR, "global_recovery_detail.json")
    with open(recovery_path, "w") as f:
        json.dump(all_recovery, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)

    # Plots
    print("\nPlotting...", flush=True)
    plot_nmse_auc(all_results, exp1_results)
    plot_global_recovery(all_recovery, k_subset=[3, 5, 8, 10, 15])
    plot_denoising_ratio(all_recovery)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}m. Results in {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
