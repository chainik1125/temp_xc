"""Low-π regime: does TFA exploit temporal structure when content matching is weak?

At π=0.5 (previous experiments), any two tokens share ~5 features, giving strong
content-based matching that drowns out temporal signal. Here we test π=0.05 with
100 features, where content overlap drops to ~0.25 features and each feature
appears in only ~3/64 context positions.

Three analyses:
  Part 1: Shuffle diagnostic (SAE vs TFA vs TFA-shuffled NMSE)
  Part 2: Attention direction quality (before proj_scale)
  Part 3: Run-length prediction projections (after proj_scale)

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_low_pi_regime.py
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

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig,
    create_tfa,
    train_tfa,
)
from src.v2_temporal_schemeC.relu_sae import (
    ReLUSAE,
    ReLUSAETrainingConfig,
    train_relu_sae,
)

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 100
HIDDEN_DIM = 100
SEQ_LEN = 64
PI_VAL = 0.05
PI = [PI_VAL] * NUM_FEATURES
RHO = [0.0] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20
DICT_WIDTH = 100

K_VALUES = [1, 2, 3, 5]
HISTORY_LEN = 5

SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 3000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "low_pi_regime"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            10000, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return (
            math.sqrt(HIDDEN_DIM)
            / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
        )


def make_generators(model, pi_t, rho_t, device, sf, shuffle=False):
    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return model(acts) * sf

    return gen_flat, gen_seq


def eval_sae(sae, eval_hidden, device):
    sae.eval()
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]
    total_se = total_signal = total_l0 = 0.0
    bs = 4096
    with torch.no_grad():
        for s in range(0, n, bs):
            x = flat[s:min(s + bs, n)].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_signal, "l0": total_l0 / n}


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    bs = 256
    total_se = total_signal = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, bs):
            x = eval_hidden[s:min(s + bs, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf = x.reshape(-1, D)
            rf = recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            n_tokens += B * T
    return {"nmse": total_se / total_signal}


# ── Attention direction extraction ───────────────────────────────────


def extract_attention_direction(tfa, x_seq):
    """Extract raw attention direction before proj_scale."""
    tfa.eval()
    E = tfa.D.T if tfa.tied_weights else tfa.E
    with torch.no_grad():
        x_input = x_seq - tfa.b
        attn_layer = tfa.attn_layers[0]
        z_input = F.relu(torch.matmul(x_input * tfa.lam, E))
        z_ctx = torch.cat(
            (torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()),
            dim=1,
        )
        z_pred_, _ = attn_layer(z_ctx, z_input, get_attn_map=False)
        z_pred_ = F.relu(z_pred_)
        Dz_pred = torch.matmul(z_pred_, tfa.D)
    return Dz_pred, x_input


def compute_direction_metrics(Dz_pred, x_input):
    """Cosine similarity and variance explained."""
    Dz_norm = Dz_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_norm = x_input.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine = (Dz_pred * x_input).sum(dim=-1) / (
        Dz_norm.squeeze(-1) * x_norm.squeeze(-1)
    )
    proj_coeff = (Dz_pred * x_input).sum(dim=-1) / Dz_norm.squeeze(-1)
    var_explained = proj_coeff.pow(2) / x_input.pow(2).sum(dim=-1).clamp(min=1e-8)
    return cosine, var_explained


# ── Prediction projection extraction ────────────────────────────────


def compute_pred_projections(tfa, eval_hidden, feature_dirs, device):
    """Compute per-token |<pred_recons, f_i>| (after proj_scale)."""
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    T = eval_hidden.shape[1]
    n_features = feature_dirs.shape[0]
    bs = 256
    all_pred = torch.zeros(n_seq, T, n_features)
    fd = feature_dirs.unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        for start in range(0, n_seq, bs):
            end = min(start + bs, n_seq)
            x = eval_hidden[start:end].to(device)
            _, inter = tfa(x)
            pr = inter["pred_recons"]
            all_pred[start:end] = (pr.unsqueeze(2) * fd).sum(dim=-1).cpu()
    return all_pred


# ── Run-length classification ────────────────────────────────────────


def classify_by_run_length(support, history_len=HISTORY_LEN):
    B, T, nf = support.shape
    h = history_len
    valid_T = T - h
    current = support[:, h:, :]
    history_sum = torch.zeros(B, valid_T, nf, device=support.device)
    for i in range(h):
        history_sum += support[:, i:i + valid_T, :]
    all_on = (history_sum == h)
    all_off = (history_sum == 0)
    curr_on = (current == 1)
    curr_off = (current == 0)
    masks = {
        "long_cont": all_on & curr_on,
        "sudden_onset": all_off & curr_on,
        "sudden_offset": all_on & curr_off,
        "long_absent": all_off & curr_off,
    }
    return masks, h


# ── Analysis ─────────────────────────────────────────────────────────


def analyze_by_category(values, support, rho_list, history_len=HISTORY_LEN,
                        feature_dirs=None, Dz_pred=None):
    """Analyze per-category metrics, optionally including per-feature direction alignment."""
    support_cpu = support.cpu()
    masks, h = classify_by_run_length(support_cpu, history_len)
    values_valid = values[:, h:]

    if Dz_pred is not None:
        Dz_valid = Dz_pred[:, h:]

    rho_groups = sorted(set(rho_list))
    n_features = support.shape[2]
    results = {}

    for cat, mask in masks.items():
        results[cat] = {}
        for rho_val in rho_groups:
            feat_indices = [
                i for i in range(n_features)
                if abs(rho_list[i] - rho_val) < 0.01
            ]
            all_vals = []
            all_feat_cos = []
            n_events = 0

            for i in feat_indices:
                m = mask[:, :, i]
                if not m.any():
                    continue
                n_events += m.sum().item()
                all_vals.append(values_valid[:, :, i][m].abs() if values_valid.dim() == 3
                                else values_valid[m])

                if Dz_pred is not None and feature_dirs is not None:
                    fi = feature_dirs[i].cpu()
                    Dz_m = Dz_valid[m]
                    Dz_hat = Dz_m / Dz_m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                    all_feat_cos.append((Dz_hat * fi.unsqueeze(0)).sum(dim=-1).abs())

            entry = {"n": n_events}
            if all_vals:
                pooled = torch.cat(all_vals)
                entry["mean"] = pooled.mean().item()
            else:
                entry["mean"] = 0.0
            if all_feat_cos:
                entry["feat_cos_mean"] = torch.cat(all_feat_cos).mean().item()
            results[cat][rho_val] = entry

    return results


def analyze_direction_global(cosine, var_explained, support, rho_list,
                              feature_dirs, Dz_pred):
    """Analyze direction metrics by category with per-feature alignment."""
    support_cpu = support.cpu()
    masks, h = classify_by_run_length(support_cpu, HISTORY_LEN)
    cos_valid = cosine[:, h:]
    ve_valid = var_explained[:, h:]
    Dz_valid = Dz_pred[:, h:]

    rho_groups = sorted(set(rho_list))
    n_features = support.shape[2]
    results = {}

    for cat, mask in masks.items():
        results[cat] = {}
        for rho_val in rho_groups:
            feat_indices = [
                i for i in range(n_features)
                if abs(rho_list[i] - rho_val) < 0.01
            ]
            all_cos = []
            all_ve = []
            all_feat_cos = []
            n_events = 0

            for i in feat_indices:
                m = mask[:, :, i]
                if not m.any():
                    continue
                n_events += m.sum().item()
                all_cos.append(cos_valid[m])
                all_ve.append(ve_valid[m])

                fi = feature_dirs[i].cpu()
                Dz_m = Dz_valid[m]
                Dz_hat = Dz_m / Dz_m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                all_feat_cos.append((Dz_hat * fi.unsqueeze(0)).sum(dim=-1).abs())

            entry = {"n": n_events}
            if all_cos:
                entry["global_cos"] = torch.cat(all_cos).mean().item()
                entry["var_expl"] = torch.cat(all_ve).mean().item()
                entry["feat_cos"] = torch.cat(all_feat_cos).mean().item()
            else:
                entry["global_cos"] = 0.0
                entry["var_expl"] = 0.0
                entry["feat_cos"] = 0.0
            results[cat][rho_val] = entry

    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_shuffle_diagnostic(all_k_results, results_dir):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # NMSE
    ax = axes[0]
    sae_n = [all_k_results[k]["sae"]["nmse"] for k in K_VALUES]
    tfa_n = [all_k_results[k]["tfa"]["nmse"] for k in K_VALUES]
    shuf_n = [all_k_results[k]["tfa_shuf"]["nmse"] for k in K_VALUES]
    ax.plot(K_VALUES, sae_n, "o-", color="tab:blue", linewidth=2, markersize=7, label="SAE")
    ax.plot(K_VALUES, tfa_n, "s-", color="tab:orange", linewidth=2, markersize=7, label="TFA")
    ax.plot(K_VALUES, shuf_n, "^-", color="tab:red", linewidth=2, markersize=7, label="TFA-shuffled")
    ax.set_xlabel("k"); ax.set_ylabel("NMSE"); ax.set_title(f"NMSE vs k (π={PI_VAL})")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    # Temporal fraction
    ax = axes[1]
    fracs = []
    for k in K_VALUES:
        s = all_k_results[k]["sae"]["nmse"]
        t = all_k_results[k]["tfa"]["nmse"]
        h = all_k_results[k]["tfa_shuf"]["nmse"]
        gap = s - t
        fracs.append((h - t) / gap * 100 if gap > 1e-10 else float("nan"))
    ax.plot(K_VALUES, fracs, "o-", color="tab:green", linewidth=2, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("Temporal fraction (%)")
    ax.set_title(f"Temporal fraction (π={PI_VAL})"); ax.grid(True, alpha=0.3)

    # Shuf/TFA ratio
    ax = axes[2]
    ratios = []
    for k in K_VALUES:
        t = all_k_results[k]["tfa"]["nmse"]
        h = all_k_results[k]["tfa_shuf"]["nmse"]
        ratios.append(h / t if t > 0 else float("inf"))
    ax.plot(K_VALUES, ratios, "s-", color="tab:purple", linewidth=2, markersize=8)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("TFA-shuffled / TFA")
    ax.set_title(f"Temporal benefit ratio (π={PI_VAL})"); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"shuffle_diagnostic.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_direction_analysis(dir_results, results_dir, k):
    rho_groups = sorted(set(RHO))
    # Only plot rho >= 0.5 (long_cont doesn't exist at rho=0.0)
    plot_rhos = [r for r in rho_groups if r >= 0.3]
    plot_idx = list(range(len(plot_rhos)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Top-left: per-feature alignment
    ax = axes[0, 0]
    for cond, color, marker in [
        ("temporal", "tab:green", "o"),
        ("shuffled", "tab:orange", "s"),
        ("random", "tab:gray", "^"),
    ]:
        for cat, ls in [("long_cont", "-"), ("sudden_onset", "--")]:
            vals = [dir_results[cond].get(cat, {}).get(r, {"feat_cos": 0})["feat_cos"]
                    for r in plot_rhos]
            ax.plot(plot_idx, vals, f"{marker}{ls}", color=color, linewidth=2,
                    markersize=6, label=f"{cond} ({cat})")
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("|cos(Dz, f_i)|")
    ax.set_title(f"Per-feature direction alignment (k={k}, π={PI_VAL})")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Top-right: global cosine
    ax = axes[0, 1]
    for cond, color, marker in [
        ("temporal", "tab:green", "o"),
        ("shuffled", "tab:orange", "s"),
        ("random", "tab:gray", "^"),
    ]:
        for cat, ls in [("long_cont", "-"), ("sudden_onset", "--")]:
            vals = [dir_results[cond].get(cat, {}).get(r, {"global_cos": 0})["global_cos"]
                    for r in plot_rhos]
            ax.plot(plot_idx, vals, f"{marker}{ls}", color=color, linewidth=2,
                    markersize=6, label=f"{cond} ({cat})")
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("cos(Dz, x_t)")
    ax.set_title(f"Global direction alignment (k={k}, π={PI_VAL})")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Bottom-left: variance explained
    ax = axes[1, 0]
    for cond, color, marker in [
        ("temporal", "tab:green", "o"),
        ("shuffled", "tab:orange", "s"),
        ("random", "tab:gray", "^"),
    ]:
        for cat, ls in [("long_cont", "-"), ("sudden_onset", "--")]:
            vals = [dir_results[cond].get(cat, {}).get(r, {"var_expl": 0})["var_expl"]
                    for r in plot_rhos]
            ax.plot(plot_idx, vals, f"{marker}{ls}", color=color, linewidth=2,
                    markersize=6, label=f"{cond} ({cat})")
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("Variance explained")
    ax.set_title(f"Variance explained (k={k}, π={PI_VAL})")
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # Bottom-right: temporal/shuffled ratio for per-feature alignment
    ax = axes[1, 1]
    for cat, ls, color in [
        ("long_cont", "-", "tab:green"),
        ("sudden_onset", "--", "tab:orange"),
    ]:
        ratios = []
        for r in plot_rhos:
            t_val = dir_results["temporal"].get(cat, {}).get(r, {"feat_cos": 0})["feat_cos"]
            s_val = dir_results["shuffled"].get(cat, {}).get(r, {"feat_cos": 0})["feat_cos"]
            ratios.append(t_val / s_val if s_val > 1e-8 else 1.0)
        ax.plot(plot_idx, ratios, f"o{ls}", color=color, linewidth=2,
                markersize=7, label=cat)
    ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5)
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("Temporal / Shuffled ratio")
    ax.set_title(f"Per-feature: temporal benefit (k={k}, π={PI_VAL})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Attention direction quality at π={PI_VAL} (k={k})", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"direction_quality_k{k}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_run_length_projections(rl_results, results_dir, k):
    rho_groups = sorted(set(RHO))
    plot_rhos = [r for r in rho_groups if r >= 0.3]
    plot_idx = list(range(len(plot_rhos)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: ON features
    ax = axes[0]
    for cat, color, label in [
        ("long_cont", "tab:green", f"Long cont ({HISTORY_LEN}×ON→ON)"),
        ("sudden_onset", "tab:orange", f"Sudden onset ({HISTORY_LEN}×OFF→ON)"),
    ]:
        vals = [rl_results.get(cat, {}).get(r, {"mean": 0})["mean"] for r in plot_rhos]
        ax.plot(plot_idx, vals, "o-", color=color, linewidth=2, markersize=7, label=label)
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("Mean |⟨pred_recons, f_i⟩|")
    ax.set_title(f"Feature ON: pred projection (k={k}, π={PI_VAL})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Right: OFF features
    ax = axes[1]
    for cat, color, label in [
        ("sudden_offset", "tab:red", f"Sudden offset ({HISTORY_LEN}×ON→OFF)"),
        ("long_absent", "tab:gray", f"Long absent ({HISTORY_LEN}×OFF→OFF)"),
    ]:
        vals = [rl_results.get(cat, {}).get(r, {"mean": 0})["mean"] for r in plot_rhos]
        ax.plot(plot_idx, vals, "o-", color=color, linewidth=2, markersize=7, label=label)
    ax.set_xticks(plot_idx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
    ax.set_xlabel("ρ"); ax.set_ylabel("Mean |⟨pred_recons, f_i⟩|")
    ax.set_title(f"Feature OFF: pred projection (k={k}, π={PI_VAL})")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.suptitle(f"Run-length prediction projections at π={PI_VAL} (k={k})",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"run_length_k{k}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    # ── Setup ────────────────────────────────────────────────────────

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    fd = model.feature_directions.to(device)

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}", flush=True)

    # ── Sanity checks ────────────────────────────────────────────────

    with torch.no_grad():
        acts_check, sup_check = generate_markov_activations(
            5000, SEQ_LEN, pi_t, rho_t, device=device
        )
        l0 = sup_check.sum(-1)
        print(f"\nSanity checks (π={PI_VAL}, {NUM_FEATURES} features):", flush=True)
        print(f"  E[L0] = {l0.mean().item():.2f} (theory: {NUM_FEATURES * PI_VAL})",
              flush=True)
        print(f"  L0 std = {l0.std().item():.2f}", flush=True)

        # Content overlap
        flat = sup_check.reshape(-1, NUM_FEATURES)
        idx = torch.randperm(flat.shape[0])[:10000]
        s1, s2 = flat[idx[:5000]], flat[idx[5000:10000]]
        overlap = (s1 * s2).sum(dim=-1).float()
        print(f"  Content overlap: {overlap.mean().item():.3f} features "
              f"(theory: {NUM_FEATURES * PI_VAL**2:.2f})", flush=True)

        # Feature availability in context
        for r in [0.0, 0.5, 0.9]:
            fi = [i for i in range(NUM_FEATURES) if abs(RHO[i] - r) < 0.01]
            ctx_counts = sup_check[:, :63, :][:, :, fi].sum(dim=1).float().mean(dim=0)
            print(f"  ρ={r}: feature in context at {ctx_counts.mean().item():.1f}/63 positions",
                  flush=True)

        # Autocorrelation check
        for r in [0.3, 0.5, 0.9]:
            fi = [i for i in range(NUM_FEATURES) if abs(RHO[i] - r) < 0.01]
            s = sup_check[:, :, fi[0]]
            a, b = s[:, :-1], s[:, 1:]
            ac = ((a - a.mean()) * (b - b.mean())).mean() / (a.std() * b.std() + 1e-8)
            print(f"  ρ={r}: empirical lag-1 autocorr = {ac.item():.3f}", flush=True)

        del acts_check, sup_check

    # Run-length category counts
    set_seed(SEED + 200)
    _, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    masks_check, h = classify_by_run_length(eval_support.cpu(), HISTORY_LEN)
    print(f"\nRun-length categories (history={HISTORY_LEN}):", flush=True)
    rho_groups = sorted(set(RHO))
    for r in rho_groups:
        fi = [i for i in range(NUM_FEATURES) if abs(RHO[i] - r) < 0.01]
        for cat in ["long_cont", "sudden_onset"]:
            n = sum(masks_check[cat][:, :, i].sum().item() for i in fi)
            print(f"  ρ={r} {cat:>14}: {n:.0f}", flush=True)
    del masks_check

    # ── Generate eval data ───────────────────────────────────────────

    set_seed(SEED + 200)
    acts_eval, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts_eval) * sf

    # ── Data generators ──────────────────────────────────────────────

    gen_flat, gen_seq = make_generators(model, pi_t, rho_t, device, sf, shuffle=False)
    gen_flat_shuf, gen_seq_shuf = make_generators(
        model, pi_t, rho_t, device, sf, shuffle=True
    )

    # ── Part 1: Shuffle diagnostic ───────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print(f"PART 1: Shuffle diagnostic (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR, log_every=TFA_TOTAL_STEPS,
    )

    all_k_results = {}
    trained_tfas = {}  # keep for parts 2-3

    for k in K_VALUES:
        print(f"\n  k={k}:", flush=True)
        k_res = {}

        # SAE
        set_seed(SEED)
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR, l1_coeff=0.0, log_every=SAE_TOTAL_STEPS,
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r_sae = eval_sae(sae, eval_hidden, device)
        print(f"    SAE:      NMSE={r_sae['nmse']:.6f}, L0={r_sae['l0']:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        k_res["sae"] = r_sae
        del sae; torch.cuda.empty_cache()

        # TFA
        set_seed(SEED)
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r_tfa = eval_tfa(tfa, eval_hidden, device)
        print(f"    TFA:      NMSE={r_tfa['nmse']:.6f} ({time.time()-t0:.1f}s)", flush=True)
        k_res["tfa"] = r_tfa
        trained_tfas[(k, "temporal")] = tfa

        # TFA-shuffled
        set_seed(SEED)
        t0 = time.time()
        tfa_shuf = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_shuf, _ = train_tfa(tfa_shuf, gen_seq_shuf, tfa_cfg, device)
        r_shuf = eval_tfa(tfa_shuf, eval_hidden, device)
        print(f"    TFA-shuf: NMSE={r_shuf['nmse']:.6f} ({time.time()-t0:.1f}s)", flush=True)
        k_res["tfa_shuf"] = r_shuf
        trained_tfas[(k, "shuffled")] = tfa_shuf

        # Derived metrics
        sae_n = r_sae["nmse"]; tfa_n = r_tfa["nmse"]; shuf_n = r_shuf["nmse"]
        gap = sae_n - tfa_n
        ratio = shuf_n / tfa_n if tfa_n > 0 else float("inf")
        tf = (shuf_n - tfa_n) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"    >> shuf/tfa={ratio:.3f} | temporal={tf:.1f}%", flush=True)

        all_k_results[k] = k_res

    plot_shuffle_diagnostic(all_k_results, RESULTS_DIR)
    print("Part 1 plots saved.", flush=True)

    # ── Part 2: Direction analysis ───────────────────────────────────

    print(f"\n{'='*70}", flush=True)
    print(f"PART 2: Attention direction analysis (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    dir_results_all_k = {}

    for k in K_VALUES:
        print(f"\n  k={k}:", flush=True)
        dir_results = {}

        for cond_name, tfa_model in [
            ("temporal", trained_tfas[(k, "temporal")]),
            ("shuffled", trained_tfas[(k, "shuffled")]),
        ]:
            all_cos = []; all_ve = []; all_Dz = []; all_x = []
            bs = 256
            with torch.no_grad():
                for start in range(0, EVAL_N_SEQ, bs):
                    end = min(start + bs, EVAL_N_SEQ)
                    x_batch = eval_hidden[start:end].to(device)
                    Dz, x_c = extract_attention_direction(tfa_model, x_batch)
                    cos, ve = compute_direction_metrics(Dz, x_c)
                    all_cos.append(cos.cpu()); all_ve.append(ve.cpu())
                    all_Dz.append(Dz.cpu()); all_x.append(x_c.cpu())

            cosine = torch.cat(all_cos, dim=0)
            var_expl = torch.cat(all_ve, dim=0)
            Dz_all = torch.cat(all_Dz, dim=0)
            x_all = torch.cat(all_x, dim=0)

            print(f"    {cond_name}: cos(Dz,x)={cosine.mean().item():.4f}, "
                  f"VE={var_expl.mean().item():.4f}", flush=True)

            res = analyze_direction_global(
                cosine, var_expl, eval_support, RHO, fd.cpu(), Dz_all,
            )
            dir_results[cond_name] = res

        # Random baseline
        x_ref = torch.cat(all_x, dim=0)  # reuse last x_centered
        rand_dir = torch.randn_like(x_ref)
        rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_r, ve_r = compute_direction_metrics(rand_dir, x_ref)
        print(f"    random: cos(Dz,x)={cos_r.mean().item():.4f}, "
              f"VE={ve_r.mean().item():.4f}", flush=True)
        res_r = analyze_direction_global(
            cos_r, ve_r, eval_support, RHO, fd.cpu(), rand_dir,
        )
        dir_results["random"] = res_r

        # Print key comparisons
        for r in [0.5, 0.9]:
            for cond in ["temporal", "shuffled"]:
                lc = dir_results[cond].get("long_cont", {}).get(r, {})
                so = dir_results[cond].get("sudden_onset", {}).get(r, {})
                if lc.get("n", 0) > 0 and so.get("n", 0) > 0:
                    ratio = lc["feat_cos"] / so["feat_cos"] if so["feat_cos"] > 1e-8 else float("inf")
                    print(f"    {cond} ρ={r}: feat_cos long_cont={lc['feat_cos']:.4f}, "
                          f"sudden_onset={so['feat_cos']:.4f}, ratio={ratio:.3f}", flush=True)

        dir_results_all_k[k] = dir_results
        plot_direction_analysis(dir_results, RESULTS_DIR, k)

    print("Part 2 plots saved.", flush=True)

    # ── Part 3: Run-length prediction projections ────────────────────

    print(f"\n{'='*70}", flush=True)
    print(f"PART 3: Run-length prediction projections (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    rl_results_all_k = {}

    for k in K_VALUES:
        print(f"\n  k={k}:", flush=True)
        tfa_model = trained_tfas[(k, "temporal")]
        pred_proj = compute_pred_projections(tfa_model, eval_hidden, fd, device)

        # Analyze by category
        rl_results = analyze_by_category(
            pred_proj, eval_support, RHO, HISTORY_LEN,
        )

        for r in [0.5, 0.9]:
            lc = rl_results.get("long_cont", {}).get(r, {"mean": 0, "n": 0})
            so = rl_results.get("sudden_onset", {}).get(r, {"mean": 0, "n": 0})
            ratio = lc["mean"] / so["mean"] if so["mean"] > 1e-8 else float("inf")
            print(f"    ρ={r}: long_cont={lc['mean']:.4f} (n={lc['n']:.0f}), "
                  f"sudden_onset={so['mean']:.4f} (n={so['n']:.0f}), "
                  f"ratio={ratio:.3f}", flush=True)

        rl_results_all_k[k] = rl_results
        plot_run_length_projections(rl_results, RESULTS_DIR, k)

    print("Part 3 plots saved.", flush=True)

    # ── Cleanup ──────────────────────────────────────────────────────

    for key in list(trained_tfas.keys()):
        del trained_tfas[key]
    torch.cuda.empty_cache()

    # ── Save ─────────────────────────────────────────────────────────

    def serialize(d):
        if isinstance(d, dict):
            return {str(k): serialize(v) for k, v in d.items()}
        return d

    save_data = {
        "config": {
            "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "pi": PI_VAL, "dict_width": DICT_WIDTH,
            "k_values": K_VALUES, "history_len": HISTORY_LEN,
            "rho": RHO, "seed": SEED, "eval_n_seq": EVAL_N_SEQ,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "sae_total_steps": SAE_TOTAL_STEPS,
        },
        "shuffle_diagnostic": serialize(all_k_results),
        "direction_analysis": serialize(dir_results_all_k),
        "run_length_projections": serialize(rl_results_all_k),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
