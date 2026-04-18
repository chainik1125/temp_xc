"""Attention direction quality analysis.

Measures how good the attention's proposed direction is *before* proj_scale,
to separate two sources of TFA's reconstruction benefit:
  1. Temporal prediction — attention produces a direction informed by context
  2. Projection onto x_t — proj_scale extracts x_t's component along any direction

Three conditions:
  - Temporal: TFA trained on temporal data
  - Shuffled: TFA trained on shuffled data
  - Random: random unit direction (no attention)

Metrics (all computed before proj_scale):
  - Cosine similarity: cos(Dz_pred, x_t)
  - Variance explained: <Dz_pred_hat, x_t>^2 / ||x_t||^2

Also breaks down by run-length category (long continuation vs sudden onset).

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_attention_direction_analysis.py
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
from src.data.toy.toy_model import ToyModel
from src.data.toy.markov import generate_markov_activations
from src.training.train_tfa import (
    TFATrainingConfig,
    create_tfa,
    train_tfa,
)

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40

K_VALUES = [5, 8]
HISTORY_LEN = 5

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 3000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "attention_direction_analysis"
)


# ── Extract raw attention direction ──────────────────────────────────


def extract_attention_direction(tfa, x_seq):
    """Run TFA forward but return the raw attention direction before proj_scale.

    Args:
        tfa: Trained TemporalSAE.
        x_seq: (B, T, dimin) input sequences.

    Returns:
        Dz_pred: (B, T, dimin) — raw attention direction in input space.
        x_centered: (B, T, dimin) — input minus bias (what proj_scale sees).
    """
    tfa.eval()
    B, L, dimin = x_seq.size()
    E = tfa.D.T if tfa.tied_weights else tfa.E

    with torch.no_grad():
        x_input = x_seq - tfa.b  # same as forward pass line 71

        # Only one attention layer in our setup
        attn_layer = tfa.attn_layers[0]

        z_input = F.relu(torch.matmul(x_input * tfa.lam, E))
        z_ctx = torch.cat(
            (torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()),
            dim=1,
        )

        z_pred_, _ = attn_layer(z_ctx, z_input, get_attn_map=False)
        z_pred_ = F.relu(z_pred_)
        Dz_pred = torch.matmul(z_pred_, tfa.D)  # (B, T, dimin)

    return Dz_pred, x_input


def compute_direction_metrics(Dz_pred, x_input):
    """Compute cosine similarity and variance explained between direction and input.

    Args:
        Dz_pred: (B, T, dimin) — proposed direction.
        x_input: (B, T, dimin) — centered input.

    Returns:
        cosine: (B, T) — cosine similarity per token.
        var_explained: (B, T) — fraction of ||x||^2 captured by projection.
    """
    # Cosine similarity
    Dz_norm = Dz_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_norm = x_input.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine = (Dz_pred * x_input).sum(dim=-1) / (Dz_norm.squeeze(-1) * x_norm.squeeze(-1))

    # Variance explained: (projection length)^2 / ||x||^2
    # projection of x onto Dz_hat: <Dz_hat, x> where Dz_hat = Dz/||Dz||
    proj_coeff = (Dz_pred * x_input).sum(dim=-1) / Dz_norm.squeeze(-1)  # scalar proj
    var_explained = proj_coeff.pow(2) / x_input.pow(2).sum(dim=-1).clamp(min=1e-8)

    return cosine, var_explained


def random_direction_metrics(x_input):
    """Compute metrics for random unit directions as a baseline.

    Args:
        x_input: (B, T, dimin).

    Returns:
        cosine: (B, T)
        var_explained: (B, T)
    """
    rand_dir = torch.randn_like(x_input)
    rand_dir = rand_dir / rand_dir.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    # Scale to match typical Dz_pred norms (doesn't affect cosine or VE)
    return compute_direction_metrics(rand_dir, x_input)


# ── Run-length classification (reused from v3) ──────────────────────


def classify_by_run_length(support, history_len=HISTORY_LEN):
    """Classify each (feature, position) by recent run-length pattern."""
    B, T, nf = support.shape
    h = history_len
    valid_T = T - h

    current = support[:, h:, :]
    history_sum = torch.zeros(B, valid_T, nf, device=support.device)
    for i in range(h):
        history_sum += support[:, i:i + valid_T, :]

    all_on_before = (history_sum == h)
    all_off_before = (history_sum == 0)
    curr_on = (current == 1)
    curr_off = (current == 0)

    masks = {
        "long_cont": all_on_before & curr_on,
        "sudden_onset": all_off_before & curr_on,
        "long_absent": all_off_before & curr_off,
        "sudden_offset": all_on_before & curr_off,
    }
    return masks, h


# ── Analysis ─────────────────────────────────────────────────────────


def analyze_direction_by_category(
    cosine, var_explained, support, feature_dirs, Dz_pred, x_input, rho_list,
):
    """Break down direction quality by run-length category and rho group.

    Also computes per-feature direction quality: cosine between Dz_pred and f_i.
    """
    support_cpu = support.cpu()
    masks, h = classify_by_run_length(support_cpu, HISTORY_LEN)

    # Trim to valid positions
    cos_valid = cosine[:, h:]
    ve_valid = var_explained[:, h:]
    Dz_valid = Dz_pred[:, h:]
    x_valid = x_input[:, h:]

    rho_groups = sorted(set(rho_list))
    n_features = support.shape[2]

    results = {}
    for cat_name, mask in masks.items():
        results[cat_name] = {}
        for rho_val in rho_groups:
            feat_indices = [
                i for i in range(n_features)
                if abs(rho_list[i] - rho_val) < 0.01
            ]

            # Global metrics (cosine, VE) — pooled across features in this rho group
            # These are per-token metrics, so we pick tokens where ANY feature
            # in this rho group matches the category
            # Actually, let's compute per-feature direction quality:
            # For each feature i, measure how much the direction aligns with f_i
            all_feat_cos = []
            all_global_cos = []
            all_global_ve = []
            n_events = 0

            for i in feat_indices:
                m = mask[:, :, i]  # (B, valid_T) — on CPU
                if not m.any():
                    continue
                n_events += m.sum().item()

                # Per-feature: cosine between Dz_pred and f_i at masked positions
                fi = feature_dirs[i].cpu()  # (dimin,)
                Dz_masked = Dz_valid[m]  # (N, dimin) — both on CPU
                Dz_hat = Dz_masked / Dz_masked.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                feat_cos = (Dz_hat * fi.unsqueeze(0)).sum(dim=-1).abs()  # |cos(Dz, f_i)|
                all_feat_cos.append(feat_cos)

                # Global direction quality at these positions
                all_global_cos.append(cos_valid[m])
                all_global_ve.append(ve_valid[m])

            if all_feat_cos:
                fc = torch.cat(all_feat_cos)
                gc = torch.cat(all_global_cos)
                gv = torch.cat(all_global_ve)
                results[cat_name][rho_val] = {
                    "feat_cos_mean": fc.mean().item(),
                    "feat_cos_std": fc.std().item(),
                    "global_cos_mean": gc.mean().item(),
                    "global_ve_mean": gv.mean().item(),
                    "n": n_events,
                }
            else:
                results[cat_name][rho_val] = {
                    "feat_cos_mean": 0, "feat_cos_std": 0,
                    "global_cos_mean": 0, "global_ve_mean": 0, "n": 0,
                }

    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_direction_comparison(all_results, results_dir):
    """Plot direction quality for temporal vs shuffled vs random."""
    rho_groups = sorted(set(RHO))

    for k, k_results in sorted(all_results.items()):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Top row: feature-specific cosine |cos(Dz, f_i)|
        # Bottom row: global direction quality

        for col, (cat_on, cat_off, title_on, title_off) in enumerate([
            ("long_cont", "sudden_onset",
             f"Feature ON: long cont (5×ON→ON)",
             f"Feature ON: sudden onset (5×OFF→ON)"),
        ]):
            # Top-left: feature cosine for long_cont vs sudden_onset
            ax = axes[0, 0]
            for cond_name, cond_results, color, marker in [
                ("Temporal", k_results["temporal"], "tab:green", "o"),
                ("Shuffled", k_results["shuffled"], "tab:orange", "s"),
                ("Random", k_results["random"], "tab:gray", "^"),
            ]:
                for cat, ls, label_suffix in [
                    ("long_cont", "-", "long cont"),
                    ("sudden_onset", "--", "sudden onset"),
                ]:
                    means = [cond_results.get(cat, {}).get(r, {"feat_cos_mean": 0})["feat_cos_mean"]
                             for r in rho_groups]
                    ax.plot(range(len(rho_groups)), means, f"{marker}{ls}",
                            color=color, linewidth=2, markersize=6,
                            label=f"{cond_name} ({label_suffix})")

            ax.set_xticks(range(len(rho_groups)))
            ax.set_xticklabels([f"{r}" for r in rho_groups])
            ax.set_xlabel("$\\rho$")
            ax.set_ylabel("$|\\cos(D z_{\\mathrm{pred}}, \\mathbf{f}_i)|$")
            ax.set_title(f"Per-feature direction alignment (k={k})")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # Top-right: global cosine cos(Dz, x_t)
            ax = axes[0, 1]
            for cond_name, cond_results, color, marker in [
                ("Temporal", k_results["temporal"], "tab:green", "o"),
                ("Shuffled", k_results["shuffled"], "tab:orange", "s"),
                ("Random", k_results["random"], "tab:gray", "^"),
            ]:
                for cat, ls, label_suffix in [
                    ("long_cont", "-", "long cont"),
                    ("sudden_onset", "--", "sudden onset"),
                ]:
                    means = [cond_results.get(cat, {}).get(r, {"global_cos_mean": 0})["global_cos_mean"]
                             for r in rho_groups]
                    ax.plot(range(len(rho_groups)), means, f"{marker}{ls}",
                            color=color, linewidth=2, markersize=6,
                            label=f"{cond_name} ({label_suffix})")

            ax.set_xticks(range(len(rho_groups)))
            ax.set_xticklabels([f"{r}" for r in rho_groups])
            ax.set_xlabel("$\\rho$")
            ax.set_ylabel("$\\cos(D z_{\\mathrm{pred}}, x_t)$")
            ax.set_title(f"Global direction–input alignment (k={k})")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # Bottom-left: variance explained
            ax = axes[1, 0]
            for cond_name, cond_results, color, marker in [
                ("Temporal", k_results["temporal"], "tab:green", "o"),
                ("Shuffled", k_results["shuffled"], "tab:orange", "s"),
                ("Random", k_results["random"], "tab:gray", "^"),
            ]:
                for cat, ls, label_suffix in [
                    ("long_cont", "-", "long cont"),
                    ("sudden_onset", "--", "sudden onset"),
                ]:
                    means = [cond_results.get(cat, {}).get(r, {"global_ve_mean": 0})["global_ve_mean"]
                             for r in rho_groups]
                    ax.plot(range(len(rho_groups)), means, f"{marker}{ls}",
                            color=color, linewidth=2, markersize=6,
                            label=f"{cond_name} ({label_suffix})")

            ax.set_xticks(range(len(rho_groups)))
            ax.set_xticklabels([f"{r}" for r in rho_groups])
            ax.set_xlabel("$\\rho$")
            ax.set_ylabel("Variance explained")
            ax.set_title(f"Variance explained by direction (k={k})")
            ax.legend(fontsize=7, ncol=2)
            ax.grid(True, alpha=0.3)

            # Bottom-right: feature cosine ratio (temporal / shuffled)
            ax = axes[1, 1]
            for cat, ls, color, label in [
                ("long_cont", "-", "tab:green", "Long continuation"),
                ("sudden_onset", "--", "tab:orange", "Sudden onset"),
            ]:
                ratios = []
                for r in rho_groups:
                    t_val = k_results["temporal"].get(cat, {}).get(r, {"feat_cos_mean": 0})["feat_cos_mean"]
                    s_val = k_results["shuffled"].get(cat, {}).get(r, {"feat_cos_mean": 0})["feat_cos_mean"]
                    ratios.append(t_val / s_val if s_val > 1e-8 else 1.0)
                ax.plot(range(len(rho_groups)), ratios, f"o{ls}", color=color,
                        linewidth=2, markersize=7, label=label)

            ax.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5,
                        label="No temporal benefit")
            ax.set_xticks(range(len(rho_groups)))
            ax.set_xticklabels([f"{r}" for r in rho_groups])
            ax.set_xlabel("$\\rho$")
            ax.set_ylabel("Temporal / Shuffled ratio")
            ax.set_title(f"Temporal benefit in direction quality (k={k})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Attention direction quality: Temporal vs Shuffled vs Random (k={k})",
            fontsize=14, y=1.02,
        )
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join(results_dir, f"direction_quality_k{k}.{ext}"),
                dpi=150, bbox_inches="tight",
            )
        plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    fd = model.feature_directions.to(device)  # (n_features, dimin)

    # Scaling factor
    with torch.no_grad():
        acts_sf, _ = generate_markov_activations(
            10000, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden_sf = model(acts_sf)
        sf = math.sqrt(HIDDEN_DIM) / hidden_sf.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    print(f"Scaling factor: {sf:.4f}", flush=True)

    # Data generators
    def gen_seq_temporal(n_seq):
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * sf

    def gen_seq_shuffled(n_seq):
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        for i in range(n_seq):
            perm = torch.randperm(SEQ_LEN, device=device)
            acts[i] = acts[i, perm]
        return model(acts) * sf

    # Fixed eval data (temporal, unshuffled)
    set_seed(SEED + 200)
    acts_eval, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts_eval) * sf

    # Report run-length category counts
    masks, h = classify_by_run_length(eval_support, HISTORY_LEN)
    print(f"\nRun-length categories (history={HISTORY_LEN}):", flush=True)
    for cat, mask in masks.items():
        print(f"  {cat:>16}: {mask.sum().item():.0f}", flush=True)

    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS,
        batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR,
        log_every=TFA_TOTAL_STEPS,
    )

    all_results = {}

    for k in K_VALUES:
        print(f"\n{'='*70}", flush=True)
        print(f"k = {k}", flush=True)
        print(f"{'='*70}", flush=True)

        k_results = {}

        # ── Train temporal TFA ───────────────────────────────────────
        set_seed(SEED)
        t0 = time.time()
        tfa_temporal = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1,
            device=device,
        )
        tfa_temporal, _ = train_tfa(tfa_temporal, gen_seq_temporal, tfa_cfg, device)
        print(f"Temporal TFA trained: {time.time()-t0:.1f}s", flush=True)

        # ── Train shuffled TFA ───────────────────────────────────────
        set_seed(SEED)
        t0 = time.time()
        tfa_shuffled = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1,
            device=device,
        )
        tfa_shuffled, _ = train_tfa(tfa_shuffled, gen_seq_shuffled, tfa_cfg, device)
        print(f"Shuffled TFA trained: {time.time()-t0:.1f}s", flush=True)

        # ── Extract directions and compute metrics ───────────────────

        batch_size = 256
        all_cos = {"temporal": [], "shuffled": [], "random": []}
        all_ve = {"temporal": [], "shuffled": [], "random": []}
        all_Dz = {"temporal": [], "shuffled": []}
        all_x_centered = []

        with torch.no_grad():
            for start in range(0, EVAL_N_SEQ, batch_size):
                end = min(start + batch_size, EVAL_N_SEQ)
                x_batch = eval_hidden[start:end].to(device)

                # Temporal
                Dz_t, x_c = extract_attention_direction(tfa_temporal, x_batch)
                cos_t, ve_t = compute_direction_metrics(Dz_t, x_c)
                all_cos["temporal"].append(cos_t.cpu())
                all_ve["temporal"].append(ve_t.cpu())
                all_Dz["temporal"].append(Dz_t.cpu())

                # Shuffled
                Dz_s, _ = extract_attention_direction(tfa_shuffled, x_batch)
                cos_s, ve_s = compute_direction_metrics(Dz_s, x_c)
                all_cos["shuffled"].append(cos_s.cpu())
                all_ve["shuffled"].append(ve_s.cpu())
                all_Dz["shuffled"].append(Dz_s.cpu())

                # Random
                cos_r, ve_r = random_direction_metrics(x_c)
                all_cos["random"].append(cos_r.cpu())
                all_ve["random"].append(ve_r.cpu())

                all_x_centered.append(x_c.cpu())

        # Concatenate
        for cond in ["temporal", "shuffled", "random"]:
            all_cos[cond] = torch.cat(all_cos[cond], dim=0)
            all_ve[cond] = torch.cat(all_ve[cond], dim=0)
        for cond in ["temporal", "shuffled"]:
            all_Dz[cond] = torch.cat(all_Dz[cond], dim=0)
        x_centered = torch.cat(all_x_centered, dim=0)

        # Global summary
        print(f"\nGlobal direction quality (all tokens):", flush=True)
        print(f"  {'Condition':>12} | {'cos(Dz,x)':>10} | {'VarExpl':>10}", flush=True)
        print(f"  {'-'*40}", flush=True)
        for cond in ["temporal", "shuffled", "random"]:
            mc = all_cos[cond].mean().item()
            mv = all_ve[cond].mean().item()
            print(f"  {cond:>12} | {mc:>10.4f} | {mv:>10.4f}", flush=True)

        # Per-category breakdown
        for cond_name, cond_cos, cond_ve, cond_Dz in [
            ("temporal", all_cos["temporal"], all_ve["temporal"], all_Dz["temporal"]),
            ("shuffled", all_cos["shuffled"], all_ve["shuffled"], all_Dz["shuffled"]),
            ("random", all_cos["random"], all_ve["random"], None),
        ]:
            results = analyze_direction_by_category(
                cond_cos, cond_ve, eval_support, fd,
                cond_Dz if cond_Dz is not None else torch.randn_like(x_centered),
                x_centered, RHO,
            )
            k_results[cond_name] = results

            rho_groups = sorted(set(RHO))
            print(f"\n  {cond_name} — per-category breakdown:", flush=True)
            print(f"  {'category':>16} {'rho':>5} | {'|cos(Dz,fi)|':>12} "
                  f"{'cos(Dz,x)':>10} {'VarExpl':>10} | {'n':>8}", flush=True)
            print(f"  {'-'*75}", flush=True)
            for cat in ["long_cont", "sudden_onset", "sudden_offset", "long_absent"]:
                for r in rho_groups:
                    d = results.get(cat, {}).get(r, {
                        "feat_cos_mean": 0, "global_cos_mean": 0,
                        "global_ve_mean": 0, "n": 0,
                    })
                    print(
                        f"  {cat:>16} {r:>5.1f} | {d['feat_cos_mean']:>12.4f} "
                        f"{d['global_cos_mean']:>10.4f} {d['global_ve_mean']:>10.4f} | "
                        f"{d['n']:>8}", flush=True,
                    )
                print(flush=True)

        all_results[k] = k_results

        del tfa_temporal, tfa_shuffled
        torch.cuda.empty_cache()

    # ── Plots ────────────────────────────────────────────────────────

    plot_direction_comparison(all_results, RESULTS_DIR)
    print("Plots saved.", flush=True)

    # ── Save ─────────────────────────────────────────────────────────

    def serialize(d):
        if isinstance(d, dict):
            return {str(k): serialize(v) for k, v in d.items()}
        return d

    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "k_values": K_VALUES,
            "history_len": HISTORY_LEN,
            "rho": RHO,
            "pi": PI,
            "seed": SEED,
            "eval_n_seq": EVAL_N_SEQ,
            "tfa_total_steps": TFA_TOTAL_STEPS,
        },
        "results": serialize(all_results),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
