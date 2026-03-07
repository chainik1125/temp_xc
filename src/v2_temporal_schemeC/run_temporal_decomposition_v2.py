"""Experiment 3 (v2): Temporal decomposition analysis.

Rewritten to address audit findings:
- Reports mean absolute projections (not ratios) to handle all event types
- Shows all k values
- Includes bootstrap confidence intervals
- Shows offset/absent false-positive predictions
"""

import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.v2_temporal_schemeC.train_tfa import (
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

K_VALUES = [3, 5, 8, 10, 15]
TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 3000  # 3000 * 63 = 189K events per feature
N_BOOTSTRAP = 200
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "temporal_decomposition_v2")


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def gen_seq_fn(model, pi_t, rho_t, device, sf):
    def fn(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return model(acts) * sf
    return fn


# ── Core analysis ────────────────────────────────────────────────────


def compute_per_token_projections(tfa, eval_hidden, feature_dirs, device):
    """Compute per-token projections of pred/novel recons onto each feature direction.

    Returns:
        pred_proj: (n_seq, T-1, n_features) — signed projection of pred_recons onto f_i
        novel_proj: (n_seq, T-1, n_features)
        x_proj: (n_seq, T-1, n_features) — signed projection of input onto f_i
    """
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    T = eval_hidden.shape[1]
    n_features = feature_dirs.shape[0]
    batch_size = 256

    all_pred = torch.zeros(n_seq, T - 1, n_features)
    all_novel = torch.zeros(n_seq, T - 1, n_features)
    all_x = torch.zeros(n_seq, T - 1, n_features)

    fd = feature_dirs.unsqueeze(0).unsqueeze(0)  # (1, 1, n_features, d)

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            x = eval_hidden[start:end].to(device)
            _, inter = tfa(x)

            pr = inter["pred_recons"][:, 1:, :]   # (B, T-1, d)
            nr = inter["novel_recons"][:, 1:, :]
            xc = x[:, 1:, :]

            # Project onto each feature direction: (B, T-1, n_features)
            all_pred[start:end] = (pr.unsqueeze(2) * fd).sum(dim=-1).cpu()
            all_novel[start:end] = (nr.unsqueeze(2) * fd).sum(dim=-1).cpu()
            all_x[start:end] = (xc.unsqueeze(2) * fd).sum(dim=-1).cpu()

    return all_pred, all_novel, all_x


def classify_events(support):
    prev = support[:, :-1, :]
    curr = support[:, 1:, :]
    return {
        "continuation": (prev == 1) & (curr == 1),
        "onset":        (prev == 0) & (curr == 1),
        "offset":       (prev == 1) & (curr == 0),
        "absent":       (prev == 0) & (curr == 0),
    }


def compute_stats_with_bootstrap(values, n_bootstrap=N_BOOTSTRAP):
    """Compute mean and 95% CI via bootstrap."""
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0}
    mean = values.mean().item()
    if n < 10:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "n": n}
    boot_means = []
    for _ in range(n_bootstrap):
        idx = torch.randint(0, n, (n,))
        boot_means.append(values[idx].mean().item())
    boot_means.sort()
    ci_low = boot_means[int(0.025 * n_bootstrap)]
    ci_high = boot_means[int(0.975 * n_bootstrap)]
    return {"mean": mean, "ci_low": ci_low, "ci_high": ci_high, "n": n}


def analyze_all(pred_proj, novel_proj, x_proj, events, rho_list):
    """Compute mean projections per (event_type, rho_group) with CIs."""
    rho_groups = sorted(set(rho_list))
    n_features = pred_proj.shape[2]

    results = {}
    for et, mask in events.items():
        mask_cpu = mask.cpu()
        results[et] = {}
        for rho_val in rho_groups:
            feat_indices = [i for i in range(n_features) if abs(rho_list[i] - rho_val) < 0.01]
            # Pool all events across features in this rho group
            all_pred_vals = []
            all_novel_vals = []
            all_x_vals = []
            for i in feat_indices:
                m = mask_cpu[:, :, i]
                if m.any():
                    all_pred_vals.append(pred_proj[:, :, i][m])
                    all_novel_vals.append(novel_proj[:, :, i][m])
                    all_x_vals.append(x_proj[:, :, i][m])

            if all_pred_vals:
                pv = torch.cat(all_pred_vals)
                nv = torch.cat(all_novel_vals)
                xv = torch.cat(all_x_vals)
                results[et][rho_val] = {
                    "pred": compute_stats_with_bootstrap(pv),
                    "novel": compute_stats_with_bootstrap(nv),
                    "x": compute_stats_with_bootstrap(xv),
                    "pred_abs": compute_stats_with_bootstrap(pv.abs()),
                    "novel_abs": compute_stats_with_bootstrap(nv.abs()),
                    "x_abs": compute_stats_with_bootstrap(xv.abs()),
                }
    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_pred_by_event_and_rho(all_k_results, results_dir):
    """Main figure: mean pred projection for each event type, across rho and k."""
    rho_groups = sorted(set(RHO))
    n_k = len(all_k_results)

    fig, axes = plt.subplots(1, n_k, figsize=(4.5 * n_k, 5), sharey=True)
    if n_k == 1:
        axes = [axes]

    event_colors = {
        "continuation": "tab:green",
        "onset": "tab:orange",
        "offset": "tab:red",
        "absent": "tab:gray",
    }
    event_labels = {
        "continuation": "Continuation (on→on)",
        "onset": "Onset (off→on)",
        "offset": "Offset (on→off)",
        "absent": "Absent (off→off)",
    }

    for ax, (k, results) in zip(axes, sorted(all_k_results.items())):
        x_pos = np.arange(len(rho_groups))
        width = 0.2
        offsets = {"continuation": -1.5, "onset": -0.5, "offset": 0.5, "absent": 1.5}

        for et in ["continuation", "onset", "offset", "absent"]:
            means = []
            ci_lows = []
            ci_highs = []
            for r in rho_groups:
                d = results[et].get(r, {}).get("pred_abs", {"mean": 0, "ci_low": 0, "ci_high": 0})
                means.append(d["mean"])
                ci_lows.append(d["mean"] - d["ci_low"])
                ci_highs.append(d["ci_high"] - d["mean"])

            pos = x_pos + offsets[et] * width
            ax.bar(pos, means, width, label=event_labels[et] if ax == axes[0] else None,
                   color=event_colors[et], alpha=0.8)
            ax.errorbar(pos, means, yerr=[ci_lows, ci_highs], fmt="none",
                        ecolor="black", capsize=2, linewidth=0.8)

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{r}" for r in rho_groups], fontsize=10)
        ax.set_xlabel("$\\rho$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Mean $|\\langle \\hat{x}_{\\mathrm{pred}}, \\mathbf{f}_i \\rangle|$",
                          fontsize=11)
        ax.set_title(f"k = {k}", fontsize=12)
        ax.grid(True, alpha=0.3, axis="y")

    axes[0].legend(fontsize=8, loc="upper left")
    plt.suptitle("Mean |pred projection| onto feature direction by event type",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"pred_proj_by_event.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cont_vs_onset_detail(all_k_results, results_dir):
    """Focused comparison: continuation vs onset pred projection with CIs."""
    rho_groups = sorted(set(RHO))

    fig, axes = plt.subplots(1, len(all_k_results), figsize=(4.5 * len(all_k_results), 5),
                              sharey=True)
    if len(all_k_results) == 1:
        axes = [axes]

    for ax, (k, results) in zip(axes, sorted(all_k_results.items())):
        cont_means = []
        cont_ci = []
        onset_means = []
        onset_ci = []
        for r in rho_groups:
            cd = results["continuation"].get(r, {}).get("pred_abs", {"mean": 0, "ci_low": 0, "ci_high": 0})
            od = results["onset"].get(r, {}).get("pred_abs", {"mean": 0, "ci_low": 0, "ci_high": 0})
            cont_means.append(cd["mean"])
            cont_ci.append([cd["mean"] - cd["ci_low"], cd["ci_high"] - cd["mean"]])
            onset_means.append(od["mean"])
            onset_ci.append([od["mean"] - od["ci_low"], od["ci_high"] - od["mean"]])

        x = np.arange(len(rho_groups))
        w = 0.3
        ax.bar(x - w/2, cont_means, w, label="Continuation", color="tab:green", alpha=0.8)
        ax.errorbar(x - w/2, cont_means, yerr=np.array(cont_ci).T, fmt="none",
                    ecolor="black", capsize=3, linewidth=1)
        ax.bar(x + w/2, onset_means, w, label="Onset", color="tab:orange", alpha=0.8)
        ax.errorbar(x + w/2, onset_means, yerr=np.array(onset_ci).T, fmt="none",
                    ecolor="black", capsize=3, linewidth=1)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}" for r in rho_groups])
        ax.set_xlabel("$\\rho$")
        if ax == axes[0]:
            ax.set_ylabel("Mean $|\\langle \\hat{x}_{\\mathrm{pred}}, \\mathbf{f}_i \\rangle|$")
        ax.set_title(f"k = {k}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Continuation vs Onset: predictable component projection (with 95% CI)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"cont_vs_onset_ci.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_false_positive_rate(all_k_results, results_dir):
    """Show pred projection when feature is OFF (offset/absent) vs ON (cont/onset)."""
    rho_groups = sorted(set(RHO))

    fig, axes = plt.subplots(1, len(all_k_results), figsize=(4.5 * len(all_k_results), 5),
                              sharey=True)
    if len(all_k_results) == 1:
        axes = [axes]

    for ax, (k, results) in zip(axes, sorted(all_k_results.items())):
        # "on" = average of continuation and onset
        # "off" = average of offset and absent
        on_means = []
        off_means = []
        for r in rho_groups:
            cont_d = results["continuation"].get(r, {}).get("pred_abs", {"mean": 0})
            onset_d = results["onset"].get(r, {}).get("pred_abs", {"mean": 0})
            offset_d = results["offset"].get(r, {}).get("pred_abs", {"mean": 0})
            absent_d = results["absent"].get(r, {}).get("pred_abs", {"mean": 0})
            on_means.append((cont_d["mean"] + onset_d["mean"]) / 2)
            off_means.append((offset_d["mean"] + absent_d["mean"]) / 2)

        x = np.arange(len(rho_groups))
        w = 0.3
        ax.bar(x - w/2, on_means, w, label="Feature ON", color="tab:green", alpha=0.8)
        ax.bar(x + w/2, off_means, w, label="Feature OFF", color="tab:red", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}" for r in rho_groups])
        ax.set_xlabel("$\\rho$")
        if ax == axes[0]:
            ax.set_ylabel("Mean $|\\langle \\hat{x}_{\\mathrm{pred}}, \\mathbf{f}_i \\rangle|$")
        ax.set_title(f"k = {k}")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Pred projection when feature is ON vs OFF (false positive analysis)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"false_positive.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    gen_seq = gen_seq_fn(model, pi_t, rho_t, device, sf)
    fd = model.feature_directions.to(device)

    # Fixed eval data
    set_seed(SEED + 200)
    acts, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts) * sf
    events = classify_events(eval_support)

    print(f"Eval: {EVAL_N_SEQ} seqs x {SEQ_LEN} pos = {EVAL_N_SEQ * (SEQ_LEN-1)} events/feature")
    for et, mask in events.items():
        print(f"  {et}: {mask.sum().item():.0f} total ({100*mask.float().mean().item():.1f}%)")

    all_k_results = {}

    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n{'='*60}")
        print(f"TFA k={k}")
        print(f"{'='*60}")

        t0 = time.time()
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
                         n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa_cfg = TFATrainingConfig(total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
                                     lr=TFA_LR, log_every=TFA_TOTAL_STEPS)
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        print(f"Training: {time.time()-t0:.1f}s")

        t0 = time.time()
        pred_proj, novel_proj, x_proj = compute_per_token_projections(
            tfa, eval_hidden, fd, device
        )
        results = analyze_all(pred_proj, novel_proj, x_proj, events, RHO)
        all_k_results[k] = results
        print(f"Analysis: {time.time()-t0:.1f}s")

        # Print summary table
        rho_groups = sorted(set(RHO))
        print(f"\n{'event':>13} {'rho':>5} | {'mean |pred|':>11} {'95% CI':>16} | "
              f"{'mean |novel|':>12} | {'mean |x|':>9} | {'n':>7}")
        print("-" * 85)
        for et in ["continuation", "onset", "offset", "absent"]:
            for r in rho_groups:
                d = results[et].get(r, {})
                pd = d.get("pred_abs", {"mean":0,"ci_low":0,"ci_high":0,"n":0})
                nd = d.get("novel_abs", {"mean":0,"ci_low":0,"ci_high":0})
                xd = d.get("x_abs", {"mean":0,"ci_low":0,"ci_high":0})
                ci = f"[{pd['ci_low']:.4f}, {pd['ci_high']:.4f}]"
                print(f"{et:>13} {r:>5.1f} | {pd['mean']:>11.4f} {ci:>16} | "
                      f"{nd['mean']:>12.4f} | {xd['mean']:>9.4f} | {pd['n']:>7}")
            print()

        del tfa
        torch.cuda.empty_cache()

    # Plots
    print("Generating plots...")
    plot_pred_by_event_and_rho(all_k_results, RESULTS_DIR)
    plot_cont_vs_onset_detail(all_k_results, RESULTS_DIR)
    plot_false_positive_rate(all_k_results, RESULTS_DIR)

    # Save results (convert to serializable)
    def serialize_stats(d):
        return {k: v for k, v in d.items()}

    save_data = {
        "config": {
            "k_values": K_VALUES, "n_features": NUM_FEATURES, "rho": RHO,
            "eval_n_seq": EVAL_N_SEQ, "n_bootstrap": N_BOOTSTRAP, "seed": SEED,
        },
        "results": {
            str(k): {
                et: {
                    str(r): {
                        comp: serialize_stats(v[comp])
                        for comp in ["pred_abs", "novel_abs", "x_abs"]
                    }
                    for r, v in rho_data.items()
                }
                for et, rho_data in results.items()
            }
            for k, results in all_k_results.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
