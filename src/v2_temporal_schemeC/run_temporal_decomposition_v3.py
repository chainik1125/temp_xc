"""Experiment 3b: Run-length temporal decomposition analysis.

Extends the v2 temporal decomposition by conditioning on the full recent
history (run length) rather than just the lag-1 transition. This is a
sharper test of whether TFA's predictable component exploits temporal
persistence.

Key comparisons:
  - Long continuation (11111→1) vs sudden onset (00001→observed at t)
  - Sudden offset (11110→0) vs long absence (00000→observed at t)

If TFA exploits temporal structure, long continuations should have much
higher prediction projections than sudden onsets — the attention has seen
the feature ON for many consecutive positions and should predict it will
stay ON.

Additionally tests whether sequence length matters by comparing T=64 vs
T=256 on a single k value.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_temporal_decomposition_v3.py
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
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40

# Run-length history window: how many past positions to condition on
HISTORY_LEN = 5  # classify based on s_{t-5}, ..., s_{t-1}, s_t

# Main experiment: T=64, k=8 (strong binding regime)
SEQ_LEN = 64
K_VALUES = [5, 8]

# Sequence length comparison: T=64 vs T=256 at k=8
SEQ_LENS_COMPARE = [64, 256]
K_FOR_SEQLEN = 8

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 3000
N_BOOTSTRAP = 200
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "temporal_decomposition_v3"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, seq_len, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            10000, seq_len, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return (
            math.sqrt(HIDDEN_DIM)
            / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
        )


def gen_seq_fn(model, pi_t, rho_t, seq_len, device, sf):
    def fn(n_seq):
        acts, _ = generate_markov_activations(
            n_seq, seq_len, pi_t, rho_t, device=device
        )
        return model(acts) * sf
    return fn


def compute_per_token_projections(tfa, eval_hidden, feature_dirs, device):
    """Compute per-token projections of pred recons onto each feature direction.

    Returns:
        pred_proj: (n_seq, T, n_features)
    """
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    T = eval_hidden.shape[1]
    n_features = feature_dirs.shape[0]
    batch_size = 256

    all_pred = torch.zeros(n_seq, T, n_features)
    fd = feature_dirs.unsqueeze(0).unsqueeze(0)  # (1, 1, n_features, d)

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            x = eval_hidden[start:end].to(device)
            _, inter = tfa(x)
            pr = inter["pred_recons"]  # (B, T, d)
            all_pred[start:end] = (pr.unsqueeze(2) * fd).sum(dim=-1).cpu()

    return all_pred


def compute_stats(values, n_bootstrap=N_BOOTSTRAP):
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


# ── Run-length classification ────────────────────────────────────────


def classify_by_run_length(support, history_len=HISTORY_LEN):
    """Classify each (feature, position) by its recent run-length pattern.

    For each position t >= history_len, we look at s_{t-history_len+1:t+1}
    (the history_len most recent values including t).

    Categories:
      - long_continuation: all 1s in the window (e.g. 11111)
      - sudden_onset: all 0s except the last position is 1 (e.g. 00001)
      - sudden_offset: all 1s except the last position is 0 (e.g. 11110)
      - long_absence: all 0s in the window (e.g. 00000)
      - short_continuation: current=1, prev=1, but not all 1s (e.g. 01011)
      - short_onset: current=1, prev=0, but not sudden onset (e.g. 10101)

    Args:
        support: (batch, seq_len, n_features) binary tensor.
        history_len: Number of positions in the history window.

    Returns:
        Dict mapping category name to boolean mask of shape
        (batch, seq_len - history_len, n_features). The time dimension
        is shifted: mask[:, t, :] corresponds to position t + history_len
        in the original support.
    """
    B, T, F = support.shape
    h = history_len
    valid_T = T - h  # positions where we have full history

    # Extract windows: (B, valid_T, F, h+1) — but we only need h positions
    # support[:, t:t+h, :] for t in range(valid_T), plus the current position
    # Actually we want positions [t, t+1, ..., t+h-1] = h positions ending at t+h-1
    # Let's build the history for each valid position

    # Current state at each valid position
    current = support[:, h:, :]  # (B, valid_T, F) — positions h, h+1, ..., T-1

    # Sum of support over the history window (including current position)
    # For position t (0-indexed in valid range), history is support[:, t:t+h, :]
    history_sum = torch.zeros(B, valid_T, F, device=support.device)
    for i in range(h):
        history_sum += support[:, i:i + valid_T, :]
    # history_sum now counts how many of the h positions before current were ON
    # (positions t, t+1, ..., t+h-1 for valid position index mapping to t+h)

    # Previous position
    prev = support[:, h - 1:T - 1, :]  # (B, valid_T, F)

    # All ON in history (h positions before current, all 1s)
    all_on_before = (history_sum == h)  # (B, valid_T, F)
    # All OFF in history
    all_off_before = (history_sum == 0)

    curr_on = (current == 1)
    curr_off = (current == 0)
    prev_on = (prev == 1)
    prev_off = (prev == 0)

    masks = {
        # Strong patterns
        "long_cont": all_on_before & curr_on,       # 1...11 → 1
        "sudden_onset": all_off_before & curr_on,    # 0...00 → 1
        "sudden_offset": all_on_before & curr_off,   # 1...11 → 0
        "long_absent": all_off_before & curr_off,    # 0...00 → 0
        # Weak patterns (lag-1 matches but not full history)
        "short_cont": prev_on & curr_on & ~all_on_before,
        "short_onset": prev_off & curr_on & ~all_off_before,
    }

    return masks, h


# ── Analysis ─────────────────────────────────────────────────────────


def analyze_run_length(pred_proj, support, rho_list, history_len=HISTORY_LEN):
    """Compute mean |pred projection| for each run-length category and rho group."""
    masks, h = classify_by_run_length(support, history_len)

    # pred_proj is (B, T, F) — trim to match valid positions
    pred_valid = pred_proj[:, h:, :]  # (B, valid_T, F)

    rho_groups = sorted(set(rho_list))
    n_features = pred_proj.shape[2]

    results = {}
    for cat_name, mask in masks.items():
        mask_cpu = mask.cpu()
        results[cat_name] = {}
        for rho_val in rho_groups:
            feat_indices = [
                i for i in range(n_features)
                if abs(rho_list[i] - rho_val) < 0.01
            ]
            all_vals = []
            for i in feat_indices:
                m = mask_cpu[:, :, i]
                if m.any():
                    all_vals.append(pred_valid[:, :, i][m].abs())
            if all_vals:
                pooled = torch.cat(all_vals)
                results[cat_name][rho_val] = compute_stats(pooled)
            else:
                results[cat_name][rho_val] = {
                    "mean": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n": 0
                }

    return results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_run_length_comparison(all_results, results_dir, suffix=""):
    """Main figure: long_cont vs sudden_onset and sudden_offset vs long_absent."""
    rho_groups = sorted(set(RHO))

    for k, results in sorted(all_results.items()):
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Left: long continuation vs sudden onset
        ax = axes[0]
        for cat, color, label in [
            ("long_cont", "tab:green", f"Long continuation ({HISTORY_LEN}×ON → ON)"),
            ("sudden_onset", "tab:orange", f"Sudden onset ({HISTORY_LEN}×OFF → ON)"),
            ("short_cont", "tab:blue", "Short continuation (mixed → ON,ON)"),
        ]:
            means = []
            ci_lo = []
            ci_hi = []
            for r in rho_groups:
                d = results.get(cat, {}).get(r, {"mean": 0, "ci_low": 0, "ci_high": 0})
                means.append(d["mean"])
                ci_lo.append(d["mean"] - d["ci_low"])
                ci_hi.append(d["ci_high"] - d["mean"])
            x = np.arange(len(rho_groups))
            ax.errorbar(x, means, yerr=[ci_lo, ci_hi], fmt="o-", color=color,
                        linewidth=2, markersize=7, capsize=4, label=label)

        ax.set_xticks(range(len(rho_groups)))
        ax.set_xticklabels([f"{r}" for r in rho_groups])
        ax.set_xlabel("$\\rho$", fontsize=12)
        ax.set_ylabel(
            "Mean $|\\langle \\hat{x}_{\\mathrm{pred}}, \\mathbf{f}_i \\rangle|$",
            fontsize=11,
        )
        ax.set_title("Feature currently ON", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Right: sudden offset vs long absence
        ax = axes[1]
        for cat, color, label in [
            ("sudden_offset", "tab:red", f"Sudden offset ({HISTORY_LEN}×ON → OFF)"),
            ("long_absent", "tab:gray", f"Long absence ({HISTORY_LEN}×OFF → OFF)"),
        ]:
            means = []
            ci_lo = []
            ci_hi = []
            for r in rho_groups:
                d = results.get(cat, {}).get(r, {"mean": 0, "ci_low": 0, "ci_high": 0})
                means.append(d["mean"])
                ci_lo.append(d["mean"] - d["ci_low"])
                ci_hi.append(d["ci_high"] - d["mean"])
            x = np.arange(len(rho_groups))
            ax.errorbar(x, means, yerr=[ci_lo, ci_hi], fmt="o-", color=color,
                        linewidth=2, markersize=7, capsize=4, label=label)

        ax.set_xticks(range(len(rho_groups)))
        ax.set_xticklabels([f"{r}" for r in rho_groups])
        ax.set_xlabel("$\\rho$", fontsize=12)
        ax.set_title("Feature currently OFF", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.suptitle(
            f"Pred projection by run-length history (k={k}, "
            f"history={HISTORY_LEN}){suffix}",
            fontsize=13, y=1.02,
        )
        plt.tight_layout()
        tag = f"_k{k}" if len(all_results) > 1 else ""
        for ext in ["png", "pdf"]:
            fig.savefig(
                os.path.join(
                    results_dir,
                    f"run_length_comparison{tag}{suffix}.{ext}",
                ),
                dpi=150, bbox_inches="tight",
            )
        plt.close(fig)


def plot_seqlen_comparison(seqlen_results, results_dir):
    """Compare temporal exploitation at different sequence lengths."""
    rho_groups = sorted(set(RHO))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, (cat_on, cat_off, title) in zip(axes, [
        ("long_cont", "sudden_onset", "ON: long cont vs sudden onset"),
        ("sudden_offset", "long_absent", "OFF: sudden offset vs long absent"),
    ]):
        for T, results, ls in seqlen_results:
            for cat, color, marker in [
                (cat_on, "tab:green", "o"),
                (cat_off, "tab:orange", "s"),
            ]:
                means = []
                for r in rho_groups:
                    d = results.get(cat, {}).get(
                        r, {"mean": 0, "ci_low": 0, "ci_high": 0}
                    )
                    means.append(d["mean"])
                x = np.arange(len(rho_groups))
                label = f"T={T} {cat}"
                ax.plot(x, means, f"{marker}{ls}", color=color, linewidth=2,
                        markersize=7, label=label, alpha=0.8)

        ax.set_xticks(range(len(rho_groups)))
        ax.set_xticklabels([f"{r}" for r in rho_groups])
        ax.set_xlabel("$\\rho$", fontsize=12)
        ax.set_ylabel(
            "Mean $|\\langle \\hat{x}_{\\mathrm{pred}}, \\mathbf{f}_i \\rangle|$",
            fontsize=11,
        )
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle(
        f"Sequence length comparison (k={K_FOR_SEQLEN}, history={HISTORY_LEN})",
        fontsize=13, y=1.02,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(results_dir, f"seqlen_comparison.{ext}"),
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

    # ── Part 1: Run-length analysis at T=64 for multiple k ──────────

    print(f"\n{'='*70}", flush=True)
    print(f"PART 1: Run-length analysis (T={SEQ_LEN}, history={HISTORY_LEN})",
          flush=True)
    print(f"{'='*70}", flush=True)

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    sf = compute_scaling_factor(model, pi_t, rho_t, SEQ_LEN, device)
    gen_seq = gen_seq_fn(model, pi_t, rho_t, SEQ_LEN, device, sf)
    fd = model.feature_directions.to(device)

    # Generate eval data
    set_seed(SEED + 200)
    acts, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts) * sf

    # Report category counts
    masks, h = classify_by_run_length(eval_support, HISTORY_LEN)
    print(f"\nRun-length categories (history={HISTORY_LEN}, "
          f"valid positions per seq={SEQ_LEN - h}):", flush=True)
    for cat, mask in masks.items():
        total = mask.sum().item()
        pct = 100 * mask.float().mean().item()
        print(f"  {cat:>16}: {total:>10.0f} events ({pct:.1f}%)", flush=True)

    # Per-rho breakdown of long_cont vs sudden_onset counts
    rho_groups = sorted(set(RHO))
    print(f"\nPer-rho event counts:", flush=True)
    for rho_val in rho_groups:
        feat_indices = [
            i for i in range(NUM_FEATURES)
            if abs(RHO[i] - rho_val) < 0.01
        ]
        for cat in ["long_cont", "sudden_onset", "sudden_offset", "long_absent"]:
            n = sum(
                masks[cat][:, :, i].sum().item() for i in feat_indices
            )
            print(f"  rho={rho_val:.1f} {cat:>16}: {n:>8.0f}", flush=True)

    all_k_results = {}

    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n{'='*60}", flush=True)
        print(f"TFA k={k}, T={SEQ_LEN}", flush=True)
        print(f"{'='*60}", flush=True)

        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1,
            device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
            log_every=TFA_TOTAL_STEPS,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        print(f"Training: {time.time() - t0:.1f}s", flush=True)

        t0 = time.time()
        pred_proj = compute_per_token_projections(tfa, eval_hidden, fd, device)
        results = analyze_run_length(pred_proj, eval_support, RHO, HISTORY_LEN)
        all_k_results[k] = results
        print(f"Analysis: {time.time() - t0:.1f}s", flush=True)

        # Summary table
        print(f"\n{'category':>16} {'rho':>5} | "
              f"{'mean |pred|':>11} {'95% CI':>20} | {'n':>8}", flush=True)
        print("-" * 75, flush=True)
        for cat in ["long_cont", "sudden_onset", "short_cont", "short_onset",
                     "sudden_offset", "long_absent"]:
            for r in rho_groups:
                d = results.get(cat, {}).get(r, {
                    "mean": 0, "ci_low": 0, "ci_high": 0, "n": 0
                })
                ci = f"[{d['ci_low']:.4f}, {d['ci_high']:.4f}]"
                print(f"{cat:>16} {r:>5.1f} | {d['mean']:>11.4f} {ci:>20} | "
                      f"{d['n']:>8}", flush=True)
            print(flush=True)

        del tfa
        torch.cuda.empty_cache()

    plot_run_length_comparison(all_k_results, RESULTS_DIR)
    print("Part 1 plots saved.", flush=True)

    # ── Part 2: Sequence length comparison ───────────────────────────

    print(f"\n{'='*70}", flush=True)
    print(f"PART 2: Sequence length comparison (k={K_FOR_SEQLEN})", flush=True)
    print(f"{'='*70}", flush=True)

    seqlen_results = []  # list of (T, results_dict, linestyle)
    linestyles = ["-", "--", "-.", ":"]

    for idx, T in enumerate(SEQ_LENS_COMPARE):
        set_seed(SEED)
        print(f"\n--- T={T} ---", flush=True)

        # Recompute scaling factor and generators for this sequence length
        sf_T = compute_scaling_factor(model, pi_t, rho_t, T, device)
        gen_seq_T = gen_seq_fn(model, pi_t, rho_t, T, device, sf_T)

        # Generate eval data at this sequence length
        set_seed(SEED + 200)
        acts_T, sup_T = generate_markov_activations(
            EVAL_N_SEQ, T, pi_t, rho_t, device=device
        )
        eval_hidden_T = model(acts_T) * sf_T

        # Train TFA
        set_seed(SEED)
        t0 = time.time()
        tfa_T = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=K_FOR_SEQLEN,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1,
            device=device,
        )
        # Scale batch size inversely with T to keep tokens/step roughly constant
        tfa_batch = max(16, TFA_BATCH_SIZE * SEQ_LEN // T)
        tfa_cfg_T = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=tfa_batch,
            lr=TFA_LR,
            log_every=TFA_TOTAL_STEPS,
        )
        tfa_T, _ = train_tfa(tfa_T, gen_seq_T, tfa_cfg_T, device)
        print(f"Training (batch={tfa_batch}): {time.time() - t0:.1f}s", flush=True)

        pred_proj_T = compute_per_token_projections(
            tfa_T, eval_hidden_T, fd, device
        )
        results_T = analyze_run_length(pred_proj_T, sup_T, RHO, HISTORY_LEN)
        seqlen_results.append((T, results_T, linestyles[idx % len(linestyles)]))

        # Print key comparisons
        print(f"\nT={T} key results:", flush=True)
        for r in [0.5, 0.9]:
            lc = results_T.get("long_cont", {}).get(r, {"mean": 0, "n": 0})
            so = results_T.get("sudden_onset", {}).get(r, {"mean": 0, "n": 0})
            ratio = lc["mean"] / so["mean"] if so["mean"] > 0 else float("inf")
            print(f"  rho={r}: long_cont={lc['mean']:.4f} (n={lc['n']}), "
                  f"sudden_onset={so['mean']:.4f} (n={so['n']}), "
                  f"ratio={ratio:.3f}", flush=True)

        del tfa_T, acts_T, eval_hidden_T
        torch.cuda.empty_cache()

    plot_seqlen_comparison(seqlen_results, RESULTS_DIR)
    print("Part 2 plots saved.", flush=True)

    # ── Save all results ─────────────────────────────────────────────

    def serialize_results(res_dict):
        return {
            cat: {str(r): d for r, d in rho_data.items()}
            for cat, rho_data in res_dict.items()
        }

    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "history_len": HISTORY_LEN,
            "k_values": K_VALUES,
            "k_for_seqlen": K_FOR_SEQLEN,
            "seq_lens_compare": SEQ_LENS_COMPARE,
            "rho": RHO,
            "pi": PI,
            "eval_n_seq": EVAL_N_SEQ,
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
            "tfa_total_steps": TFA_TOTAL_STEPS,
        },
        "part1_run_length": {
            str(k): serialize_results(results)
            for k, results in all_k_results.items()
        },
        "part2_seqlen": {
            str(T): serialize_results(results)
            for T, results, _ in seqlen_results
        },
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
