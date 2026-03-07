"""Step 3: Temporal decomposition analysis.

Uses ground-truth Markov chain state to classify each (feature i, position t)
event as continuation/onset/offset/absent, then measures how much of each
feature's energy is captured by TFA's predictable vs novel component.

Tests: does TFA learn to route continuations to prediction and onsets to novel?
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
EXPECTED_L0 = 10.0
DICT_WIDTH = 40

K_VALUES = [5, 8, 10, 15]
TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 3000  # 3000 * 64 = 192K tokens
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "temporal_decomposition")


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()
    return math.sqrt(HIDDEN_DIM) / mean_norm if mean_norm > 0 else 1.0


def gen_seq_fn(model, pi_t, rho_t, device, sf):
    def fn(n_seq):
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * sf
    return fn


# ── Temporal event classification ────────────────────────────────────


def classify_events(support):
    """Classify each (feature, position) as continuation/onset/offset/absent.

    Args:
        support: Binary tensor (n_seq, T, n_features)

    Returns:
        Dict with boolean tensors for each event type, shape (n_seq, T, n_features).
        Position t=0 is excluded (no previous state).
    """
    prev = support[:, :-1, :]  # (n_seq, T-1, n_features)
    curr = support[:, 1:, :]   # (n_seq, T-1, n_features)

    return {
        "continuation": (prev == 1) & (curr == 1),  # on -> on
        "onset":        (prev == 0) & (curr == 1),  # off -> on
        "offset":       (prev == 1) & (curr == 0),  # on -> off
        "absent":       (prev == 0) & (curr == 0),  # off -> off
    }


# ── Per-event decomposition ─────────────────────────────────────────


def analyze_decomposition(tfa, model, eval_hidden, eval_support, device):
    """For each event type and feature, measure pred/novel energy fractions.

    Projects TFA's predictable and novel reconstructions onto each ground-truth
    feature direction, conditioned on the event type at each position.
    """
    tfa.eval()
    feature_dirs = model.feature_directions.to(device)  # (n_features, d)
    n_features = feature_dirs.shape[0]

    # Classify events (positions 1..T-1)
    events = classify_events(eval_support)

    # Accumulators: for each (event_type, feature), accumulate |projection|
    event_types = ["continuation", "onset", "offset", "absent"]
    pred_proj = {et: torch.zeros(n_features, device=device) for et in event_types}
    novel_proj = {et: torch.zeros(n_features, device=device) for et in event_types}
    true_proj = {et: torch.zeros(n_features, device=device) for et in event_types}
    counts = {et: torch.zeros(n_features, device=device) for et in event_types}

    n_seq = eval_hidden.shape[0]
    batch_size = 256

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            x = eval_hidden[start:end].to(device)  # (B, T, d)

            recons, inter = tfa(x)
            pred_r = inter["pred_recons"][:, 1:, :]  # skip t=0
            novel_r = inter["novel_recons"][:, 1:, :]
            x_curr = x[:, 1:, :]  # positions 1..T-1

            # Project onto each feature direction
            # feature_dirs: (n_features, d) -> (1, 1, n_features, d)
            fd = feature_dirs.unsqueeze(0).unsqueeze(0)

            x_proj = (x_curr.unsqueeze(2) * fd).sum(dim=-1)       # (B, T-1, n_features)
            pred_p = (pred_r.unsqueeze(2) * fd).sum(dim=-1)        # (B, T-1, n_features)
            novel_p = (novel_r.unsqueeze(2) * fd).sum(dim=-1)      # (B, T-1, n_features)

            for et in event_types:
                mask = events[et][start:end].to(device)  # (B, T-1, n_features)
                for i in range(n_features):
                    m = mask[:, :, i]  # (B, T-1)
                    if m.any():
                        pred_proj[et][i] += pred_p[:, :, i][m].abs().sum()
                        novel_proj[et][i] += novel_p[:, :, i][m].abs().sum()
                        true_proj[et][i] += x_proj[:, :, i][m].abs().sum()
                        counts[et][i] += m.float().sum()

    # Compute fractions
    results = {}
    for et in event_types:
        results[et] = []
        for i in range(n_features):
            total = true_proj[et][i].item()
            if total > 1e-8:
                pf = pred_proj[et][i].item() / total
                nf = novel_proj[et][i].item() / total
            else:
                pf = nf = 0.0
            results[et].append({
                "feature": i,
                "rho": RHO[i],
                "pred_frac": pf,
                "novel_frac": nf,
                "count": int(counts[et][i].item()),
            })

    return results


def aggregate_by_rho(decomp_results):
    """Aggregate per-feature results by rho group."""
    rho_groups = sorted(set(RHO))
    agg = {}
    for et, features in decomp_results.items():
        agg[et] = {}
        for rho_val in rho_groups:
            group = [f for f in features if abs(f["rho"] - rho_val) < 0.01]
            if group:
                total_count = sum(f["count"] for f in group)
                if total_count > 0:
                    # Weighted average by count
                    avg_pred = sum(f["pred_frac"] * f["count"] for f in group) / total_count
                    avg_novel = sum(f["novel_frac"] * f["count"] for f in group) / total_count
                else:
                    avg_pred = avg_novel = 0.0
                agg[et][rho_val] = {
                    "pred_frac": avg_pred,
                    "novel_frac": avg_novel,
                    "count": total_count,
                }
    return agg


# ── Plotting ─────────────────────────────────────────────────────────


def plot_event_decomposition(agg_by_rho, k, results_dir):
    """Bar chart: pred/novel fraction for continuation vs onset, grouped by rho."""
    rho_groups = sorted(set(RHO))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, et, title in zip(axes, ["continuation", "onset"],
                              ["Continuations (on→on)", "Onsets (off→on)"]):
        pred_fracs = [agg_by_rho[et].get(r, {}).get("pred_frac", 0) for r in rho_groups]
        novel_fracs = [agg_by_rho[et].get(r, {}).get("novel_frac", 0) for r in rho_groups]

        x = np.arange(len(rho_groups))
        width = 0.35
        ax.bar(x - width/2, pred_fracs, width, label="Predictable",
               color="tab:green", alpha=0.8)
        ax.bar(x + width/2, novel_fracs, width, label="Novel",
               color="tab:red", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"$\\rho$={r}" for r in rho_groups], fontsize=10)
        ax.set_ylabel("Fraction of feature energy", fontsize=11)
        ax.set_title(f"{title} (k={k})", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(max(pred_fracs + novel_fracs) * 1.2, 1.0))

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"event_decomp_k{k}.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_continuation_vs_onset(all_agg, results_dir):
    """Compare pred fraction for continuations vs onsets across k and rho."""
    rho_groups = sorted(set(RHO))
    n_k = len(all_agg)

    fig, axes = plt.subplots(1, n_k, figsize=(5 * n_k, 5), sharey=True)
    if n_k == 1:
        axes = [axes]

    for ax, (k, agg) in zip(axes, sorted(all_agg.items())):
        cont_pred = [agg["continuation"].get(r, {}).get("pred_frac", 0) for r in rho_groups]
        onset_pred = [agg["onset"].get(r, {}).get("pred_frac", 0) for r in rho_groups]

        x = np.arange(len(rho_groups))
        width = 0.35
        ax.bar(x - width/2, cont_pred, width, label="Continuation pred frac",
               color="tab:green", alpha=0.8)
        ax.bar(x + width/2, onset_pred, width, label="Onset pred frac",
               color="tab:orange", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([f"{r}" for r in rho_groups], fontsize=10)
        ax.set_xlabel("$\\rho$", fontsize=11)
        if ax == axes[0]:
            ax.set_ylabel("Predictable fraction", fontsize=11)
        ax.set_title(f"k = {k}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, 1.5)

    plt.suptitle("TFA Predictable Fraction: Continuations vs Onsets by $\\rho$",
                 fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"cont_vs_onset_by_rho.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_heatmap(all_agg, results_dir):
    """Heatmap: pred_frac for each (event_type, rho) at a selected k."""
    rho_groups = sorted(set(RHO))
    event_types = ["continuation", "onset", "offset", "absent"]
    event_labels = ["Continuation\n(on→on)", "Onset\n(off→on)",
                    "Offset\n(on→off)", "Absent\n(off→off)"]

    # Use the most interesting k (lowest binding)
    k_select = min(all_agg.keys())
    agg = all_agg[k_select]

    data = np.zeros((len(event_types), len(rho_groups)))
    for i, et in enumerate(event_types):
        for j, r in enumerate(rho_groups):
            data[i, j] = agg[et].get(r, {}).get("pred_frac", 0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1.2)
    ax.set_xticks(range(len(rho_groups)))
    ax.set_xticklabels([f"$\\rho$={r}" for r in rho_groups], fontsize=10)
    ax.set_yticks(range(len(event_types)))
    ax.set_yticklabels(event_labels, fontsize=10)
    ax.set_title(f"Predictable Fraction by Event Type and $\\rho$ (k={k_select})",
                 fontsize=12)

    # Annotate
    for i in range(len(event_types)):
        for j in range(len(rho_groups)):
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if data[i, j] < 0.8 else "white")

    plt.colorbar(im, ax=ax, label="Predictable fraction")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"pred_frac_heatmap_k{k_select}.{ext}"),
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
    print(f"Scaling factor: {sf:.4f}")

    gen_seq = gen_seq_fn(model, pi_t, rho_t, device, sf)

    # Fixed eval data with ground-truth support
    set_seed(SEED + 200)
    acts, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts) * sf
    print(f"Eval: {eval_hidden.shape[0]}x{eval_hidden.shape[1]} = "
          f"{eval_hidden.shape[0] * eval_hidden.shape[1]} tokens")

    # Event statistics
    events = classify_events(eval_support)
    n_total = eval_support.shape[0] * (eval_support.shape[1] - 1) * eval_support.shape[2]
    print(f"\nEvent counts (total {n_total}):")
    for et, mask in events.items():
        ct = mask.sum().item()
        print(f"  {et:15s}: {ct:10.0f} ({100*ct/n_total:.1f}%)")

    all_results = {}
    all_agg = {}

    for k in K_VALUES:
        set_seed(SEED)
        print(f"\n{'='*60}")
        print(f"TFA k={k}")
        print(f"{'='*60}")

        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_cfg = TFATrainingConfig(
            total_steps=TFA_TOTAL_STEPS,
            batch_size=TFA_BATCH_SIZE,
            lr=TFA_LR,
            log_every=10000,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        dt_train = time.time() - t0
        print(f"Training: {dt_train:.1f}s")

        # Analyze decomposition
        t0 = time.time()
        decomp = analyze_decomposition(tfa, model, eval_hidden, eval_support, device)
        dt_eval = time.time() - t0
        print(f"Analysis: {dt_eval:.1f}s")

        agg = aggregate_by_rho(decomp)
        all_results[k] = decomp
        all_agg[k] = agg

        # Print summary
        rho_groups = sorted(set(RHO))
        print(f"\n{'Event':<15} {'rho':>5} {'pred_frac':>10} {'novel_frac':>11} {'count':>8}")
        print("-" * 55)
        for et in ["continuation", "onset"]:
            for r in rho_groups:
                d = agg[et].get(r, {"pred_frac": 0, "novel_frac": 0, "count": 0})
                print(f"{et:<15} {r:>5.1f} {d['pred_frac']:>10.3f} "
                      f"{d['novel_frac']:>11.3f} {d['count']:>8}")
            print()

        # Plot per-k
        plot_event_decomposition(agg, k, RESULTS_DIR)

        del tfa
        torch.cuda.empty_cache()

    # Cross-k plots
    print("\nGenerating summary plots...")
    plot_continuation_vs_onset(all_agg, RESULTS_DIR)
    plot_summary_heatmap(all_agg, RESULTS_DIR)

    # Save
    # Convert to serializable
    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "k_values": K_VALUES,
            "total_steps": TFA_TOTAL_STEPS,
            "seed": SEED,
        },
        "aggregated": {
            str(k): {
                et: {str(r): v for r, v in rho_data.items()}
                for et, rho_data in agg.items()
            }
            for k, agg in all_agg.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
