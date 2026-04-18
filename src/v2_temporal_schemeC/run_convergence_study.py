"""Convergence study: TFA MSE vs training steps.

Trains TFA at key conditions and evaluates at checkpoints to
show how MSE evolves with training. Also evaluates SAE for comparison.
Reinforces the finding that TFA requires more training steps than SAE.
"""

import json
import math
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.shared.configs import TrainingConfig
from src.shared.train_sae import create_sae, train_sae
from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import generate_markov_activations
from src.bench.architectures._tfa_module import TemporalSAE
from src.v2_temporal_schemeC.train_tfa import (
    TFATrainingConfig,
    create_tfa,
)

# ── Configuration ────────────────────────────────────────────────────

NUM_FEATURES = 10
HIDDEN_DIM = 40
SEQ_LEN = 64

PI = [0.2] * 10
RHO = [0.0, 0.0, 0.0, 0.3, 0.3, 0.6, 0.6, 0.9, 0.9, 0.95]

# Conditions to study convergence
CONDITIONS = [
    {"k": 2},
    {"k": 4},
]

# Checkpoints (steps)
CHECKPOINTS = [500, 1000, 2000, 3000, 5000, 7500, 10000, 15000, 20000, 30000]
MAX_STEPS = max(CHECKPOINTS)

BATCH_SIZE = 64
LR = 1e-3
MIN_LR = 9e-4
WARMUP_STEPS = 200

SAE_TOTAL_SAMPLES = 10_000_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
SAE_D_SAE = 40

TFA_WIDTH = 40
EVAL_N_SAMPLES = 100_000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "convergence_study")


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device, n_samples=10000):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            n_samples, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
        mean_norm = norms.mean().item()
    target_norm = math.sqrt(HIDDEN_DIM)
    sf = target_norm / mean_norm if mean_norm > 0 else 1.0
    return sf, mean_norm


def make_data_generators(model, pi_t, rho_t, device, scaling_factor):
    def generate_flattened(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return hidden.reshape(-1, HIDDEN_DIM)[:batch_size] * scaling_factor

    def generate_sequences(n_seq: int) -> torch.Tensor:
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * scaling_factor

    return generate_flattened, generate_sequences


def eval_tfa(tfa, gen_fn, device, n_samples=EVAL_N_SAMPLES):
    """Evaluate TFA and return MSE and pred energy."""
    tfa.eval()
    total_mse = 0.0
    pred_energy_sum = 0.0
    novel_energy_sum = 0.0
    n_tokens = 0
    batch_seqs = 256
    with torch.no_grad():
        while n_tokens < n_samples:
            n_seq = min(batch_seqs, max(1, (n_samples - n_tokens) // SEQ_LEN))
            x = gen_fn(n_seq).to(device)
            recons, intermediates = tfa(x)
            B, T, D = x.shape
            n_tok = B * T
            x_flat = x.reshape(-1, D)
            r_flat = recons.reshape(-1, D)
            total_mse += (x_flat - r_flat).pow(2).sum().item()
            pred_r = intermediates["pred_recons"]
            novel_r = intermediates["novel_recons"]
            pred_energy_sum += pred_r.norm(dim=-1).pow(2).sum().item()
            novel_energy_sum += novel_r.norm(dim=-1).pow(2).sum().item()
            n_tokens += n_tok
    mse = total_mse / n_tokens
    total_energy = pred_energy_sum + novel_energy_sum + 1e-12
    rel_energy_pred = pred_energy_sum / total_energy
    tfa.train()
    return {"mse": mse, "rel_energy_pred": rel_energy_pred}


def eval_sae(sae, gen_fn, device, n_samples=EVAL_N_SAMPLES):
    sae.eval()
    total_mse = 0.0
    n_processed = 0
    batch_size = 4096
    with torch.no_grad():
        while n_processed < n_samples:
            bs = min(batch_size, n_samples - n_processed)
            x = gen_fn(bs).to(device)
            x_hat = sae(x)
            total_mse += (x - x_hat).pow(2).sum().item()
            n_processed += x.shape[0]
    mse = total_mse / n_processed
    return {"mse": mse}


def _get_lr(step: int) -> float:
    if step < WARMUP_STEPS:
        return LR * step / WARMUP_STEPS
    decay_ratio = (step - WARMUP_STEPS) / max(1, MAX_STEPS - WARMUP_STEPS)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return MIN_LR + coeff * (LR - MIN_LR)


def train_tfa_with_checkpoints(tfa, gen_fn, checkpoints, device):
    """Train TFA and evaluate at checkpoint steps."""
    # Setup optimizer
    decay_params = []
    no_decay_params = []
    for name, param in tfa.named_parameters():
        if param.requires_grad:
            if param.dim() >= 2:
                decay_params.append(param)
            else:
                no_decay_params.append(param)
    param_groups = [
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=LR, betas=(0.9, 0.95))

    tfa.train()
    checkpoint_results = []
    checkpoint_set = set(checkpoints)

    for step in range(MAX_STEPS):
        lr = _get_lr(step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        batch = gen_fn(BATCH_SIZE).to(device)
        recons, intermediates = tfa(batch)
        batch_flat = batch.reshape(-1, batch.shape[-1])
        recons_flat = recons.reshape(-1, recons.shape[-1])
        n_tokens = batch_flat.shape[0]
        loss = F.mse_loss(recons_flat, batch_flat, reduction="sum") / n_tokens

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tfa.parameters(), 1.0)
        optimizer.step()

        if (step + 1) in checkpoint_set:
            result = eval_tfa(tfa, gen_fn, device)
            result["step"] = step + 1
            checkpoint_results.append(result)
            print(
                f"      step {step + 1:6d}: MSE={result['mse']:.6f}, "
                f"pred_E={result['rel_energy_pred']:.3f}"
            )
            tfa.train()

        if step % 5000 == 0 and step > 0:
            print(f"      [training step {step}, loss={loss.item():.6f}]")

    return checkpoint_results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_convergence_curves(all_results, results_dir):
    """MSE vs training steps for each condition."""
    n_conds = len(all_results)
    fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 5), sharey=True)
    if n_conds == 1:
        axes = [axes]

    for idx, cond_result in enumerate(all_results):
        ax = axes[idx]
        cond = cond_result["condition"]
        steps = [r["step"] for r in cond_result["tfa_checkpoints"]]
        mses = [r["mse"] for r in cond_result["tfa_checkpoints"]]
        sae_mse = cond_result["sae_mse"]

        ax.plot(steps, mses, "o-", color="tab:orange", linewidth=2, markersize=6,
                label="TFA")
        ax.axhline(y=sae_mse, color="tab:blue", linestyle="--", linewidth=2,
                   label=f"SAE (MSE={sae_mse:.6f})")

        # Mark where TFA drops below SAE
        for i, (s, m) in enumerate(zip(steps, mses)):
            if m <= sae_mse and (i == 0 or mses[i - 1] > sae_mse):
                ax.axvline(x=s, color="red", linestyle=":", alpha=0.5)
                ax.annotate(f"TFA <= SAE\nat step {s}",
                           (s, m), textcoords="offset points",
                           xytext=(10, 10), fontsize=8, color="red")
                break

        ax.set_xlabel("Training steps")
        ax.set_title(f"k={cond['k']}")
        if idx == 0:
            ax.set_ylabel("MSE")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")

    fig.suptitle("TFA Convergence: MSE vs Training Steps", fontsize=14)
    plt.tight_layout()
    path = os.path.join(results_dir, "convergence_curves.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_pred_energy_evolution(all_results, results_dir):
    """Pred energy fraction vs training steps."""
    n_conds = len(all_results)
    fig, axes = plt.subplots(1, n_conds, figsize=(6 * n_conds, 5), sharey=True)
    if n_conds == 1:
        axes = [axes]

    for idx, cond_result in enumerate(all_results):
        ax = axes[idx]
        cond = cond_result["condition"]
        steps = [r["step"] for r in cond_result["tfa_checkpoints"]]
        pred_e = [r["rel_energy_pred"] for r in cond_result["tfa_checkpoints"]]

        ax.plot(steps, pred_e, "s-", color="tab:green", linewidth=2, markersize=6)
        ax.set_xlabel("Training steps")
        ax.set_title(f"k={cond['k']}")
        if idx == 0:
            ax.set_ylabel("Pred energy fraction")
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log")
        ax.set_ylim(0, 1)

    fig.suptitle("Predictable Component Energy vs Training Steps", fontsize=14)
    plt.tight_layout()
    path = os.path.join(results_dir, "pred_energy_evolution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def plot_convergence_overlay(all_results, results_dir):
    """All conditions overlaid on one plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    for idx, cond_result in enumerate(all_results):
        cond = cond_result["condition"]
        steps = [r["step"] for r in cond_result["tfa_checkpoints"]]
        mses = [r["mse"] for r in cond_result["tfa_checkpoints"]]
        sae_mse = cond_result["sae_mse"]
        label = f"k={cond['k']}"
        color = colors[idx % len(colors)]

        ax.plot(steps, mses, "o-", color=color, linewidth=2, markersize=5,
                label=f"TFA ({label})")
        ax.axhline(y=sae_mse, color=color, linestyle="--", alpha=0.5,
                   label=f"SAE ({label}): {sae_mse:.6f}")

    ax.set_xlabel("Training steps")
    ax.set_ylabel("MSE")
    ax.set_title("TFA Convergence Across Conditions")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log")

    plt.tight_layout()
    path = os.path.join(results_dir, "convergence_overlay.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    print(f"\nConvergence study")
    print(f"Conditions: {CONDITIONS}")
    print(f"Checkpoints: {CHECKPOINTS}")
    print(f"Max steps: {MAX_STEPS}\n")

    all_results = []
    t_start = time.time()

    # Build model once
    set_seed(42)
    model = ToyModel(
        num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"scaling_factor={scaling_factor:.4f}")

    gen_flat, gen_seq = make_data_generators(
        model, pi_t, rho_t, device, scaling_factor
    )

    for cond_idx, cond in enumerate(CONDITIONS):
        k = cond["k"]
        print(f"\n{'=' * 60}")
        print(f"  Condition {cond_idx + 1}/{len(CONDITIONS)}: k={k}")
        print(f"{'=' * 60}")

        # Train SAE baseline
        print(f"\n  Training SAE (k={k})...")
        set_seed(42)
        sae = create_sae(HIDDEN_DIM, SAE_D_SAE, k=float(k), device=device)
        cfg = TrainingConfig(
            k=float(k), d_sae=SAE_D_SAE,
            total_training_samples=SAE_TOTAL_SAMPLES,
            batch_size=SAE_BATCH_SIZE, lr=SAE_LR, seed=42,
        )
        sae = train_sae(sae, gen_flat, cfg, device)
        sae_result = eval_sae(sae, gen_flat, device)
        print(f"  SAE MSE: {sae_result['mse']:.6f}")

        # Train TFA with checkpoints
        print(f"\n  Training TFA (k={k}, {MAX_STEPS} steps with checkpoints)...")
        set_seed(42)
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=TFA_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        t0 = time.time()
        tfa_checkpoints = train_tfa_with_checkpoints(
            tfa, gen_seq, CHECKPOINTS, device
        )
        dt = time.time() - t0
        print(f"  TFA training done ({dt:.1f}s)")

        cond_result = {
            "condition": cond,
            "sae_mse": sae_result["mse"],
            "scaling_factor": scaling_factor,
            "tfa_checkpoints": tfa_checkpoints,
        }
        all_results.append(cond_result)

        # Print summary for this condition
        print(f"\n  Step-by-step MSE (SAE baseline: {sae_result['mse']:.6f}):")
        for cp in tfa_checkpoints:
            delta = sae_result["mse"] - cp["mse"]  # positive = TFA wins
            sign = "+" if delta >= 0 else ""
            winner = "TFA" if delta > 0 else "SAE"
            print(
                f"    step {cp['step']:6d}: TFA MSE={cp['mse']:.6f} "
                f"(delta={sign}{delta:.6f}, winner={winner})"
            )

    total_time = time.time() - t_start
    print(f"\n{'=' * 60}")
    print(f"Total runtime: {total_time / 60:.1f} min")
    print(f"{'=' * 60}")

    # Plots
    plot_convergence_curves(all_results, RESULTS_DIR)
    plot_pred_energy_evolution(all_results, RESULTS_DIR)
    plot_convergence_overlay(all_results, RESULTS_DIR)

    # Save
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump({
            "config": {
                "conditions": CONDITIONS,
                "checkpoints": CHECKPOINTS,
                "max_steps": MAX_STEPS,
                "seed": 42,
            },
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {json_path}")


if __name__ == "__main__":
    main()
