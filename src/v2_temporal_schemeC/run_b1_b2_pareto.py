"""Part B: Scaled-up Pareto frontier with ReLU+L1 SAEs.

B1: n=20 features, pi=0.5, E[L0]=10, rho spread across [0, 0.9].
B2: Sweep L1 coefficient for both ReLU SAE and TFA (ReLU novel component).
    Plot (NMSE, L0) Pareto frontiers.
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
import torch.nn.functional as F

from src.utils.device import DEFAULT_DEVICE
from src.utils.seed import set_seed
from src.v2_temporal_schemeC.toy_model import ToyModel
from src.v2_temporal_schemeC.markov_data_generation import (
    generate_markov_activations,
    generate_markov_support,
)
from src.v2_temporal_schemeC.tfa import TemporalSAE
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

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = (
    [0.0] * 4
    + [0.3] * 4
    + [0.5] * 4
    + [0.7] * 4
    + [0.9] * 4
)
EXPECTED_L0 = sum(PI)  # 10.0

# Dictionary width for both models
DICT_WIDTH = 40

# Training
TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
TFA_SEQ_BATCH = 64  # sequences per batch (64 * 64 = 4096 tokens)
TFA_N_HEADS = 4
TFA_N_ATTN_LAYERS = 1
TFA_BOTTLENECK = 1

# Evaluation
EVAL_N_SEQ = 2000  # 2000 * 64 = 128K tokens

SEED = 42
LOG_EVERY = 5000

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "b1_b2_pareto")


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
    """Return (generate_flat_tokens, generate_sequences)."""

    def generate_flat(batch_size: int) -> torch.Tensor:
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return hidden.reshape(-1, HIDDEN_DIM)[:batch_size] * scaling_factor

    def generate_seq(n_seq: int) -> torch.Tensor:
        acts, _ = generate_markov_activations(
            n_seq, SEQ_LEN, pi_t, rho_t, device=device
        )
        return model(acts) * scaling_factor

    return generate_flat, generate_seq


def generate_eval_data(model, pi_t, rho_t, device, scaling_factor, n_seq=EVAL_N_SEQ):
    """Generate fixed evaluation dataset."""
    acts, support = generate_markov_activations(
        n_seq, SEQ_LEN, pi_t, rho_t, device=device
    )
    hidden = model(acts) * scaling_factor
    return hidden, support


# ── Evaluation ───────────────────────────────────────────────────────


def eval_sae_on_data(sae, eval_hidden, device):
    """Evaluate ReLU SAE on pre-generated data. Returns dict with nmse, l0, mse."""
    sae.eval()
    batch_size = 4096
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]

    total_se = 0.0
    total_signal_energy = 0.0
    total_l0 = 0.0

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = flat[start:end].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal_energy += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()

    nmse = total_se / total_signal_energy
    mse = total_se / n
    l0 = total_l0 / n
    return {"nmse": nmse, "mse": mse, "l0": l0}


def eval_tfa_on_data(tfa, eval_hidden, device):
    """Evaluate TFA on pre-generated sequence data. Returns dict with nmse, l0 variants."""
    tfa.eval()
    batch_size = 256  # sequences
    n_seq = eval_hidden.shape[0]

    total_se = 0.0
    total_signal_energy = 0.0
    total_novel_l0 = 0.0
    total_total_l0 = 0.0
    total_pred_energy = 0.0
    total_novel_energy = 0.0
    n_tokens = 0

    with torch.no_grad():
        for start in range(0, n_seq, batch_size):
            end = min(start + batch_size, n_seq)
            x = eval_hidden[start:end].to(device)  # (B, T, d)
            recons, intermediates = tfa(x)

            B, T, D = x.shape
            n_tok = B * T
            x_flat = x.reshape(-1, D)
            r_flat = recons.reshape(-1, D)

            total_se += (x_flat - r_flat).pow(2).sum().item()
            total_signal_energy += x_flat.pow(2).sum().item()

            novel_codes = intermediates["novel_codes"]  # (B, T, width)
            pred_codes = intermediates["pred_codes"]    # (B, T, width)
            total_codes = novel_codes + pred_codes

            total_novel_l0 += (novel_codes > 0).float().sum(dim=-1).sum().item()
            total_total_l0 += (total_codes.abs() > 1e-8).float().sum(dim=-1).sum().item()

            pred_r = intermediates["pred_recons"]
            novel_r = intermediates["novel_recons"]
            total_pred_energy += pred_r.norm(dim=-1).pow(2).sum().item()
            total_novel_energy += novel_r.norm(dim=-1).pow(2).sum().item()

            n_tokens += n_tok

    nmse = total_se / total_signal_energy
    mse = total_se / n_tokens
    novel_l0 = total_novel_l0 / n_tokens
    total_l0 = total_total_l0 / n_tokens
    total_e = total_pred_energy + total_novel_energy + 1e-12
    pred_energy_frac = total_pred_energy / total_e

    return {
        "nmse": nmse,
        "mse": mse,
        "novel_l0": novel_l0,
        "total_l0": total_l0,
        "pred_energy_frac": pred_energy_frac,
    }


# ── Data Sanity Check ────────────────────────────────────────────────


def run_sanity_check(model, pi_t, rho_t, scaling_factor, device):
    """Verify data statistics match theory."""
    print("=" * 60)
    print("DATA SANITY CHECK")
    print("=" * 60)

    set_seed(SEED)
    n_seq = 5000
    acts, support = generate_markov_activations(
        n_seq, SEQ_LEN, pi_t, rho_t, device=device
    )
    hidden = model(acts) * scaling_factor

    # Marginal activation rates
    marginal_rates = support.mean(dim=(0, 1))
    print(f"\nMarginal rates (target={PI[0]}):")
    for i, (rate, rho_val) in enumerate(zip(marginal_rates.tolist(), RHO)):
        print(f"  f{i:2d} (rho={rho_val}): pi={rate:.4f}")

    # L0 distribution
    l0_per_token = support.sum(dim=-1)  # (n_seq, T)
    l0_mean = l0_per_token.mean().item()
    l0_std = l0_per_token.std().item()
    print(f"\nL0: mean={l0_mean:.3f} (target {EXPECTED_L0}), "
          f"std={l0_std:.3f} (theory {math.sqrt(NUM_FEATURES * PI[0] * (1 - PI[0])):.3f})")

    # Lag-1 autocorrelation
    print(f"\nLag-1 autocorrelation:")
    for i in range(NUM_FEATURES):
        s = support[:, :, i]  # (n_seq, T)
        s_t = s[:, 1:]
        s_prev = s[:, :-1]
        mean_s = s.mean()
        var_s = s.var()
        if var_s > 1e-8:
            cov = ((s_t - mean_s) * (s_prev - mean_s)).mean()
            empirical_rho = (cov / var_s).item()
        else:
            empirical_rho = 0.0
        print(f"  f{i:2d}: rho_emp={empirical_rho:.4f} (target={RHO[i]:.1f})")

    # Input scaling
    flat = hidden.reshape(-1, HIDDEN_DIM)
    norms = flat.norm(dim=-1)
    print(f"\nScaled norms: mean={norms.mean().item():.4f} "
          f"(target={math.sqrt(HIDDEN_DIM):.4f})")

    # Feature directions orthogonality
    fd = model.feature_directions.to(device)
    cos_sim = fd @ fd.T
    off_diag = cos_sim - torch.eye(NUM_FEATURES, device=device)
    print(f"Feature directions max off-diag cosine: {off_diag.abs().max().item():.6f}")

    print("\nSanity check PASSED.\n")


# ── Pilot Runs ───────────────────────────────────────────────────────


def run_pilot(gen_flat, gen_seq, eval_hidden, device):
    """Quick pilot to find L1 ranges."""
    print("=" * 60)
    print("PILOT RUNS")
    print("=" * 60)

    pilot_steps = 5000

    # SAE pilots — wide L1 range to find where L0 goes from ~1 to ~30
    print("\n--- SAE Pilots ---")
    sae_pilots = []
    for l1c in [1e-3, 1e-2, 5e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0]:
        set_seed(SEED)
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=pilot_steps,
            batch_size=SAE_BATCH_SIZE,
            l1_coeff=l1c,
            log_every=pilot_steps,  # only log final
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        result = eval_sae_on_data(sae, eval_hidden, device)
        sae_pilots.append({"l1": l1c, **result})
        print(f"  l1={l1c:.1e} -> NMSE={result['nmse']:.6f}, L0={result['l0']:.2f}")
        del sae
        torch.cuda.empty_cache()

    # TFA pilots — wide L1 range
    print("\n--- TFA Pilots ---")
    tfa_pilots = []
    for l1c in [1e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0, 30.0]:
        set_seed(SEED)
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=999,  # k unused for relu
            n_heads=TFA_N_HEADS, n_attn_layers=TFA_N_ATTN_LAYERS,
            bottleneck_factor=TFA_BOTTLENECK, device=device,
        )
        # Override sae_diff_type to relu
        tfa.sae_diff_type = 'relu'
        tfa_cfg = TFATrainingConfig(
            total_steps=pilot_steps,
            batch_size=TFA_SEQ_BATCH,
            l1_coeff=l1c,
            log_every=pilot_steps,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa_on_data(tfa, eval_hidden, device)
        tfa_pilots.append({"l1": l1c, **result})
        print(f"  l1={l1c:.1e} -> NMSE={result['nmse']:.6f}, "
              f"novel_L0={result['novel_l0']:.2f}, total_L0={result['total_l0']:.2f}, "
              f"pred_E={result['pred_energy_frac']:.3f}")
        del tfa
        torch.cuda.empty_cache()

    return sae_pilots, tfa_pilots


# ── Full Sweep ───────────────────────────────────────────────────────


def run_full_sweep(l1_values_sae, l1_values_tfa, gen_flat, gen_seq,
                   eval_hidden, device, results_dir):
    """Train both models across L1 sweeps and produce Pareto data."""

    print("\n" + "=" * 60)
    print("FULL SAE SWEEP")
    print("=" * 60)

    sae_results = []
    for i, l1c in enumerate(l1_values_sae):
        set_seed(SEED)
        print(f"\n--- SAE [{i+1}/{len(l1_values_sae)}] l1={l1c:.2e} ---")
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=TOTAL_STEPS,
            batch_size=SAE_BATCH_SIZE,
            l1_coeff=l1c,
            log_every=LOG_EVERY,
        )
        sae, sae_log = train_relu_sae(sae, gen_flat, cfg, device)
        result = eval_sae_on_data(sae, eval_hidden, device)
        dt = time.time() - t0
        print(f"  EVAL: NMSE={result['nmse']:.6f}, L0={result['l0']:.2f} ({dt:.1f}s)")
        sae_results.append({"l1_coeff": l1c, **result})
        del sae
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("FULL TFA SWEEP")
    print("=" * 60)

    tfa_results = []
    for i, l1c in enumerate(l1_values_tfa):
        set_seed(SEED)
        print(f"\n--- TFA [{i+1}/{len(l1_values_tfa)}] l1={l1c:.2e} ---")
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=999,
            n_heads=TFA_N_HEADS, n_attn_layers=TFA_N_ATTN_LAYERS,
            bottleneck_factor=TFA_BOTTLENECK, device=device,
        )
        tfa.sae_diff_type = 'relu'
        tfa_cfg = TFATrainingConfig(
            total_steps=TOTAL_STEPS,
            batch_size=TFA_SEQ_BATCH,
            l1_coeff=l1c,
            log_every=LOG_EVERY,
        )
        tfa, tfa_log = train_tfa(tfa, gen_seq, tfa_cfg, device)
        result = eval_tfa_on_data(tfa, eval_hidden, device)
        dt = time.time() - t0
        print(f"  EVAL: NMSE={result['nmse']:.6f}, novel_L0={result['novel_l0']:.2f}, "
              f"total_L0={result['total_l0']:.2f}, pred_E={result['pred_energy_frac']:.3f} "
              f"({dt:.1f}s)")
        tfa_results.append({"l1_coeff": l1c, **result})
        del tfa
        torch.cuda.empty_cache()

    return sae_results, tfa_results


# ── Plotting ─────────────────────────────────────────────────────────


def plot_pareto_frontier(sae_results, tfa_results, results_dir):
    """Three-curve Pareto plot: SAE, TFA-novel-L0, TFA-total-L0."""

    fig, ax = plt.subplots(figsize=(10, 7))

    # SAE
    sae_l0 = [r["l0"] for r in sae_results]
    sae_nmse = [r["nmse"] for r in sae_results]
    # Sort by L0
    sae_order = np.argsort(sae_l0)
    sae_l0_sorted = [sae_l0[i] for i in sae_order]
    sae_nmse_sorted = [sae_nmse[i] for i in sae_order]
    ax.plot(sae_l0_sorted, sae_nmse_sorted, "o-", color="tab:blue",
            linewidth=2, markersize=7, label="Standard SAE", zorder=3)

    # TFA novel L0
    tfa_novel_l0 = [r["novel_l0"] for r in tfa_results]
    tfa_nmse = [r["nmse"] for r in tfa_results]
    tfa_order_novel = np.argsort(tfa_novel_l0)
    tfa_nl0_sorted = [tfa_novel_l0[i] for i in tfa_order_novel]
    tfa_nmse_novel_sorted = [tfa_nmse[i] for i in tfa_order_novel]
    ax.plot(tfa_nl0_sorted, tfa_nmse_novel_sorted, "s-", color="tab:orange",
            linewidth=2, markersize=7, label="TFA (novel L0)", zorder=3)

    # TFA total L0
    tfa_total_l0 = [r["total_l0"] for r in tfa_results]
    tfa_order_total = np.argsort(tfa_total_l0)
    tfa_tl0_sorted = [tfa_total_l0[i] for i in tfa_order_total]
    tfa_nmse_total_sorted = [tfa_nmse[i] for i in tfa_order_total]
    ax.plot(tfa_tl0_sorted, tfa_nmse_total_sorted, "^--", color="tab:red",
            linewidth=1.5, markersize=7, alpha=0.7, label="TFA (total L0)", zorder=2)

    ax.axvline(x=EXPECTED_L0, color="gray", linestyle="--", alpha=0.5,
               label=f"$E[L_0]$ = {EXPECTED_L0:.0f}")

    ax.set_xlabel("L0 (avg active features)", fontsize=13)
    ax.set_ylabel("NMSE", fontsize=13)
    ax.set_title("Pareto Frontier: NMSE vs L0\n"
                 f"(n={NUM_FEATURES}, d={HIDDEN_DIM}, $\\pi$={PI[0]}, "
                 f"$E[L_0]$={EXPECTED_L0:.0f})", fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    ax.set_xlim(left=0)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"pareto_nmse_vs_l0.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved pareto_nmse_vs_l0.png/pdf")


def plot_tfa_decomposition(tfa_results, results_dir):
    """Predictable energy fraction vs novel L0."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    novel_l0 = [r["novel_l0"] for r in tfa_results]
    pred_frac = [r["pred_energy_frac"] for r in tfa_results]
    total_l0 = [r["total_l0"] for r in tfa_results]

    order = np.argsort(novel_l0)
    nl0 = [novel_l0[i] for i in order]
    pf = [pred_frac[i] for i in order]
    tl0 = [total_l0[i] for i in order]

    ax1.plot(nl0, pf, "o-", color="tab:green", linewidth=2, markersize=7)
    ax1.set_xlabel("Novel L0", fontsize=12)
    ax1.set_ylabel("Predictable energy fraction", fontsize=12)
    ax1.set_title("Predictable vs Novel Energy", fontsize=12)
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)

    ax2.plot(nl0, tl0, "o-", color="tab:red", linewidth=2, markersize=7)
    ax2.set_xlabel("Novel L0", fontsize=12)
    ax2.set_ylabel("Total L0", fontsize=12)
    ax2.set_title("Total vs Novel L0", fontsize=12)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(results_dir, f"tfa_decomposition.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved tfa_decomposition.png/pdf")


def plot_summary_table(sae_results, tfa_results, results_dir):
    """Print and save a summary comparison."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Model':<6} {'L1':>8} {'NMSE':>10} {'L0':>8} {'novel_L0':>10} "
          f"{'total_L0':>10} {'pred_E':>8}")
    print("-" * 80)
    for r in sorted(sae_results, key=lambda x: x["l0"]):
        print(f"{'SAE':<6} {r['l1_coeff']:>8.2e} {r['nmse']:>10.6f} "
              f"{r['l0']:>8.2f} {'—':>10} {'—':>10} {'—':>8}")
    print("-" * 80)
    for r in sorted(tfa_results, key=lambda x: x["novel_l0"]):
        print(f"{'TFA':<6} {r['l1_coeff']:>8.2e} {r['nmse']:>10.6f} "
              f"{'—':>8} {r['novel_l0']:>10.2f} {r['total_l0']:>10.2f} "
              f"{r['pred_energy_frac']:>8.3f}")


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}")
    print(f"Config: n={NUM_FEATURES}, d={HIDDEN_DIM}, T={SEQ_LEN}")
    print(f"pi={PI[0]}, E[L0]={EXPECTED_L0}")
    print(f"rho: {RHO}")
    print(f"Dict width: {DICT_WIDTH}")
    print(f"Total steps: {TOTAL_STEPS}")

    # ── Setup ──
    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(
        num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM,
    ).to(device)
    model.eval()

    scaling_factor, mean_norm = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"\nScaling: mean_norm={mean_norm:.4f}, factor={scaling_factor:.4f}, "
          f"target={math.sqrt(HIDDEN_DIM):.4f}\n")

    gen_flat, gen_seq = make_data_generators(model, pi_t, rho_t, device, scaling_factor)

    # ── Sanity check ──
    run_sanity_check(model, pi_t, rho_t, scaling_factor, device)

    # ── Fixed eval data ──
    set_seed(SEED + 100)
    eval_hidden, eval_support = generate_eval_data(
        model, pi_t, rho_t, device, scaling_factor
    )
    print(f"Eval data: {eval_hidden.shape} ({eval_hidden.shape[0] * eval_hidden.shape[1]} tokens)")

    # ── Pilot runs ──
    sae_pilots, tfa_pilots = run_pilot(gen_flat, gen_seq, eval_hidden, device)

    # Determine L1 ranges from pilots
    # SAE: find range where L0 goes from ~1 to ~25
    print("\n--- Determining L1 sweep ranges ---")
    print("SAE pilot results:")
    for p in sae_pilots:
        print(f"  l1={p['l1']:.1e}: L0={p['l0']:.2f}, NMSE={p['nmse']:.6f}")
    print("TFA pilot results:")
    for p in tfa_pilots:
        print(f"  l1={p['l1']:.1e}: novel_L0={p['novel_l0']:.2f}, "
              f"total_L0={p['total_l0']:.2f}, NMSE={p['nmse']:.6f}")

    # Build sweep ranges: we want L0 to span roughly [1, 25] for SAE
    # and novel_L0 to span [1, 25] for TFA.
    # Find L1 values that bracket these L0 ranges from pilots.
    def find_l1_range(pilots, l0_key, target_lo=2.0, target_hi=25.0):
        """Interpolate to find L1 values giving target L0 range."""
        sorted_p = sorted(pilots, key=lambda p: p["l1"])
        l1s = [p["l1"] for p in sorted_p]
        l0s = [p[l0_key] for p in sorted_p]
        # L0 decreases with increasing L1 (generally)
        l1_for_hi_l0 = l1s[0]  # lowest L1 -> highest L0
        l1_for_lo_l0 = l1s[-1]  # highest L1 -> lowest L0
        # Find bracketing L1 for target_hi L0
        for i in range(len(l0s) - 1):
            if l0s[i] >= target_hi >= l0s[i + 1]:
                l1_for_hi_l0 = l1s[i]
                break
        # Find bracketing L1 for target_lo L0
        for i in range(len(l0s) - 1):
            if l0s[i] >= target_lo >= l0s[i + 1]:
                l1_for_lo_l0 = l1s[i + 1]
                break
        # Extend slightly beyond brackets
        return l1_for_hi_l0 / 2, l1_for_lo_l0 * 2

    sae_l1_min, sae_l1_max = find_l1_range(sae_pilots, "l0")
    tfa_l1_min, tfa_l1_max = find_l1_range(tfa_pilots, "novel_l0")

    n_points = 15
    sae_l1_values = np.logspace(
        np.log10(sae_l1_min), np.log10(sae_l1_max), n_points
    ).tolist()
    tfa_l1_values = np.logspace(
        np.log10(tfa_l1_min), np.log10(tfa_l1_max), n_points
    ).tolist()

    print(f"\nSAE L1 sweep: {sae_l1_min:.2e} to {sae_l1_max:.2e} ({n_points} points)")
    print(f"TFA L1 sweep: {tfa_l1_min:.2e} to {tfa_l1_max:.2e} ({n_points} points)")

    # ── Full sweep ──
    sae_results, tfa_results = run_full_sweep(
        sae_l1_values, tfa_l1_values, gen_flat, gen_seq,
        eval_hidden, device, RESULTS_DIR
    )

    # ── Summary and plots ──
    plot_summary_table(sae_results, tfa_results, RESULTS_DIR)

    print("\nGenerating plots...")
    plot_pareto_frontier(sae_results, tfa_results, RESULTS_DIR)
    plot_tfa_decomposition(tfa_results, RESULTS_DIR)

    # ── Save ──
    results_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "expected_l0": EXPECTED_L0,
            "dict_width": DICT_WIDTH,
            "total_steps": TOTAL_STEPS,
            "scaling_factor": scaling_factor,
            "seed": SEED,
        },
        "sae_pilots": sae_pilots,
        "tfa_pilots": tfa_pilots,
        "sae_l1_values": sae_l1_values,
        "tfa_l1_values": tfa_l1_values,
        "sae_results": sae_results,
        "tfa_results": tfa_results,
    }
    json_path = os.path.join(RESULTS_DIR, "results.json")
    with open(json_path, "w") as f:
        json.dump(results_data, f, indent=2)
    print(f"\nResults saved to {json_path}")
    print("Done!")


if __name__ == "__main__":
    main()
