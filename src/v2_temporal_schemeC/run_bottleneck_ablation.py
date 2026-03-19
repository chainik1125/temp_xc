"""Bottleneck ablation: does reducing TFA's attention capacity
make temporal correlations more important for reconstruction?

Hypothesis: at full attention capacity, TFA's advantage over a
standard SAE is mostly architectural (content-based retrieval via
the attention mechanism). As we bottleneck the K/Q projections,
content matching degrades, and temporal prediction should become
a relatively larger fraction of TFA's advantage.

We sweep:
  Part 1 — bottleneck_factor ∈ {1, 2, 5, 10} with n_heads=4 fixed
  Part 2 — n_heads ∈ {1, 2, 4} with bottleneck_factor=1 fixed

For each capacity configuration and each TopK budget k, we train:
  - TopK SAE        (baseline, per-token, no temporal info)
  - TFA             (trained on temporally ordered sequences)
  - TFA-shuffled    (trained on position-shuffled sequences)

All models evaluated on the same unshuffled temporal eval data.

Key metric: temporal fraction
  = (NMSE_shuf - NMSE_tfa) / (NMSE_sae - NMSE_tfa)
This is the share of TFA's total advantage over the SAE that comes
from temporal information (as opposed to architectural capacity).

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_bottleneck_ablation.py
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

# ── Data configuration ──────────────────────────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4

DICT_WIDTH = 40

# ── Capacity sweep configurations ───────────────────────────────────

# Part 1: bottleneck sweep (n_heads=4 fixed)
# width=40, n_heads=4 → valid bf where 40 % (bf*4) == 0: 1, 2, 5, 10
BOTTLENECK_VALUES = [1, 2, 5, 10]
BOTTLENECK_NHEADS = 4

# Part 2: n_heads sweep (bf=1 fixed)
NHEADS_VALUES = [1, 2, 4]
NHEADS_BF = 1

K_VALUES = [3, 5, 8, 10]

# ── Training configuration ──────────────────────────────────────────

SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "bottleneck_ablation"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            10000, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
    return math.sqrt(HIDDEN_DIM) / norms.mean().item()


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
            x = flat[s : min(s + bs, n)].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_signal, "l0": total_l0 / n}


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    bs = 256
    total_se = total_signal = total_novel_l0 = total_total_l0 = 0.0
    total_pred_e = total_novel_e = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, bs):
            x = eval_hidden[s : min(s + bs, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf = x.reshape(-1, D)
            rf = recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            nc = inter["novel_codes"]
            pc = inter["pred_codes"]
            total_novel_l0 += (nc > 0).float().sum(dim=-1).sum().item()
            total_total_l0 += (
                ((nc + pc).abs() > 1e-8).float().sum(dim=-1).sum().item()
            )
            total_pred_e += inter["pred_recons"].norm(dim=-1).pow(2).sum().item()
            total_novel_e += (
                inter["novel_recons"].norm(dim=-1).pow(2).sum().item()
            )
            n_tokens += B * T
    te = total_pred_e + total_novel_e + 1e-12
    return {
        "nmse": total_se / total_signal,
        "novel_l0": total_novel_l0 / n_tokens,
        "total_l0": total_total_l0 / n_tokens,
        "pred_energy_frac": total_pred_e / te,
    }


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def tfa_label(n_heads, bottleneck_factor):
    """Short label for a TFA capacity configuration."""
    return f"nh{n_heads}_bf{bottleneck_factor}"


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    set_seed(SEED)
    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}", flush=True)

    # Data generators
    gen_flat, gen_seq = make_generators(
        model, pi_t, rho_t, device, sf, shuffle=False
    )
    gen_flat_shuf, gen_seq_shuf = make_generators(
        model, pi_t, rho_t, device, sf, shuffle=True
    )

    # Eval data (unshuffled, held-out seed)
    set_seed(SEED + 100)
    acts, _ = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts) * sf
    print(
        f"Eval: {eval_hidden.shape[0]}x{eval_hidden.shape[1]} tokens",
        flush=True,
    )

    # ── Build capacity configurations to sweep ──────────────────────

    capacity_configs = []

    # Part 1: bottleneck sweep (n_heads=4)
    for bf in BOTTLENECK_VALUES:
        capacity_configs.append(
            {"n_heads": BOTTLENECK_NHEADS, "bottleneck_factor": bf}
        )

    # Part 2: n_heads sweep (bf=1), skip n_heads=4 (already in Part 1)
    for nh in NHEADS_VALUES:
        if nh == BOTTLENECK_NHEADS and NHEADS_BF == 1:
            continue  # already covered by Part 1 bf=1
        capacity_configs.append(
            {"n_heads": nh, "bottleneck_factor": NHEADS_BF}
        )

    # Print capacity configurations with param counts
    print(f"\nCapacity configurations:", flush=True)
    for cc in capacity_configs:
        test_tfa = create_tfa(
            dimin=HIDDEN_DIM,
            width=DICT_WIDTH,
            k=5,
            n_heads=cc["n_heads"],
            bottleneck_factor=cc["bottleneck_factor"],
            device="cpu",
        )
        n_embds = DICT_WIDTH // cc["bottleneck_factor"]
        kq_per_head = n_embds // cc["n_heads"]
        print(
            f"  {tfa_label(**cc):>10s}: "
            f"params={count_params(test_tfa):,}, "
            f"n_embds={n_embds}, "
            f"KQ/head={kq_per_head}",
            flush=True,
        )
        del test_tfa

    # ── Step 1: Train SAE baselines (one per k) ────────────────────

    print(f"\n{'='*70}", flush=True)
    print("STEP 1: Training SAE baselines", flush=True)
    print(f"{'='*70}", flush=True)

    sae_results = {}
    for k in K_VALUES:
        set_seed(SEED)
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=SAE_TOTAL_STEPS,
            batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR,
            l1_coeff=0.0,
            log_every=SAE_TOTAL_STEPS,
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        print(
            f"  k={k:>2d}: NMSE={r['nmse']:.6f}, "
            f"L0={r['l0']:.2f} ({time.time()-t0:.1f}s)",
            flush=True,
        )
        sae_results[k] = r
        del sae
        torch.cuda.empty_cache()

    # ── Step 2: Train TFA and TFA-shuffled at each capacity config ──

    # Results dict: key = tfa_label, value = {k: {tfa: {...}, tfa_shuf: {...}}}
    all_tfa_results = {}

    for ci, cc in enumerate(capacity_configs):
        label = tfa_label(**cc)
        print(f"\n{'='*70}", flush=True)
        print(
            f"STEP 2.{ci+1}: Capacity config {label} "
            f"(n_heads={cc['n_heads']}, bf={cc['bottleneck_factor']})",
            flush=True,
        )
        print(f"{'='*70}", flush=True)

        all_tfa_results[label] = {}

        for k in K_VALUES:
            print(f"\n  k={k}:", flush=True)

            # TFA (temporal)
            set_seed(SEED)
            t0 = time.time()
            tfa = create_tfa(
                dimin=HIDDEN_DIM,
                width=DICT_WIDTH,
                k=k,
                n_heads=cc["n_heads"],
                bottleneck_factor=cc["bottleneck_factor"],
                device=device,
            )
            tfa_cfg = TFATrainingConfig(
                total_steps=TFA_TOTAL_STEPS,
                batch_size=TFA_BATCH_SIZE,
                lr=TFA_LR,
                log_every=TFA_TOTAL_STEPS,
            )
            tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
            r_tfa = eval_tfa(tfa, eval_hidden, device)
            print(
                f"    TFA:     NMSE={r_tfa['nmse']:.6f}, "
                f"novel_L0={r_tfa['novel_l0']:.2f}, "
                f"total_L0={r_tfa['total_l0']:.2f}, "
                f"pred_E={r_tfa['pred_energy_frac']:.3f} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )
            del tfa
            torch.cuda.empty_cache()

            # TFA-shuffled
            set_seed(SEED)
            t0 = time.time()
            tfa_shuf = create_tfa(
                dimin=HIDDEN_DIM,
                width=DICT_WIDTH,
                k=k,
                n_heads=cc["n_heads"],
                bottleneck_factor=cc["bottleneck_factor"],
                device=device,
            )
            tfa_shuf, _ = train_tfa(tfa_shuf, gen_seq_shuf, tfa_cfg, device)
            r_shuf = eval_tfa(tfa_shuf, eval_hidden, device)
            print(
                f"    TFA-shuf: NMSE={r_shuf['nmse']:.6f}, "
                f"novel_L0={r_shuf['novel_l0']:.2f}, "
                f"total_L0={r_shuf['total_l0']:.2f}, "
                f"pred_E={r_shuf['pred_energy_frac']:.3f} "
                f"({time.time()-t0:.1f}s)",
                flush=True,
            )
            del tfa_shuf
            torch.cuda.empty_cache()

            # Compute derived metrics
            sae_nmse = sae_results[k]["nmse"]
            tfa_nmse = r_tfa["nmse"]
            shuf_nmse = r_shuf["nmse"]

            total_gap = sae_nmse - tfa_nmse
            temporal_gap = shuf_nmse - tfa_nmse
            shuf_tfa_ratio = shuf_nmse / tfa_nmse if tfa_nmse > 0 else float("inf")

            if total_gap > 1e-10:
                temporal_frac = temporal_gap / total_gap
                arch_frac = 1.0 - temporal_frac
            else:
                temporal_frac = float("nan")
                arch_frac = float("nan")

            print(
                f"    SAE NMSE={sae_nmse:.6f} | "
                f"shuf/tfa ratio={shuf_tfa_ratio:.3f} | "
                f"temporal={temporal_frac*100:.1f}% | "
                f"architecture={arch_frac*100:.1f}%",
                flush=True,
            )

            all_tfa_results[label][k] = {
                "tfa": r_tfa,
                "tfa_shuf": r_shuf,
                "sae_nmse": sae_nmse,
                "shuf_tfa_ratio": shuf_tfa_ratio,
                "temporal_frac": temporal_frac,
                "arch_frac": arch_frac,
            }

    # ── Summary tables ──────────────────────────────────────────────

    print(f"\n\n{'='*90}", flush=True)
    print("SUMMARY: Part 1 — Bottleneck Sweep (n_heads=4)", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"{'bf':>4} | {'KQ/hd':>5}"
    for k in K_VALUES:
        header += f" | k={k} shuf/tfa"
        header += f" | k={k} temp%"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for bf in BOTTLENECK_VALUES:
        label = tfa_label(BOTTLENECK_NHEADS, bf)
        n_embds = DICT_WIDTH // bf
        kq_per_head = n_embds // BOTTLENECK_NHEADS
        row = f"{bf:>4d} | {kq_per_head:>5d}"
        for k in K_VALUES:
            r = all_tfa_results[label][k]
            row += f" |       {r['shuf_tfa_ratio']:>6.3f}"
            tf = r["temporal_frac"]
            row += f" |     {tf*100:>5.1f}%"
        print(row, flush=True)

    print(f"\n{'='*90}", flush=True)
    print("SUMMARY: Part 2 — n_heads Sweep (bf=1)", flush=True)
    print(f"{'='*90}", flush=True)

    header = f"{'nh':>4} | {'KQ/hd':>5}"
    for k in K_VALUES:
        header += f" | k={k} shuf/tfa"
        header += f" | k={k} temp%"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for nh in NHEADS_VALUES:
        label = tfa_label(nh, NHEADS_BF)
        kq_per_head = DICT_WIDTH // nh
        row = f"{nh:>4d} | {kq_per_head:>5d}"
        for k in K_VALUES:
            r = all_tfa_results[label][k]
            row += f" |       {r['shuf_tfa_ratio']:>6.3f}"
            tf = r["temporal_frac"]
            row += f" |     {tf*100:>5.1f}%"
        print(row, flush=True)

    # ── Detailed NMSE table ──

    print(f"\n{'='*90}", flush=True)
    print("DETAILED NMSE TABLE", flush=True)
    print(f"{'='*90}", flush=True)

    print(f"{'config':>12s}", end="", flush=True)
    for k in K_VALUES:
        print(f" | k={k:>2d} SAE    TFA    shuf", end="", flush=True)
    print(flush=True)
    print("-" * 110, flush=True)

    for label in all_tfa_results:
        print(f"{label:>12s}", end="", flush=True)
        for k in K_VALUES:
            r = all_tfa_results[label][k]
            print(
                f" |     {r['sae_nmse']:.4f} {r['tfa']['nmse']:.4f} {r['tfa_shuf']['nmse']:.4f}",
                end="",
                flush=True,
            )
        print(flush=True)

    # ── Plots ───────────────────────────────────────────────────────

    _make_plots(sae_results, all_tfa_results)

    # ── Save ────────────────────────────────────────────────────────

    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "pi": PI,
            "rho": RHO,
            "dict_width": DICT_WIDTH,
            "k_values": K_VALUES,
            "bottleneck_values": BOTTLENECK_VALUES,
            "bottleneck_nheads": BOTTLENECK_NHEADS,
            "nheads_values": NHEADS_VALUES,
            "nheads_bf": NHEADS_BF,
            "sae_total_steps": SAE_TOTAL_STEPS,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "seed": SEED,
        },
        "sae_results": {str(k): v for k, v in sae_results.items()},
        "tfa_results": {
            label: {str(k): v for k, v in kv.items()}
            for label, kv in all_tfa_results.items()
        },
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}", flush=True)


def _make_plots(sae_results, all_tfa_results):
    """Generate all plots for the bottleneck ablation experiment."""

    colors_bf = {1: "tab:blue", 2: "tab:green", 5: "tab:orange", 10: "tab:red"}
    colors_nh = {1: "tab:purple", 2: "tab:cyan", 4: "tab:blue"}

    # ── Plot 1: Temporal fraction vs bottleneck_factor ──────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: temporal fraction vs bottleneck (Part 1)
    ax = axes[0]
    for k in K_VALUES:
        temporal_fracs = []
        for bf in BOTTLENECK_VALUES:
            label = tfa_label(BOTTLENECK_NHEADS, bf)
            tf = all_tfa_results[label][k]["temporal_frac"]
            temporal_fracs.append(tf * 100)
        ax.plot(
            BOTTLENECK_VALUES,
            temporal_fracs,
            "o-",
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )

    ax.set_xlabel("Bottleneck factor (higher = less K/Q capacity)", fontsize=12)
    ax.set_ylabel("Temporal fraction of TFA advantage (%)", fontsize=12)
    ax.set_title("Part 1: Bottleneck sweep (n_heads=4)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BOTTLENECK_VALUES)

    # Right: shuf/tfa ratio vs bottleneck (Part 1)
    ax = axes[1]
    for k in K_VALUES:
        ratios = []
        for bf in BOTTLENECK_VALUES:
            label = tfa_label(BOTTLENECK_NHEADS, bf)
            ratios.append(all_tfa_results[label][k]["shuf_tfa_ratio"])
        ax.plot(
            BOTTLENECK_VALUES,
            ratios,
            "s-",
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No temporal benefit")
    ax.set_xlabel("Bottleneck factor", fontsize=12)
    ax.set_ylabel("TFA-shuffled / TFA NMSE ratio", fontsize=12)
    ax.set_title("Part 1: Temporal benefit ratio", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(BOTTLENECK_VALUES)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(RESULTS_DIR, f"bottleneck_sweep.{ext}"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ── Plot 2: NMSE curves at each bottleneck level ───────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, bf in enumerate(BOTTLENECK_VALUES):
        ax = axes[idx]
        label = tfa_label(BOTTLENECK_NHEADS, bf)

        sae_nmse = [sae_results[k]["nmse"] for k in K_VALUES]
        tfa_nmse = [all_tfa_results[label][k]["tfa"]["nmse"] for k in K_VALUES]
        shuf_nmse = [
            all_tfa_results[label][k]["tfa_shuf"]["nmse"] for k in K_VALUES
        ]

        ax.plot(
            K_VALUES,
            sae_nmse,
            "o-",
            color="tab:blue",
            linewidth=2,
            markersize=7,
            label="TopK SAE",
        )
        ax.plot(
            K_VALUES,
            tfa_nmse,
            "s-",
            color="tab:orange",
            linewidth=2,
            markersize=7,
            label="TFA (temporal)",
        )
        ax.plot(
            K_VALUES,
            shuf_nmse,
            "^--",
            color="tab:red",
            linewidth=2,
            markersize=7,
            label="TFA-shuffled",
        )

        n_embds = DICT_WIDTH // bf
        kq_per_head = n_embds // BOTTLENECK_NHEADS

        ax.set_xlabel("k (TopK budget)", fontsize=11)
        ax.set_ylabel("NMSE", fontsize=11)
        ax.set_title(f"bf={bf} (KQ/head={kq_per_head})", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    plt.suptitle(
        "NMSE vs k at each bottleneck level (n_heads=4)", fontsize=14, y=1.01
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(RESULTS_DIR, f"nmse_by_bottleneck.{ext}"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ── Plot 3: n_heads sweep ──────────────────────────────────────

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: temporal fraction vs n_heads
    ax = axes[0]
    for k in K_VALUES:
        temporal_fracs = []
        for nh in NHEADS_VALUES:
            label = tfa_label(nh, NHEADS_BF)
            tf = all_tfa_results[label][k]["temporal_frac"]
            temporal_fracs.append(tf * 100)
        ax.plot(
            NHEADS_VALUES,
            temporal_fracs,
            "o-",
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )

    ax.set_xlabel("Number of attention heads", fontsize=12)
    ax.set_ylabel("Temporal fraction (%)", fontsize=12)
    ax.set_title("Part 2: n_heads sweep (bf=1)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(NHEADS_VALUES)

    # Right: shuf/tfa ratio vs n_heads
    ax = axes[1]
    for k in K_VALUES:
        ratios = []
        for nh in NHEADS_VALUES:
            label = tfa_label(nh, NHEADS_BF)
            ratios.append(all_tfa_results[label][k]["shuf_tfa_ratio"])
        ax.plot(
            NHEADS_VALUES,
            ratios,
            "s-",
            linewidth=2,
            markersize=7,
            label=f"k={k}",
        )

    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="No temporal benefit")
    ax.set_xlabel("Number of attention heads", fontsize=12)
    ax.set_ylabel("TFA-shuffled / TFA NMSE ratio", fontsize=12)
    ax.set_title("Part 2: Temporal benefit ratio", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(NHEADS_VALUES)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(RESULTS_DIR, f"nheads_sweep.{ext}"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)

    # ── Plot 4: Combined summary ───────────────────────────────────

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))

    # Plot temporal fraction as heatmap-like scatter for bottleneck sweep
    for ki, k in enumerate(K_VALUES):
        for bi, bf in enumerate(BOTTLENECK_VALUES):
            label = tfa_label(BOTTLENECK_NHEADS, bf)
            tf = all_tfa_results[label][k]["temporal_frac"] * 100
            size = max(50, min(300, tf * 10))
            color = plt.cm.RdYlGn(tf / 100.0)
            ax.scatter(bf, k, s=size, c=[color], edgecolors="black", linewidth=1, zorder=5)
            ax.annotate(
                f"{tf:.0f}%",
                (bf, k),
                textcoords="offset points",
                xytext=(15, 0),
                fontsize=10,
                ha="left",
            )

    ax.set_xlabel("Bottleneck factor", fontsize=13)
    ax.set_ylabel("k (TopK budget)", fontsize=13)
    ax.set_title(
        "Temporal fraction (%) at each (bottleneck, k)\n"
        "Larger/greener = more temporal contribution",
        fontsize=12,
    )
    ax.set_xticks(BOTTLENECK_VALUES)
    ax.set_yticks(K_VALUES)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(RESULTS_DIR, f"temporal_fraction_grid.{ext}"),
            dpi=150,
            bbox_inches="tight",
        )
    plt.close(fig)

    print("Plots saved.", flush=True)


if __name__ == "__main__":
    main()
