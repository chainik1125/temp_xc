"""Shuffle diagnostic on event-structured vs independent data.

Compares TFA's temporal fraction under three data generation regimes:
  1. Independent features: each feature has its own Markov chain (baseline)
  2. Event-structured (non-overlapping): features grouped into events, each
     event has a single Markov chain driving all its member features
  3. Event-structured (overlapping): like (2) but some features belong to
     two events, creating cross-event correlations

The question: does event-level structure (co-activating feature groups)
make temporal information more important for TFA, compared to the
per-feature autocorrelation tested in the original shuffle diagnostic?

Additionally tests n_heads=2 for event data, since the bottleneck ablation
found n_heads=2 is the sweet spot for temporal exploitation.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_event_shuffle_diagnostic.py
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
from src.v2_temporal_schemeC.factorial_hmm import (
    generate_event_activations,
    generate_event_activations_general,
    create_overlapping_membership,
    compute_marginal_pi,
)
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

# ── Shared configuration ────────────────────────────────────────────

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
DICT_WIDTH = 40

K_VALUES = [3, 5, 8, 10]

SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
SEED = 42

# ── Data mode configurations ────────────────────────────────────────

# Independent: 20 features, each with own Markov chain
INDEP_PI = [0.5] * NUM_FEATURES
INDEP_RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4

# Event (non-overlapping): 4 events × 5 features
N_EVENTS = 4
FEATURES_PER_EVENT = 5  # 4 × 5 = 20 features total
EVENT_PI = [0.5] * N_EVENTS  # E[active events] = 2, E[L0] = 2*5 = 10
EVENT_RHO = [0.3, 0.5, 0.7, 0.9]  # one rho per event

# Event (overlapping): 4 events, 4 exclusive features each + 4 shared
OVERLAP_N_EVENTS = 4
OVERLAP_BASE_PER_EVENT = 4  # 16 exclusive features
OVERLAP_N_SHARED = 4  # 4 features belong to 2 events
OVERLAP_PI = [0.5] * OVERLAP_N_EVENTS
OVERLAP_RHO = [0.3, 0.5, 0.7, 0.9]

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "event_shuffle_diagnostic"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, gen_acts_fn, device):
    """Compute scaling factor sqrt(d) / mean(||x||) for any data generator."""
    with torch.no_grad():
        hidden = model(gen_acts_fn())
        norms = hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1)
    return math.sqrt(HIDDEN_DIM) / norms.mean().item()


def compute_cross_correlation(support: torch.Tensor) -> torch.Tensor:
    """Compute (n_features, n_features) correlation matrix across features.

    Args:
        support: (batch, seq_len, n_features) binary support tensor.

    Returns:
        (n_features, n_features) Pearson correlation matrix.
    """
    flat = support.reshape(-1, support.shape[-1]).float()  # (B*T, F)
    centered = flat - flat.mean(dim=0)
    stds = centered.std(dim=0).clamp(min=1e-8)
    normed = centered / stds
    corr = (normed.T @ normed) / normed.shape[0]
    return corr


def plot_cross_correlations(corr_dict, save_dir):
    """Plot cross-feature correlation heatmaps for each data mode."""
    n_modes = len(corr_dict)
    fig, axes = plt.subplots(1, n_modes, figsize=(6 * n_modes, 5))
    if n_modes == 1:
        axes = [axes]

    for ax, (name, corr) in zip(axes, corr_dict.items()):
        corr_np = corr.cpu().numpy()
        im = ax.imshow(corr_np, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"Feature correlation: {name}", fontsize=11)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature index")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(save_dir, f"cross_feature_correlations.{ext}"),
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)


def make_indep_generators(model, pi_t, rho_t, device, sf, shuffle=False):
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


def make_event_generators(model, device, sf, shuffle=False):
    pi_t = torch.tensor(EVENT_PI)
    rho_t = torch.tensor(EVENT_RHO)

    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _, _ = generate_event_activations(
            n_seq, SEQ_LEN, N_EVENTS, FEATURES_PER_EVENT,
            pi_t, rho_t, device=device,
        )
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _, _ = generate_event_activations(
            n_seq, SEQ_LEN, N_EVENTS, FEATURES_PER_EVENT,
            pi_t, rho_t, device=device,
        )
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return model(acts) * sf

    return gen_flat, gen_seq


def make_overlap_generators(model, membership, device, sf, shuffle=False):
    pi_t = torch.tensor(OVERLAP_PI)
    rho_t = torch.tensor(OVERLAP_RHO)

    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _, _ = generate_event_activations_general(
            n_seq, SEQ_LEN, pi_t, rho_t, membership, device=device,
        )
        if shuffle:
            for i in range(n_seq):
                perm = torch.randperm(SEQ_LEN, device=device)
                acts[i] = acts[i, perm]
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _, _ = generate_event_activations_general(
            n_seq, SEQ_LEN, pi_t, rho_t, membership, device=device,
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


def run_data_mode(
    mode_name, gen_flat, gen_seq, gen_flat_shuf, gen_seq_shuf,
    eval_hidden, device, n_heads=4,
):
    """Run the full shuffle diagnostic for one data mode."""
    print(f"\n{'='*70}", flush=True)
    print(f"DATA MODE: {mode_name} (n_heads={n_heads})", flush=True)
    print(f"{'='*70}", flush=True)

    results = {"sae": [], "tfa": [], "tfa_shuffled": []}
    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS,
        batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR,
        log_every=TFA_TOTAL_STEPS,
    )

    for k in K_VALUES:
        print(f"\n  k={k}:", flush=True)

        # SAE
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
        r_sae = eval_sae(sae, eval_hidden, device)
        print(
            f"    SAE:      NMSE={r_sae['nmse']:.6f}, L0={r_sae['l0']:.2f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        results["sae"].append({"k": k, **r_sae})
        del sae
        torch.cuda.empty_cache()

        # TFA
        set_seed(SEED)
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=n_heads, n_attn_layers=1, bottleneck_factor=1,
            device=device,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r_tfa = eval_tfa(tfa, eval_hidden, device)
        print(
            f"    TFA:      NMSE={r_tfa['nmse']:.6f}, "
            f"novel_L0={r_tfa['novel_l0']:.2f}, "
            f"total_L0={r_tfa['total_l0']:.2f}, "
            f"pred_E={r_tfa['pred_energy_frac']:.3f} "
            f"({time.time()-t0:.1f}s)",
            flush=True,
        )
        results["tfa"].append({"k": k, **r_tfa})
        del tfa
        torch.cuda.empty_cache()

        # TFA-shuffled
        set_seed(SEED)
        t0 = time.time()
        tfa_shuf = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=n_heads, n_attn_layers=1, bottleneck_factor=1,
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
        results["tfa_shuffled"].append({"k": k, **r_shuf})
        del tfa_shuf
        torch.cuda.empty_cache()

        # Derived metrics
        sae_nmse = r_sae["nmse"]
        tfa_nmse = r_tfa["nmse"]
        shuf_nmse = r_shuf["nmse"]
        total_gap = sae_nmse - tfa_nmse
        temporal_gap = shuf_nmse - tfa_nmse
        ratio = shuf_nmse / tfa_nmse if tfa_nmse > 0 else float("inf")

        if total_gap > 1e-10:
            temporal_frac = temporal_gap / total_gap
        else:
            temporal_frac = float("nan")

        print(
            f"    >> shuf/tfa={ratio:.3f} | "
            f"temporal={temporal_frac*100:.1f}% | "
            f"architecture={100-temporal_frac*100:.1f}%",
            flush=True,
        )

    return results


def compute_temporal_frac(results):
    """Extract temporal fraction per k from results dict."""
    fracs = []
    for i in range(len(K_VALUES)):
        s = results["sae"][i]["nmse"]
        t = results["tfa"][i]["nmse"]
        h = results["tfa_shuffled"][i]["nmse"]
        gap = s - t
        tf = (h - t) / gap * 100 if gap > 1e-10 else float("nan")
        fracs.append(tf)
    return fracs


def compute_shuf_ratio(results):
    """Extract shuf/tfa ratio per k from results dict."""
    ratios = []
    for i in range(len(K_VALUES)):
        t = results["tfa"][i]["nmse"]
        h = results["tfa_shuffled"][i]["nmse"]
        ratios.append(h / t if t > 0 else float("inf"))
    return ratios


# ── Main ─────────────────────────────────────────────────────────────


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()

    # ── Build overlapping membership matrix ────────────────────────

    membership = create_overlapping_membership(
        n_features=NUM_FEATURES,
        n_events=OVERLAP_N_EVENTS,
        base_features_per_event=OVERLAP_BASE_PER_EVENT,
        overlap_features=OVERLAP_N_SHARED,
        seed=SEED,
    )
    print(f"\nOverlapping membership matrix ({NUM_FEATURES}x{OVERLAP_N_EVENTS}):",
          flush=True)
    print(f"  Features per event: {membership.sum(dim=0).int().tolist()}", flush=True)
    print(f"  Events per feature: {membership.sum(dim=1).int().tolist()}", flush=True)
    n_multi = (membership.sum(dim=1) > 1).sum().item()
    print(f"  Features in >1 event: {n_multi}", flush=True)

    # Compute marginal pi for overlapping config
    pi_overlap = compute_marginal_pi(
        torch.tensor(OVERLAP_PI), membership
    )
    expected_l0_overlap = pi_overlap.sum().item()
    print(f"  Per-feature pi: {pi_overlap.tolist()}", flush=True)
    print(f"  Expected E[L0] = {expected_l0_overlap:.2f} "
          f"(vs independent/block: 10.00)", flush=True)

    # ── Data sanity checks ──────────────────────────────────────────

    pi_ev = torch.tensor(EVENT_PI)
    rho_ev = torch.tensor(EVENT_RHO)

    # Collect cross-correlation matrices for all three data modes
    corr_dict = {}

    # Independent data cross-correlations
    pi_indep = torch.tensor(INDEP_PI)
    rho_indep = torch.tensor(INDEP_RHO)
    with torch.no_grad():
        acts_check, sup_check = generate_markov_activations(
            5000, SEQ_LEN, pi_indep, rho_indep, device=device
        )
        corr_dict["Independent"] = compute_cross_correlation(sup_check)
        l0_indep = sup_check.sum(dim=-1)
        print(f"\nIndependent data sanity check:", flush=True)
        print(f"  E[L0] = {l0_indep.mean().item():.2f}", flush=True)
        del acts_check, sup_check

    # Block event data sanity check + cross-correlations
    with torch.no_grad():
        _, feat_sup, evt_sup = generate_event_activations(
            5000, SEQ_LEN, N_EVENTS, FEATURES_PER_EVENT,
            pi_ev, rho_ev, device=device,
        )
        corr_dict["Event (block)"] = compute_cross_correlation(feat_sup)
        l0_event = feat_sup.sum(dim=-1)
        print(f"\nBlock event data sanity check:", flush=True)
        print(f"  E[L0] = {l0_event.mean().item():.2f} "
              f"(theory: {N_EVENTS * 0.5 * FEATURES_PER_EVENT})", flush=True)
        print(f"  L0 std = {l0_event.std().item():.2f}", flush=True)
        for g in range(N_EVENTS):
            es = evt_sup[:, :, g]
            a = es[:, :-1]
            b = es[:, 1:]
            ac = (
                ((a - a.mean()) * (b - b.mean())).mean()
                / (a.std() * b.std() + 1e-8)
            ).item()
            print(
                f"  Event {g} (rho={EVENT_RHO[g]:.1f}): "
                f"empirical lag-1 autocorr = {ac:.3f}",
                flush=True,
            )
        del feat_sup, evt_sup

    # Overlapping event data sanity check + cross-correlations
    pi_ov = torch.tensor(OVERLAP_PI)
    rho_ov = torch.tensor(OVERLAP_RHO)
    with torch.no_grad():
        _, feat_sup_ov, evt_sup_ov = generate_event_activations_general(
            5000, SEQ_LEN, pi_ov, rho_ov, membership, device=device,
        )
        corr_dict["Event (overlap)"] = compute_cross_correlation(feat_sup_ov)
        l0_overlap = feat_sup_ov.sum(dim=-1)
        print(f"\nOverlapping event data sanity check:", flush=True)
        print(f"  E[L0] = {l0_overlap.mean().item():.2f} "
              f"(theory: {expected_l0_overlap:.2f})", flush=True)
        print(f"  L0 std = {l0_overlap.std().item():.2f}", flush=True)
        del feat_sup_ov, evt_sup_ov

    # Plot cross-correlation heatmaps
    plot_cross_correlations(corr_dict, RESULTS_DIR)
    print("Cross-correlation heatmaps saved.", flush=True)

    # ── Scaling factors ─────────────────────────────────────────────

    def gen_indep_acts():
        acts, _ = generate_markov_activations(
            10000, SEQ_LEN, pi_indep, rho_indep, device=device
        )
        return acts

    def gen_event_acts():
        acts, _, _ = generate_event_activations(
            10000, SEQ_LEN, N_EVENTS, FEATURES_PER_EVENT,
            pi_ev, rho_ev, device=device,
        )
        return acts

    def gen_overlap_acts():
        acts, _, _ = generate_event_activations_general(
            10000, SEQ_LEN, pi_ov, rho_ov, membership, device=device,
        )
        return acts

    sf_indep = compute_scaling_factor(model, gen_indep_acts, device)
    sf_event = compute_scaling_factor(model, gen_event_acts, device)
    sf_overlap = compute_scaling_factor(model, gen_overlap_acts, device)
    print(f"\nScaling factors: independent={sf_indep:.4f}, "
          f"event={sf_event:.4f}, overlap={sf_overlap:.4f}", flush=True)

    # ── Independent data generators ─────────────────────────────────

    gen_flat_indep, gen_seq_indep = make_indep_generators(
        model, pi_indep, rho_indep, device, sf_indep, shuffle=False
    )
    gen_flat_indep_shuf, gen_seq_indep_shuf = make_indep_generators(
        model, pi_indep, rho_indep, device, sf_indep, shuffle=True
    )

    set_seed(SEED + 100)
    acts_indep, _ = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_indep, rho_indep, device=device
    )
    eval_hidden_indep = model(acts_indep) * sf_indep

    # ── Block event data generators ─────────────────────────────────

    gen_flat_event, gen_seq_event = make_event_generators(
        model, device, sf_event, shuffle=False
    )
    gen_flat_event_shuf, gen_seq_event_shuf = make_event_generators(
        model, device, sf_event, shuffle=True
    )

    set_seed(SEED + 100)
    acts_event, _, _ = generate_event_activations(
        EVAL_N_SEQ, SEQ_LEN, N_EVENTS, FEATURES_PER_EVENT,
        pi_ev, rho_ev, device=device,
    )
    eval_hidden_event = model(acts_event) * sf_event

    # ── Overlapping event data generators ───────────────────────────

    gen_flat_overlap, gen_seq_overlap = make_overlap_generators(
        model, membership, device, sf_overlap, shuffle=False
    )
    gen_flat_overlap_shuf, gen_seq_overlap_shuf = make_overlap_generators(
        model, membership, device, sf_overlap, shuffle=True
    )

    set_seed(SEED + 100)
    acts_overlap, _, _ = generate_event_activations_general(
        EVAL_N_SEQ, SEQ_LEN, pi_ov, rho_ov, membership, device=device,
    )
    eval_hidden_overlap = model(acts_overlap) * sf_overlap

    print(f"\nEval tokens: {EVAL_N_SEQ}x{SEQ_LEN} = {EVAL_N_SEQ*SEQ_LEN}",
          flush=True)

    # ── Run diagnostics ─────────────────────────────────────────────

    all_results = {}

    all_results["independent"] = run_data_mode(
        "independent", gen_flat_indep, gen_seq_indep,
        gen_flat_indep_shuf, gen_seq_indep_shuf,
        eval_hidden_indep, device,
    )

    all_results["event_block"] = run_data_mode(
        "event (4×5 block)", gen_flat_event, gen_seq_event,
        gen_flat_event_shuf, gen_seq_event_shuf,
        eval_hidden_event, device,
    )

    all_results["event_overlap"] = run_data_mode(
        "event (4×4+4 overlap)", gen_flat_overlap, gen_seq_overlap,
        gen_flat_overlap_shuf, gen_seq_overlap_shuf,
        eval_hidden_overlap, device,
    )

    # n_heads=2 variant for block events (bottleneck ablation sweet spot)
    all_results["event_block_nh2"] = run_data_mode(
        "event (4×5 block) n_heads=2", gen_flat_event, gen_seq_event,
        gen_flat_event_shuf, gen_seq_event_shuf,
        eval_hidden_event, device, n_heads=2,
    )

    # ── Summary comparison ──────────────────────────────────────────

    print(f"\n\n{'='*100}", flush=True)
    print("COMPARISON: Independent vs Event (block) vs Event (overlap) vs Event (block, nh=2)",
          flush=True)
    print(f"{'='*100}", flush=True)

    mode_labels = [
        ("independent", "Independent"),
        ("event_block", "Event block"),
        ("event_overlap", "Event overlap"),
        ("event_block_nh2", "Event nh=2"),
    ]

    # Header
    header_parts = [f"{'k':>3}"]
    for _, label in mode_labels:
        header_parts.append(f"{'SAE':>8} {'TFA':>8} {'shuf':>8} {'temp%':>6}")
    print(" | ".join(header_parts), flush=True)
    print("-" * 100, flush=True)

    for i, k in enumerate(K_VALUES):
        parts = [f"{k:>3}"]
        for key, _ in mode_labels:
            res = all_results[key]
            s = res["sae"][i]["nmse"]
            t = res["tfa"][i]["nmse"]
            h = res["tfa_shuffled"][i]["nmse"]
            gap = s - t
            tf = (h - t) / gap * 100 if gap > 1e-10 else float("nan")
            parts.append(f"{s:>8.4f} {t:>8.4f} {h:>8.4f} {tf:>5.1f}%")
        print(" | ".join(parts), flush=True)

    # ── Plots ───────────────────────────────────────────────────────

    plot_configs = [
        ("independent", "Independent", "tab:blue", "-"),
        ("event_block", "Event (block)", "tab:orange", "--"),
        ("event_overlap", "Event (overlap)", "tab:green", "-."),
        ("event_block_nh2", "Event (block, nh=2)", "tab:red", ":"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot 1: NMSE comparison
    ax = axes[0]
    for key, label, color, ls in plot_configs:
        results = all_results[key]
        sae_n = [r["nmse"] for r in results["sae"]]
        tfa_n = [r["nmse"] for r in results["tfa"]]
        shuf_n = [r["nmse"] for r in results["tfa_shuffled"]]
        ax.plot(K_VALUES, sae_n, f"o{ls}", color=color, linewidth=2,
                markersize=7, label=f"SAE ({label})", alpha=0.7)
        ax.plot(K_VALUES, tfa_n, f"s{ls}", color=color, linewidth=2,
                markersize=7, alpha=0.9)
        ax.plot(K_VALUES, shuf_n, f"^{ls}", color=color, linewidth=2,
                markersize=7, alpha=0.5)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("NMSE", fontsize=12)
    ax.set_title("NMSE vs k", fontsize=12)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    # Plot 2: Temporal fraction comparison
    ax = axes[1]
    for key, label, color, ls in plot_configs:
        fracs = compute_temporal_frac(all_results[key])
        ax.plot(K_VALUES, fracs, f"o{ls}", color=color, linewidth=2,
                markersize=8, label=label)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("Temporal fraction (%)", fontsize=12)
    ax.set_title("Temporal fraction by data mode", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 3: Shuf/TFA ratio
    ax = axes[2]
    for key, label, color, ls in plot_configs:
        ratios = compute_shuf_ratio(all_results[key])
        ax.plot(K_VALUES, ratios, f"s{ls}", color=color, linewidth=2,
                markersize=8, label=label)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5,
               label="No temporal benefit")
    ax.set_xlabel("k", fontsize=12)
    ax.set_ylabel("TFA-shuffled / TFA", fontsize=12)
    ax.set_title("Temporal benefit ratio", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(
            os.path.join(RESULTS_DIR, f"event_vs_independent.{ext}"),
            dpi=150, bbox_inches="tight",
        )
    plt.close(fig)
    print("\nPlots saved.", flush=True)

    # ── Save ────────────────────────────────────────────────────────

    save_data = {
        "config": {
            "num_features": NUM_FEATURES,
            "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN,
            "dict_width": DICT_WIDTH,
            "k_values": K_VALUES,
            "seed": SEED,
            "sae_total_steps": SAE_TOTAL_STEPS,
            "tfa_total_steps": TFA_TOTAL_STEPS,
            "indep_pi": INDEP_PI,
            "indep_rho": INDEP_RHO,
            "n_events": N_EVENTS,
            "features_per_event": FEATURES_PER_EVENT,
            "event_pi": EVENT_PI,
            "event_rho": EVENT_RHO,
            "overlap_n_events": OVERLAP_N_EVENTS,
            "overlap_base_per_event": OVERLAP_BASE_PER_EVENT,
            "overlap_n_shared": OVERLAP_N_SHARED,
            "overlap_pi": OVERLAP_PI,
            "overlap_rho": OVERLAP_RHO,
            "overlap_membership": membership.tolist(),
            "overlap_marginal_pi": pi_overlap.tolist(),
            "overlap_expected_l0": expected_l0_overlap,
        },
        "results_independent": all_results["independent"],
        "results_event_block": all_results["event_block"],
        "results_event_overlap": all_results["event_overlap"],
        "results_event_block_nh2": all_results["event_block_nh2"],
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"Results saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
