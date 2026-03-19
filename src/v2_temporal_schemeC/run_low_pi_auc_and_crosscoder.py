"""AUC + crosscoder for low-π regime (π=0.05, 100 features).

Analogous to run_auc_and_crosscoder_v2.py but for the low-π setup.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_low_pi_auc_and_crosscoder.py
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
    TFATrainingConfig, create_tfa, train_tfa,
)
from src.v2_temporal_schemeC.relu_sae import (
    ReLUSAE, ReLUSAETrainingConfig, train_relu_sae,
)
from src.v2_temporal_schemeC.temporal_crosscoder import (
    TemporalCrosscoder, CrosscoderTrainingConfig, train_crosscoder,
)
from src.v2_temporal_schemeC.feature_recovery import (
    feature_recovery_score, sae_decoder_directions, tfa_decoder_directions,
)

# ── Configuration (low-π) ────────────────────────────────────────────

NUM_FEATURES = 100
HIDDEN_DIM = 100
SEQ_LEN = 64
PI_VAL = 0.05
PI = [PI_VAL] * NUM_FEATURES
RHO = [0.0] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20
DICT_WIDTH = 100

TOPK_K_VALUES = [1, 2, 3, 5, 7, 10, 15]
TXCDR_T = 2

SAE_STEPS = 30_000
TFA_STEPS = 30_000
TXCDR_STEPS = 80_000

SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3
TXCDR_BATCH_SIZE = 2048
TXCDR_LR = 3e-4

L1_COEFFS_SAE = np.logspace(-2, 2, 15).tolist()
L1_COEFFS_TFA = np.logspace(-1, 2.5, 12).tolist()
L1_COEFFS_TXCDR = np.logspace(-1, 2, 12).tolist()

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "low_pi_auc_and_crosscoder"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def make_flat_gen(model, pi_t, rho_t, device, sf):
    def gen(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]
    return gen


def make_seq_gen(model, pi_t, rho_t, device, sf, shuffle=False):
    def gen(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(SEQ_LEN, device=device)]
        return model(acts) * sf
    return gen


def make_window_gen(model, pi_t, rho_t, device, sf, T):
    def gen(batch_size):
        n_seq = max(1, batch_size // (SEQ_LEN - T + 1)) + 1
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts) * sf
        windows = [hidden[:, t:t+T, :] for t in range(SEQ_LEN - T + 1)]
        all_w = torch.cat(windows, dim=0)
        idx = torch.randperm(all_w.shape[0], device=device)[:batch_size]
        return all_w[idx]
    return gen


def eval_sae(sae, eval_hidden, device):
    sae.eval()
    flat = eval_hidden.reshape(-1, HIDDEN_DIM)
    n = flat.shape[0]
    total_se = total_signal = total_l0 = 0.0
    with torch.no_grad():
        for s in range(0, n, 4096):
            x = flat[s:min(s+4096, n)].to(device)
            x_hat, z = sae(x)
            total_se += (x - x_hat).pow(2).sum().item()
            total_signal += x.pow(2).sum().item()
            total_l0 += (z > 0).float().sum(dim=-1).sum().item()
    return {"nmse": total_se / total_signal, "l0": total_l0 / n}


def eval_tfa(tfa, eval_hidden, device):
    tfa.eval()
    n_seq = eval_hidden.shape[0]
    total_se = total_signal = total_novel_l0 = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s+256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            total_novel_l0 += (inter["novel_codes"] > 0).float().sum(dim=-1).sum().item()
            n_tokens += B * T
    return {"nmse": total_se / total_signal, "novel_l0": total_novel_l0 / n_tokens}


def eval_crosscoder(txcdr, eval_hidden, device, T):
    txcdr.eval()
    total_se = total_signal = total_l0 = 0.0
    n_windows = 0
    with torch.no_grad():
        for s in range(0, eval_hidden.shape[0], 256):
            seqs = eval_hidden[s:min(s+256, eval_hidden.shape[0])].to(device)
            for t in range(SEQ_LEN - T + 1):
                w = seqs[:, t:t+T, :]
                loss, x_hat, z = txcdr(w)
                total_se += (x_hat - w).pow(2).sum().item()
                total_signal += w.pow(2).sum().item()
                total_l0 += (z > 0).float().sum(dim=-1).sum().item()
                n_windows += w.shape[0]
    return {"nmse": total_se / total_signal, "l0": total_l0 / n_windows}


def compute_auc(decoder_dirs, true_features, device):
    dd = decoder_dirs.to(device)
    tf = true_features.T.to(device)
    return feature_recovery_score(dd, tf)


def txcdr_avg_auc(txcdr, true_features, device, T):
    aucs = []
    tf = true_features.T.to(device)
    for pos in range(T):
        dd = txcdr.decoder_directions(pos).to(device)
        a = feature_recovery_score(dd, tf)
        aucs.append(a["auc"])
    return np.mean(aucs)


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
    true_features = model.feature_directions

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}", flush=True)

    gen_flat = make_flat_gen(model, pi_t, rho_t, device, sf)
    gen_seq = make_seq_gen(model, pi_t, rho_t, device, sf)
    gen_seq_shuf = make_seq_gen(model, pi_t, rho_t, device, sf, shuffle=True)
    gen_windows = make_window_gen(model, pi_t, rho_t, device, sf, TXCDR_T)

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    tfa_cfg = TFATrainingConfig(total_steps=TFA_STEPS, batch_size=TFA_BATCH_SIZE,
                                 lr=TFA_LR, log_every=TFA_STEPS)
    txcdr_cfg = CrosscoderTrainingConfig(total_steps=TXCDR_STEPS, batch_size=TXCDR_BATCH_SIZE,
                                          lr=TXCDR_LR, log_every=TXCDR_STEPS)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"EXP 1: TopK sweep (π={PI_VAL}, TXCDR {TXCDR_STEPS//1000}K steps)", flush=True)
    print(f"{'='*70}", flush=True)

    exp1 = {"sae": [], "tfa": [], "tfa_shuf": [], "txcdr": []}

    for k in TOPK_K_VALUES:
        print(f"\n  k={k}:", flush=True)

        set_seed(SEED); t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_STEPS, batch_size=SAE_BATCH_SIZE,
                                     lr=SAE_LR, l1_coeff=0.0, log_every=SAE_STEPS)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        auc = compute_auc(sae_decoder_directions(sae), true_features, device)
        r.update({"auc": auc["auc"], "r90": auc["frac_recovered_90"]})
        print(f"    SAE:      NMSE={r['nmse']:.6f} AUC={r['auc']:.4f} R90={r['r90']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        exp1["sae"].append({"k": k, **r})
        del sae; torch.cuda.empty_cache()

        set_seed(SEED); t0 = time.time()
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                         n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r = eval_tfa(tfa, eval_hidden, device)
        auc = compute_auc(tfa_decoder_directions(tfa), true_features, device)
        r.update({"auc": auc["auc"], "r90": auc["frac_recovered_90"]})
        print(f"    TFA:      NMSE={r['nmse']:.6f} AUC={r['auc']:.4f} R90={r['r90']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        exp1["tfa"].append({"k": k, **r})
        del tfa; torch.cuda.empty_cache()

        set_seed(SEED); t0 = time.time()
        tfa_s = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                           n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa_s, _ = train_tfa(tfa_s, gen_seq_shuf, tfa_cfg, device)
        r = eval_tfa(tfa_s, eval_hidden, device)
        auc = compute_auc(tfa_decoder_directions(tfa_s), true_features, device)
        r.update({"auc": auc["auc"], "r90": auc["frac_recovered_90"]})
        print(f"    TFA-shuf: NMSE={r['nmse']:.6f} AUC={r['auc']:.4f} R90={r['r90']:.2f} ({time.time()-t0:.0f}s)", flush=True)
        exp1["tfa_shuf"].append({"k": k, **r})
        del tfa_s; torch.cuda.empty_cache()

        set_seed(SEED); t0 = time.time()
        txcdr = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, TXCDR_T, k).to(device)
        txcdr, _ = train_crosscoder(txcdr, gen_windows, txcdr_cfg, device)
        r = eval_crosscoder(txcdr, eval_hidden, device, TXCDR_T)
        r["auc"] = txcdr_avg_auc(txcdr, true_features, device, TXCDR_T)
        r["r90"] = 0.0
        print(f"    TXCDR:    NMSE={r['nmse']:.6f} AUC={r['auc']:.4f} ({time.time()-t0:.0f}s)", flush=True)
        exp1["txcdr"].append({"k": k, **r})
        del txcdr; torch.cuda.empty_cache()

        s, t, h, x = exp1["sae"][-1], exp1["tfa"][-1], exp1["tfa_shuf"][-1], exp1["txcdr"][-1]
        gap = s["nmse"] - t["nmse"]
        tf = (h["nmse"] - t["nmse"]) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"    >> TFA/SAE={s['nmse']/t['nmse']:.2f}x  temp={tf:.1f}%", flush=True)

    # Summary
    print(f"\n{'k':>3} | {'SAE NMSE':>9} {'AUC':>5} | {'TFA NMSE':>9} {'AUC':>5} | "
          f"{'shuf NMSE':>9} {'AUC':>5} | {'XC NMSE':>9} {'AUC':>5}", flush=True)
    print("-" * 85, flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        s, t, h, x = exp1["sae"][i], exp1["tfa"][i], exp1["tfa_shuf"][i], exp1["txcdr"][i]
        print(f"{k:>3} | {s['nmse']:>9.4f} {s['auc']:>.3f} | {t['nmse']:>9.4f} {t['auc']:>.3f} | "
              f"{h['nmse']:>9.4f} {h['auc']:>.3f} | {x['nmse']:>9.4f} {x['auc']:>.3f}", flush=True)

    # Exp 1 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    pd = [("SAE", exp1["sae"], "tab:blue", "o", "-"), ("TFA", exp1["tfa"], "tab:orange", "s", "-"),
          ("TFA-shuf", exp1["tfa_shuf"], "tab:red", "^", "--"), (f"TXCDR T={TXCDR_T}", exp1["txcdr"], "tab:purple", "D", "-.")]

    ax = axes[0]
    for label, data, c, m, ls in pd:
        ax.plot(TOPK_K_VALUES, [r["nmse"] for r in data], f"{m}{ls}", color=c, lw=2, ms=7, label=label)
    ax.set_xlabel("k"); ax.set_ylabel("NMSE"); ax.set_title(f"NMSE vs k (π={PI_VAL})"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    ax = axes[1]
    for label, data, c, m, ls in pd:
        ax.plot(TOPK_K_VALUES, [r["auc"] for r in data], f"{m}{ls}", color=c, lw=2, ms=7, label=label)
    ax.set_xlabel("k"); ax.set_ylabel("AUC"); ax.set_title(f"AUC vs k (π={PI_VAL})"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    for label, data, c, m, _ in pd:
        ax.scatter([r["nmse"] for r in data], [r["auc"] for r in data], color=c, marker=m, s=60, label=label, zorder=3)
        ax.plot([r["nmse"] for r in data], [r["auc"] for r in data], color=c, alpha=0.3, lw=1)
    ax.set_xlabel("NMSE"); ax.set_ylabel("AUC"); ax.set_title("NMSE vs AUC"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xscale("log")

    plt.suptitle(f"Exp 1: TopK sweep π={PI_VAL} (TXCDR {TXCDR_STEPS//1000}K steps)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"exp1_topk_auc.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Exp 1 plots saved.", flush=True)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"EXP 2: ReLU+L1 Pareto (π={PI_VAL}, including TXCDR)", flush=True)
    print(f"{'='*70}", flush=True)

    exp2_sae = []
    for l1c in L1_COEFFS_SAE:
        set_seed(SEED)
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=None).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_STEPS, batch_size=SAE_BATCH_SIZE,
                                     lr=SAE_LR, l1_coeff=l1c, log_every=SAE_STEPS)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        auc = compute_auc(sae_decoder_directions(sae), true_features, device)
        r.update({"auc": auc["auc"], "l1": l1c})
        exp2_sae.append(r)
        print(f"  SAE l1={l1c:.4f}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f} AUC={r['auc']:.4f}", flush=True)
        del sae; torch.cuda.empty_cache()

    tfa_relu_cfg = TFATrainingConfig(total_steps=TFA_STEPS, batch_size=TFA_BATCH_SIZE,
                                      lr=TFA_LR, log_every=TFA_STEPS)
    exp2_tfa = []
    for l1c in L1_COEFFS_TFA:
        set_seed(SEED)
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=None, n_heads=4,
                         n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa.sae_diff_type = "relu"
        tfa_relu_cfg.l1_coeff = l1c
        tfa, _ = train_tfa(tfa, gen_seq, tfa_relu_cfg, device)
        r = eval_tfa(tfa, eval_hidden, device)
        auc = compute_auc(tfa_decoder_directions(tfa), true_features, device)
        r.update({"auc": auc["auc"], "l1": l1c})
        exp2_tfa.append(r)
        print(f"  TFA l1={l1c:.4f}: NMSE={r['nmse']:.6f} nL0={r['novel_l0']:.2f} AUC={r['auc']:.4f}", flush=True)
        del tfa; torch.cuda.empty_cache()

    exp2_txcdr = []
    for l1c in L1_COEFFS_TXCDR:
        set_seed(SEED)
        txcdr = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, TXCDR_T, k=None).to(device)
        l1_cfg = CrosscoderTrainingConfig(total_steps=TXCDR_STEPS, batch_size=TXCDR_BATCH_SIZE,
                                           lr=TXCDR_LR, l1_coeff=l1c, log_every=TXCDR_STEPS)
        txcdr, _ = train_crosscoder(txcdr, gen_windows, l1_cfg, device)
        r = eval_crosscoder(txcdr, eval_hidden, device, TXCDR_T)
        r["auc"] = txcdr_avg_auc(txcdr, true_features, device, TXCDR_T)
        r["l1"] = l1c
        exp2_txcdr.append(r)
        print(f"  TXCDR l1={l1c:.4f}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f} AUC={r['auc']:.4f}", flush=True)
        del txcdr; torch.cuda.empty_cache()

    # Exp 2 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    def pareto_min(xs, ys):
        pts = sorted(zip(xs, ys)); fx, fy = [], []; best = float("inf")
        for x, y in pts:
            if y < best: fx.append(x); fy.append(y); best = y
        return fx, fy

    def pareto_max(xs, ys):
        pts = sorted(zip(xs, ys)); fx, fy = [], []; best = -float("inf")
        for x, y in pts:
            if y > best: fx.append(x); fy.append(y); best = y
        return fx, fy

    sae_l0 = [r["l0"] for r in exp2_sae]; sae_nmse = [r["nmse"] for r in exp2_sae]; sae_auc = [r["auc"] for r in exp2_sae]
    tfa_l0 = [r["novel_l0"] for r in exp2_tfa]; tfa_nmse = [r["nmse"] for r in exp2_tfa]; tfa_auc = [r["auc"] for r in exp2_tfa]
    xc_l0 = [r["l0"] for r in exp2_txcdr]; xc_nmse = [r["nmse"] for r in exp2_txcdr]; xc_auc = [r["auc"] for r in exp2_txcdr]

    ax = axes[0]
    ax.scatter(sae_l0, sae_nmse, color="tab:blue", alpha=0.3, s=30)
    ax.scatter(tfa_l0, tfa_nmse, color="tab:orange", alpha=0.3, s=30)
    ax.scatter(xc_l0, xc_nmse, color="tab:purple", alpha=0.3, s=30)
    fx, fy = pareto_min(sae_l0, sae_nmse); ax.plot(fx, fy, "o-", color="tab:blue", lw=2, ms=7, label="ReLU SAE")
    fx, fy = pareto_min(tfa_l0, tfa_nmse); ax.plot(fx, fy, "s-", color="tab:orange", lw=2, ms=7, label="TFA (novel L0)")
    fx, fy = pareto_min(xc_l0, xc_nmse); ax.plot(fx, fy, "D-", color="tab:purple", lw=2, ms=7, label=f"TXCDR T={TXCDR_T}")
    ax.set_xlabel("L0"); ax.set_ylabel("NMSE"); ax.set_title("L0 vs NMSE"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    ax = axes[1]
    ax.scatter(sae_l0, sae_auc, color="tab:blue", alpha=0.3, s=30)
    ax.scatter(tfa_l0, tfa_auc, color="tab:orange", alpha=0.3, s=30)
    ax.scatter(xc_l0, xc_auc, color="tab:purple", alpha=0.3, s=30)
    fx, fy = pareto_max(sae_l0, sae_auc); ax.plot(fx, fy, "o-", color="tab:blue", lw=2, ms=7, label="ReLU SAE")
    fx, fy = pareto_max(tfa_l0, tfa_auc); ax.plot(fx, fy, "s-", color="tab:orange", lw=2, ms=7, label="TFA (novel L0)")
    fx, fy = pareto_max(xc_l0, xc_auc); ax.plot(fx, fy, "D-", color="tab:purple", lw=2, ms=7, label=f"TXCDR T={TXCDR_T}")
    ax.set_xlabel("L0"); ax.set_ylabel("AUC"); ax.set_title("L0 vs AUC"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.scatter(sae_nmse, sae_auc, color="tab:blue", alpha=0.4, s=30, label="ReLU SAE")
    ax.scatter(tfa_nmse, tfa_auc, color="tab:orange", alpha=0.4, s=30, label="TFA")
    ax.scatter(xc_nmse, xc_auc, color="tab:purple", alpha=0.4, s=30, label=f"TXCDR T={TXCDR_T}")
    ax.set_xlabel("NMSE"); ax.set_ylabel("AUC"); ax.set_title("NMSE vs AUC"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_xscale("log")

    plt.suptitle(f"Exp 2: ReLU+L1 Pareto π={PI_VAL} (TXCDR {TXCDR_STEPS//1000}K steps)", fontsize=14, y=1.02)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"exp2_pareto_auc.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Exp 2 plots saved.", flush=True)

    # ── Save ─────────────────────────────────────────────────────────

    def ser(d):
        if isinstance(d, dict):
            return {str(k): ser(v) for k, v in d.items()}
        if isinstance(d, np.ndarray):
            return d.tolist()
        return d

    save_data = {
        "config": {"pi": PI_VAL, "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM,
                    "txcdr_T": TXCDR_T, "txcdr_steps": TXCDR_STEPS, "seed": SEED},
        "exp1": ser(exp1),
        "exp2_sae": ser(exp2_sae), "exp2_tfa": ser(exp2_tfa), "exp2_txcdr": ser(exp2_txcdr),
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
