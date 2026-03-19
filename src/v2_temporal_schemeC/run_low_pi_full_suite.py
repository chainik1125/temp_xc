"""Full experiment suite at low π (π=0.05, 100 features).

Replicates Experiments 1-4 from the π=0.5 study in a regime where
content-based matching is weak (overlap ~0.25 features vs ~5).

  Experiment 1: TopK sweep (k=1,...,15)
  Experiment 2: ReLU+L1 Pareto frontier
  Experiment 3: Temporal decomposition (run-length + direction analysis)
  Experiment 4: Shuffle diagnostic (included in Exp 1)

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_low_pi_full_suite.py
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
EXPECTED_L0 = NUM_FEATURES * PI_VAL  # 5.0

# Experiment 1: TopK sweep
TOPK_K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15]

# Experiment 2: ReLU+L1 Pareto
L1_COEFFS_SAE = np.logspace(-2, 2, 15).tolist()
L1_COEFFS_TFA = np.logspace(-1, 2.5, 15).tolist()

# Experiment 3: Direction + run-length analysis
ANALYSIS_K_VALUES = [2, 3, 5]
HISTORY_LEN = 5

# Training
SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
ANALYSIS_N_SEQ = 3000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "low_pi_regime"
)


# ── Helpers ──────────────────────────────────────────────────────────


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def make_generators(model, pi_t, rho_t, device, sf, shuffle=False):
    def gen_flat(batch_size):
        n_seq = max(1, batch_size // SEQ_LEN)
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(SEQ_LEN, device=device)]
        return (model(acts) * sf).reshape(-1, HIDDEN_DIM)[:batch_size]

    def gen_seq(n_seq):
        acts, _ = generate_markov_activations(n_seq, SEQ_LEN, pi_t, rho_t, device=device)
        if shuffle:
            for i in range(n_seq):
                acts[i] = acts[i, torch.randperm(SEQ_LEN, device=device)]
        return model(acts) * sf

    return gen_flat, gen_seq


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
    total_se = total_signal = total_novel_l0 = total_total_l0 = 0.0
    total_pred_e = total_novel_e = 0.0
    n_tokens = 0
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s+256, n_seq)].to(device)
            recons, inter = tfa(x)
            B, T, D = x.shape
            xf, rf = x.reshape(-1, D), recons.reshape(-1, D)
            total_se += (xf - rf).pow(2).sum().item()
            total_signal += xf.pow(2).sum().item()
            nc, pc = inter["novel_codes"], inter["pred_codes"]
            total_novel_l0 += (nc > 0).float().sum(dim=-1).sum().item()
            total_total_l0 += ((nc + pc).abs() > 1e-8).float().sum(dim=-1).sum().item()
            total_pred_e += inter["pred_recons"].norm(dim=-1).pow(2).sum().item()
            total_novel_e += inter["novel_recons"].norm(dim=-1).pow(2).sum().item()
            n_tokens += B * T
    te = total_pred_e + total_novel_e + 1e-12
    return {
        "nmse": total_se / total_signal,
        "novel_l0": total_novel_l0 / n_tokens,
        "total_l0": total_total_l0 / n_tokens,
        "pred_energy_frac": total_pred_e / te,
    }


def extract_attention_direction(tfa, x_seq):
    tfa.eval()
    E = tfa.D.T if tfa.tied_weights else tfa.E
    with torch.no_grad():
        x_input = x_seq - tfa.b
        z_input = F.relu(torch.matmul(x_input * tfa.lam, E))
        z_ctx = torch.cat((torch.zeros_like(z_input[:, :1, :]), z_input[:, :-1, :].clone()), dim=1)
        z_pred_, _ = tfa.attn_layers[0](z_ctx, z_input, get_attn_map=False)
        z_pred_ = F.relu(z_pred_)
        Dz_pred = torch.matmul(z_pred_, tfa.D)
    return Dz_pred, x_input


def compute_direction_metrics(Dz_pred, x_input):
    Dz_norm = Dz_pred.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    x_norm = x_input.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    cosine = (Dz_pred * x_input).sum(dim=-1) / (Dz_norm.squeeze(-1) * x_norm.squeeze(-1))
    proj = (Dz_pred * x_input).sum(dim=-1) / Dz_norm.squeeze(-1)
    ve = proj.pow(2) / x_input.pow(2).sum(dim=-1).clamp(min=1e-8)
    return cosine, ve


def classify_by_run_length(support, h=HISTORY_LEN):
    B, T, F = support.shape
    vT = T - h
    current = support[:, h:, :]
    hsum = torch.zeros(B, vT, F, device=support.device)
    for i in range(h):
        hsum += support[:, i:i+vT, :]
    return {
        "long_cont": (hsum == h) & (current == 1),
        "sudden_onset": (hsum == 0) & (current == 1),
        "sudden_offset": (hsum == h) & (current == 0),
        "long_absent": (hsum == 0) & (current == 0),
    }, h


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
    fd = model.feature_directions.to(device)

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}", flush=True)

    gen_flat, gen_seq = make_generators(model, pi_t, rho_t, device, sf)
    gen_flat_shuf, gen_seq_shuf = make_generators(model, pi_t, rho_t, device, sf, shuffle=True)

    # Eval data
    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device)
    eval_hidden = model(acts_eval) * sf

    # Analysis data (larger, with support)
    set_seed(SEED + 200)
    acts_analysis, analysis_support = generate_markov_activations(
        ANALYSIS_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    analysis_hidden = model(acts_analysis) * sf

    # Sanity checks
    with torch.no_grad():
        sup_check = analysis_support
        l0 = sup_check.sum(-1)
        print(f"\nSanity checks (π={PI_VAL}, n={NUM_FEATURES}):", flush=True)
        print(f"  E[L0] = {l0.mean().item():.2f} (theory: {EXPECTED_L0})", flush=True)
        flat = sup_check.reshape(-1, NUM_FEATURES)
        idx = torch.randperm(flat.shape[0])[:10000]
        overlap = (flat[idx[:5000]] * flat[idx[5000:10000]]).sum(dim=-1).float()
        print(f"  Content overlap: {overlap.mean().item():.3f} (theory: {NUM_FEATURES * PI_VAL**2:.2f})",
              flush=True)

    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR, log_every=TFA_TOTAL_STEPS,
    )

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: TopK sweep + shuffle diagnostic
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 1: TopK sweep (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    exp1 = {"sae": [], "tfa": [], "tfa_shuffled": []}
    trained_tfas = {}

    for k in TOPK_K_VALUES:
        print(f"\n  k={k}:", flush=True)

        set_seed(SEED); t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
                                     lr=SAE_LR, l1_coeff=0.0, log_every=SAE_TOTAL_STEPS)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        print(f"    SAE:      NMSE={r['nmse']:.6f} L0={r['l0']:.2f} ({time.time()-t0:.1f}s)", flush=True)
        exp1["sae"].append({"k": k, **r})
        del sae; torch.cuda.empty_cache()

        set_seed(SEED); t0 = time.time()
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
                         n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r = eval_tfa(tfa, eval_hidden, device)
        print(f"    TFA:      NMSE={r['nmse']:.6f} tL0={r['total_l0']:.1f} predE={r['pred_energy_frac']:.3f} ({time.time()-t0:.1f}s)", flush=True)
        exp1["tfa"].append({"k": k, **r})
        if k in ANALYSIS_K_VALUES:
            trained_tfas[(k, "temporal")] = tfa
        else:
            del tfa; torch.cuda.empty_cache()

        set_seed(SEED); t0 = time.time()
        tfa_s = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
                           n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa_s, _ = train_tfa(tfa_s, gen_seq_shuf, tfa_cfg, device)
        r = eval_tfa(tfa_s, eval_hidden, device)
        print(f"    TFA-shuf: NMSE={r['nmse']:.6f} tL0={r['total_l0']:.1f} predE={r['pred_energy_frac']:.3f} ({time.time()-t0:.1f}s)", flush=True)
        exp1["tfa_shuffled"].append({"k": k, **r})
        if k in ANALYSIS_K_VALUES:
            trained_tfas[(k, "shuffled")] = tfa_s
        else:
            del tfa_s; torch.cuda.empty_cache()

        s_n = exp1["sae"][-1]["nmse"]; t_n = exp1["tfa"][-1]["nmse"]; h_n = exp1["tfa_shuffled"][-1]["nmse"]
        gap = s_n - t_n
        tf = (h_n - t_n) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"    >> TFA/SAE={s_n/t_n:.2f}x  shuf/tfa={h_n/t_n:.3f}  temporal={tf:.1f}%", flush=True)

    # Exp 1 summary table
    print(f"\n{'k':>3} | {'SAE':>10} | {'TFA':>10} {'tL0':>6} {'predE':>6} | {'TFA-shuf':>10} | {'TFA/SAE':>8} {'shuf/TFA':>9} {'temp%':>6}", flush=True)
    print("-" * 85, flush=True)
    for i, k in enumerate(TOPK_K_VALUES):
        s = exp1["sae"][i]; t = exp1["tfa"][i]; h = exp1["tfa_shuffled"][i]
        gap = s["nmse"] - t["nmse"]
        tf = (h["nmse"] - t["nmse"]) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"{k:>3} | {s['nmse']:>10.6f} | {t['nmse']:>10.6f} {t['total_l0']:>6.1f} {t['pred_energy_frac']:>6.3f} | "
              f"{h['nmse']:>10.6f} | {s['nmse']/t['nmse']:>8.2f}x {h['nmse']/t['nmse']:>9.3f} {tf:>5.1f}%", flush=True)

    # Exp 1 plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    ax = axes[0]
    ax.plot(TOPK_K_VALUES, [r["nmse"] for r in exp1["sae"]], "o-", color="tab:blue", lw=2, ms=7, label="TopK SAE")
    ax.plot(TOPK_K_VALUES, [r["nmse"] for r in exp1["tfa"]], "s-", color="tab:orange", lw=2, ms=7, label="TFA")
    ax.plot(TOPK_K_VALUES, [r["nmse"] for r in exp1["tfa_shuffled"]], "^-", color="tab:red", lw=2, ms=7, label="TFA-shuffled")
    ax.axvline(x=EXPECTED_L0, color="gray", ls=":", alpha=0.5, label=f"E[L0]={EXPECTED_L0:.0f}")
    ax.set_xlabel("k"); ax.set_ylabel("NMSE"); ax.set_title(f"Exp 1: NMSE vs k (π={PI_VAL})"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    ax = axes[1]
    fracs = []
    for i in range(len(TOPK_K_VALUES)):
        s, t, h = exp1["sae"][i]["nmse"], exp1["tfa"][i]["nmse"], exp1["tfa_shuffled"][i]["nmse"]
        fracs.append((h - t) / (s - t) * 100 if (s - t) > 1e-10 else float("nan"))
    ax.plot(TOPK_K_VALUES, fracs, "o-", color="tab:green", lw=2, ms=8)
    ax.axhline(y=0, color="gray", ls="--", alpha=0.5); ax.axvline(x=EXPECTED_L0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("Temporal fraction (%)"); ax.set_title("Temporal fraction"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(TOPK_K_VALUES, [exp1["tfa"][i]["pred_energy_frac"] for i in range(len(TOPK_K_VALUES))],
            "s-", color="tab:orange", lw=2, ms=7, label="TFA pred energy")
    ax.plot(TOPK_K_VALUES, [exp1["tfa"][i]["total_l0"] for i in range(len(TOPK_K_VALUES))],
            "o--", color="tab:purple", lw=2, ms=7, label="TFA total L0")
    ax.axvline(x=EXPECTED_L0, color="gray", ls=":", alpha=0.5)
    ax.set_xlabel("k"); ax.set_title("TFA decomposition"); ax.legend(); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"exp1_topk_sweep.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Exp 1 plots saved.", flush=True)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: ReLU+L1 Pareto frontier
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 2: ReLU+L1 Pareto (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    exp2_sae = []
    for l1c in L1_COEFFS_SAE:
        set_seed(SEED)
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=None).to(device)
        cfg = ReLUSAETrainingConfig(total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
                                     lr=SAE_LR, l1_coeff=l1c, log_every=SAE_TOTAL_STEPS)
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r = eval_sae(sae, eval_hidden, device)
        exp2_sae.append({"l1_coeff": l1c, **r})
        print(f"  ReLU SAE l1={l1c:.4f}: NMSE={r['nmse']:.6f} L0={r['l0']:.2f}", flush=True)
        del sae; torch.cuda.empty_cache()

    tfa_relu_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR, log_every=TFA_TOTAL_STEPS,
    )
    exp2_tfa = []
    for l1c in L1_COEFFS_TFA:
        set_seed(SEED)
        tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=None,
                         n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device)
        tfa.sae_diff_type = "relu"
        tfa_relu_cfg.l1_coeff = l1c
        tfa, _ = train_tfa(tfa, gen_seq, tfa_relu_cfg, device)
        r = eval_tfa(tfa, eval_hidden, device)
        exp2_tfa.append({"l1_coeff": l1c, **r})
        print(f"  TFA l1={l1c:.4f}: NMSE={r['nmse']:.6f} novelL0={r['novel_l0']:.2f} totalL0={r['total_l0']:.1f}", flush=True)
        del tfa; torch.cuda.empty_cache()

    # Pareto plot
    fig, ax = plt.subplots(figsize=(10, 7))
    sae_l0s = [r["l0"] for r in exp2_sae]
    sae_nmses = [r["nmse"] for r in exp2_sae]
    tfa_novel_l0s = [r["novel_l0"] for r in exp2_tfa]
    tfa_total_l0s = [r["total_l0"] for r in exp2_tfa]
    tfa_nmses = [r["nmse"] for r in exp2_tfa]

    ax.scatter(sae_l0s, sae_nmses, color="tab:blue", alpha=0.3, s=30)
    ax.scatter(tfa_novel_l0s, tfa_nmses, color="tab:orange", alpha=0.3, s=30)
    ax.scatter(tfa_total_l0s, tfa_nmses, color="tab:red", alpha=0.3, s=30)

    # Pareto frontiers
    def pareto_front(xs, ys):
        pts = sorted(zip(xs, ys))
        front_x, front_y = [], []
        best_y = float("inf")
        for x, y in pts:
            if y < best_y:
                front_x.append(x); front_y.append(y); best_y = y
        return front_x, front_y

    fx, fy = pareto_front(sae_l0s, sae_nmses)
    ax.plot(fx, fy, "o-", color="tab:blue", lw=2, ms=8, label="ReLU SAE")
    fx, fy = pareto_front(tfa_novel_l0s, tfa_nmses)
    ax.plot(fx, fy, "s-", color="tab:orange", lw=2, ms=8, label="TFA (novel L0)")
    fx, fy = pareto_front(tfa_total_l0s, tfa_nmses)
    ax.plot(fx, fy, "^--", color="tab:red", lw=2, ms=8, label="TFA (total L0)")

    ax.set_xlabel("L0", fontsize=12); ax.set_ylabel("NMSE", fontsize=12)
    ax.set_title(f"Exp 2: ReLU+L1 Pareto (π={PI_VAL})", fontsize=13)
    ax.legend(fontsize=10); ax.grid(True, alpha=0.3); ax.set_yscale("log")
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"exp2_pareto.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Exp 2 plots saved.", flush=True)

    # ════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: Direction analysis + run-length decomposition
    # ════════════════════════════════════════════════════════════════

    print(f"\n{'='*70}", flush=True)
    print(f"EXPERIMENT 3: Direction + run-length analysis (π={PI_VAL})", flush=True)
    print(f"{'='*70}", flush=True)

    rho_groups = sorted(set(RHO))
    exp3 = {}

    for k in ANALYSIS_K_VALUES:
        print(f"\n  k={k}:", flush=True)
        k_results = {"direction": {}, "run_length": {}}

        for cond_name in ["temporal", "shuffled"]:
            tfa_model = trained_tfas.get((k, cond_name))
            if tfa_model is None:
                print(f"    Skipping {cond_name} (not in trained_tfas)", flush=True)
                continue

            # Direction analysis
            all_cos, all_ve, all_Dz = [], [], []
            with torch.no_grad():
                for s in range(0, ANALYSIS_N_SEQ, 256):
                    e = min(s + 256, ANALYSIS_N_SEQ)
                    x = analysis_hidden[s:e].to(device)
                    Dz, xc = extract_attention_direction(tfa_model, x)
                    cos, ve = compute_direction_metrics(Dz, xc)
                    all_cos.append(cos.cpu()); all_ve.append(ve.cpu()); all_Dz.append(Dz.cpu())

            cosine = torch.cat(all_cos); var_expl = torch.cat(all_ve); Dz_all = torch.cat(all_Dz)
            print(f"    {cond_name}: cos(Dz,x)={cosine.mean().item():.4f} VE={var_expl.mean().item():.4f}", flush=True)

            # Per-category direction quality
            sup_cpu = analysis_support.cpu()
            masks, h = classify_by_run_length(sup_cpu)
            cos_v, ve_v, Dz_v = cosine[:, h:], var_expl[:, h:], Dz_all[:, h:]

            dir_cat = {}
            for cat, mask in masks.items():
                dir_cat[cat] = {}
                for rho_val in rho_groups:
                    fi_list = [i for i in range(NUM_FEATURES) if abs(RHO[i] - rho_val) < 0.01]
                    fc_vals, gc_vals, gv_vals = [], [], []
                    n_ev = 0
                    for i in fi_list:
                        m = mask[:, :, i]
                        if not m.any(): continue
                        n_ev += m.sum().item()
                        gc_vals.append(cos_v[m]); gv_vals.append(ve_v[m])
                        Dz_m = Dz_v[m]
                        Dz_hat = Dz_m / Dz_m.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                        fc_vals.append((Dz_hat * fd[i].cpu().unsqueeze(0)).sum(-1).abs())
                    if fc_vals:
                        dir_cat[cat][rho_val] = {
                            "feat_cos": torch.cat(fc_vals).mean().item(),
                            "global_cos": torch.cat(gc_vals).mean().item(),
                            "var_expl": torch.cat(gv_vals).mean().item(),
                            "n": n_ev,
                        }
                    else:
                        dir_cat[cat][rho_val] = {"feat_cos": 0, "global_cos": 0, "var_expl": 0, "n": 0}

            k_results["direction"][cond_name] = dir_cat

            # Print key results
            for r in [0.5, 0.9]:
                lc = dir_cat.get("long_cont", {}).get(r, {"feat_cos": 0, "n": 0})
                so = dir_cat.get("sudden_onset", {}).get(r, {"feat_cos": 0, "n": 0})
                ratio = lc["feat_cos"] / so["feat_cos"] if so["feat_cos"] > 1e-8 else float("inf")
                print(f"    {cond_name} ρ={r}: |cos(Dz,fi)| long_cont={lc['feat_cos']:.4f} "
                      f"sudden_onset={so['feat_cos']:.4f} ratio={ratio:.2f}", flush=True)

        # Run-length prediction projections (temporal model only)
        tfa_t = trained_tfas.get((k, "temporal"))
        if tfa_t:
            all_pred = torch.zeros(ANALYSIS_N_SEQ, SEQ_LEN, NUM_FEATURES)
            fd_exp = fd.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                for s in range(0, ANALYSIS_N_SEQ, 256):
                    e = min(s + 256, ANALYSIS_N_SEQ)
                    x = analysis_hidden[s:e].to(device)
                    _, inter = tfa_t(x)
                    all_pred[s:e] = (inter["pred_recons"].unsqueeze(2) * fd_exp).sum(-1).cpu()

            sup_cpu = analysis_support.cpu()
            masks, h = classify_by_run_length(sup_cpu)
            pred_v = all_pred[:, h:]

            rl_results = {}
            for cat, mask in masks.items():
                rl_results[cat] = {}
                for rho_val in rho_groups:
                    fi_list = [i for i in range(NUM_FEATURES) if abs(RHO[i] - rho_val) < 0.01]
                    vals = []
                    for i in fi_list:
                        m = mask[:, :, i]
                        if m.any():
                            vals.append(pred_v[:, :, i][m].abs())
                    if vals:
                        pooled = torch.cat(vals)
                        rl_results[cat][rho_val] = {"mean": pooled.mean().item(), "n": pooled.shape[0]}
                    else:
                        rl_results[cat][rho_val] = {"mean": 0, "n": 0}

            k_results["run_length"] = rl_results

            for r in [0.5, 0.9]:
                lc = rl_results.get("long_cont", {}).get(r, {"mean": 0, "n": 0})
                so = rl_results.get("sudden_onset", {}).get(r, {"mean": 0, "n": 0})
                ratio = lc["mean"] / so["mean"] if so["mean"] > 1e-8 else float("inf")
                print(f"    pred_proj ρ={r}: long_cont={lc['mean']:.4f} sudden_onset={so['mean']:.4f} ratio={ratio:.2f}", flush=True)

        exp3[k] = k_results

    # Exp 3 plots: direction quality
    for k in ANALYSIS_K_VALUES:
        if k not in exp3: continue
        kr = exp3[k]
        plot_rhos = [r for r in rho_groups if r >= 0.3]
        pidx = list(range(len(plot_rhos)))

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # Per-feature alignment
        ax = axes[0]
        for cond, color, marker in [("temporal", "tab:green", "o"), ("shuffled", "tab:orange", "s")]:
            dr = kr["direction"].get(cond, {})
            for cat, ls in [("long_cont", "-"), ("sudden_onset", "--")]:
                vals = [dr.get(cat, {}).get(r, {"feat_cos": 0})["feat_cos"] for r in plot_rhos]
                ax.plot(pidx, vals, f"{marker}{ls}", color=color, lw=2, ms=6, label=f"{cond} ({cat})")
        ax.set_xticks(pidx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
        ax.set_xlabel("ρ"); ax.set_ylabel("|cos(Dz, f_i)|")
        ax.set_title(f"Per-feature direction alignment (k={k})"); ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

        # Run-length projections
        ax = axes[1]
        rl = kr.get("run_length", {})
        for cat, color, label in [
            ("long_cont", "tab:green", "Long cont"), ("sudden_onset", "tab:orange", "Sudden onset"),
            ("sudden_offset", "tab:red", "Sudden offset"), ("long_absent", "tab:gray", "Long absent"),
        ]:
            vals = [rl.get(cat, {}).get(r, {"mean": 0})["mean"] for r in plot_rhos]
            ax.plot(pidx, vals, "o-", color=color, lw=2, ms=7, label=label)
        ax.set_xticks(pidx); ax.set_xticklabels([f"{r}" for r in plot_rhos])
        ax.set_xlabel("ρ"); ax.set_ylabel("Mean |⟨pred_recons, f_i⟩|")
        ax.set_title(f"Run-length pred projections (k={k})"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        plt.suptitle(f"Experiment 3: π={PI_VAL}, k={k}", fontsize=14, y=1.02)
        plt.tight_layout()
        for ext in ["png", "pdf"]:
            fig.savefig(os.path.join(RESULTS_DIR, f"exp3_k{k}.{ext}"), dpi=150, bbox_inches="tight")
        plt.close(fig)

    print("Exp 3 plots saved.", flush=True)

    # Cleanup trained models
    for key in list(trained_tfas.keys()):
        del trained_tfas[key]
    torch.cuda.empty_cache()

    # ── Save all results ─────────────────────────────────────────────

    def ser(d):
        if isinstance(d, dict):
            return {str(k): ser(v) for k, v in d.items()}
        return d

    save_data = {
        "config": {
            "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM, "seq_len": SEQ_LEN,
            "pi": PI_VAL, "dict_width": DICT_WIDTH, "expected_l0": EXPECTED_L0,
            "topk_k_values": TOPK_K_VALUES, "analysis_k_values": ANALYSIS_K_VALUES,
            "history_len": HISTORY_LEN, "rho": RHO, "seed": SEED,
        },
        "exp1_topk_sweep": ser(exp1),
        "exp2_pareto_sae": ser(exp2_sae),
        "exp2_pareto_tfa": ser(exp2_tfa),
        "exp3_analysis": ser(exp3),
    }
    with open(os.path.join(RESULTS_DIR, "full_suite_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nAll results saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
