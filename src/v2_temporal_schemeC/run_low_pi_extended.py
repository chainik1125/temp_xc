"""Extended experiments at low π: wider k sweep + ReLU Pareto.

Supplements run_low_pi_regime.py with:
  - Extended k sweep: k=1,...,15 for full NMSE curve
  - Includes k values above E[L0]=5 where SAE should catch up

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 python -u src/v2_temporal_schemeC/run_low_pi_extended.py
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
from src.v2_temporal_schemeC.relu_sae import (
    ReLUSAE,
    ReLUSAETrainingConfig,
    train_relu_sae,
)

# ── Configuration (same as run_low_pi_regime.py) ─────────────────────

NUM_FEATURES = 100
HIDDEN_DIM = 100
SEQ_LEN = 64
PI_VAL = 0.05
PI = [PI_VAL] * NUM_FEATURES
RHO = [0.0] * 20 + [0.3] * 20 + [0.5] * 20 + [0.7] * 20 + [0.9] * 20
DICT_WIDTH = 100

K_VALUES = [1, 2, 3, 4, 5, 7, 10, 15]

SAE_TOTAL_STEPS = 30_000
SAE_BATCH_SIZE = 4096
SAE_LR = 3e-4

TFA_TOTAL_STEPS = 30_000
TFA_BATCH_SIZE = 64
TFA_LR = 1e-3

EVAL_N_SEQ = 2000
SEED = 42

RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "results", "low_pi_regime"
)


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(
            10000, SEQ_LEN, pi_t, rho_t, device=device
        )
        hidden = model(acts)
        return (
            math.sqrt(HIDDEN_DIM)
            / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
        )


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
            x = flat[s:min(s + bs, n)].to(device)
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
            x = eval_hidden[s:min(s + bs, n_seq)].to(device)
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
            total_novel_e += inter["novel_recons"].norm(dim=-1).pow(2).sum().item()
            n_tokens += B * T
    te = total_pred_e + total_novel_e + 1e-12
    return {
        "nmse": total_se / total_signal,
        "novel_l0": total_novel_l0 / n_tokens,
        "total_l0": total_total_l0 / n_tokens,
        "pred_energy_frac": total_pred_e / te,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t = torch.tensor(PI)
    rho_t = torch.tensor(RHO)

    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()

    sf = compute_scaling_factor(model, pi_t, rho_t, device)
    print(f"Scaling factor: {sf:.4f}", flush=True)

    gen_flat, gen_seq = make_generators(model, pi_t, rho_t, device, sf)
    gen_flat_shuf, gen_seq_shuf = make_generators(model, pi_t, rho_t, device, sf, shuffle=True)

    set_seed(SEED + 100)
    acts_eval, _ = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(acts_eval) * sf
    print(f"Eval: {EVAL_N_SEQ}x{SEQ_LEN} = {EVAL_N_SEQ * SEQ_LEN} tokens", flush=True)

    tfa_cfg = TFATrainingConfig(
        total_steps=TFA_TOTAL_STEPS, batch_size=TFA_BATCH_SIZE,
        lr=TFA_LR, log_every=TFA_TOTAL_STEPS,
    )

    results = {"sae": [], "tfa": [], "tfa_shuffled": []}

    for k in K_VALUES:
        print(f"\n{'='*60}", flush=True)
        print(f"k = {k}", flush=True)
        print(f"{'='*60}", flush=True)

        # SAE
        set_seed(SEED)
        t0 = time.time()
        sae = ReLUSAE(HIDDEN_DIM, DICT_WIDTH, k=k).to(device)
        cfg = ReLUSAETrainingConfig(
            total_steps=SAE_TOTAL_STEPS, batch_size=SAE_BATCH_SIZE,
            lr=SAE_LR, l1_coeff=0.0, log_every=SAE_TOTAL_STEPS,
        )
        sae, _ = train_relu_sae(sae, gen_flat, cfg, device)
        r_sae = eval_sae(sae, eval_hidden, device)
        print(f"  SAE:      NMSE={r_sae['nmse']:.6f}, L0={r_sae['l0']:.2f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        results["sae"].append({"k": k, **r_sae})
        del sae; torch.cuda.empty_cache()

        # TFA
        set_seed(SEED)
        t0 = time.time()
        tfa = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)
        r_tfa = eval_tfa(tfa, eval_hidden, device)
        print(f"  TFA:      NMSE={r_tfa['nmse']:.6f}, novel_L0={r_tfa['novel_l0']:.2f}, "
              f"total_L0={r_tfa['total_l0']:.2f}, pred_E={r_tfa['pred_energy_frac']:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        results["tfa"].append({"k": k, **r_tfa})
        del tfa; torch.cuda.empty_cache()

        # TFA-shuffled
        set_seed(SEED)
        t0 = time.time()
        tfa_shuf = create_tfa(
            dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k,
            n_heads=4, n_attn_layers=1, bottleneck_factor=1, device=device,
        )
        tfa_shuf, _ = train_tfa(tfa_shuf, gen_seq_shuf, tfa_cfg, device)
        r_shuf = eval_tfa(tfa_shuf, eval_hidden, device)
        print(f"  TFA-shuf: NMSE={r_shuf['nmse']:.6f}, novel_L0={r_shuf['novel_l0']:.2f}, "
              f"total_L0={r_shuf['total_l0']:.2f}, pred_E={r_shuf['pred_energy_frac']:.3f} "
              f"({time.time()-t0:.1f}s)", flush=True)
        results["tfa_shuffled"].append({"k": k, **r_shuf})
        del tfa_shuf; torch.cuda.empty_cache()

        # Summary
        sae_n = r_sae["nmse"]; tfa_n = r_tfa["nmse"]; shuf_n = r_shuf["nmse"]
        gap = sae_n - tfa_n
        ratio = shuf_n / tfa_n if tfa_n > 0 else float("inf")
        tfa_vs_sae = sae_n / tfa_n if tfa_n > 0 else float("inf")
        tf = (shuf_n - tfa_n) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"  >> TFA/SAE={tfa_vs_sae:.2f}x | shuf/tfa={ratio:.3f} | "
              f"temporal={tf:.1f}%", flush=True)

    # ── Summary table ────────────────────────────────────────────────

    print(f"\n\n{'='*90}", flush=True)
    print(f"FULL TOPK SWEEP (π={PI_VAL}, {NUM_FEATURES} features, E[L0]={NUM_FEATURES*PI_VAL})",
          flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'k':>3} | {'TopK SAE':>10} | {'TFA':>10} {'TFA tL0':>8} {'pred_E':>7} | "
          f"{'TFA-shuf':>10} {'shuf tL0':>9} | {'TFA/SAE':>8} {'shuf/TFA':>9} {'temp%':>6}",
          flush=True)
    print("-" * 90, flush=True)

    for i, k in enumerate(K_VALUES):
        s = results["sae"][i]
        t = results["tfa"][i]
        h = results["tfa_shuffled"][i]
        gap = s["nmse"] - t["nmse"]
        ratio = h["nmse"] / t["nmse"] if t["nmse"] > 0 else float("inf")
        tfa_vs_sae = s["nmse"] / t["nmse"] if t["nmse"] > 0 else float("inf")
        tf = (h["nmse"] - t["nmse"]) / gap * 100 if gap > 1e-10 else float("nan")
        print(f"{k:>3} | {s['nmse']:>10.6f} | {t['nmse']:>10.6f} {t['total_l0']:>8.1f} "
              f"{t['pred_energy_frac']:>7.3f} | {h['nmse']:>10.6f} {h['total_l0']:>9.1f} | "
              f"{tfa_vs_sae:>8.2f}x {ratio:>9.3f} {tf:>5.1f}%", flush=True)

    # ── Plots ────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    ax = axes[0]
    sae_n = [r["nmse"] for r in results["sae"]]
    tfa_n = [r["nmse"] for r in results["tfa"]]
    shuf_n = [r["nmse"] for r in results["tfa_shuffled"]]
    ax.plot(K_VALUES, sae_n, "o-", color="tab:blue", linewidth=2, markersize=7, label="TopK SAE")
    ax.plot(K_VALUES, tfa_n, "s-", color="tab:orange", linewidth=2, markersize=7, label="TFA")
    ax.plot(K_VALUES, shuf_n, "^-", color="tab:red", linewidth=2, markersize=7, label="TFA-shuffled")
    ax.axvline(x=NUM_FEATURES * PI_VAL, color="gray", linestyle=":", alpha=0.5, label=f"E[L0]={NUM_FEATURES*PI_VAL:.0f}")
    ax.set_xlabel("k"); ax.set_ylabel("NMSE"); ax.set_title(f"NMSE vs k (π={PI_VAL})")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_yscale("log")

    ax = axes[1]
    fracs = []
    for i in range(len(K_VALUES)):
        s = results["sae"][i]["nmse"]; t = results["tfa"][i]["nmse"]
        h = results["tfa_shuffled"][i]["nmse"]
        gap = s - t
        fracs.append((h - t) / gap * 100 if gap > 1e-10 else float("nan"))
    ax.plot(K_VALUES, fracs, "o-", color="tab:green", linewidth=2, markersize=8)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=NUM_FEATURES * PI_VAL, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("Temporal fraction (%)")
    ax.set_title(f"Temporal fraction (π={PI_VAL})"); ax.grid(True, alpha=0.3)

    ax = axes[2]
    tfa_vs = [results["sae"][i]["nmse"] / results["tfa"][i]["nmse"]
              if results["tfa"][i]["nmse"] > 0 else 0 for i in range(len(K_VALUES))]
    ax.plot(K_VALUES, tfa_vs, "s-", color="tab:orange", linewidth=2, markersize=7, label="TFA / SAE")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.axvline(x=NUM_FEATURES * PI_VAL, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("k"); ax.set_ylabel("NMSE ratio (SAE / TFA)")
    ax.set_title(f"TFA advantage over SAE (π={PI_VAL})"); ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(RESULTS_DIR, f"extended_topk_sweep.{ext}"),
                    dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save
    save_data = {
        "config": {
            "num_features": NUM_FEATURES, "hidden_dim": HIDDEN_DIM,
            "seq_len": SEQ_LEN, "pi": PI_VAL, "dict_width": DICT_WIDTH,
            "k_values": K_VALUES, "seed": SEED,
        },
        "results": results,
    }
    with open(os.path.join(RESULTS_DIR, "extended_results.json"), "w") as f:
        json.dump(save_data, f, indent=2)

    print(f"\nResults saved to {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
