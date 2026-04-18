"""Experiment 3 for TFA-pos: run-length temporal decomposition.

Compares TFA vs TFA-pos on the run-length analysis from Experiment 3b.
Key question: does TFA-pos's attention distinguish long continuations
(11111) from sudden onsets (00001)?

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_exp3_tfa_pos.py
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
from src.plotting.save_figure import save_figure
from src.utils.seed import set_seed
from src.data.toy.toy_model import ToyModel
from src.data.toy.markov import generate_markov_activations
from src.training.train_tfa import create_tfa, train_tfa, TFATrainingConfig

NUM_FEATURES = 20
HIDDEN_DIM = 40
SEQ_LEN = 64
PI = [0.5] * NUM_FEATURES
RHO = [0.0] * 4 + [0.3] * 4 + [0.5] * 4 + [0.7] * 4 + [0.9] * 4
DICT_WIDTH = 40
HISTORY_LEN = 5
K_VALUES = [5, 8]
EVAL_N_SEQ = 3000
SEED = 42

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "exp3_tfa_pos")


def compute_scaling_factor(model, pi_t, rho_t, device):
    with torch.no_grad():
        acts, _ = generate_markov_activations(10000, SEQ_LEN, pi_t, rho_t, device=device)
        hidden = model(acts)
        return math.sqrt(HIDDEN_DIM) / hidden.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()


def classify_by_run_length(support, history_len=HISTORY_LEN):
    B, T, F = support.shape
    h = history_len
    valid_T = T - h
    current = support[:, h:, :]
    history_sum = torch.zeros(B, valid_T, F, device=support.device)
    for i in range(h):
        history_sum += support[:, i:i + valid_T, :]
    prev = support[:, h - 1:T - 1, :]
    all_on = (history_sum == h)
    all_off = (history_sum == 0)
    curr_on = (current == 1)
    curr_off = (current == 0)
    return {
        "long_cont": all_on & curr_on,
        "sudden_onset": all_off & curr_on,
        "sudden_offset": all_on & curr_off,
        "long_absent": all_off & curr_off,
    }, h


def compute_pred_projections(tfa, eval_hidden, feature_dirs, device):
    tfa.eval()
    n_seq, T, d = eval_hidden.shape
    n_features = feature_dirs.shape[0]
    fd = feature_dirs.unsqueeze(0).unsqueeze(0)
    all_pred = torch.zeros(n_seq, T, n_features)
    with torch.no_grad():
        for s in range(0, n_seq, 256):
            x = eval_hidden[s:min(s + 256, n_seq)].to(device)
            _, inter = tfa(x)
            pr = inter["pred_recons"]
            all_pred[s:s + x.shape[0]] = (pr.unsqueeze(2) * fd).sum(dim=-1).cpu()
    return all_pred


def analyze(pred_proj, support, rho_list):
    masks, h = classify_by_run_length(support)
    pred_valid = pred_proj[:, h:, :].abs()
    rho_groups = sorted(set(rho_list))
    results = {}
    for cat, mask in masks.items():
        results[cat] = {}
        for rho in rho_groups:
            feat_idx = [i for i in range(len(rho_list)) if abs(rho_list[i] - rho) < 0.01]
            vals = []
            for i in feat_idx:
                m = mask[:, :, i].cpu()
                if m.any():
                    vals.append(pred_valid[:, :, i][m].mean().item())
            results[cat][rho] = np.mean(vals) if vals else 0.0
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    device = DEFAULT_DEVICE
    print(f"Device: {device}", flush=True)

    pi_t, rho_t = torch.tensor(PI), torch.tensor(RHO)
    set_seed(SEED)
    model = ToyModel(num_features=NUM_FEATURES, hidden_dim=HIDDEN_DIM).to(device)
    model.eval()
    feature_dirs = model.feature_directions.to(device)
    sf = compute_scaling_factor(model, pi_t, rho_t, device)

    def gen_seq(n):
        acts, _ = generate_markov_activations(n, SEQ_LEN, pi_t, rho_t, device=device)
        return model(acts) * sf

    set_seed(SEED + 100)
    eval_acts, eval_support = generate_markov_activations(
        EVAL_N_SEQ, SEQ_LEN, pi_t, rho_t, device=device
    )
    eval_hidden = model(eval_acts) * sf

    tfa_cfg = TFATrainingConfig(
        total_steps=30_000, batch_size=64, lr=1e-3, log_every=30_000,
    )

    all_results = {}

    for model_name, use_pos in [("TFA", False), ("TFA-pos", True)]:
        print(f"\n{'='*60}", flush=True)
        print(f"{model_name}", flush=True)
        print(f"{'='*60}", flush=True)

        all_results[model_name] = {}

        for k in K_VALUES:
            set_seed(SEED)
            t0 = time.time()
            tfa = create_tfa(dimin=HIDDEN_DIM, width=DICT_WIDTH, k=k, n_heads=4,
                             n_attn_layers=1, bottleneck_factor=1,
                             use_pos_encoding=use_pos, device=device)
            tfa, _ = train_tfa(tfa, gen_seq, tfa_cfg, device)

            pred_proj = compute_pred_projections(tfa, eval_hidden, feature_dirs, device)
            results = analyze(pred_proj, eval_support, RHO)

            all_results[model_name][k] = results
            elapsed = time.time() - t0

            # Print key results
            for rho in [0.5, 0.9]:
                lc = results["long_cont"].get(rho, 0)
                so = results["sudden_onset"].get(rho, 0)
                ratio = lc / so if so > 0.01 else float("inf")
                print(f"  k={k} ρ={rho}: long_cont={lc:.3f} sudden_onset={so:.3f} "
                      f"ratio={ratio:.2f} ({elapsed:.0f}s)", flush=True)

            del tfa; torch.cuda.empty_cache()

    # ── Summary ──────────────────────────────────────────────────────

    print(f"\n{'='*60}", flush=True)
    print("Summary: long_cont / sudden_onset ratio at ρ=0.9", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Model':>10} {'k':>3} {'long_cont':>10} {'sudden_onset':>13} {'ratio':>8}", flush=True)
    for model_name in ["TFA", "TFA-pos"]:
        for k in K_VALUES:
            lc = all_results[model_name][k]["long_cont"].get(0.9, 0)
            so = all_results[model_name][k]["sudden_onset"].get(0.9, 0)
            ratio = lc / so if so > 0.01 else float("inf")
            print(f"{model_name:>10} {k:>3} {lc:>10.3f} {so:>13.3f} {ratio:>8.2f}", flush=True)

    # ── Save ──────────────────────────────────────────────────────────

    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Plot ──────────────────────────────────────────────────────────

    rho_groups = sorted(set(RHO))
    categories = ["long_cont", "sudden_onset", "sudden_offset", "long_absent"]
    cat_colors = {"long_cont": "tab:blue", "sudden_onset": "tab:red",
                  "sudden_offset": "tab:orange", "long_absent": "tab:gray"}

    for k in K_VALUES:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        for ax, model_name in zip(axes, ["TFA", "TFA-pos"]):
            res = all_results[model_name][k]
            for cat in categories:
                vals = [res[cat].get(rho, 0) for rho in rho_groups]
                ax.plot(rho_groups, vals, "o-", color=cat_colors[cat], lw=2, ms=8, label=cat)
            ax.set_xlabel("ρ")
            ax.set_ylabel("Mean |pred projection|")
            ax.set_title(f"{model_name} (k={k})")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        plt.suptitle(f"Experiment 3: Run-length prediction projections (k={k})", fontsize=13)
        plt.tight_layout()
        save_figure(fig, os.path.join(RESULTS_DIR, f"run_length_k{k}.png"))
        plt.close(fig)

    print(f"\nResults saved to {RESULTS_DIR}/", flush=True)


if __name__ == "__main__":
    main()
