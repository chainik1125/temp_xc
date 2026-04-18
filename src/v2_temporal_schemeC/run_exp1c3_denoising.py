"""Experiment 1c3 denoising: correlation ratio and linear probe for coupled features.

Complements the gAUC analysis from the bench sweep with two additional metrics:
  (i)  Single-latent correlation ratio (emission vs hidden state matching)
  (ii) Linear probe R² (z → emission support, z → hidden state)

Uses Aniket's coupled data pipeline + our model training code + denoising.py.

Usage:
  TQDM_DISABLE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/home/elysium/temp_xc \
    /home/elysium/miniforge3/envs/torchgpu/bin/python -u \
    src/v2_temporal_schemeC/run_exp1c3_denoising.py
"""

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch

from src.utils.plot import save_figure
from src.utils.seed import set_seed
from src.data.toy.configs import (
    CoupledDataGenerationConfig, CouplingConfig,
    TransitionConfig, SequenceConfig, MagnitudeConfig,
)
from src.data.toy.coupled_dataset import generate_coupled_dataset
from src.pipeline.toy_models import (TFAModelSpec, TXCDRv2ModelSpec, StackedSAEModelSpec)
from src.eval.toy_unified import evaluate_model
from src.eval.denoising import (
    extract_latents_tfa, extract_latents_windowed,
    compute_correlation_against_targets, run_linear_probes_general,
)
from src.eval.toy_unified import _compute_auc
from src.eval.feature_recovery import cos_sims
from src.training.train_tfa import create_tfa, train_tfa, TFATrainingConfig
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.stacked_sae import StackedSAE
from src.training.train_crosscoder import (
    CrosscoderTrainingConfig, train_crosscoder,
)
from src.training.train_stacked_sae import (
    StackedSAETrainingConfig, train_stacked_sae,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Match bench sweep parameters ──
K_HIDDEN = 10
M_EMISSION = 20
N_PARENTS = 2
HIDDEN_DIM = 256
DICT_WIDTH = 40   # d_sae (matches bench --d-sae 40)
SEQ_LEN = 64
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500
PI = 0.05
RHO = 0.7

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]
TXCDR_T_VALUES = [2, 3, 4, 5, 6, 8, 10, 12]

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c3_coupled")
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c3")

MODELS = [
    {"name": "Stacked-T2", "type": "stacked", "T": 2,
     "spec": StackedSAEModelSpec(T=2),
     "steps": 30_000, "batch": 2048, "lr": 3e-4},
    {"name": "Stacked-T5", "type": "stacked", "T": 5,
     "spec": StackedSAEModelSpec(T=5),
     "steps": 30_000, "batch": 2048, "lr": 3e-4},
] + [
    {"name": f"TXCDRv2-T{T}", "type": "txcdrv2", "T": T,
     "spec": TXCDRv2ModelSpec(T=T),
     "steps": 30_000, "batch": 2048, "lr": 3e-4}
    for T in TXCDR_T_VALUES
]


def generate_data():
    cfg = CoupledDataGenerationConfig(
        transition=TransitionConfig.from_reset_process(lam=1 - RHO, p=PI),
        coupling=CouplingConfig(K_hidden=K_HIDDEN, M_emission=M_EMISSION,
                                n_parents=N_PARENTS),
        magnitude=MagnitudeConfig(distribution="folded_normal", mu=1.0, sigma=0.15),
        sequence=SequenceConfig(T=SEQ_LEN, n_sequences=EVAL_N_SEQ + TRAIN_N_SEQ),
        hidden_dim=HIDDEN_DIM,
        seed=SEED,
    )
    result = generate_coupled_dataset(cfg)
    x_all = result["x"]

    # Scale
    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf
    print(f"  Scaling factor: {sf:.4f}", flush=True)

    return {
        "eval_x": x_all[:EVAL_N_SEQ].to(DEVICE),
        "train_x": x_all[EVAL_N_SEQ:],
        "emission_features": result["emission_features"],
        "hidden_features": result["hidden_features"],
        "eval_support": result["support"][:EVAL_N_SEQ],      # (n_eval, M, T)
        "eval_hidden": result["hidden_states"][:EVAL_N_SEQ],  # (n_eval, K, T)
        "coupling_matrix": result["coupling_matrix"],
        "sf": sf,
    }


def make_seq_gen(train_x):
    seqs = train_x.to(DEVICE)
    def gen(n):
        idx = torch.randint(0, seqs.shape[0], (n,), device=DEVICE)
        return seqs[idx]
    return gen


def make_window_gen(train_x, T):
    seqs = train_x.to(DEVICE)
    n_win = SEQ_LEN - T + 1
    def gen(batch_size):
        n_seq = max(1, batch_size // n_win) + 1
        idx = torch.randint(0, seqs.shape[0], (n_seq,), device=DEVICE)
        batch = seqs[idx]
        windows = torch.cat([batch[:, t:t+T, :] for t in range(n_win)], dim=0)
        sel = torch.randperm(windows.shape[0], device=DEVICE)[:batch_size]
        return windows[sel]
    return gen


def _cache_path(name, k):
    return os.path.join(CACHE_DIR, f"{name}_k{k}.pt")


def _try_load(spec, name, k):
    path = _cache_path(name, k)
    if not os.path.exists(path):
        return None
    model = spec.create(d_in=HIDDEN_DIM, d_sae=DICT_WIDTH, k=k, device=DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE, weights_only=True))
    return model


def _save(model, name, k):
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), _cache_path(name, k))


def train_model(mcfg, k, gen_fns):
    name, spec = mcfg["name"], mcfg["spec"]
    cached = _try_load(spec, name, k)
    if cached is not None:
        return cached, spec, True

    set_seed(SEED)
    if mcfg["type"] == "stacked":
        T = mcfg["T"]
        model = StackedSAE(HIDDEN_DIM, DICT_WIDTH, T, k=k).to(DEVICE)
        cfg = StackedSAETrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"])
        model, _ = train_stacked_sae(model, gen_fns[f"window_{T}"], cfg, DEVICE)
    elif mcfg["type"] == "txcdrv2":
        T = mcfg["T"]
        k_eff = k * T
        if k_eff > DICT_WIDTH:
            return None, spec, False
        model = TemporalCrosscoder(HIDDEN_DIM, DICT_WIDTH, T, k=k_eff).to(DEVICE)
        cfg = CrosscoderTrainingConfig(
            total_steps=mcfg["steps"], batch_size=mcfg["batch"],
            lr=mcfg["lr"], log_every=mcfg["steps"])
        model, _ = train_crosscoder(model, gen_fns[f"window_{T}"], cfg, DEVICE)
    else:
        raise ValueError(f"Unknown type: {mcfg['type']}")

    _save(model, name, k)
    return model, spec, False


def get_decoder_directions(spec, model):
    """Get decoder-averaged directions for a model."""
    if spec.n_decoder_positions is None:
        return spec.decoder_directions(model).to(DEVICE)
    dds = [spec.decoder_directions(model, pos=p).to(DEVICE)
           for p in range(spec.n_decoder_positions)]
    return torch.stack(dds).mean(dim=0)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    print(f"Exp 1c3 denoising: K={K_HIDDEN}, M={M_EMISSION}, n_parents={N_PARENTS}", flush=True)
    print(f"  d={HIDDEN_DIM}, d_sae={DICT_WIDTH}, rho={RHO}, pi={PI}", flush=True)
    t_start = time.time()

    print("\nGenerating data...", flush=True)
    data = generate_data()
    eval_x = data["eval_x"]
    train_x = data["train_x"]
    emission_features = data["emission_features"]
    hidden_features = data["hidden_features"]
    eval_support = data["eval_support"]    # (n_eval, M, T)
    eval_hidden = data["eval_hidden"]      # (n_eval, K, T)

    print(f"  eval_x: {eval_x.shape}", flush=True)
    print(f"  emission support: {eval_support.shape} (M={M_EMISSION})", flush=True)
    print(f"  hidden states: {eval_hidden.shape} (K={K_HIDDEN})", flush=True)

    # Build generators
    gen_fns = {"seq": make_seq_gen(train_x)}
    for T in sorted(set(m["T"] for m in MODELS)):
        gen_fns[f"window_{T}"] = make_window_gen(train_x, T)

    all_results = {}
    for mcfg in MODELS:
        all_results[mcfg["name"]] = []

    for k in K_VALUES:
        print(f"\n{'='*60}\nk = {k}\n{'='*60}", flush=True)

        for mcfg in MODELS:
            name, spec = mcfg["name"], mcfg["spec"]

            if mcfg["type"] == "txcdrv2" and k * mcfg["T"] > DICT_WIDTH:
                print(f"  {name:>15}: SKIPPED (k*T={k*mcfg['T']} > d_sae)", flush=True)
                continue

            t0 = time.time()
            model, spec, cached = train_model(mcfg, k, gen_fns)
            if model is None:
                continue
            model.eval()

            # Extract latents
            T_win = mcfg["T"]
            is_xc = mcfg["type"] == "txcdrv2"
            z = extract_latents_windowed(
                model, eval_x, T_win, DICT_WIDTH, SEQ_LEN, is_xc)

            # Get decoder directions
            dd = get_decoder_directions(spec, model)

            # (i) gAUC: decoder cosine similarity
            emission_auc = _compute_auc(spec, model, emission_features, DEVICE)[0]
            hidden_auc = _compute_auc(spec, model, hidden_features, DEVICE)[0]

            # (ii) Single-latent correlation
            emission_corr = compute_correlation_against_targets(
                z, eval_support, dd, emission_features, DEVICE)
            hidden_corr = compute_correlation_against_targets(
                z, eval_hidden, dd, hidden_features, DEVICE)

            # (iii) Linear probe
            emission_probe = run_linear_probes_general(
                z, eval_support, M_EMISSION)
            hidden_probe = run_linear_probes_general(
                z, eval_hidden, K_HIDDEN)

            corr_ratio = (hidden_corr["mean"] / emission_corr["mean"]
                          if emission_corr["mean"] > 0.01 else 0)
            probe_ratio = (hidden_probe["mean_r2"] / emission_probe["mean_r2"]
                           if emission_probe["mean_r2"] > 0.01 else 0)

            result = {
                "k": k,
                "emission_auc": emission_auc, "hidden_auc": hidden_auc,
                "emission_corr": emission_corr["mean"],
                "hidden_corr": hidden_corr["mean"],
                "corr_ratio": corr_ratio,
                "emission_r2": emission_probe["mean_r2"],
                "hidden_r2": hidden_probe["mean_r2"],
                "probe_ratio": probe_ratio,
                "emission_corrs": emission_corr["corrs"],
                "hidden_corrs": hidden_corr["corrs"],
                "emission_r2s": emission_probe["r2"],
                "hidden_r2s": hidden_probe["r2"],
            }
            all_results[name].append(result)

            src = "cached" if cached else f"{time.time()-t0:.0f}s"
            print(f"  {name:>15}: eAUC={emission_auc:.3f} hAUC={hidden_auc:.3f} "
                  f"corr_r={corr_ratio:.2f} probe_r={probe_ratio:.2f} ({src})",
                  flush=True)

            del model
            torch.cuda.empty_cache()

    # Save
    save_path = os.path.join(RESULTS_DIR, "denoising_results.json")
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, 'item') else x)
    print(f"\nResults saved to {save_path}", flush=True)

    elapsed = time.time() - t_start
    print(f"Done in {elapsed/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
