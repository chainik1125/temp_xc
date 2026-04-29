"""Add paper-faithful T-SAE to Han's Experiment 1c3 (coupled features).

Reproduces the data generation from run_exp1c3_denoising.py and trains
TemporalMatryoshkaBatchTopKSAE (Ye et al. 2025 paper-faithful T-SAE) for
each k in the same sweep, then merges results into the existing
denoising_results.json.

Why standalone: the original sweep took ~68 min on a 5090 to retrain all
10 architectures × 12 k values. We just need to add 1 more architecture
× 12 k values ≈ 10 min.

Usage:
    PYTHONPATH=/root/temp_xc TQDM_DISABLE=1 \
    .venv/bin/python -m experiments.phase3_coupled.run_exp1c3_tsae
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.seed import set_seed
from src.data.toy.configs import (
    CoupledDataGenerationConfig, CouplingConfig,
    TransitionConfig, SequenceConfig, MagnitudeConfig,
)
from src.data.toy.coupled_dataset import generate_coupled_dataset
from src.eval.denoising import (
    compute_correlation_against_targets, run_linear_probes_general,
)
from src.architectures.tsae_paper import (
    TemporalMatryoshkaBatchTopKSAE,
    TemporalMatryoshkaBatchTopKTrainerLite,
    geometric_median,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Match the original run exactly
K_HIDDEN = 10
M_EMISSION = 20
N_PARENTS = 2
HIDDEN_DIM = 256
DICT_WIDTH = 40
SEQ_LEN = 64
SEED = 42
EVAL_N_SEQ = 2000
TRAIN_N_SEQ = 500
PI = 0.05
RHO = 0.7

K_VALUES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20]

BASE = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(BASE, "results", "experiment1c3_coupled")
RESULTS_PATH = os.path.join(RESULTS_DIR, "denoising_results.json")
CACHE_DIR = os.path.join(BASE, "model_cache", "exp1c3_tsae")


def generate_data() -> dict:
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

    sf = math.sqrt(HIDDEN_DIM) / x_all.reshape(-1, HIDDEN_DIM).norm(dim=-1).mean().item()
    x_all = x_all * sf
    print(f"  Scaling factor: {sf:.4f}", flush=True)

    return {
        "eval_x": x_all[:EVAL_N_SEQ].to(DEVICE),
        "train_x": x_all[EVAL_N_SEQ:].to(DEVICE),
        "emission_features": result["emission_features"],
        "hidden_features": result["hidden_features"],
        "eval_support": result["support"][:EVAL_N_SEQ],
        "eval_hidden": result["hidden_states"][:EVAL_N_SEQ],
    }


def make_pair_gen(train_x: torch.Tensor):
    """Returns a callable that yields (B, 2, d) — adjacent token pairs."""
    N, L, d = train_x.shape
    n_wins = L - 1

    def gen(batch_size: int):
        seq = torch.randint(0, N, (batch_size,), device=DEVICE)
        off = torch.randint(0, n_wins, (batch_size,), device=DEVICE)
        a = train_x[seq, off]
        b = train_x[seq, off + 1]
        return torch.stack([a, b], dim=1).float()

    return gen


def train_tsae(k: int, pair_gen) -> TemporalMatryoshkaBatchTopKSAE:
    """Train paper-faithful T-SAE at average-active-per-token = k."""
    cache_path = os.path.join(CACHE_DIR, f"tsae_k{k}.pt")
    # 2 matryoshka groups at 20 / 80 (paper default).
    g1 = max(1, int(0.2 * DICT_WIDTH))
    group_sizes = [g1, DICT_WIDTH - g1]
    group_weights = [0.5, 0.5]

    model = TemporalMatryoshkaBatchTopKSAE(
        activation_dim=HIDDEN_DIM, dict_size=DICT_WIDTH,
        k=k, group_sizes=group_sizes,
    ).to(DEVICE)

    if os.path.exists(cache_path):
        model.load_state_dict(torch.load(cache_path, map_location=DEVICE, weights_only=True))
        return model, True

    set_seed(SEED)
    total_steps = 30_000
    batch_size = 2048
    lr = 2e-4 / math.sqrt(DICT_WIDTH / 16384)  # paper's lr law (gives ~1.4e-3 at d_sae=40)

    trainer = TemporalMatryoshkaBatchTopKTrainerLite(
        model, group_weights=group_weights,
        total_steps=total_steps, lr=lr,
        warmup_steps=1000, decay_start=int(0.8 * total_steps),
        device=DEVICE,
    )
    bdec = geometric_median(pair_gen(batch_size)[:, 0])

    for step in range(total_steps):
        x_pair = pair_gen(batch_size)
        trainer.update(step, x_pair, b_dec_init=bdec if step == 0 else None)
    model.eval()

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(model.state_dict(), cache_path)
    return model, False


@torch.no_grad()
def extract_latents_tsae(model: TemporalMatryoshkaBatchTopKSAE,
                        eval_x: torch.Tensor) -> np.ndarray:
    """Per-token latents under threshold-based inference. Returns (n_tokens, d_sae)."""
    n, L, d = eval_x.shape
    flat = eval_x.reshape(-1, d).float()
    out = []
    bs = 4096
    for i in range(0, flat.shape[0], bs):
        z = model.encode(flat[i:i + bs], use_threshold=True)
        if isinstance(z, tuple):
            z = z[0]
        out.append(z.cpu())
    return torch.cat(out, dim=0).numpy()


def compute_auc(decoder_dirs: torch.Tensor, gt: torch.Tensor) -> float:
    """Decoder cosine-sim AUC against gt directions.

    decoder_dirs: (d_in, d_sae). gt: (M, d_in). Per-row best match → AUC.
    """
    dn = F.normalize(decoder_dirs.T.float(), dim=-1)            # (d_sae, d_in)
    gn = F.normalize(gt.float().to(decoder_dirs.device), dim=-1) # (M, d_in)
    sim = (gn @ dn.T).abs()                                      # (M, d_sae)
    return float(sim.max(dim=1).values.mean().item())


def main() -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Device: {DEVICE}", flush=True)
    t_start = time.time()

    print("Generating data ...", flush=True)
    data = generate_data()
    eval_x = data["eval_x"]
    train_x = data["train_x"]
    emission_features = data["emission_features"]
    hidden_features = data["hidden_features"]
    eval_support = data["eval_support"]
    eval_hidden = data["eval_hidden"]
    print(f"  eval_x: {tuple(eval_x.shape)}, support: {tuple(eval_support.shape)}, "
          f"hidden: {tuple(eval_hidden.shape)}", flush=True)

    pair_gen = make_pair_gen(train_x)

    # Load existing results to merge into.
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH) as f:
            all_results = json.load(f)
    else:
        all_results = {}
    all_results["TSAE"] = []

    for k in K_VALUES:
        print(f"\n=== k = {k} ===", flush=True)
        t0 = time.time()
        model, cached = train_tsae(k, pair_gen)

        # Decoder directions: TSAE's W_dec is (d_sae, d_in); transpose to (d_in, d_sae)
        dd = model.W_dec.data.t().contiguous()                   # (d_in, d_sae)

        # gAUC + eAUC
        emission_auc = compute_auc(dd, emission_features)
        hidden_auc = compute_auc(dd, hidden_features)

        # Per-token latents
        z = extract_latents_tsae(model, eval_x)

        # Single-latent correlation against per-token targets
        emission_corr = compute_correlation_against_targets(
            z, eval_support, dd, emission_features, DEVICE)
        hidden_corr = compute_correlation_against_targets(
            z, eval_hidden, dd, hidden_features, DEVICE)

        # Linear probe R²
        emission_probe = run_linear_probes_general(z, eval_support, M_EMISSION)
        hidden_probe = run_linear_probes_general(z, eval_hidden, K_HIDDEN)

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
        all_results["TSAE"].append(result)

        src = "cached" if cached else f"{time.time() - t0:.0f}s"
        print(f"  TSAE: eAUC={emission_auc:.3f} hAUC={hidden_auc:.3f} "
              f"corr_r={corr_ratio:.2f} probe_r={probe_ratio:.2f} ({src})",
              flush=True)

        del model
        torch.cuda.empty_cache()

    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "item") else x)
    print(f"\nMerged results saved to {RESULTS_PATH}", flush=True)
    print(f"Done in {(time.time() - t_start) / 60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
