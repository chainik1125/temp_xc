"""Three-arch AUC sweep with Han's locked TXC recipe.

Re-runs Bill's three-arch synthetic AUC bench (Fig 5/6/7) — the experiment
that shows the big TXC vs Stacked-SAE AUC gap — but swaps in Han's two
locked TXCs (`txc_base`, `txc_pro` from `configs/locked_archs.yaml`) and
expands all four arches to Han's `d_sae = 8 * d_in`.

Bill's three-arch DataConfig (matches `scripts/run_three_arch_sweep.py`):
    n_features = 128
    d_model    = 256
    pi         = 0.05
    rho        in {0.0, 0.6, 0.9}
    deterministic emissions  (p_A=0, p_B=1)

Han recipe applied here:
    d_sae      = 8 * d_in = 2048   (paper expansion, all four arches)
    k_pos      = 20                 (Han's locked sparsity)
    T          = 5                  (locked window for txc_base + baselines)
    T_max      = 10, t_sample = 5   (txc_pro subseq encoder)
    auxk_alpha = 1/32, dead_threshold = 1e7  (txc_base, txc_pro)
    contrastive shifts (1, 2), inverse-distance weighting  (txc_pro)

Training is the existing run_han_txcs_10k convention so the new numbers
are directly comparable: 10k steps, batch 64, lr 3e-4, grad clip 1.

The window-level L0 for the four arches is:
    regular_sae:  k_win = 20 (per-token; total k_win*T = 100 latents/window)
    stacked_sae:  k_win = 20 per position; total k_win*T = 100 latents/window
    txc_base:     k_win = 100 (window-level shared latent)
    txc_pro:      k_inference = 200 (window-level), k_train = 100

Writes results/han_three_arch/results.json. Run from repo root:
    uv run python scripts/run_han_three_arch_sweep.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

sys.path.insert(0, "src")

from temporal_bench.config import DataConfig
from temporal_bench.data.pipeline import DataPipeline
from temporal_bench.metrics import evaluate
from temporal_bench.models.stacked_sae import StackedSAE
from temporal_bench.models.topk_sae import TopKSAE
from temporal_bench.models.txc_base import TXCBase
from temporal_bench.models.txc_pro import TXCPro
from temporal_bench.utils import get_device, set_seed


# ── Bill's three-arch DataConfig (matches scripts/run_three_arch_sweep.py) ──
N_FEATURES = 128
D_MODEL = 256
PI = 0.05
P_A = 0.0
P_B = 1.0
RHO_VALUES = [0.0, 0.6, 0.9]

# ── Han locked params (configs/locked_archs.yaml on origin/final) ──
EXPANSION = 8
D_SAE = D_MODEL * EXPANSION  # 2048
K_POS = 20
T_BASE = 5
T_MAX_PRO = 10
T_SAMPLE_PRO = 5

# ── Training (matches run_han_txcs_10k.py for direct comparability) ──
N_STEPS = 10_000
BATCH_SIZE = 64
LR = 3e-4
GRAD_CLIP = 1.0
SEED = 42
N_EVAL_SEQUENCES = 200


def build_pipeline(device: torch.device) -> DataPipeline:
    cfg = DataConfig(
        n_features=N_FEATURES,
        d_model=D_MODEL,
        pi=PI,
        seed=SEED,
        p_A=P_A,
        p_B=P_B,
    )
    return DataPipeline(cfg, device=device)


def train_loop(model, data_fn, n_steps: int, desc: str) -> dict[str, float]:
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    model.train()
    pbar = tqdm(range(n_steps), desc=desc)
    last_metrics: dict[str, float] = {}
    for step in pbar:
        x = data_fn(BATCH_SIZE)
        out = model(x)
        optimizer.zero_grad()
        out.loss.backward()
        clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        model.normalize_decoder()
        if step % 200 == 0 or step == n_steps - 1:
            last_metrics = dict(out.metrics)
            recon = last_metrics.get("recon_loss", 0.0)
            l0 = last_metrics.get("l0", 0.0)
            dead = last_metrics.get("dead", 0)
            pbar.set_postfix(
                recon=f"{recon:.4f}", l0=f"{l0:.1f}", dead=f"{dead}"
            )
    return last_metrics


def make_model(name: str, device: torch.device):
    """Instantiate one of the four arches at Han's d_sae expansion."""
    if name == "regular_sae":
        return TopKSAE(d_in=D_MODEL, d_sae=D_SAE, k=K_POS).to(device)
    if name == "stacked_sae":
        return StackedSAE(d_in=D_MODEL, d_sae=D_SAE, T=T_BASE, k=K_POS).to(device)
    if name == "txc_base":
        return TXCBase(
            d_in=D_MODEL, d_sae=D_SAE, T=T_BASE, k_pos=K_POS,
            auxk_alpha=1.0 / 32.0,
        ).to(device)
    if name == "txc_pro":
        return TXCPro(
            d_in=D_MODEL, d_sae=D_SAE,
            T_max=T_MAX_PRO, t_sample=T_SAMPLE_PRO, k_pos=K_POS,
            contrastive_shifts=(1, 2),
            contrastive_inverse_distance_weight=True,
            auxk_alpha=1.0 / 32.0,
            bdec_geom_median_init=True,
        ).to(device)
    raise ValueError(f"Unknown arch: {name}")


def windows_for(name: str, model) -> tuple[int, int]:
    """Return (train_T, eval_T) per arch."""
    if name == "txc_pro":
        return model.train_window, model.T_max  # 12, 10
    return T_BASE, T_BASE  # 5, 5


def run_one(name: str, rho: float, pipeline: DataPipeline, device: torch.device) -> dict:
    set_seed(SEED)
    model = make_model(name, device)
    train_T, eval_T = windows_for(name, model)

    eval_x = pipeline.eval_data(
        n_sequences=N_EVAL_SEQUENCES, T=eval_T, rho=rho, seed=9999
    )

    def data_fn(batch_size: int, _T=train_T, _rho=rho) -> torch.Tensor:
        return pipeline.sample_windows(batch_size, _T, _rho)

    desc = f"{name} | rho={rho}"
    t0 = time.time()
    last_train = train_loop(model, data_fn, N_STEPS, desc=desc)
    train_secs = time.time() - t0

    final = evaluate(model, eval_x, pipeline.true_features)

    return {
        "model": name,
        "rho": rho,
        "T_train": train_T,
        "T_eval": eval_T,
        "k_pos": K_POS,
        "d_sae": D_SAE,
        "n_steps": N_STEPS,
        "train_seconds": train_secs,
        "device": str(device),
        "final_train_metrics": last_train,
        "auc": final.auc,
        "nmse": final.nmse,
        "l0": final.l0,
        "r_at_90": final.r_at_90,
        "r_at_80": final.r_at_80,
        "mean_max_cos": final.mean_max_cos,
    }


def main() -> None:
    out_dir = "results/han_three_arch"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")
    tmp_path = out_path + ".tmp"

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Data: n_features={N_FEATURES}, d_model={D_MODEL}, pi={PI}, "
        f"p_A={P_A}, p_B={P_B}"
    )
    print(
        f"Han params: d_sae={D_SAE} (=8*d_in), k_pos={K_POS}, "
        f"n_steps={N_STEPS}, batch={BATCH_SIZE}"
    )
    print(f"Sweep: 4 arches x {len(RHO_VALUES)} rhos = {4 * len(RHO_VALUES)} cells")

    pipeline = build_pipeline(device)
    results: list[dict] = []

    arches = ["regular_sae", "stacked_sae", "txc_base", "txc_pro"]
    for rho in RHO_VALUES:
        for name in arches:
            r = run_one(name, rho, pipeline, device)
            results.append(r)
            print(
                f"  {name:<12} rho={rho:.1f}  AUC={r['auc']:.3f}  "
                f"NMSE={r['nmse']:.4f}  L0={r['l0']:.1f}"
            )
            with open(tmp_path, "w") as f:
                json.dump(results, f, indent=2)
            os.replace(tmp_path, out_path)

    print("\n=== Summary (AUC) ===")
    fmt = "{:<12} " + "  ".join("{:>8}" for _ in RHO_VALUES)
    print(fmt.format("model", *[f"rho={r}" for r in RHO_VALUES]))
    for name in arches:
        cells = [
            next(x for x in results if x["model"] == name and x["rho"] == r)
            for r in RHO_VALUES
        ]
        print(fmt.format(name, *[f"{c['auc']:.3f}" for c in cells]))

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
