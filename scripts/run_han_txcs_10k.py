"""Han's two locked TXCs (txc_base, txc_pro) on Bill's HMM denoising bench.

Runs each TXC at Han's locked params (configs/locked_archs.yaml on
origin/final) for 10,000 training steps on the Fig 8/9 noisy-emission
HMM setup (n_features=40, hetero rho, p_A=0, p_B=0.625), then reports
the same denoising metrics Bill's run_hmm_denoising_sweep.py prints.

Locked params:
    txc_base: T=5,  k_pos=20, d_sae = 8 * d_in
    txc_pro:  T_max=10, t_sample=5, k_pos=20, d_sae = 8 * d_in,
              shifts=(1,2), inverse-distance contrastive weighting

Bill's d_in=80, so d_sae=640 (16x over-complete vs n_features=40). This
is faithful to Han's recipe but produces a wide dictionary; the
best-match-latent metrics still go through (cos-sim picks the closest
column out of 640).

Writes results/han_txcs_10k/results.json. Run from repo root:
    uv run python scripts/run_han_txcs_10k.py
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
from temporal_bench.metrics import evaluate, evaluate_denoising
from temporal_bench.models.txc_base import TXCBase
from temporal_bench.models.txc_pro import TXCPro
from temporal_bench.utils import get_device, set_seed


# Mirror the HMM sweep script.
RHO_GROUPS = [0.1, 0.4, 0.7, 0.95]
GROUP_SIZE = 10
N_FEATURES = len(RHO_GROUPS) * GROUP_SIZE  # 40
D_MODEL = 80
PI = 0.15
P_A = 0.0
P_B = 0.625

# Han locked params.
EXPANSION = 8
D_SAE = D_MODEL * EXPANSION  # 640
K_POS = 20

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
        rho_per_feature=[r for r in RHO_GROUPS for _ in range(GROUP_SIZE)],
    )
    return DataPipeline(cfg, device=device)


def train_loop(
    model,
    data_fn,
    n_steps: int,
    desc: str,
) -> dict[str, float]:
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


def run_one(
    name: str,
    model,
    pipeline: DataPipeline,
    train_T: int,
    eval_T: int,
) -> dict:
    device = next(model.parameters()).device

    eval_x, eval_s, eval_h = pipeline.eval_data_with_support(
        n_sequences=N_EVAL_SEQUENCES, T=eval_T, rho=0.0, seed=9999
    )

    def data_fn(batch_size: int, _T=train_T) -> torch.Tensor:
        return pipeline.sample_windows(batch_size, _T, rho=0.0)

    t0 = time.time()
    last_train = train_loop(model, data_fn, N_STEPS, desc=name)
    train_secs = time.time() - t0

    final = evaluate(model, eval_x, pipeline.true_features)
    denoise = evaluate_denoising(
        model,
        eval_x,
        eval_s,
        eval_h,
        pipeline.true_features,
        feature_rho=pipeline.config.rho_per_feature,
    )

    return {
        "model": name,
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
        "corr_local": denoise.corr_local,
        "corr_global": denoise.corr_global,
        "r2_local": denoise.r2_local,
        "r2_global": denoise.r2_global,
        "denoising_ratio_corr": denoise.denoising_ratio_corr,
        "denoising_ratio_r2": denoise.denoising_ratio_r2,
        "corr_local_per_feature": denoise.corr_local_per_feature,
        "corr_global_per_feature": denoise.corr_global_per_feature,
        "r2_local_per_feature": denoise.r2_local_per_feature,
        "r2_global_per_feature": denoise.r2_global_per_feature,
        "feature_rho": denoise.feature_rho,
    }


def main() -> None:
    out_dir = "results/han_txcs_10k"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "results.json")

    device = get_device()
    print(f"Device: {device}")
    print(
        f"Data: n_features={N_FEATURES}, d_model={D_MODEL}, pi={PI}, "
        f"p_A={P_A}, p_B={P_B}"
    )
    print(f"Han params: d_sae={D_SAE} (=8*d_in), k_pos={K_POS}, n_steps={N_STEPS}")

    pipeline = build_pipeline(device)

    results = []

    # ── txc_base ────────────────────────────────────────────────────────
    set_seed(SEED)
    base = TXCBase(
        d_in=D_MODEL,
        d_sae=D_SAE,
        T=5,
        k_pos=K_POS,
        auxk_alpha=1.0 / 32.0,
    ).to(device)
    base_result = run_one("txc_base", base, pipeline, train_T=5, eval_T=5)
    results.append(base_result)

    # ── txc_pro ─────────────────────────────────────────────────────────
    set_seed(SEED)
    pro = TXCPro(
        d_in=D_MODEL,
        d_sae=D_SAE,
        T_max=10,
        t_sample=5,
        k_pos=K_POS,
        contrastive_shifts=(1, 2),
        contrastive_inverse_distance_weight=True,
        auxk_alpha=1.0 / 32.0,
        bdec_geom_median_init=True,
    ).to(device)
    # Train at T_input = T_max + max_shift = 12; eval at T_max = 10.
    pro_result = run_one(
        "txc_pro", pro, pipeline,
        train_T=pro.train_window, eval_T=pro.T_max,
    )
    results.append(pro_result)

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Console summary
    print("\n=== Summary ===")
    fmt = "{:<10} {:>6} {:>6} {:>8} {:>8} {:>8} {:>10}"
    print(fmt.format("model", "T_eval", "k_pos", "AUC", "L0", "ratio_corr", "ratio_r2"))
    for r in results:
        print(fmt.format(
            r["model"], r["T_eval"], r["k_pos"],
            f"{r['auc']:.3f}", f"{r['l0']:.1f}",
            f"{r['denoising_ratio_corr']:.3f}",
            f"{r['denoising_ratio_r2']:.3f}",
        ))
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
