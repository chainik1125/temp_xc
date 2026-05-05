"""ρ × k sweep: regular SAE, plain TXCDR (T=2, T=5), and Han's TXC-pro / H8.

Same DataConfig as Bill's three-arch sweep (n_features=128, d_model=256,
pi=0.05, deterministic emissions). All four arches are run at Han's
`d_sae = 8 × d_in = 2048`, with **per-token k** swept over {1, 2, 5, 10}
and ρ swept over {0.0, 0.6, 0.9}.

Per-token k semantics:
  - regular_sae:   TopK budget per token  = k
  - txcdr T=2:     window-level TopK      = k * 2
  - txcdr T=5:     window-level TopK      = k * 5
  - txc_pro/H8:    inference TopK         = k * 10  (T_max = 10)

Each cell trains 10k steps batch 64 lr 3e-4, matching
run_han_three_arch_sweep.py. The complete sweep is 4 archs × 3 ρ × 4 k
= 48 cells.

Results: results/rho_k_sweep/results.json. Run from repo root:
    uv run python scripts/run_rho_k_sweep.py
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
from temporal_bench.models.temporal_crosscoder import TemporalCrosscoder
from temporal_bench.models.topk_sae import TopKSAE
from temporal_bench.models.txc_pro import TXCPro
from temporal_bench.utils import get_device, set_seed


# ── Bill's three-arch DataConfig ──
N_FEATURES = 128
D_MODEL = 256
PI = 0.05
P_A = 0.0
P_B = 1.0
RHO_VALUES = [0.0, 0.6, 0.9]

# ── Han's d_sae expansion ──
D_SAE = D_MODEL * 8  # 2048

# ── Per-token k sweep ──
K_VALUES = [1, 2, 5, 10]

# ── Training ──
N_STEPS = 10_000
BATCH_SIZE = 64
LR = 3e-4
GRAD_CLIP = 1.0
SEED = 42
N_EVAL_SEQUENCES = 200


def build_pipeline(device: torch.device) -> DataPipeline:
    cfg = DataConfig(
        n_features=N_FEATURES, d_model=D_MODEL, pi=PI, seed=SEED,
        p_A=P_A, p_B=P_B,
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
            pbar.set_postfix(recon=f"{recon:.4f}", l0=f"{l0:.1f}")
    return last_metrics


def make_model(name: str, k_pos: int, device: torch.device):
    """Per-token k semantics; each model converts internally as needed."""
    if name == "regular_sae":
        return TopKSAE(d_in=D_MODEL, d_sae=D_SAE, k=k_pos).to(device), 5
    if name == "txcdr_t2":
        return TemporalCrosscoder(
            d_in=D_MODEL, d_sae=D_SAE, T=2, k_per_pos=k_pos
        ).to(device), 2
    if name == "txcdr_t5":
        return TemporalCrosscoder(
            d_in=D_MODEL, d_sae=D_SAE, T=5, k_per_pos=k_pos
        ).to(device), 5
    if name == "txc_pro":
        return TXCPro(
            d_in=D_MODEL, d_sae=D_SAE,
            T_max=10, t_sample=5, k_pos=k_pos,
            contrastive_shifts=(1, 2),
            contrastive_inverse_distance_weight=True,
            auxk_alpha=1.0 / 32.0,
            bdec_geom_median_init=True,
        ).to(device), 10
    raise ValueError(f"Unknown arch: {name}")


def run_one(name: str, rho: float, k_pos: int, pipeline: DataPipeline,
            device: torch.device) -> dict:
    set_seed(SEED)
    model, model_T = make_model(name, k_pos, device)

    if name == "txc_pro":
        train_T, eval_T = model.train_window, model.T_max
    else:
        train_T, eval_T = model_T, model_T

    eval_x = pipeline.eval_data(
        n_sequences=N_EVAL_SEQUENCES, T=eval_T, rho=rho, seed=9999
    )

    def data_fn(batch_size: int, _T=train_T, _rho=rho) -> torch.Tensor:
        return pipeline.sample_windows(batch_size, _T, _rho)

    desc = f"{name} | rho={rho} k={k_pos}"
    t0 = time.time()
    last = train_loop(model, data_fn, N_STEPS, desc=desc)
    train_secs = time.time() - t0

    final = evaluate(model, eval_x, pipeline.true_features)

    # Window-level total nonzeros per encode call ("raw k").
    raw_k = {
        "regular_sae": k_pos,             # per token
        "txcdr_t2": k_pos * 2,
        "txcdr_t5": k_pos * 5,
        "txc_pro": k_pos * 10,
    }[name]

    return {
        "model": name, "rho": rho, "k_pos": k_pos, "raw_k": raw_k,
        "T_train": train_T, "T_eval": eval_T,
        "d_sae": D_SAE, "n_steps": N_STEPS,
        "train_seconds": train_secs, "device": str(device),
        "final_train_metrics": last,
        "auc": final.auc, "nmse": final.nmse, "l0": final.l0,
        "r_at_90": final.r_at_90, "r_at_80": final.r_at_80,
        "mean_max_cos": final.mean_max_cos,
    }


def main() -> None:
    out_dir = "results/rho_k_sweep"
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
        f"Han d_sae={D_SAE} (=8*d_in); per-token k_pos sweep={K_VALUES}; "
        f"rho sweep={RHO_VALUES}"
    )

    archs = ["regular_sae", "txcdr_t2", "txcdr_t5", "txc_pro"]
    print(f"Sweep: {len(archs)} archs × {len(RHO_VALUES)} rhos × {len(K_VALUES)} ks "
          f"= {len(archs) * len(RHO_VALUES) * len(K_VALUES)} cells")

    pipeline = build_pipeline(device)
    results: list[dict] = []

    for rho in RHO_VALUES:
        for k_pos in K_VALUES:
            for name in archs:
                r = run_one(name, rho, k_pos, pipeline, device)
                results.append(r)
                print(
                    f"  {name:<14} rho={rho:.1f} k={k_pos:>2}  "
                    f"raw_k={r['raw_k']:>3}  AUC={r['auc']:.3f}  "
                    f"NMSE={r['nmse']:.4f}  L0={r['l0']:.1f}"
                )
                with open(tmp_path, "w") as f:
                    json.dump(results, f, indent=2)
                os.replace(tmp_path, out_path)

    # Console summary: AUC table per (arch, rho, k)
    print("\n=== AUC by (arch, rho, k_pos) ===")
    for name in archs:
        print(f"\n{name}:")
        hdr = "%-10s" % "rho"
        for k in K_VALUES:
            hdr += "  k=%-7d" % k
        print(hdr)
        for rho in RHO_VALUES:
            row = "%-10.2f" % rho
            for k in K_VALUES:
                cell = next(
                    (x for x in results
                     if x["model"] == name and x["rho"] == rho
                     and x["k_pos"] == k),
                    None,
                )
                row += "  %-7.3f " % (cell["auc"] if cell else float("nan"))
            print(row)

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
