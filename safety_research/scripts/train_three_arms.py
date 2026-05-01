"""
Train three architectures on mid_res for the safety / meta-autointerp study:
  arm A: SAE       (FastStackedSAE, T=1, k=100)
  arm B: T-SAE     (FastStackedSAE, T=5, k=100  -- per-position, k per slot)
  arm C: TXC       (FastTemporalCrosscoder, T=5, k=100*5=500 window-level L0 to match)

All three share d_in=2304 (mid_res), d_sae=18432, lr=3e-4. Trained for STEPS
steps each. Checkpoints land under safety_research/results/checkpoints/.
Each run streams to wandb (project: temporal-crosscoders-safety).

Run: uv run python safety_research/scripts/train_three_arms.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

NLP_DIR = "/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP"
SAFETY_DIR = "/home/cs29824/andre/temp_xc/safety_research"
sys.path.insert(0, NLP_DIR)
os.chdir(NLP_DIR)

from config import (  # type: ignore  # noqa: E402
    D_SAE, LEARNING_RATE, ADAM_BETAS, GRAD_CLIP, DEVICE,
    LAYER_SPECS, SEED,
)
from data import CachedActivationSource, WindowIterator  # type: ignore # noqa: E402
from fast_models import FastStackedSAE, FastTemporalCrosscoder  # type: ignore # noqa: E402

import wandb

LAYER = "mid_res"
K = 100
STEPS = 3000     # tight: ~5 min/run on RTX 4090, plenty for fvu to settle
BATCH = 1024
WANDB_PROJECT = "temporal-crosscoders-safety"

CKPT_DIR = Path(SAFETY_DIR) / "results" / "checkpoints"
LOG_DIR = Path(SAFETY_DIR) / "results" / "training_logs"
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


ARMS = [
    dict(arm="sae",   arch="stacked_sae",  T=1, k=K),
    dict(arm="tsae",  arch="stacked_sae",  T=5, k=K),
    dict(arm="txc",   arch="txcdr",        T=5, k=K),
]


def make_model(arch: str, T: int, k: int, d_in: int) -> torch.nn.Module:
    if arch == "stacked_sae":
        return FastStackedSAE(d_in=d_in, d_sae=D_SAE, T=T, k=k)
    if arch == "txcdr":
        return FastTemporalCrosscoder(d_in=d_in, d_sae=D_SAE, T=T, k=k)
    raise ValueError(arch)


def train_arm(source: CachedActivationSource, cfg: dict, d_in: int) -> dict:
    arm = cfg["arm"]; arch = cfg["arch"]; T = cfg["T"]; k = cfg["k"]
    name = f"{arm}__{LAYER}__k{k}__T{T}"

    run = wandb.init(
        project=WANDB_PROJECT,
        name=name,
        tags=["safety", "three-arms", arm, LAYER],
        config=dict(arm=arm, arch=arch, T=T, k=k, layer=LAYER,
                    d_in=d_in, d_sae=D_SAE, steps=STEPS, batch=BATCH,
                    lr=LEARNING_RATE),
        reinit=True,
    )
    print(f"\n=== {name} ===\n  wandb: {run.url}")

    torch.manual_seed(SEED)
    iterator = WindowIterator(source, BATCH, T=T)
    model = make_model(arch, T, k, d_in).to(DEVICE)
    compiled = torch.compile(model, mode="reduce-overhead")
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=ADAM_BETAS)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    history: list[dict] = []
    pbar = tqdm(range(STEPS), desc=name, ncols=100)
    t0 = time.time()
    for step in pbar:
        x = next(iterator)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            loss, x_hat, u = compiled(x)
        optim.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optim)
        scaler.update()

        if step % 100 == 0 or step == STEPS - 1:
            with torch.no_grad():
                fvu = ((x_hat - x).pow(2).sum() / (x - x.mean(0)).pow(2).sum()).item()
                if u.dim() == 3:  # stacked: (B,T,h) → window-level L0 = sum across T
                    window_l0 = (u > 0).float().sum(dim=(-1, -2)).mean().item()
                else:             # txc: (B,h)
                    window_l0 = (u > 0).float().sum(dim=-1).mean().item()
            row = dict(step=step, loss=float(loss.item()), fvu=fvu, window_l0=window_l0)
            history.append(row)
            wandb.log(row, step=step)
            pbar.set_postfix(loss=f"{row['loss']:.3f}", fvu=f"{fvu:.3f}", L0=int(window_l0))

    elapsed = time.time() - t0

    ckpt_path = CKPT_DIR / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(),
                "arch": arch, "T": T, "k": k, "d_in": d_in,
                "d_sae": D_SAE, "layer": LAYER}, ckpt_path)
    log_path = LOG_DIR / f"{name}.json"
    with open(log_path, "w") as f:
        json.dump(dict(history=history, elapsed_s=elapsed,
                       wandb_url=run.url, ckpt=str(ckpt_path)), f, indent=1)
    print(f"  saved: {ckpt_path}\n  log:   {log_path}\n  time:  {elapsed:.0f}s")
    final = history[-1]
    wandb.summary.update({"final_fvu": final["fvu"],
                          "final_loss": final["loss"],
                          "final_window_l0": final["window_l0"],
                          "elapsed_s": elapsed})
    run.finish()
    return dict(name=name, final=final, ckpt=str(ckpt_path),
                wandb_url=run.url, elapsed_s=elapsed)


def main() -> None:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    spec = LAYER_SPECS[LAYER]
    d_in = spec["d_act"]
    print(f"Layer {LAYER}  d_in={d_in}  d_sae={D_SAE}  steps/arm={STEPS}")

    source = CachedActivationSource(LAYER)

    results = []
    for cfg in ARMS:
        results.append(train_arm(source, cfg, d_in))

    summary_path = Path(SAFETY_DIR) / "results" / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSUMMARY → {summary_path}")
    for r in results:
        print(f"  {r['name']:30s} fvu={r['final']['fvu']:.4f}  "
              f"L0={r['final']['window_l0']:.0f}  {r['elapsed_s']:.0f}s  "
              f"{r['wandb_url']}")


if __name__ == "__main__":
    main()
