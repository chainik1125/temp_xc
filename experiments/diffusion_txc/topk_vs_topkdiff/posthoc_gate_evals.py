"""Post-hoc gate swap: evaluate trained TopK checkpoints with a per-latent
threshold gate (JumpReLU semantics) instead of TopK at inference.

This is the zero-training ablation of the "bayes gate" proposal: the
sigma->0 slice of the MMSE activation is a hard threshold gate passing the
magnitude through (see docs/dmitry/proposals/2026-08-11_jumprelu_mmse_note.md).
Thresholds are rate-calibrated on held-out activations: theta_i is the
(1 - p_i)-quantile of latent i's preactivation, where p_i is its firing
rate under TopK — so the average L0 matches by construction and only the
gate's *shape* changes (per-token variable L0, no winner-take-all).
Latents dead under TopK stay off (theta = +inf): we evaluate the trained
dictionary as-is, not a resurrection of it.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import torch

from experiments.diffusion_txc.topk_vs_topkdiff.run_evals import (
    _load_sae,
    absorption_and_fragility,
    sparse_probing,
)


class GateSAE:
    """Wraps a trained TopKSAE; encode() uses thresholds instead of TopK."""

    def __init__(self, sae, theta: torch.Tensor):
        self.sae, self.theta = sae, theta

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.sae.b_dec) @ self.sae.W_enc + self.sae.b_enc
        return torch.relu(pre) * (pre > self.theta)


def calibrate(sae, X: torch.Tensor, dev, col_chunk: int = 2048) -> tuple[torch.Tensor, dict]:
    """Rate-matched per-latent thresholds from calibration activations."""
    with torch.no_grad():
        pres = []
        for b0 in range(0, X.shape[0], 8192):
            xb = X[b0 : b0 + 8192].to(dev, torch.float32)
            pres.append(((xb - sae.b_dec) @ sae.W_enc + sae.b_enc).cpu())
        P = torch.cat(pres)                                    # (N, H) cpu
        vals, idx = P.topk(sae.k, dim=1)
        fired = torch.zeros_like(P, dtype=torch.bool)
        fired.scatter_(1, idx, vals > 0)
        rate = fired.float().mean(0)                           # (H,)
        H = P.shape[1]
        theta = torch.full((H,), float("inf"))
        alive = rate > 0
        N = P.shape[0]
        for c0 in range(0, H, col_chunk):
            cols = slice(c0, min(c0 + col_chunk, H))
            # per-column quantile at per-column levels: sort + gather
            # (torch.quantile only takes a shared scalar/1D q)
            P_sorted, _ = P[:, cols].float().sort(dim=0)
            idx = ((1 - rate[cols]).clamp(0, 1) * (N - 1)).round().long()
            block = P_sorted.gather(0, idx.unsqueeze(0)).squeeze(0)
            theta[cols] = torch.where(alive[cols], block, theta[cols])
    stats = {"topk_mean_l0": float(fired.float().sum(1).mean()),
             "alive_under_topk": int(alive.sum())}
    return theta.to(dev), stats


def run(device: str = "cuda", vol: str = "/vol", arm: str = "recon",
        seed: int = 0, ckpt_dir: str = "logs_100M",
        calib_rows: int = 50_000) -> dict:
    from transformers import AutoTokenizer

    dev = torch.device(device)
    volp = pathlib.Path(vol)
    meta = json.loads((volp / "cache" / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model"])
    sae = _load_sae(volp, arm, seed, meta["d"], dev, ckpt_dir=ckpt_dir)

    ev = torch.load(volp / "cache" / "eval_shard.pt", weights_only=True)
    X_cal = ev["acts"][-calib_rows:]                           # tail: disjoint
    theta, stats = calibrate(sae, X_cal, dev)                  # from eval rows
    gate = GateSAE(sae, theta)

    with torch.no_grad():
        Xs = ev["acts"][:20000].to(dev, torch.float32)
        z = gate.encode(Xs)
        stats["gate_mean_l0"] = float((z > 0).float().sum(1).mean())
        stats["gate_l0_std"] = float((z > 0).float().sum(1).std())

    out = {"experiment": "posthoc_gate", "arm": arm, "seed": seed,
           "ckpt_dir": ckpt_dir, "calibration": stats}
    out["sparse_probing"] = sparse_probing(gate, volp / "concepts", dev)
    out["absorption"], out["fragility"] = absorption_and_fragility(
        gate, volp, tok, dev, rms=meta["rms"])
    return out
