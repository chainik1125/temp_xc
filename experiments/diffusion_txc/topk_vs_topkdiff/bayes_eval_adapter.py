"""Eval-time adapter exposing BayesGateSAE checkpoints to run_evals.py.

Pinned eval semantics (pre-registered; do not deviate):

- gate conditioned at sigma = 0, i.e. u = (sigma/rms)^2 = 0 — the JumpReLU
  limit of the model (threshold theta0, temperature softplus(tau0_raw)+1e-3,
  Wiener shrink s2/(s2+u) = 1);
- gate hard-thresholded at 0.5;
- z_i = m_i * 1[g_i > 0.5] with m_i the model's shrunk-linear magnitude
  output (= relu(pre_i) exactly, at u = 0).

The gate and magnitude come from the training class's own ``encode()``
(psc/psc_train_sae.py:BayesGateSAE) rather than a reimplementation; its soft
output is z_soft = g * m, so the magnitude is recovered as z_soft / g, which
is numerically safe exactly where the hard mask is nonzero (g > 0.5).
Exposes ``.encode(x) -> (n, H)``, the interface run_evals.py expects.
The TopK eval path is untouched.
"""

from __future__ import annotations

import pathlib

import torch
from torch import nn

from experiments.diffusion_txc.psc.psc_train_sae import BayesGateSAE


class BayesGateEvalSAE(nn.Module):
    def __init__(self, d: int, H: int, u: float = 0.0):
        super().__init__()
        self.sae = BayesGateSAE(d, H)
        self.u = u

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        z_soft, g = self.sae.encode(x, u=self.u)
        return torch.where(g > 0.5, z_soft / g, torch.zeros_like(z_soft))

    @torch.no_grad()
    def gate_l0(self, x: torch.Tensor) -> float:
        """Mean count of g_i > 0.5 — the quantity training logged as
        ``l0_gate_half`` (also measured at u=0). encode()'s support further
        requires relu(pre) > 0, so mean L0 of z can only be <= this."""
        _, g = self.sae.encode(x, u=self.u)
        return float((g > 0.5).float().sum(1).mean())


def load_bayes_sae(ckpt: pathlib.Path | str, dev, d: int | None = None,
                   u: float = 0.0) -> BayesGateEvalSAE:
    sd = torch.load(ckpt, weights_only=True, map_location="cpu")
    d_ck, H_ck = sd["W_enc"].shape
    if d is not None and d_ck != d:
        raise ValueError(f"{ckpt}: checkpoint d={d_ck}, expected {d}")
    m = BayesGateEvalSAE(d_ck, H_ck, u=u)
    # rate_ema / steps_since_fire are train-only buffers that may or may not
    # be in the checkpoint; strict=False tolerates either, but every
    # learnable parameter must be present.
    missing, _unexpected = m.sae.load_state_dict(sd, strict=False)
    missing_params = {n for n, _ in m.sae.named_parameters()} & set(missing)
    if missing_params:
        raise RuntimeError(f"{ckpt}: missing parameters {sorted(missing_params)}")
    m.to(dev).eval()
    return m
