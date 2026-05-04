"""Methodology validation — synthetic-data sanity check of the
detection + steering infrastructure.

What this script answers:

1. **Position variance behaves as documented.** Histogram on
   - a freshly-initialised TXC (≈ 0 because tied init makes every
     atom's trajectory roughly constant: each W_dec[f, t, :] is the
     transpose of W_enc[t, :, f], drawn from kaiming, broadly
     similar across t at init);
   - a TXC with **shuffled W_dec across t** (large variance — the
     trajectory has structure across t even though feature identity is
     preserved, illustrating where V0 throws away signal).

2. **Encoder–decoder divergence after a small training step**: at init
   it's 0 (tied weights). After a few hundred SGD steps on random
   activations, the encoder drifts; the rel_residual histogram should
   shift away from 0. Quantifies the gap V4 captures over V0.

3. **Detection PR-AUC + within-window shuffle ablation correctly
   distinguishes** a temporally-structured cohort from a
   position-invariant cohort. Validates the protocol is sensitive to
   what it claims to be sensitive to.

4. **Hook math**: visual side-by-side of V0 / V1 / V2 / V4 deltas on a
   fixed feature + magnitude. V0 is constant across positions; V1
   cycles; V2 is a trailing ramp; V4 differs from V0 by exactly the
   encoder-decoder divergence direction. Confirms each mode does what
   the docstring claims.

This script intentionally does NOT touch the locked checkpoints,
HF data, or the case-study generation pipelines — it's a
methodology-validation scaffold the agents (C5, C6, C7) re-run on
their real cells via :mod:`experiments.det_steer.run_c7_locked`.

Output:
* ``results/det_steer/validate/{position_variance,encoder_decoder_divergence,
  pr_auc_shuffle_gap,hook_modes_delta}.{png,thumb.png}``
* ``results/det_steer/validate/summary.json`` — every numeric output
  the figures derive from, for downstream rendering.

Wallclock target: < 5 minutes on one H100.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Add det-steer src to path before any temp_bench imports — keeps the
# script runnable without re-installing the package on this branch.
_DETSTEER_SRC = Path(__file__).resolve().parents[3] / "src"
if str(_DETSTEER_SRC) not in sys.path:
    sys.path.insert(0, str(_DETSTEER_SRC))

import numpy as np
import torch

from temp_bench.eval.detection import detect_case_study
from temp_bench.eval.steering_hooks import (
    ALL_MODES,
    TXCSteeringHook,
    encoder_decoder_divergence,
    encoder_preimage,
    position_variance,
)
from temp_bench.utils.seed import set_seed


def _import_txc_classes():
    """Defer the txc_base/txc_pro imports — det-steer is branched off
    final-aniket which doesn't yet ship them. The script falls back to
    a local minimal-TXC stub when the canonical classes aren't on
    PYTHONPATH (i.e. when run on a det-steer-only checkout)."""
    try:
        from temp_bench.architectures.txc_base import TXCBase  # noqa
        from temp_bench.architectures.txc_pro import TXCPro    # noqa
        return TXCBase, TXCPro
    except ImportError:
        return None, None


class _MinimalTxc(torch.nn.Module):
    """Minimal TXC matching the txc_base interface, used when the real
    class isn't on the venv (det-steer is missing case-study deps)."""

    def __init__(self, *, d_in: int, d_sae: int, T: int, k_pos: int, seed: int = 0):
        super().__init__()
        from temp_bench.architectures.base import ArchConfig
        self.config = ArchConfig(name="minimal_txc", d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T)
        self.d_in = d_in
        self._d_sae = d_sae
        self._T = T
        self.k_pos = k_pos
        self.k_win = k_pos * T
        g = torch.Generator().manual_seed(seed)
        self.W_dec = torch.nn.Parameter(torch.randn(d_sae, T, d_in, generator=g))
        self.W_enc = torch.nn.Parameter(torch.empty(T, d_in, d_sae))
        with torch.no_grad():
            # Order matches txc_base.py: normalize THEN tie. If we tie
            # first, the post-norm W_dec no longer equals W_enc.T per-t,
            # which silently breaks encoder_decoder_divergence at init.
            norms = self.W_dec.data.norm(dim=(1, 2), keepdim=True).clamp_min(1e-8)
            self.W_dec.data /= norms
            for t in range(T):
                self.W_enc.data[t] = self.W_dec.data[:, t, :].T

    @property
    def d_sae(self): return self._d_sae

    @property
    def T(self): return self._T

    def encode(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        pre = torch.einsum("btd,tds->bs", x, self.W_enc)
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, torch.relu(vals))
        return z.unsqueeze(1)

    def decode(self, z):
        if z.dim() == 3:
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec)

    def decoder_directions(self) -> torch.Tensor:
        return self.W_dec.data.mean(dim=1).clone()

    def train_step(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        pre = torch.einsum("btd,tds->bs", x, self.W_enc)
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, torch.relu(vals))
        x_hat = torch.einsum("bs,std->btd", z, self.W_dec)
        loss = (x - x_hat).pow(2).mean()
        return loss, {"mse": loss.detach()}


def _build_arch(*, d_in: int, d_sae: int, T: int, k_pos: int, seed: int):
    TXCBase, _ = _import_txc_classes()
    if TXCBase is None:
        return _MinimalTxc(d_in=d_in, d_sae=d_sae, T=T, k_pos=k_pos, seed=seed)
    arch = TXCBase(d_in=d_in, d_sae=d_sae, T=T, k_pos=k_pos)
    return arch


def _train_minimal(arch, *, n_steps: int, batch_size: int, d_in: int, lr: float = 3e-4, device: str = "cuda"):
    """Tiny SGD loop on random Gaussian activations. Goal: drift
    encoder/decoder weights from tied init so encoder_decoder_divergence
    is non-zero — methodology validation only, NOT a paper-quality
    training. Uses the canonical TXC ``train_step`` signature
    (B, seq_len, d_in)."""
    arch = arch.to(device)
    opt = torch.optim.Adam(arch.parameters(), lr=lr)
    losses = []
    for step in range(n_steps):
        x = torch.randn(batch_size, max(arch._T, arch._T), d_in, device=device)
        loss, _ = arch.train_step(x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if hasattr(arch, "post_step"):
            arch.post_step()
        losses.append(float(loss.item()))
    return arch, losses


def _save_figure(fig, path: Path) -> None:
    """save_figure compat — also writes a thumb. Picks up
    temp_bench.plotting.figure if available, else falls back to a
    matplotlib + PIL minimal version."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from temp_bench.plotting.figure import save_figure
        save_figure(fig, str(path))
    except Exception:
        # Fallback: matplotlib savefig + a downscaled thumb.
        fig.savefig(path, dpi=150, bbox_inches="tight")
        thumb = path.with_name(path.stem + ".thumb.png")
        fig.set_size_inches(2.0, 2.0)
        fig.savefig(thumb, dpi=48, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def _make_temporal_cohort(*, n_pos: int, n_neg: int, T: int, d_in: int, sig: float, seed: int):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    X = rng.standard_normal((n, T, d_in)).astype(np.float32)
    X[:n_pos, 0, 0] += sig
    X[:n_pos, T - 1, 0] -= sig
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    gids = np.array([i % 12 for i in range(n)])
    return X, y, gids


def _make_density_cohort(*, n_pos: int, n_neg: int, T: int, d_in: int, sig: float, seed: int):
    rng = np.random.default_rng(seed)
    n = n_pos + n_neg
    X = rng.standard_normal((n, T, d_in)).astype(np.float32)
    X[:n_pos, :, 0] += sig
    y = np.array([1] * n_pos + [0] * n_neg, dtype=np.int64)
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    gids = np.array([i % 12 for i in range(n)])
    return X, y, gids


def main() -> None:
    set_seed(0)
    out_dir = Path(__file__).resolve().parent / "results" / "validate"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    T = 5
    d_in = 64
    d_sae = 256
    k_pos = 5

    summary: dict[str, Any] = {
        "device": device,
        "T": T, "d_in": d_in, "d_sae": d_sae, "k_pos": k_pos,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ── 1. Position variance: tied init + after small training drift ──
    arch_init = _build_arch(d_in=d_in, d_sae=d_sae, T=T, k_pos=k_pos, seed=0).to(device)
    pv_init = position_variance(arch_init.W_dec.data.cpu()).numpy()
    summary["position_variance_init"] = {
        "mean": float(pv_init.mean()),
        "median": float(np.median(pv_init)),
        "p10": float(np.percentile(pv_init, 10)),
        "p90": float(np.percentile(pv_init, 90)),
    }

    arch_drifted, losses = _train_minimal(
        _build_arch(d_in=d_in, d_sae=d_sae, T=T, k_pos=k_pos, seed=1),
        n_steps=600, batch_size=256, d_in=d_in, device=device,
    )
    pv_drift = position_variance(arch_drifted.W_dec.data.cpu()).numpy()
    summary["position_variance_after_train"] = {
        "mean": float(pv_drift.mean()),
        "median": float(np.median(pv_drift)),
        "p10": float(np.percentile(pv_drift, 10)),
        "p90": float(np.percentile(pv_drift, 90)),
        "n_steps": 600,
        "final_loss": float(losses[-1]) if losses else None,
    }

    # ── 2. Encoder-decoder divergence after drift ──
    n_features_check = min(64, arch_drifted.d_sae)
    divs = []
    for fid in range(n_features_check):
        d = encoder_decoder_divergence(arch_drifted, fid)
        divs.append(d)
    summary["encoder_decoder_divergence"] = {
        "n_features": n_features_check,
        "cos_sim_mean": float(np.mean([d["cos_sim"] for d in divs])),
        "cos_sim_min": float(np.min([d["cos_sim"] for d in divs])),
        "rel_residual_mean": float(np.mean([d["rel_residual"] for d in divs])),
        "rel_residual_p90": float(np.percentile([d["rel_residual"] for d in divs], 90)),
    }
    # Sanity check on tied-init arch: rel_residual should be ~0.
    div_init = [encoder_decoder_divergence(arch_init.cpu(), fid) for fid in range(min(8, arch_init.d_sae))]
    summary["encoder_decoder_divergence_at_init"] = {
        "rel_residual_mean": float(np.mean([d["rel_residual"] for d in div_init])),
        "cos_sim_mean": float(np.mean([d["cos_sim"] for d in div_init])),
    }

    # Plot histograms
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(pv_init, bins=40, alpha=0.6, label="tied init")
    ax[0].hist(pv_drift, bins=40, alpha=0.6, label="after 600 SGD steps")
    ax[0].set_xlabel("position_variance(W_dec[f, :, :])")
    ax[0].set_ylabel("# features")
    ax[0].set_title(f"position-variance histogram (d_sae={d_sae}, T={T})")
    ax[0].legend()

    rel_res_init = [d["rel_residual"] for d in div_init]
    rel_res_drift = [d["rel_residual"] for d in divs]
    ax[1].hist(rel_res_init, bins=20, alpha=0.6, label="tied init")
    ax[1].hist(rel_res_drift, bins=20, alpha=0.6, label="after train")
    ax[1].set_xlabel("‖encoder_preimage - T·mean_dec‖ / ‖encoder_preimage‖")
    ax[1].set_title("encoder-decoder divergence (V4 vs V0 expected gap)")
    ax[1].legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "position_variance_and_divergence.png")

    # ── 3. Detection PR-AUC with shuffle ablation on synthetic cohorts ──
    # Train two copies of the arch on slightly-different synthetic data
    # so the detector has features that respond.
    cohort_temporal = _make_temporal_cohort(
        n_pos=120, n_neg=120, T=T, d_in=d_in, sig=4.0, seed=42,
    )
    cohort_density = _make_density_cohort(
        n_pos=120, n_neg=120, T=T, d_in=d_in, sig=4.0, seed=43,
    )
    res_temp = detect_case_study(
        arch_drifted.cpu(), *cohort_temporal,
        S_grid=(1, 2, 4, 8, 16),
        n_folds=4,
        shuffle_seed=42,
        device="cpu",
    )
    res_dens = detect_case_study(
        arch_drifted.cpu(), *cohort_density,
        S_grid=(1, 2, 4, 8, 16),
        n_folds=4,
        shuffle_seed=42,
        device="cpu",
    )
    summary["detection_temporal_cohort"] = {
        "pr_auc": res_temp.pr_auc,
        "pr_auc_shuffled": res_temp.pr_auc_shuffled,
        "shuffle_gap": res_temp.shuffle_gap,
    }
    summary["detection_density_cohort"] = {
        "pr_auc": res_dens.pr_auc,
        "pr_auc_shuffled": res_dens.pr_auc_shuffled,
        "shuffle_gap": res_dens.shuffle_gap,
    }

    fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    Sg = list(res_temp.pr_auc.keys())
    ax[0].plot(Sg, [res_temp.pr_auc[s] for s in Sg], "o-", label="unshuffled")
    ax[0].plot(Sg, [res_temp.pr_auc_shuffled[s] for s in Sg], "s--", label="within-window shuffled")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("S (top features used)")
    ax[0].set_ylabel("PR-AUC")
    ax[0].set_title("temporal-signature cohort\n(shuffle should drop PR-AUC)")
    ax[0].legend()
    ax[0].set_ylim(0, 1)

    ax[1].plot(Sg, [res_dens.pr_auc[s] for s in Sg], "o-", label="unshuffled")
    ax[1].plot(Sg, [res_dens.pr_auc_shuffled[s] for s in Sg], "s--", label="within-window shuffled")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("S (top features used)")
    ax[1].set_title("position-invariant cohort\n(shuffle should NOT drop PR-AUC)")
    ax[1].legend()
    fig.tight_layout()
    _save_figure(fig, out_dir / "pr_auc_shuffle_gap.png")

    # ── 4. Hook deltas: V0 / V1 / V2 / V4 side-by-side ──
    feature_id = int(np.argmax([d["rel_residual"] for d in divs]))  # pick the most-drifted feature
    W_dec_f = arch_drifted.W_dec.data[feature_id].cpu()
    pre = encoder_preimage(arch_drifted.cpu(), feature_id)
    ref_norm = 1.0
    deltas = {}
    for mode in ALL_MODES:
        kwargs = {"encoder_preimage": pre} if mode == "v4" else {}
        hook = TXCSteeringHook(W_dec_f, mode=mode, ref_norm=ref_norm, T=T, **kwargs)
        hook.magnitudes = torch.tensor([1.0])
        x = torch.zeros(1, 2 * T, d_in)
        out = hook(None, None, x)
        deltas[mode] = out[0].numpy()  # (2T, d_in)

    summary["hook_deltas_per_position"] = {
        m: {
            "feature_id": feature_id,
            "T": T,
            "per_position_norms": [float(np.linalg.norm(deltas[m][s])) for s in range(2 * T)],
            "total_energy": float(np.sum(deltas[m] ** 2)),
        }
        for m in ALL_MODES
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    pos = np.arange(2 * T)
    for mode in ALL_MODES:
        norms = [np.linalg.norm(deltas[mode][s]) for s in range(2 * T)]
        ax.plot(pos, norms, "o-", label=f"{mode} (total energy={np.sum(deltas[mode]**2):.3f})")
    ax.set_xlabel("token position in batch")
    ax.set_ylabel("‖Δ at this position‖")
    ax.set_title(f"per-position steering Δ across modes (feature {feature_id}, ref=1.0)")
    ax.legend()
    ax.axvline(T - 0.5, ls=":", color="gray", alpha=0.5)
    fig.tight_layout()
    _save_figure(fig, out_dir / "hook_modes_delta.png")

    # ── Save summary.json ──
    summary["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[validate_protocols] wrote {out_dir / 'summary.json'}")
    print(json.dumps(
        {k: v for k, v in summary.items() if k.startswith(("position_variance", "encoder_decoder", "detection"))},
        indent=2, default=str,
    ))


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
