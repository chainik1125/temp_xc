"""Multi-lane Reed-Solomon experiment: TXC-global vs local SAE.

Spec at `docs/aniket/experiments/synthetic/notes/multilane_rs_hmm_txc_proposal.tex`.

Headline metric is **reconstruction loss** (the proposal's main resource-
separation claim). The local alphabet code with `k_total` active features
across a length-`W` window has `||x - x_hat||^2 / (m*W) >= 1 - k_total/(m*W)`
(in normalized signal units). The TXC-global trajectory solution at
`k_total = m` reaches the noise floor exactly.

Architectures compared:

- **Local alphabet SAE** trained on iid tokens (regular `TopKSAE` from
  `temporal_crosscoders/models.py`). Per-token `k`. We sweep `k ∈ {m,
  m*(h+1), m*W}` to map the predicted lower bound. With `k = m*(h+1)`
  per-token (= per-position alphabet decomposition), the SAE captures all
  m lane components per position → noise floor.
- **TXC-global** (`TXCBareAntidead`, `k_total ∈ {m, ...}`) — the proposal's
  prescribed temporal-template architecture.

Per cell we report:

- Mean per-token reconstruction MSE on a held-out set of windows.
- Predicted local-alphabet lower bound: `1 - min(k_total, m*W) / (m*W)`.
- Lane-level temporal-atom recovery `Rec_temp_lane`.
- Probe accuracy on `Y` from the architecture's latent.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[2]
_TXC_DIR = _REPO_ROOT / "temporal_crosscoders"
if str(_TXC_DIR) not in sys.path:
    sys.path.insert(0, str(_TXC_DIR))

from models import TopKSAE  # noqa: E402

from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead  # noqa: E402

from .data_adapter import ColoredSourceCache  # noqa: E402
from .multilane_rs import (  # noqa: E402
    MultilaneRSConfig,
    all_lane_atoms,
    generate_multilane_dataset,
    lane_alphabet_lower_bound,
)
from .run_pair_experiment import _train_logistic_probe  # noqa: E402


# ---------- Trainers (inline so we can keep models for evaluation) -----------


def _train_topk_sae_tokens(
    cache: ColoredSourceCache,
    *,
    d: int, H: int, k: int,
    n_steps: int, batch_size: int, lr: float, device: torch.device,
    log_interval: int = 1000,
) -> TopKSAE:
    sae = TopKSAE(d_in=d, d_sae=H, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, 1).squeeze(1)
        loss, _, _ = sae(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"    [sae k={k}] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return sae


def _train_txc_global(
    cache: ColoredSourceCache,
    *,
    d: int, H: int, W: int, k_window: int,
    n_steps: int, batch_size: int, lr: float, device: torch.device,
    log_interval: int = 1000,
) -> TXCBareAntidead:
    model = TXCBareAntidead(d_in=d, d_sae=H, T=W, k=k_window).to(device)
    with torch.no_grad():
        init_x = cache.sample_windows(min(batch_size * 4, 256), W)
        model.init_b_dec_geometric_median(init_x.float())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, W)
        loss, _, _ = model(x)
        opt.zero_grad()
        loss.backward()
        model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        model._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"    [txc k={k_window} W={W}] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return model


# ---------- Evaluation ---------------------------------------------------------


@torch.no_grad()
def _sae_window_recon_mse(
    sae: TopKSAE, cache: ColoredSourceCache, W: int, n_eval: int
) -> float:
    """Per-token MSE on a length-W window: encode each token, decode, average."""
    batch = min(n_eval, 4096)
    n_done, total_se, total_tokens = 0, 0.0, 0
    while n_done < n_eval:
        b = min(batch, n_eval - n_done)
        x = cache.sample_windows(b, W)                          # (b, W, d)
        x_flat = x.reshape(-1, x.shape[-1])                     # (b*W, d)
        _, x_hat, _ = sae(x_flat)
        total_se += (x_hat - x_flat).pow(2).sum().item()
        total_tokens += x_flat.shape[0]
        n_done += b
    return total_se / total_tokens


@torch.no_grad()
def _txc_window_recon_mse(
    txc: TXCBareAntidead, cache: ColoredSourceCache, W: int, n_eval: int
) -> float:
    """Per-token MSE for TXC: average squared error across all (B, W, d)."""
    batch = min(n_eval, 4096)
    n_done, total_se, total_tokens = 0, 0.0, 0
    while n_done < n_eval:
        b = min(batch, n_eval - n_done)
        x = cache.sample_windows(b, W)                          # (b, W, d)
        _, x_hat, _ = txc(x)
        total_se += (x_hat - x).pow(2).sum().item()
        total_tokens += b * W
        n_done += b
    return total_se / total_tokens


@torch.no_grad()
def _lane_rec_temp(
    txc_W_dec: torch.Tensor, lane_atoms: torch.Tensor
) -> float:
    """Lane-level Rec_temp:
        (1 / (m * M)) * sum_{ell, beta} max_j |<G_{ell, beta}, w_j>|^2.

    Args:
        txc_W_dec: (H, W, d_in) learned per-atom temporal templates.
        lane_atoms: (m, M, W, d_in) ground-truth lane-level atoms.

    Returns:
        Scalar in [0, 1].
    """
    H = txc_W_dec.shape[0]
    m, M, W, d = lane_atoms.shape
    flat_learned = txc_W_dec.reshape(H, -1)
    flat_learned = flat_learned / flat_learned.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat_atoms = lane_atoms.reshape(m * M, -1)
    flat_atoms = flat_atoms / flat_atoms.norm(dim=1, keepdim=True).clamp_min(1e-12)
    inner = flat_atoms @ flat_learned.T                              # (m*M, H)
    max_per_atom = (inner ** 2).max(dim=1).values
    return float(max_per_atom.mean().item())


def _gather_anchor_indices(
    data: dict, W: int, *, max_samples: int, device: torch.device, seed: int
) -> dict:
    cfg: MultilaneRSConfig = data["config"]
    n_seq = cfg.n_seq
    max_start = cfg.T_chain - W
    if max_start < 0:
        raise ValueError(f"T_chain={cfg.T_chain} too short for W={W}")
    n = min(max_samples, n_seq * (max_start + 1))
    gen = torch.Generator(device="cpu").manual_seed(seed)
    seq = torch.randint(0, n_seq, (n,), generator=gen)
    starts = torch.randint(0, max_start + 1, (n,), generator=gen)
    return {
        "chain_idx": seq.to(device),
        "t_start": starts.to(device),
        "Y": data["Y"][seq].to(device),
    }


def _windows_at_anchors(
    cache: ColoredSourceCache, anchors: dict, W: int
) -> torch.Tensor:
    chains = anchors["chain_idx"]
    starts = anchors["t_start"]
    offsets = torch.arange(W, device=cache.device).unsqueeze(0)
    pos = starts.unsqueeze(1) + offsets
    chain_exp = chains.unsqueeze(1).expand(-1, W)
    return cache.act_chains[chain_exp, pos]


# ---------- Stage runner -------------------------------------------------------


_STAGE_GRID = {
    "smoke": dict(h=1, q=11, m=32, W_grid=[1, 2, 3, 4]),
    "main": dict(h=2, q=7, m=32, W_grid=[1, 2, 3, 4, 5, 6, 7]),
}


def run_stage(
    *,
    h: int, q: int, m: int, d: int, sigma: float,
    n_seq: int, T_chain: int,
    H_txc: int, H_sae: int,
    n_steps: int, batch_size: int, lr: float,
    W_grid: list[int],
    n_probe_samples: int,
    n_recon_eval: int,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
    label: str = "smoke",
) -> dict:
    cfg = MultilaneRSConfig(
        h=h, q=q, m=m, d=d, sigma=sigma,
        n_seq=n_seq, T_chain=T_chain, seed=seed,
    )
    print(f"\n=== Multilane RS stage {label}: {cfg} ===")
    data = generate_multilane_dataset(cfg)
    cache = ColoredSourceCache(data["x"], device)

    cells: list[dict] = []
    t0 = time.time()
    for W in W_grid:
        if W > T_chain:
            print(f"\n  Skipping W={W} > T_chain={T_chain}")
            continue
        print(f"\n--- W={W} ---")
        atoms_full = all_lane_atoms(data["alphabet"], h, q, W).to(device).float()
        # Predicted local-alphabet lower bound at k_total = m and k_total = m*W.
        lb_at_km = lane_alphabet_lower_bound(m=m, W=W, k_window=m)
        lb_at_kmW = lane_alphabet_lower_bound(m=m, W=W, k_window=m * W)
        # Noise-floor per-token MSE: sigma^2 * d (each token is 1 + sigma^2 * d).
        noise_floor = cfg.sigma ** 2 * cfg.d
        clean_norm_sq = 1.0  # signal energy per token
        print(
            f"  Reference: noise floor MSE = sigma^2 * d = {noise_floor:.4f}; "
            f"clean signal energy per token = {clean_norm_sq:.4f}"
        )
        print(
            f"  Local-alphabet predicted err: at k_total=m={m} → {lb_at_km:.3f}, "
            f"at k_total=m*W={m * W} → {lb_at_kmW:.3f}"
        )

        anchors = _gather_anchor_indices(
            data, W, max_samples=n_probe_samples, device=device, seed=seed + W,
        )

        cell: dict = {
            "W": W,
            "noise_floor_mse_per_token": noise_floor,
            "local_alphabet_lb_signal_err": {
                "k_eq_m": lb_at_km,
                "k_eq_m_times_W": lb_at_kmW,
            },
        }

        # 1. Local SAE per-token at k_pos values that map the resource gap.
        #    k_pos = 1: minimal local; captures 1 lane per token → big error.
        #    k_pos = m (= h+1 components per token in lane sense): full per-token
        #    alphabet recovery → noise floor.
        for k_pos in (1, m):
            print(f"\n    [SAE token-level k={k_pos}]")
            sae = _train_topk_sae_tokens(
                cache, d=d, H=H_sae, k=k_pos,
                n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
            )
            sae_mse = _sae_window_recon_mse(sae, cache, W, n_recon_eval)
            # Effective k_total across window:
            k_total_eff = k_pos * W
            # Probe Y from per-token SAE latents concatenated across the window.
            x_anchor = _windows_at_anchors(cache, anchors, W)
            B = x_anchor.shape[0]
            with torch.no_grad():
                z_flat = sae.encode(x_anchor.reshape(B * W, -1)).detach()
            z_concat = z_flat.reshape(B, W * z_flat.shape[-1])
            sae_probe = _train_logistic_probe(
                z_concat, anchors["Y"], R=cfg.q, device=device,
                seed=seed + 200 + 10 * k_pos + W,
            )
            cell[f"sae_k{k_pos}"] = {
                "k_pos": k_pos,
                "k_total_window_equivalent": k_total_eff,
                "recon_mse_per_token": sae_mse,
                "signal_excess_err": max(0.0, sae_mse - noise_floor) / clean_norm_sq,
                "probe_val_acc": sae_probe["val_accuracy"],
            }
            print(
                f"      SAE k_pos={k_pos} (k_total={k_total_eff}) "
                f"recon_mse={sae_mse:.4f}  excess={cell[f'sae_k{k_pos}']['signal_excess_err']:.3f} "
                f"probe={sae_probe['val_accuracy']:.3f}"
            )

        # 2. TXC-global at the proposal's prescription k_total = m, plus
        #    k_total = m*W for an upper-bound sanity check.
        for k_window in (m, m * W):
            print(f"\n    [TXC-global k_total={k_window} W={W}]")
            txc = _train_txc_global(
                cache, d=d, H=H_txc, W=W, k_window=k_window,
                n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
            )
            txc_mse = _txc_window_recon_mse(txc, cache, W, n_recon_eval)
            rec_temp_lane = _lane_rec_temp(txc.W_dec.detach(), atoms_full)
            # Probe Y from window-shared latent.
            x_anchor = _windows_at_anchors(cache, anchors, W)
            with torch.no_grad():
                z = txc.encode(x_anchor).detach()
            probe = _train_logistic_probe(
                z, anchors["Y"], R=cfg.q, device=device,
                seed=seed + 100 + k_window + W,
            )
            cell[f"txc_k{k_window}"] = {
                "k_window": k_window,
                "recon_mse_per_token": txc_mse,
                "signal_excess_err": max(0.0, txc_mse - noise_floor) / clean_norm_sq,
                "rec_temp_lane": rec_temp_lane,
                "probe_val_acc": probe["val_accuracy"],
                "n_atoms_total": cfg.total_temporal_atoms,
            }
            print(
                f"      TXC k_total={k_window} recon_mse={txc_mse:.4f} "
                f"excess={cell[f'txc_k{k_window}']['signal_excess_err']:.3f} "
                f"Rec_temp_lane={rec_temp_lane:.3f} probe={probe['val_accuracy']:.3f}"
            )

        cells.append(cell)

    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_label": f"multilane_rs_{label}",
        "config": asdict(cfg),
        "H_txc": H_txc,
        "H_sae": H_sae,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "W_grid": W_grid,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "cells": cells,
    }
    out_path = out_dir / f"multilane_rs_{label}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}  (elapsed {elapsed/60:.1f} min)")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Run multilane Reed-Solomon HMM sweep.")
    p.add_argument("--label", choices=["smoke", "main"], required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results/v6_colored_sources")
    p.add_argument("--n_seq", type=int, default=4096)
    p.add_argument("--T_chain", type=int, default=8)
    p.add_argument("--n_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_probe_samples", type=int, default=20000)
    p.add_argument("--n_recon_eval", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sigma", type=float, default=0.1)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    grid = _STAGE_GRID[args.label]
    h, q, m = grid["h"], grid["q"], grid["m"]
    W_grid = grid["W_grid"]
    d = max(512, m * q)
    H_sae = max(1024, m * q * 2)
    H_txc = max(4096, m * q ** (h + 1))
    if args.label == "main":
        H_txc = max(16384, m * q ** (h + 1))

    T_chain = max(args.T_chain, max(W_grid))

    run_stage(
        h=h, q=q, m=m, d=d, sigma=args.sigma,
        n_seq=args.n_seq, T_chain=T_chain,
        H_txc=H_txc, H_sae=H_sae,
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        W_grid=W_grid,
        n_probe_samples=args.n_probe_samples,
        n_recon_eval=args.n_recon_eval,
        device=device,
        out_dir=Path(args.out_dir),
        seed=args.seed,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
