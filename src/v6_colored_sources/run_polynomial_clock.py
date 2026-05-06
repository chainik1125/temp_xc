"""Train SAE / Stacked SAE / TXC-global on polynomial-clock data, run latent
probes for ``Y`` recovery, and report alphabet (``Rec_local``) and temporal-
template (``Rec_temp``) recovery against the ground-truth bank.

This is the load-bearing experiment for proposal Section 4
(`docs/aniket/experiments/synthetic/notes/polynomial_clock_experiment.tex`):
Stage 1/2/3 sweeps over ``(h, q)`` and window length ``W``, with the
prediction that

  - For ``W <= h``: every probe is at chance ``1/q`` (impossibility theorem).
  - For ``W >= h+1``: every windowed probe (raw, stacked SAE, TXC-global)
    hits ``Acc(Y) = 1`` — the latent-prediction metric does not differentiate
    architectures.
  - But ``Rec_temp`` (decoder-atom recovery against the ``q^(h+1)`` polynomial
    templates ``G_β``) DOES differentiate: only TXC-global with window-level
    top-1 has the dictionary capacity to learn polynomial templates; stacked
    SAE learns alphabet atoms per position so its ``Rec_temp`` stays at
    chance.
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

from models import StackedSAE, TopKSAE  # noqa: E402

from .data_adapter import ColoredSourceCache  # noqa: E402
from .metrics import squared_axis_recovery  # noqa: E402
from .polynomial_clock import (  # noqa: E402
    PolynomialClockConfig,
    all_polynomial_atoms,
    generate_polynomial_clock_dataset,
)
from .run_pair_experiment import _train_logistic_probe  # noqa: E402
from .train_runner import TrainConfig  # noqa: E402


# ---------- Architecture training -----------------------------------------


def _train_topk_sae_local(
    cache: ColoredSourceCache,
    *,
    d: int,
    H: int,
    k: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_interval: int = 1000,
) -> TopKSAE:
    """Train regular TopKSAE on iid tokens; return the model."""
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
                f"  [token_sae] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return sae


def _train_stacked_sae(
    cache: ColoredSourceCache,
    *,
    d: int,
    H: int,
    W: int,
    k_pos: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_interval: int = 1000,
) -> StackedSAE:
    """Train ``T`` independent per-position TopK SAEs on aligned windows."""
    if W < 1:
        raise ValueError(f"W must be >= 1; got {W}")
    sae = StackedSAE(d_in=d, d_sae=H, T=W, k=k_pos).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, W)
        loss, _, _ = sae(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        opt.step()
        sae._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  [stacked_sae W={W}] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return sae


# ---------- Recovery metrics ----------------------------------------------


def rec_local_alphabet(W_dec: torch.Tensor, alphabet: torch.Tensor) -> float:
    """Mean over alphabet symbols ``a`` of ``max_j |<u_a, w_j>|^2``.

    Args:
        W_dec: ``(d, H)`` decoder columns (one atom per column). For SAE/SAE
            on tokens this is just ``sae.W_dec`` of shape ``(d, H)``. For
            stacked / TXC architectures with shape ``(H, T, d)`` you should
            pass ``W_dec.mean(dim=1).T`` (the per-position-averaged direction).
        alphabet: ``(q, d)`` orthonormal alphabet rows.

    Returns:
        Scalar in ``[0, 1]``.
    """
    return squared_axis_recovery(alphabet, W_dec.T.contiguous())


def rec_temp_polynomial(W_dec_window: torch.Tensor, atoms: torch.Tensor) -> float:
    """Temporal-atom recovery score from the proposal's Section 7.

    Args:
        W_dec_window: ``(H, W, d)`` learned per-atom temporal templates. For
            ``TXCBareAntidead`` this is just ``model.W_dec``. Each atom
            assumed unit-norm (Frobenius); we normalize defensively anyway.
        atoms: ``(M, W, d)`` ground-truth ``G_β`` bank where ``M = q^(h+1)``.

    Returns:
        ``Rec_temp = (1/M) sum_β max_j |<G_β, W_dec[j]>|^2``.
    """
    H = W_dec_window.shape[0]
    M = atoms.shape[0]
    flat_learned = W_dec_window.reshape(H, -1)             # (H, W*d)
    flat_learned = flat_learned / flat_learned.norm(dim=1, keepdim=True).clamp_min(1e-12)
    flat_atoms = atoms.reshape(M, -1)                      # (M, W*d)
    flat_atoms = flat_atoms / flat_atoms.norm(dim=1, keepdim=True).clamp_min(1e-12)
    inner = flat_atoms @ flat_learned.T                    # (M, H)
    max_per_atom = (inner ** 2).max(dim=1).values          # (M,)
    return float(max_per_atom.mean().item())


# ---------- Latent extraction at "anchor" positions -----------------------


def _gather_anchor_indices(
    data: dict, W: int, *, max_samples: int, device: torch.device, seed: int = 0
) -> dict:
    """Pick anchor (chain, t_start) indices uniformly at random from the
    valid range ``t_start in [0, T_chain - W]``.

    Each anchor's window will start at ``t_start`` and span ``W`` positions,
    covering polynomial evaluations at ``[t_start, ..., t_start + W - 1]``.
    The label we predict is the leading coefficient ``Y`` of the original
    episode (constant across the chain).
    """
    cfg: PolynomialClockConfig = data["config"]
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
    return cache.act_chains[chain_exp, pos]                # (B, W, d)


def _sae_latents_at_first_position(
    sae: TopKSAE, cache: ColoredSourceCache, anchors: dict
) -> torch.Tensor:
    """Encode the *first* position of each anchor through the token-SAE.

    For ``W = 1`` this is just the position itself; for ``W > 1`` it
    represents the regular (no-window-context) baseline — the SAE only sees
    one token at a time, so probing it at ``W > 1`` would be cheating. We
    intentionally pass only the first position to keep the regular SAE
    properly local.
    """
    chains = anchors["chain_idx"]
    starts = anchors["t_start"]
    x = cache.act_chains[chains, starts]                   # (B, d)
    with torch.no_grad():
        return sae.encode(x).detach()


def _stacked_sae_latents_at(
    sae: StackedSAE, cache: ColoredSourceCache, anchors: dict, W: int
) -> torch.Tensor:
    """Encode windows of length ``W`` and return the flattened
    ``(B, T*H)`` stacked latents."""
    x = _windows_at_anchors(cache, anchors, W)
    with torch.no_grad():
        _, _, u = sae(x)                                    # (B, T, H)
    return u.reshape(u.shape[0], -1)


def _txc_global_latents_at(
    txc, cache: ColoredSourceCache, anchors: dict, W: int
) -> torch.Tensor:
    """Encode windows of length ``W`` through the window-shared latent."""
    x = _windows_at_anchors(cache, anchors, W)
    with torch.no_grad():
        return txc.encode(x).detach()                       # (B, H)


def _raw_window_features(cache: ColoredSourceCache, anchors: dict, W: int) -> torch.Tensor:
    """Flattened raw window — the architecture-free ceiling for ``Acc(Y)``."""
    return _windows_at_anchors(cache, anchors, W).reshape(anchors["chain_idx"].shape[0], -1)


# ---------- Stage runner --------------------------------------------------


def run_stage(
    *,
    h: int,
    q: int,
    d: int,
    sigma: float,
    n_seq: int,
    T_chain: int,
    H: int,
    k_pos_local: int,
    k_pos_stacked: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    W_grid: list[int],
    n_probe_samples: int,
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
) -> dict:
    cfg = PolynomialClockConfig(
        h=h, q=q, d=d, sigma=sigma, n_seq=n_seq, T_chain=T_chain, seed=seed,
    )
    print(f"\n=== Stage h={h}, q={q} ({cfg.num_atoms} polynomial atoms) ===")
    data = generate_polynomial_clock_dataset(cfg)
    cache = ColoredSourceCache(data["x"], device)
    alphabet = data["alphabet"].to(device).float()

    # Train regular SAE once on iid tokens — its latent doesn't see windows.
    # Use H = q square dictionary so the SAE has exactly enough atoms for
    # the alphabet (Rec_local should be ~1).
    print("\n--- regular TopKSAE (iid tokens) ---")
    sae = _train_topk_sae_local(
        cache, d=d, H=q, k=k_pos_local,
        n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
    )
    rec_local_sae = rec_local_alphabet(sae.W_dec.detach(), alphabet)
    print(f"    Rec_local (alphabet) = {rec_local_sae:.3f}")

    cells: list[dict] = []
    t0 = time.time()
    for W in W_grid:
        if W < 1:
            continue
        if W > T_chain - 1:
            print(f"  Skipping W={W} (> T_chain-1)")
            continue
        print(f"\n--- W={W} ---")
        anchors = _gather_anchor_indices(
            data, W, max_samples=n_probe_samples, device=device, seed=seed + W,
        )

        # 1. Raw-window probe (architecture-free ceiling).
        raw_X = _raw_window_features(cache, anchors, W)
        raw_probe = _train_logistic_probe(
            raw_X, anchors["Y"], R=cfg.q, device=device, seed=seed + 100 + W,
        )
        print(f"    raw window probe          val={raw_probe['val_accuracy']:.3f}")

        # 2. Regular SAE probe — only sees first token of each anchor.
        sae_X = _sae_latents_at_first_position(sae, cache, anchors)
        sae_probe = _train_logistic_probe(
            sae_X, anchors["Y"], R=cfg.q, device=device, seed=seed + 200 + W,
        )
        print(f"    regular SAE (W=1 latent)  val={sae_probe['val_accuracy']:.3f}")

        # 3. Stacked SAE: train fresh per W, then probe on flattened (T*H) latents.
        stacked = _train_stacked_sae(
            cache, d=d, H=H, W=W, k_pos=k_pos_stacked,
            n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
        )
        stacked_X = _stacked_sae_latents_at(stacked, cache, anchors, W)
        stacked_probe = _train_logistic_probe(
            stacked_X, anchors["Y"], R=cfg.q, device=device, seed=seed + 300 + W,
        )
        print(f"    stacked SAE (T*H latent)  val={stacked_probe['val_accuracy']:.3f}")
        # Stacked SAE has per-position decoders saes[t].W_dec of shape (d, H).
        # decoder_directions returns the mean over T → (d, H).
        stacked_avg = stacked.decoder_directions.detach()
        rec_local_stacked = rec_local_alphabet(stacked_avg, alphabet)
        # For Rec_temp build (H, W, d) by stacking each position's W_dec.
        per_pos_decoders = [stacked.saes[t].W_dec.detach() for t in range(W)]  # each (d, H)
        # Stack to (W, d, H); permute to (H, W, d).
        stacked_temporal_HWd = torch.stack(per_pos_decoders, dim=0).permute(2, 0, 1)
        atoms = all_polynomial_atoms(alphabet, h, q, W).to(device).float()
        rec_temp_stacked = rec_temp_polynomial(stacked_temporal_HWd, atoms)
        print(
            f"      stacked SAE Rec_local={rec_local_stacked:.3f}  "
            f"Rec_temp={rec_temp_stacked:.3f}  (M={atoms.shape[0]})"
        )

        # 4. TXC-global: window-level k=1 — the proposal's prescription.
        # Train inline to get access to the model for latent + decoder extraction.
        from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead
        txc_model = TXCBareAntidead(d_in=d, d_sae=H, T=W, k=1).to(device)
        with torch.no_grad():
            init_x = cache.sample_windows(min(batch_size * 4, 256), W)
            txc_model.init_b_dec_geometric_median(init_x.float())
        opt = torch.optim.Adam(txc_model.parameters(), lr=lr)
        for step in range(n_steps):
            x = cache.sample_windows(batch_size, W)
            loss, _, _ = txc_model(x)
            opt.zero_grad(); loss.backward()
            txc_model.remove_gradient_parallel_to_decoder()
            nn.utils.clip_grad_norm_(txc_model.parameters(), 1.0)
            opt.step(); txc_model._normalize_decoder()
            if step % 1000 == 0:
                elapsed = time.time() - t0
                print(
                    f"    [txc_global W={W}] step={step:>5d} loss={loss.item():.4f}",
                    flush=True,
                )

        txc_X = _txc_global_latents_at(txc_model, cache, anchors, W)
        txc_probe = _train_logistic_probe(
            txc_X, anchors["Y"], R=cfg.q, device=device, seed=seed + 400 + W,
        )
        print(f"    TXC-global (k_win=1)      val={txc_probe['val_accuracy']:.3f}")
        rec_local_txc = rec_local_alphabet(
            txc_model.decoder_dirs_averaged.detach(), alphabet
        )
        rec_temp_txc = rec_temp_polynomial(txc_model.W_dec.detach(), atoms)
        print(
            f"      TXC-global Rec_local={rec_local_txc:.3f}  "
            f"Rec_temp={rec_temp_txc:.3f}  (M={atoms.shape[0]})"
        )

        cells.append({
            "W": W,
            "raw_probe": raw_probe,
            "sae_probe": sae_probe,
            "stacked_probe": stacked_probe,
            "stacked_rec_local": rec_local_stacked,
            "stacked_rec_temp": rec_temp_stacked,
            "txc_probe": txc_probe,
            "txc_rec_local": rec_local_txc,
            "txc_rec_temp": rec_temp_txc,
            "n_atoms": atoms.shape[0],
            "txc_final_loss": float(loss.item()),
        })

    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage_label": f"polynomial_clock_h{h}_q{q}",
        "config": asdict(cfg),
        "H": H,
        "k_pos_local": k_pos_local,
        "k_pos_stacked": k_pos_stacked,
        "k_window_txc": 1,
        "n_steps": n_steps,
        "batch_size": batch_size,
        "lr": lr,
        "W_grid": W_grid,
        "device": str(device),
        "elapsed_seconds": elapsed,
        "sae_rec_local": rec_local_sae,
        "cells": cells,
    }
    out_path = out_dir / f"polynomial_clock_h{h}_q{q}.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}  (elapsed {elapsed/60:.1f} min)")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Run polynomial-clock sweeps.")
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results/v6_colored_sources")
    p.add_argument("--n_seq", type=int, default=4096)
    p.add_argument("--T_chain", type=int, default=16)
    p.add_argument("--n_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n_probe_samples", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    if args.stage == 1:
        h, q, W_grid, H = 1, 31, [1, 2, 3, 4], 1024
    elif args.stage == 2:
        h, q, W_grid, H = 2, 11, [1, 2, 3, 4, 5], 2048
    else:
        h, q, W_grid, H = 3, 7, [1, 2, 3, 4, 5, 6], 4096

    run_stage(
        h=h, q=q, d=max(64, q * 2), sigma=0.1,
        n_seq=args.n_seq, T_chain=args.T_chain,
        H=H, k_pos_local=1, k_pos_stacked=1,
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        W_grid=W_grid, n_probe_samples=args.n_probe_samples,
        device=device, out_dir=Path(args.out_dir), seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
