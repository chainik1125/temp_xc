"""Train SAE + TXC on ambiguous-pair HMM data, run a logistic probe at
middle positions, report SAE pair-classification accuracy vs. TXC.

Implements the proposal's "weaker" Section-4 alternative as an empirical
sanity check:

    Local one-position learner has Acc <= 1/R + leakage.
    Temporal learner with W >= 3 covering [cue, middle, readout] has
        Acc -> 1.

Usage:
    python -m src.v6_colored_sources.run_pair_experiment
        [--device cuda] [--n_steps 4000] ...
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure temporal_crosscoders/ is importable for the model classes.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_TXC_DIR = _REPO_ROOT / "temporal_crosscoders"
if str(_TXC_DIR) not in sys.path:
    sys.path.insert(0, str(_TXC_DIR))

from models import TemporalCrosscoder, TopKSAE  # noqa: E402

from .ambiguous_pair import (  # noqa: E402
    AmbiguousPairConfig,
    generate_ambiguous_pair_dataset,
    isotropy_diagnostics,
)
from .data_adapter import ColoredSourceCache  # noqa: E402


def _train_logistic_probe(
    X: torch.Tensor,
    y: torch.Tensor,
    R: int,
    *,
    n_steps: int = 1500,
    batch_size: int = 256,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    val_frac: float = 0.2,
    device: torch.device,
    seed: int = 0,
) -> dict:
    """Multi-class logistic regression probe on (X, y). Returns dict with
    train_accuracy, val_accuracy, and a chance-baseline 1/R."""
    n = X.shape[0]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    perm = torch.randperm(n, generator=gen)
    n_val = max(1, int(n * val_frac))
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    X_tr = X[tr_idx].to(device).float()
    y_tr = y[tr_idx].to(device).long()
    X_val = X[val_idx].to(device).float()
    y_val = y[val_idx].to(device).long()

    h = X_tr.shape[1]
    probe = nn.Linear(h, R, bias=True).to(device)
    opt = torch.optim.Adam(
        probe.parameters(), lr=lr, weight_decay=weight_decay
    )
    n_tr = X_tr.shape[0]
    for step in range(n_steps):
        idx = torch.randint(0, n_tr, (min(batch_size, n_tr),), device=device)
        logits = probe(X_tr[idx])
        loss = F.cross_entropy(logits, y_tr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        train_acc = (probe(X_tr).argmax(dim=-1) == y_tr).float().mean().item()
        val_acc = (probe(X_val).argmax(dim=-1) == y_val).float().mean().item()
    return {
        "train_accuracy": float(train_acc),
        "val_accuracy": float(val_acc),
        "chance": 1.0 / R,
        "n_train": int(X_tr.shape[0]),
        "n_val": int(X_val.shape[0]),
    }


def _train_sae(
    cache: ColoredSourceCache,
    *,
    d: int,
    H: int,
    k: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    log_interval: int = 500,
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
                f"  [SAE] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return sae


def _train_txc(
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
    log_interval: int = 500,
) -> TemporalCrosscoder:
    if W < 2:
        raise ValueError(f"TXC requires W >= 2; got {W}")
    txc = TemporalCrosscoder(d_in=d, d_sae=H, T=W, k=k_pos).to(device)
    opt = torch.optim.Adam(txc.parameters(), lr=lr)
    t0 = time.time()
    for step in range(n_steps):
        x = cache.sample_windows(batch_size, W)
        loss, _, _ = txc(x)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(txc.parameters(), 1.0)
        opt.step()
        txc._normalize_decoder()
        if step % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"  [TXC W={W}] step={step:>5d} loss={loss.item():.4f} "
                f"| {(step + 1) / max(elapsed, 1e-6):.1f} it/s",
                flush=True,
            )
    return txc


def _gather_middle_positions(
    data: dict, device: torch.device, *, max_samples: int = 30_000, seed: int = 0
) -> dict:
    """Pick up to `max_samples` middle-position indices uniformly at random.

    Returns dict with chain_idx, t_idx (the time index into the chain),
    y_at_middle, on `device`.
    """
    pos = data["position_kind"]
    y = data["y"]
    middle_mask = pos == 1
    seq_idx, t_idx = torch.where(middle_mask)
    if seq_idx.numel() > max_samples:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        sel = torch.randperm(seq_idx.numel(), generator=gen)[:max_samples]
        seq_idx = seq_idx[sel]
        t_idx = t_idx[sel]
    return {
        "chain_idx": seq_idx.to(device),
        "t_idx": t_idx.to(device),
        "y_at_middle": y[seq_idx, t_idx].to(device),
    }


def _sae_latents_at(
    sae: TopKSAE, cache: ColoredSourceCache, idx: dict
) -> torch.Tensor:
    """Encode middle-position activations one-by-one through the SAE."""
    chains = idx["chain_idx"]
    ts = idx["t_idx"]
    x = cache.act_chains[chains, ts]    # (N, d)
    with torch.no_grad():
        return sae.encode(x).detach()


def _txc_latents_at(
    txc: TemporalCrosscoder,
    cache: ColoredSourceCache,
    idx: dict,
    W: int,
) -> torch.Tensor:
    """Encode windows of length W centered as tightly around each middle
    position as possible. The middle is at offset `m = (W - 1) // 2` so
    we read [t - m, ..., t - m + W - 1].

    For W=3 this is exactly [cue, middle, readout]. For W=2 we use
    [cue, middle].
    """
    chains = idx["chain_idx"]
    ts = idx["t_idx"]
    L = cache.chain_length
    d = cache.d

    m = (W - 1) // 2
    # Clamp to keep windows inside the chain.
    starts = (ts - m).clamp(min=0, max=L - W)
    offsets = torch.arange(W, device=cache.device).unsqueeze(0)
    pos_idx = starts.unsqueeze(1) + offsets        # (N, W)
    chain_exp = chains.unsqueeze(1).expand(-1, W)  # (N, W)
    x = cache.act_chains[chain_exp, pos_idx]       # (N, W, d)
    with torch.no_grad():
        return txc.encode(x).detach()


def run(
    *,
    R: int,
    d: int,
    a: float,
    sigma: float,
    n_seq: int,
    n_segments_per_seq: int,
    n_steps: int,
    batch_size: int,
    lr: float,
    H: int,
    k_pos: int,
    W_grid: list[int],
    device: torch.device,
    out_dir: Path,
    seed: int = 0,
) -> dict:
    cfg = AmbiguousPairConfig(
        R=R, d=d, a=a, sigma=sigma,
        n_seq=n_seq, n_segments_per_seq=n_segments_per_seq, seed=seed,
    )
    print(f"Generating ambiguous-pair dataset: {cfg}")
    data = generate_ambiguous_pair_dataset(cfg)

    diag = isotropy_diagnostics(data)
    print(
        f"Middle-position drift: max ||per_class_mean - global_mean||_2 "
        f"= {diag['max_class_drift']:.4f}  (N={diag['n_middle_samples']})"
    )

    cache = ColoredSourceCache(data["x"], device)

    # Train SAE on iid tokens.
    print("\n=== Training regular TopKSAE on iid tokens ===")
    sae = _train_sae(
        cache, d=cfg.d, H=H, k=k_pos,
        n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
    )

    # Probe SAE latents at middle positions.
    idx = _gather_middle_positions(data, device, seed=seed + 7)
    print(f"\nProbing on {idx['y_at_middle'].numel()} middle positions")
    z_sae_mid = _sae_latents_at(sae, cache, idx)
    sae_probe = _train_logistic_probe(
        z_sae_mid, idx["y_at_middle"], R=cfg.R, device=device, seed=seed + 11
    )
    print(
        f"  SAE probe:  train={sae_probe['train_accuracy']:.3f}  "
        f"val={sae_probe['val_accuracy']:.3f}  chance={sae_probe['chance']:.3f}"
    )

    # Direct probe on raw x for sanity (should be near chance — ambiguity
    # is built into the activations, not the dictionary).
    raw_x_mid = cache.act_chains[idx["chain_idx"], idx["t_idx"]]
    raw_probe = _train_logistic_probe(
        raw_x_mid, idx["y_at_middle"], R=cfg.R, device=device, seed=seed + 13
    )
    print(
        f"  Raw x probe: train={raw_probe['train_accuracy']:.3f}  "
        f"val={raw_probe['val_accuracy']:.3f}  (sanity)"
    )

    txc_results: list[dict] = []
    for W in W_grid:
        print(f"\n=== Training TXC W={W} ===")
        txc = _train_txc(
            cache, d=cfg.d, H=H, W=W, k_pos=k_pos,
            n_steps=n_steps, batch_size=batch_size, lr=lr, device=device,
        )
        z_txc_mid = _txc_latents_at(txc, cache, idx, W)
        txc_probe = _train_logistic_probe(
            z_txc_mid, idx["y_at_middle"], R=cfg.R, device=device, seed=seed + W,
        )
        print(
            f"  TXC W={W} probe:  train={txc_probe['train_accuracy']:.3f}  "
            f"val={txc_probe['val_accuracy']:.3f}  chance={txc_probe['chance']:.3f}"
        )

        # Direct probe on the raw window (sanity ceiling — what's
        # achievable without going through any architecture).
        raw_w = cache.act_chains[
            idx["chain_idx"].unsqueeze(1).expand(-1, W),
            (idx["t_idx"] - (W - 1) // 2).clamp(min=0, max=cache.chain_length - W).unsqueeze(1)
            + torch.arange(W, device=device).unsqueeze(0),
        ].reshape(idx["t_idx"].numel(), W * cfg.d)
        raw_w_probe = _train_logistic_probe(
            raw_w, idx["y_at_middle"], R=cfg.R, device=device, seed=seed + W + 100
        )
        print(
            f"  Raw window W={W} probe: val={raw_w_probe['val_accuracy']:.3f}  (sanity ceiling)"
        )

        txc_results.append({
            "W": W,
            "txc_probe": txc_probe,
            "raw_window_probe": raw_w_probe,
        })

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": "ambiguous_pair",
        "config": asdict(cfg),
        "isotropy_max_class_drift": diag["max_class_drift"],
        "n_middle_samples": diag["n_middle_samples"],
        "n_steps": n_steps,
        "batch_size": batch_size,
        "H": H,
        "k_pos": k_pos,
        "W_grid": W_grid,
        "device": str(device),
        "sae_probe": sae_probe,
        "raw_token_probe": raw_probe,
        "txc": txc_results,
    }
    out_path = out_dir / "ambiguous_pair.json"
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved {out_path}")
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description="Run ambiguous-pair HMM probing.")
    p.add_argument("--R", type=int, default=8)
    p.add_argument("--d", type=int, default=64)
    p.add_argument("--a", type=float, default=0.7071067811865476)
    p.add_argument("--sigma", type=float, default=0.1)
    p.add_argument("--n_seq", type=int, default=128)
    p.add_argument("--n_segments_per_seq", type=int, default=256)
    p.add_argument("--n_steps", type=int, default=4000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--H", type=int, default=64)
    p.add_argument("--k_pos", type=int, default=8)
    p.add_argument("--W_grid", type=int, nargs="+", default=[2, 3, 5])
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out_dir", type=str, default="results/v6_colored_sources")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    run(
        R=args.R, d=args.d, a=args.a, sigma=args.sigma,
        n_seq=args.n_seq, n_segments_per_seq=args.n_segments_per_seq,
        n_steps=args.n_steps, batch_size=args.batch_size, lr=args.lr,
        H=args.H, k_pos=args.k_pos, W_grid=args.W_grid, device=device,
        out_dir=Path(args.out_dir), seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
