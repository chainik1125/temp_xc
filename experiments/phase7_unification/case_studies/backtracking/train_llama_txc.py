#!/usr/bin/env python3
"""Train a TXC / TopKSAE / TemporalSAE on a Llama-3.1-8B L10 hook cache.

Lean trainer that bypasses Phase 7's hardcoded Gemma + L12 + d_in=2304 paths.
Reads from `data/llama_3_1_8b/<hook>/L<layer>/{token_ids.npy, activations.fp16.npy}`
(produced by `build_llama_finetune_cache.py`), reshapes into T-token windows,
and trains a single architecture for `--steps` steps.

Architectures supported (selected by `--arch`):

    topk_sae    — per-token TopK SAE baseline (T=1, d_in=4096). Window
                  reshape collapses (B, T, d_in) → (B*T, d_in).
    txc_bare    — TXCBareAntidead, the TXC variant the user wanted at
                  ln1 / attn_out. d_in=4096, T configurable, k=k_pos*T.
    tsae_paper  — TemporalMatryoshkaBatchTopKSAE, paper-faithful T-SAE.

Output: `data/llama_3_1_8b/<hook>/L<layer>/<arch>/{ckpt.pt, log.json, meta.json}`.

Run from repo root after the cache exists:

    .venv/bin/python -m experiments.phase7_unification.case_studies.backtracking.train_llama_txc \
        --hook attn --arch txc_bare --T 5 --d-sae 32768 --k-pos 100 --steps 10000

For a fast smoke run, use --steps 1000 (~5 min on H100).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TMPDIR", "/workspace/tmp")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.architectures.topk_sae import TopKSAE  # noqa: E402
from src.architectures.tsae_paper import (  # noqa: E402
    TemporalMatryoshkaBatchTopKSAE,
    TemporalMatryoshkaBatchTopKTrainerLite,
)
from src.architectures.txc_bare_antidead import TXCBareAntidead  # noqa: E402


def _open_cache(cache_dir: Path) -> tuple[np.memmap, dict]:
    act = np.load(cache_dir / "activations.fp16.npy", mmap_mode="r")
    meta = json.loads((cache_dir / "meta.json").read_text())
    return act, meta


def _make_token_window_gen(activations: np.memmap, T: int, batch_size: int,
                           device: str, dtype: torch.dtype, seed: int = 42):
    """Yield (B, T, d_in) windows sampled uniformly from the cache.

    Each cache row is (ctx, d_in) where ctx > T. We pick a random row
    and a random offset [0, ctx-T] inside it, take the T contiguous
    window. Repeat until batch is filled.
    """
    n, ctx, d_in = activations.shape
    rng = np.random.default_rng(seed)

    def gen(batch: int) -> torch.Tensor:
        rows = rng.integers(0, n, size=batch)
        offs = rng.integers(0, ctx - T + 1, size=batch)
        out = np.empty((batch, T, d_in), dtype=np.float32)
        for i, (r, o) in enumerate(zip(rows, offs)):
            out[i] = activations[r, o:o + T].astype(np.float32, copy=False)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    return gen


def _flat_token_gen(activations: np.memmap, batch_size: int,
                    device: str, dtype: torch.dtype, seed: int = 42):
    """Per-token generator — flattens (B, T, d_in) -> (B, d_in)."""
    n, ctx, d_in = activations.shape
    rng = np.random.default_rng(seed)

    def gen(batch: int) -> torch.Tensor:
        rows = rng.integers(0, n, size=batch)
        offs = rng.integers(0, ctx, size=batch)
        out = np.empty((batch, d_in), dtype=np.float32)
        for i, (r, o) in enumerate(zip(rows, offs)):
            out[i] = activations[r, o].astype(np.float32, copy=False)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    return gen


def _temporal_pair_gen(activations: np.memmap, batch_size: int,
                       device: str, dtype: torch.dtype, offset: int = 1,
                       seed: int = 42):
    """Pair generator for T-SAE: yields (B, 2, d_in) where the second
    column is the activation at position t+offset in the same trace.
    """
    n, ctx, d_in = activations.shape
    rng = np.random.default_rng(seed)

    def gen(batch: int) -> torch.Tensor:
        rows = rng.integers(0, n, size=batch)
        offs = rng.integers(0, ctx - offset, size=batch)
        out = np.empty((batch, 2, d_in), dtype=np.float32)
        for i, (r, o) in enumerate(zip(rows, offs)):
            out[i, 0] = activations[r, o].astype(np.float32, copy=False)
            out[i, 1] = activations[r, o + offset].astype(np.float32, copy=False)
        return torch.from_numpy(out).to(device=device, dtype=dtype)

    return gen


def train_tsae_paper(model: TemporalMatryoshkaBatchTopKSAE, gen_fn, steps: int,
                     batch_size: int, lr: float, device: str,
                     log_every: int = 200) -> dict:
    """Run the Bhalla et al. 2025 T-SAE trainer wrapper for `steps` steps.

    Paper params: group_weights=[1.0, 1.0] for the two matryoshka groups,
    auxk_alpha=1/32, temp_alpha=1/10, threshold_start_step=1000.
    """
    trainer = TemporalMatryoshkaBatchTopKTrainerLite(
        model=model,
        group_weights=[1.0, 1.0],
        total_steps=steps,
        lr=lr,
        warmup_steps=min(1000, steps // 10),
        contrastive=True,
        device=torch.device(device),
    )
    log: dict[str, list[float]] = {"step": [], "l2": [], "auxk": [], "temp": [], "dead": []}
    model.train()
    t0 = time.time()
    for step in range(steps):
        x_pair = gen_fn(batch_size)
        trainer.update(step, x_pair, b_dec_init=None)
        if step % log_every == 0 or step == steps - 1:
            # Re-run loss without backward to grab fresh stats
            with torch.no_grad():
                _, stats, _, _ = trainer.loss(x_pair, step=step)
            log["step"].append(step)
            log["l2"].append(stats["l2"])
            log["auxk"].append(stats["auxk"])
            log["temp"].append(stats["temp"])
            log["dead"].append(stats["dead"])
            print(
                f"  step={step:>6}  l2={stats['l2']:.4f}  auxk={stats['auxk']:.4f}"
                f"  temp={stats['temp']:.4f}  dead={stats['dead']:>5}  "
                f"elapsed={time.time()-t0:.0f}s"
            )
    return log


def train_topk_sae(model: TopKSAE, gen_fn, steps: int, batch_size: int, lr: float,
                   device: str, log_every: int = 200) -> dict:
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log: dict[str, list[float]] = {"step": [], "mse": [], "l0": [], "alive": []}
    model.train()
    t0 = time.time()
    for step in range(steps):
        x = gen_fn(batch_size)
        mse, _, z = model(x)
        loss = mse
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        with torch.no_grad():
            model._normalize_decoder()
        if step % log_every == 0 or step == steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
                alive = (z.sum(dim=0) > 0).float().mean().item()
            log["step"].append(step)
            log["mse"].append(float(mse))
            log["l0"].append(l0)
            log["alive"].append(alive)
            print(f"  step={step:>6}  mse={float(mse):.4f}  l0={l0:.1f}  alive={alive:.3f}  elapsed={time.time()-t0:.0f}s")
    return log


def train_txc_bare(model: TXCBareAntidead, gen_fn, steps: int, batch_size: int,
                   lr: float, device: str, log_every: int = 200) -> dict:
    """Bare reconstruction loop for TXCBareAntidead. Calls model.forward(x)
    where x is (B, T, d_in); model returns (loss, recon, z, ...) — we just
    take the recon loss + standard SGD."""
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    log: dict[str, list[float]] = {"step": [], "mse": [], "l0_window": []}
    model.train()
    t0 = time.time()
    for step in range(steps):
        x = gen_fn(batch_size)
        out = model(x)
        # txc_bare_antidead returns (loss_total, x_hat, z, info_dict) typically.
        if isinstance(out, tuple):
            loss = out[0]
            z = out[2] if len(out) >= 3 else None
        else:
            loss = out
            z = None
        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        if step % log_every == 0 or step == steps - 1:
            l0_str = ""
            if z is not None:
                with torch.no_grad():
                    l0 = (z > 0).float().sum(dim=-1).mean().item()
                    log["l0_window"].append(l0)
                    l0_str = f"  l0_win={l0:.1f}"
            log["step"].append(step)
            log["mse"].append(float(loss))
            print(f"  step={step:>6}  loss={float(loss):.4f}{l0_str}  elapsed={time.time()-t0:.0f}s")
    return log


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hook", choices=("resid", "ln1", "attn"), required=True)
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--arch", choices=("topk_sae", "txc_bare", "tsae_paper"),
                        default="txc_bare",
                        help="topk_sae=per-token TopK; txc_bare=window encoder T tokens; "
                             "tsae_paper=Bhalla et al. 2025 TemporalMatryoshkaBatchTopKSAE "
                             "(per-token + temporal-pair contrastive loss + matryoshka groups)")
    parser.add_argument("--T", type=int, default=5, help="window length (txc_bare only)")
    parser.add_argument("--d-sae", type=int, default=32768)
    parser.add_argument("--k-pos", type=int, default=100, help="per-position TopK budget")
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--out-suffix", default=None, help="default arch name")
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    cache_root = Path(args.cache_root) if args.cache_root else _REPO / "data" / "llama_3_1_8b"
    cache_dir = cache_root / args.hook / f"L{args.layer}"
    if not (cache_dir / "activations.fp16.npy").exists():
        raise SystemExit(f"missing cache at {cache_dir}; run build_llama_finetune_cache.py first")
    activations, cache_meta = _open_cache(cache_dir)
    n, ctx, d_in = activations.shape
    print(f"[train] cache {activations.shape} from {cache_dir}; d_in={d_in}")

    suffix = args.out_suffix or args.arch
    out_dir = cache_dir / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists() and not args.force:
        print(f"[train] {ckpt_path} exists; use --force to retrain")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32  # train in fp32 for stability; fp16 cache promotes per-batch

    if args.arch == "topk_sae":
        model = TopKSAE(d_in=d_in, d_sae=args.d_sae, k=args.k_pos).to(device=device, dtype=dtype)
        gen_fn = _flat_token_gen(activations, args.batch, device, dtype)
        print(f"[train] TopKSAE d_in={d_in} d_sae={args.d_sae} k={args.k_pos} → {out_dir}")
        log = train_topk_sae(model, gen_fn, args.steps, args.batch, args.lr, device)
    elif args.arch == "txc_bare":
        if ctx < args.T:
            raise SystemExit(f"cache ctx={ctx} < T={args.T}")
        k_win = args.k_pos * args.T
        model = TXCBareAntidead(
            d_in=d_in, d_sae=args.d_sae, T=args.T, k=k_win,
        ).to(device=device, dtype=dtype)
        gen_fn = _make_token_window_gen(activations, args.T, args.batch, device, dtype)
        print(f"[train] TXCBareAntidead d_in={d_in} d_sae={args.d_sae} T={args.T} k_win={k_win} → {out_dir}")
        log = train_txc_bare(model, gen_fn, args.steps, args.batch, args.lr, device)
    elif args.arch == "tsae_paper":
        # Bhalla et al. 2025 — d_sae split into [0.2, 0.8] matryoshka groups,
        # k applied at the BATCH level (BatchTopK across all (B*d_sae) preacts).
        g0 = args.d_sae // 5
        g1 = args.d_sae - g0
        model = TemporalMatryoshkaBatchTopKSAE(
            activation_dim=d_in, dict_size=args.d_sae, k=args.k_pos,
            group_sizes=[g0, g1],
        ).to(device=device, dtype=dtype)
        gen_fn = _temporal_pair_gen(activations, args.batch, device, dtype, offset=1)
        print(f"[train] TemporalMatryoshkaBatchTopKSAE (Bhalla et al.) d_in={d_in} "
              f"d_sae={args.d_sae} k={args.k_pos} groups=[{g0}, {g1}] → {out_dir}")
        log = train_tsae_paper(model, gen_fn, args.steps, args.batch, args.lr, device)
    else:
        raise ValueError(args.arch)

    torch.save(model.state_dict(), ckpt_path)
    (out_dir / "log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "meta.json").write_text(json.dumps({
        "arch": args.arch,
        "d_in": int(d_in),
        "d_sae": int(args.d_sae),
        "T": int(args.T) if args.arch == "txc_bare" else 1,
        "k_pos": int(args.k_pos),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "lr": args.lr,
        "hook": args.hook,
        "layer": int(args.layer),
        "cache_meta": cache_meta,
    }, indent=2))
    print(f"[train] saved {ckpt_path}")


if __name__ == "__main__":
    main()
