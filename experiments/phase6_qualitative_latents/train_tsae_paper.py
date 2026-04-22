"""Train the paper-faithful T-SAE on our Gemma-2-2b-IT L13 anchor buf.

Mirrors Ye et al. 2025's `TemporalMatryoshkaBatchTopKTrainer` training
loop with our pair generator. Unlike the simpler `tsae_ours` port, this
includes:

    - BatchTopK activation (k=20 average per token by default)
    - Matryoshka groups [0.2, 0.8] × d_sae (paper default)
    - AuxK loss on dead features (auxk_alpha=1/32)
    - Geometric median init for b_dec
    - Decoder-parallel gradient removal + unit-norm renormalisation
    - Linear warmup + decay LR schedule
    - Threshold-EMA for inference-time encoding
    - Temporal InfoNCE (raw dot product) on high-level group

Disk-saves ckpt as `tsae_paper__seed<seed>.pt` compatible with our
encoding pipeline via a flat `state_dict + arch + meta` envelope.

Usage:
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
        experiments/phase6_qualitative_latents/train_tsae_paper.py \
        [--steps N] [--k K] [--d-sae D] [--seed S]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
# Force unbuffered stdout so background-launched runs show progress live.
import sys as _sys  # noqa: E402
_sys.stdout.reconfigure(line_buffering=True)
_sys.stderr.reconfigure(line_buffering=True)

import numpy as np  # noqa: E402
import torch as t  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
LOGS_DIR = REPO / "experiments/phase5_downstream_utility/results/training_logs"
INDEX_PATH = REPO / "experiments/phase5_downstream_utility/results/training_index.jsonl"

D_IN = 2304


def preload_anchor(n_seqs: int, device) -> t.Tensor:
    """Load first n_seqs sequences of L13 activations onto GPU as fp16."""
    arr = np.load(CACHE / "resid_L13.npy", mmap_mode="r")
    sub = np.asarray(arr[:n_seqs], dtype=np.float16)
    return t.from_numpy(sub).to(device)


def make_pair_gen(buf: t.Tensor, batch_size: int):
    """Sample (B, 2, d_in) adjacent-token pairs from `buf` (shape N, L, d).

    Returns a callable: pair_gen(step) -> (B, 2, d_in) on buf.device.
    """
    N, L, d = buf.shape
    n_wins = L - 1  # positions 0..L-2 for pair (t, t+1)

    def gen(_step: int) -> t.Tensor:
        seq = t.randint(0, N, (batch_size,), device=buf.device)
        off = t.randint(0, n_wins, (batch_size,), device=buf.device)
        # (B, 2, d): stack (buf[seq, off], buf[seq, off+1])
        a = buf[seq, off]
        b = buf[seq, off + 1]
        return t.stack([a, b], dim=1).float()
    return gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=25_000)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--k", type=int, default=20,
                   help="BatchTopK budget (paper default 20)")
    p.add_argument("--d-sae", type=int, default=18432,
                   help="Dict size; 18432 to match Phase 5.7 bench, 16384 to match paper")
    p.add_argument("--group-fractions", type=float, nargs="+",
                   default=[0.2, 0.8])
    p.add_argument("--group-weights", type=float, nargs="+",
                   default=None, help="per-group loss weights; default equal")
    p.add_argument("--lr", type=float, default=None,
                   help="LR; if None, use paper's scaling law 2e-4/sqrt(d_sae/16384)")
    p.add_argument("--warmup-steps", type=int, default=1000)
    p.add_argument("--decay-start", type=int, default=None,
                   help="Step at which linear decay begins; default int(0.8*steps)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--preload-seqs", type=int, default=6000)
    p.add_argument("--run-name", type=str, default=None,
                   help="Override run id; default tsae_paper__seed{seed}")
    args = p.parse_args()

    t.manual_seed(args.seed)
    np.random.seed(args.seed)

    run_id = args.run_name or f"tsae_paper__seed{args.seed}"
    print(f"=== {run_id} ===")

    # ---- setup -----------------------------------------------------------
    device = t.device("cuda")
    print(f"  [data] preloading {args.preload_seqs} seqs L13 to GPU...")
    t0 = time.time()
    buf = preload_anchor(args.preload_seqs, device)
    print(f"  [data] loaded {buf.shape} ({buf.element_size()*buf.numel()/1e9:.1f}GB) in {time.time()-t0:.1f}s")
    pair_gen = make_pair_gen(buf, args.batch_size)

    # group sizes
    assert abs(sum(args.group_fractions) - 1.0) < 1e-6
    gs = [int(f * args.d_sae) for f in args.group_fractions[:-1]]
    gs.append(args.d_sae - sum(gs))
    gw = args.group_weights or [1.0 / len(gs)] * len(gs)
    print(f"  [model] d_in={D_IN} d_sae={args.d_sae} k={args.k} groups={gs} weights={gw}")

    from src.architectures.tsae_paper import (
        TemporalMatryoshkaBatchTopKSAE,
        TemporalMatryoshkaBatchTopKTrainerLite,
        geometric_median,
    )
    ae = TemporalMatryoshkaBatchTopKSAE(
        D_IN, args.d_sae, k=args.k, group_sizes=gs,
    ).to(device)

    decay_start = args.decay_start or int(0.8 * args.steps)
    lr = args.lr if args.lr is not None else (2e-4 / (args.d_sae / 16384) ** 0.5)
    print(f"  [train] steps={args.steps} lr={lr:.2e} warmup={args.warmup_steps} decay_start={decay_start}")

    trainer = TemporalMatryoshkaBatchTopKTrainerLite(
        ae, group_weights=gw, total_steps=args.steps, lr=lr,
        warmup_steps=args.warmup_steps, decay_start=decay_start,
        device=device,
    )

    # geometric median init for b_dec
    x0 = pair_gen(0)[:, 0]  # (B, d)
    bdec = geometric_median(x0)
    print(f"  [init] b_dec geometric median ({bdec.shape}, norm={float(bdec.norm()):.3f})")

    # ---- train loop ------------------------------------------------------
    losses, auxks, temps, deads, l0s = [], [], [], [], []
    steps_logged = []
    t0 = time.time()
    log_every = 200
    for step in range(args.steps):
        x_pair = pair_gen(step)
        stats = trainer.update(step, x_pair, b_dec_init=bdec if step == 0 else None)
        if step % log_every == 0 or step == args.steps - 1:
            # Measure effective per-token L0 on-the-fly (BatchTopK == avg k)
            with t.no_grad():
                z, active, _ = ae.encode(x_pair[:, 0], return_active=True, use_threshold=False)
                l0 = float((z > 0).sum(dim=-1).float().mean().item())
            losses.append(stats["l2"])
            auxks.append(stats["auxk"])
            temps.append(stats["temp"])
            deads.append(stats["dead"])
            l0s.append(l0)
            steps_logged.append(step)
            elapsed = time.time() - t0
            print(f"  step {step:6d}  l2={stats['l2']:.3f}  aux={stats['auxk']:.3f}  temp={stats['temp']:.3f}  dead={stats['dead']:>6d}  l0={l0:.1f}  [{elapsed:.0f}s]")

    elapsed = time.time() - t0
    print(f"  [done] {elapsed:.1f}s")

    # ---- save ------------------------------------------------------------
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    # Mirror the fp16 save convention of train_primary_archs
    fp16_state = {k: v.to(t.float16) if v.dtype == t.float32 else v
                  for k, v in ae.state_dict().items()}
    meta = dict(
        seed=args.seed, k_pos=args.k, k_win=None, T=2, layer=13,
        d_sae=args.d_sae, group_sizes=gs, group_weights=gw,
        variant="tsae_paper_matryoshka_batchtopk_auxk",
        arch_class="TemporalMatryoshkaBatchTopKSAE",
    )
    t.save(
        {"state_dict": fp16_state, "arch": "tsae_paper",
         "meta": meta, "state_dict_dtype": "float16"},
        ckpt_path,
    )
    print(f"  [save] {ckpt_path}")

    log = {
        "run_id": run_id, "arch": "tsae_paper", **meta,
        "loss": losses, "auxk": auxks, "temp": temps, "dead": deads,
        "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "elapsed_s": elapsed,
    }
    (LOGS_DIR / f"{run_id}.json").write_text(json.dumps(log, indent=2, default=str))

    with INDEX_PATH.open("a") as f:
        row = {"run_id": run_id, "arch": "tsae_paper", **meta,
               "final_step": log["final_step"], "converged": True,
               "final_loss": losses[-1] if losses else None,
               "final_l0": l0s[-1] if l0s else None,
               "elapsed_s": elapsed}
        f.write(json.dumps(row, default=str) + "\n")


if __name__ == "__main__":
    main()
