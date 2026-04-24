"""Train a TemporalCrosscoder once to --total_steps, writing intermediate
checkpoints at each ``--snapshot_at`` step. This is the tool for the
"how does TXC quality scale with d_sae and training length" deep-dive in
the em_features pipeline — a single training run produces 40k / 100k /
300k checkpoints.

    uv run python -m experiments.em_features.run_training_txc_snapshots \
        --config experiments/em_features/config.yaml \
        --d_sae 65536 --T 5 --k_total 128 \
        --total_steps 300000 --snapshot_at 40000 100000 300000 \
        --out_prefix experiments/em_features/checkpoints/qwen_l15_txc_big

Produces qwen_l15_txc_big_step40000.pt, qwen_l15_txc_big_step100000.pt,
qwen_l15_txc_big_step300000.pt.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import TemporalCrosscoder  # noqa: E402

from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer,
    StreamingBufferConfig,
    mixed_text_iter,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path, required=True)
    p.add_argument("--d_sae", type=int, required=True)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--total_steps", type=int, required=True)
    p.add_argument("--snapshot_at", type=int, nargs="+", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_every", type=int, default=200)
    p.add_argument("--normalize_every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    snapshots = sorted(set(args.snapshot_at))
    if snapshots[-1] != args.total_steps:
        raise ValueError(
            f"last snapshot ({snapshots[-1]}) must equal --total_steps ({args.total_steps})"
        )

    print(f"Loading subject model: {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    assert model.config.hidden_size == d_model

    buf_cfg = StreamingBufferConfig(
        layer=int(cfg["layer_txc"]),
        d_model=d_model,
        buffer_seqs=max(1, int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"])),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    print(f"Buffer: {buf_cfg.buffer_seqs} seqs × {buf_cfg.chunk_len} toks "
          f"≈ {buf_cfg.buffer_seqs*buf_cfg.chunk_len*d_model*2/1e9:.1f} GB VRAM", flush=True)

    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = StreamingActivationBuffer(model, tok, text_iter, buf_cfg)

    print("warmup...", flush=True)
    t0 = time.time()
    buffer.warmup()
    print(f"  warmup done in {time.time()-t0:.1f}s", flush=True)

    txc = TemporalCrosscoder(
        d_in=d_model, d_sae=args.d_sae, T=args.T, k_total=args.k_total,
    ).to(args.device)
    print(f"TXC params: d_in={d_model}  d_sae={args.d_sae}  T={args.T}  k_total={args.k_total}", flush=True)
    n_params = sum(p.numel() for p in txc.parameters())
    print(f"  → {n_params/1e6:.1f} M parameters "
          f"(~{n_params*4/1e9:.1f} GB fp32 params, ~{n_params*4*4/1e9:.1f} GB with Adam fp32 state)",
          flush=True)

    optim = torch.optim.Adam(txc.parameters(), lr=args.lr)
    loss_history: list[float] = []
    l0_history: list[tuple[int, float]] = []
    best = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0

    for step in range(args.total_steps):
        x = buffer.sample_txc_windows(args.batch_size, args.T).float()
        x_hat, z = txc(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % args.normalize_every == 0:
            txc.normalize_decoder()
        loss_history.append(float(loss.detach()))
        best = min(best, loss_history[-1])
        current_step = step + 1
        if current_step % args.log_every == 0:
            with torch.no_grad():
                l0 = (z != 0).float().sum(dim=-1).mean().item()
            l0_history.append((current_step, l0))
            elapsed = time.time() - train_t0
            print(f"[train] step {current_step:>7}/{args.total_steps}  "
                  f"loss={loss_history[-1]:.4f}  L0={l0:.2f}  elapsed={elapsed/60:.1f}m", flush=True)
        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            ckpt = {
                "state_dict": txc.state_dict(),
                "config": {
                    "d_in": d_model, "d_sae": args.d_sae, "T": args.T,
                    "k_total": args.k_total,
                    "subject_model": cfg["subject_model"],
                    "layer": int(cfg["layer_txc"]),
                    "steps_trained": snap_step,
                },
                "loss_history_snapshot_len": len(loss_history),
                "best_loss_so_far": best,
            }
            torch.save(ckpt, ckpt_path)
            print(f"  snapshot saved: {ckpt_path}  (step {snap_step}, best loss {best:.4f})", flush=True)
            next_snap_idx += 1

    # Also write the full loss history next to the last snapshot.
    meta_path = args.out_prefix.with_name(f"{args.out_prefix.name}_training.meta.json")
    with meta_path.open("w") as f:
        json.dump({
            "d_sae": args.d_sae, "T": args.T, "k_total": args.k_total,
            "total_steps": args.total_steps, "snapshot_at": snapshots,
            "best_loss": best, "final_loss": loss_history[-1],
            "l0_history": l0_history,
            "loss_history": loss_history,
        }, f)
    print(f"wrote {meta_path}  (best loss={best:.4f})", flush=True)


if __name__ == "__main__":
    main()
