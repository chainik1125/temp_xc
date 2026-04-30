"""Train a T-SAE (Bhalla et al. 2025, arXiv:2511.05541) on streaming
base-Qwen-7B-Instruct activations.

Per-token TopK SAE with an adjacent-token contrastive loss. Same training
infrastructure as ``run_training_han_champion.py`` (streaming buffer,
windowed batches of size T) so the contrastive loss has consecutive tokens
to compare; per-token encode/decode at inference is identical to a
plain TopKSAE.

Resume-capable: saves optimizer state + RNG state into each snapshot.

    uv run python -m experiments.em_features.run_training_tsae \\
        --config experiments/em_features/config.yaml \\
        --out_prefix /root/em_features/checkpoints/qwen_l15_tsae_k128 \\
        --total_steps 30000 --snapshot_at 10000 20000 30000 \\
        --d_sae 32768 --k 128 --T 5 --contrastive_alpha 1.0 \\
        --batch_size 256 --lr 3e-4 --layer 15
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
    HookpointStreamingBuffer, HookpointBufferConfig, VALID_HOOKPOINTS,
)
from experiments.em_features.architectures.tsae_adjacent_contrastive import (  # noqa: E402
    TSAEAdjacentContrastive,
)
from experiments.em_features.architectures.windowed_tsae import (  # noqa: E402
    WindowedTSAE,
)
from experiments.em_features.hf_upload import upload_if_enabled  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path, required=True)
    p.add_argument("--total_steps", type=int, required=True)
    p.add_argument("--snapshot_at", type=int, nargs="+", required=True)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--T", type=int, default=5,
                   help="Window length used for the contrastive loss (uses T-1 adjacent pairs).")
    p.add_argument("--contrastive_alpha", type=float, default=1.0,
                   help="Bhalla 2025 paper default: 0.1")
    p.add_argument("--batch_topk", action="store_true",
                   help="Use BatchTopK during training (Bhalla 2025 paper default).")
    p.add_argument("--arch", choices=["tsae", "windowed_tsae"], default="tsae",
                   help="Architecture: tsae (per-token + adjacency contrastive, "
                        "default) or windowed_tsae (lifts T-SAE to T-token window "
                        "with optional cross-position mixing).")
    p.add_argument("--mix_positions", action="store_true",
                   help="(windowed_tsae only) Learn a (T, T) cross-position mixing matrix.")
    p.add_argument("--n_temporal_features", type=int, default=None,
                   help="(windowed_tsae only) First N features participate in the "
                        "adjacency contrastive loss (Bhalla 2025 matryoshka split). "
                        "Default: all d_sae.")
    p.add_argument("--auxk_alpha", type=float, default=1.0/32.0)
    p.add_argument("--aux_k", type=int, default=512)
    p.add_argument("--dead_threshold_tokens", type=int, default=640_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--layer", type=int, default=15)
    p.add_argument("--hookpoint", choices=list(VALID_HOOKPOINTS), default="resid_post",
                   help="Where in the transformer block to extract activations.")
    p.add_argument("--resume_from", type=Path, default=None)
    p.add_argument("--upload_category", default="tsae")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading subject model: {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model_lm = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    buf_cfg = HookpointBufferConfig(
        hookpoint=args.hookpoint,
        layer=args.layer,
        d_model=d_model,
        buffer_seqs=max(1, int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"])),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    print(f"Buffer: {buf_cfg.buffer_seqs} seqs × {buf_cfg.chunk_len} toks @ fp16  hookpoint={args.hookpoint} layer={args.layer}", flush=True)

    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = HookpointStreamingBuffer(model_lm, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    t0 = time.time()
    buffer.warmup()
    print(f"  warmup done in {time.time()-t0:.1f}s", flush=True)

    sample_fn = buffer.sample_flat                              # (B, d) — for probing
    pair_sample_fn = lambda B: buffer.sample_txc_windows(B, args.T)  # (B, T, d) — windowed for contrastive

    if args.arch == "tsae":
        sae = TSAEAdjacentContrastive(
            d_in=d_model, d_sae=args.d_sae, k=args.k,
            contrastive_alpha=args.contrastive_alpha,
            batch_topk=args.batch_topk,
            aux_k=args.aux_k,
            dead_threshold_tokens=args.dead_threshold_tokens,
            auxk_alpha=args.auxk_alpha,
        ).to(args.device)
        print(f"T-SAE: d_sae={args.d_sae} k={args.k} T={args.T} "
              f"α_contrast={args.contrastive_alpha} batch_topk={args.batch_topk}",
              flush=True)
    else:  # windowed_tsae
        sae = WindowedTSAE(
            d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
            contrastive_alpha=args.contrastive_alpha,
            n_temporal_features=args.n_temporal_features,
            mix_positions=args.mix_positions,
            aux_k=args.aux_k,
            dead_threshold_tokens=args.dead_threshold_tokens,
            auxk_alpha=args.auxk_alpha,
        ).to(args.device)
        print(f"Windowed-T-SAE: d_sae={args.d_sae} k={args.k} T={args.T} "
              f"α_contrast={args.contrastive_alpha} mix_positions={args.mix_positions} "
              f"n_temporal_features={sae.n_temporal_features}",
              flush=True)

    start_step = 0
    rckpt = None
    if args.resume_from is not None and args.resume_from.exists():
        rckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        sae.load_state_dict(rckpt["state_dict"])
        start_step = int(rckpt.get("steps_trained", 0))

    opt = torch.optim.Adam(sae.parameters(), lr=args.lr)
    if rckpt is not None:
        if "optimizer_state" in rckpt:
            opt.load_state_dict(rckpt["optimizer_state"])
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam state restored)", flush=True)
        else:
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam fresh)", flush=True)
        if "rng_state" in rckpt:
            rs = rckpt["rng_state"]
            try:
                torch.set_rng_state(rs["cpu"].cpu() if hasattr(rs["cpu"], "cpu") else rs["cpu"])
                if torch.cuda.is_available() and rs.get("cuda"):
                    torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in rs["cuda"]])
                random.setstate(rs["python"])
                np.random.set_state(rs["numpy"])
                print("  RNG states restored", flush=True)
            except Exception as e:
                print(f"  warning: RNG restore failed ({e}); continuing", flush=True)

    snapshots = sorted(set(args.snapshot_at))
    if snapshots[-1] != args.total_steps:
        raise ValueError(f"last snapshot ({snapshots[-1]}) must equal --total_steps ({args.total_steps})")

    history: list[dict] = []
    best_loss = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0
    while next_snap_idx < len(snapshots) and snapshots[next_snap_idx] <= start_step:
        next_snap_idx += 1

    for step in range(start_step, args.total_steps):
        windows = pair_sample_fn(args.batch_size)  # (B, T, d_model)
        out = sae.training_loss(windows.float())
        loss = out["total_loss"]
        z = out["z"]

        opt.zero_grad()
        loss.backward()
        if hasattr(sae, "remove_gradient_parallel_to_decoder"):
            sae.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(sae.parameters(), args.grad_clip)
        opt.step()
        with torch.no_grad():
            sae._normalize_decoder_()
            sae.update_dead_counter(z.detach())

        loss_scalar = float(loss.detach())
        best_loss = min(best_loss, loss_scalar)

        current_step = step + 1
        if current_step % args.log_every == 0:
            with torch.no_grad():
                probe = sample_fn(2048).to(args.device).float()
                z_probe = sae.encode(probe)
                fire_count = (z_probe > 0).sum(dim=0)
            print(
                f"[tsae] step {current_step:>5}/{args.total_steps}  "
                f"loss={loss_scalar:.4f}  recon={float(out['recon_loss']):.4f}  "
                f"contrast={float(out['contrast_loss']):.4f}  "
                f"dead={int((fire_count == 0).sum().item())}/{args.d_sae}  "
                f"elapsed={(time.time()-train_t0)/60:.1f}m",
                flush=True,
            )

        # Snapshots
        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            ckpt = {
                "state_dict": sae.state_dict(),
                "optimizer_state": opt.state_dict(),
                "rng_state": {
                    "cpu": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    "python": random.getstate(),
                    "numpy": np.random.get_state(),
                },
                "steps_trained": current_step,
                "config": {
                    "arch": ("tsae_adjacent_contrastive" if args.arch == "tsae"
                             else "windowed_tsae"),
                    "d_in": d_model, "d_sae": args.d_sae, "k": args.k, "T": args.T,
                    "contrastive_alpha": args.contrastive_alpha,
                    "batch_topk": args.batch_topk,
                    "mix_positions": args.mix_positions if args.arch == "windowed_tsae" else False,
                    "n_temporal_features": (sae.n_temporal_features if args.arch == "windowed_tsae" else None),
                    "aux_k": args.aux_k, "auxk_alpha": args.auxk_alpha,
                    "dead_threshold_tokens": args.dead_threshold_tokens,
                    "subject_model": cfg["subject_model"],
                    "layer": args.layer, "hookpoint": args.hookpoint,
                    "lr": args.lr, "batch_size": args.batch_size,
                    "best_loss": best_loss,
                },
            }
            torch.save(ckpt, ckpt_path)
            print(f"  Saved {ckpt_path}  (step {snap_step}, best loss={best_loss:.6f})", flush=True)
            try:
                upload_if_enabled(ckpt_path, category=args.upload_category)
            except Exception as e:
                print(f"  hf_upload failed: {e}", flush=True)
            next_snap_idx += 1

    print(f"\nTraining done after {args.total_steps} steps. Best loss={best_loss:.6f}", flush=True)


if __name__ == "__main__":
    main()
