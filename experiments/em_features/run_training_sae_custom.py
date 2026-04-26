"""Train a TopK SAE on streaming Qwen-7B activations at one hookpoint.

    uv run python -m experiments.em_features.run_training_sae_custom \
        --config experiments/em_features/config.yaml \
        --hookpoint resid_mid --layer 15 \
        --out experiments/em_features/checkpoints/qwen_l15_sae_resid_mid_k128.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
if str(VENDOR_SRC) not in sys.path:
    sys.path.insert(0, str(VENDOR_SRC))

from sae_day.sae import TopKSAE  # noqa: E402

from experiments.em_features.streaming_buffer import (  # noqa: E402
    HookpointStreamingBuffer,
    HookpointBufferConfig,
    mixed_text_iter,
    VALID_HOOKPOINTS,
)
from experiments.em_features.hf_upload import upload_if_enabled  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--hookpoint", choices=VALID_HOOKPOINTS, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--out", type=Path, required=False, help="Single-output mode (legacy)")
    p.add_argument("--out_prefix", type=Path, required=False,
                   help="Snapshot mode: writes <prefix>_step<N>.pt for each --snapshot_at")
    p.add_argument("--total_steps", type=int, default=None,
                   help="Override config sae.steps (or txc.steps if no sae block)")
    p.add_argument("--snapshot_at", type=int, nargs="*", default=None,
                   help="Step counts to write checkpoints at (requires --out_prefix)")
    p.add_argument("--d_sae", type=int, default=None, help="Override config")
    p.add_argument("--k", type=int, default=None, help="Override config k_total")
    p.add_argument("--batch_size", type=int, default=None, help="Override config")
    p.add_argument("--lr", type=float, default=None, help="Override config")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--resume_from", type=Path, default=None,
                   help="Resume from a snapshot ckpt (loads state_dict + start_step from config['steps_trained']). Adam resets.")
    args = p.parse_args()
    if not args.out and not args.out_prefix:
        p.error("must pass --out (single output) or --out_prefix (snapshot mode)")
    if args.snapshot_at and not args.out_prefix:
        p.error("--snapshot_at requires --out_prefix")
    return args


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out_prefix is not None:
        args.out_prefix.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading subject model: {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    # Use the TXC sub-config as default SAE training hyperparams to match scale.
    sae_cfg = cfg.get("sae_custom", cfg["txc"])
    d_sae = args.d_sae if args.d_sae is not None else int(sae_cfg["d_sae"])
    k = args.k if args.k is not None else int(sae_cfg["k_total"])
    batch_size = args.batch_size if args.batch_size is not None else int(sae_cfg["batch_size"])
    lr = args.lr if args.lr is not None else float(sae_cfg["lr"])
    n_steps = args.total_steps if args.total_steps is not None else int(sae_cfg["steps"])
    snapshots = sorted(set(args.snapshot_at)) if args.snapshot_at else []
    if snapshots and snapshots[-1] != n_steps:
        raise ValueError(f"last snapshot ({snapshots[-1]}) must equal --total_steps ({n_steps})")
    buf_cfg = HookpointBufferConfig(
        hookpoint=args.hookpoint,
        layer=args.layer,
        d_model=d_model,
        buffer_seqs=int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"]),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    print(f"Buffer: {buf_cfg.buffer_seqs} seqs × {buf_cfg.chunk_len} toks @ fp16 — hookpoint {args.hookpoint} L{args.layer}",
          flush=True)

    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = HookpointStreamingBuffer(model, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    t0 = time.time()
    buffer.warmup()
    print(f"  warmup done in {time.time()-t0:.1f}s", flush=True)

    sae = TopKSAE(
        d_in=d_model,
        d_sae=d_sae,
        k=k,
    ).to(args.device)

    start_step = 0
    rckpt = None
    if args.resume_from is not None and args.resume_from.exists():
        rckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        sae.load_state_dict(rckpt["state_dict"])
        start_step = int(rckpt["config"].get("steps_trained", 0))

    optim = torch.optim.Adam(sae.parameters(), lr=lr)
    if rckpt is not None:
        if "optimizer_state" in rckpt:
            optim.load_state_dict(rckpt["optimizer_state"])
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam state restored)", flush=True)
        else:
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam fresh — no optimizer_state in ckpt)", flush=True)
        # Restore all RNG states for bit-exact reproducibility (chunked == continuous).
        if "rng_state" in rckpt:
            rs = rckpt["rng_state"]
            torch.set_rng_state(rs["cpu"].cpu() if hasattr(rs["cpu"], "cpu") else rs["cpu"])
            if torch.cuda.is_available():
                torch.cuda.set_rng_state_all([s.cpu() if hasattr(s, "cpu") else s for s in rs["cuda"]])
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            print("  RNG states restored", flush=True)

    loss_history: list[float] = []
    l0_history: list[tuple[int, float]] = []
    best = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0
    while next_snap_idx < len(snapshots) and snapshots[next_snap_idx] <= start_step:
        next_snap_idx += 1

    def write_ckpt(path: Path, step_done: int) -> None:
        ckpt = {
            "state_dict": sae.state_dict(),
            "optimizer_state": optim.state_dict(),
            "rng_state": {
                "cpu": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                "python": random.getstate(),
                "numpy": np.random.get_state(),
            },
            "config": {
                "d_in": d_model,
                "d_sae": d_sae,
                "k": k,
                "hookpoint": args.hookpoint,
                "layer": args.layer,
                "subject_model": cfg["subject_model"],
                "steps_trained": step_done,
                "training_recipe": "topk_sae_arditi-style",
            },
            "loss_history": loss_history,
            "l0_history": l0_history,
            "best_loss": best,
        }
        torch.save(ckpt, path)
        with path.with_suffix(".meta.json").open("w") as f:
            json.dump({kk: vv for kk, vv in ckpt.items() if kk != "state_dict"}, f, indent=2)
        print(f"Saved {path}  (step {step_done}, best loss={best:.6f})", flush=True)
        upload_if_enabled(path, category="sae")

    for step in range(start_step, n_steps):
        x = buffer.sample_flat(batch_size).float()
        x_hat, z = sae(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % 100 == 0 and hasattr(sae, "normalize_decoder"):
            sae.normalize_decoder()
        loss_history.append(float(loss.detach()))
        best = min(best, loss_history[-1])
        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                probe = buffer.sample_flat(2048).float()
                _, z_probe = sae(probe)
                fire_count = (z_probe != 0).sum(dim=0)
                n_dead = int((fire_count == 0).sum().item())
                l0 = (z_probe != 0).float().sum(dim=-1).mean().item()
            l0_history.append((step + 1, l0))
            elapsed = time.time() - train_t0
            print(f"[sae] step {step+1:>6}/{n_steps}  loss={loss_history[-1]:.4f}  "
                  f"L0={l0:.2f}  dead={n_dead}/{d_sae} ({100*n_dead/d_sae:.1f}%)  "
                  f"elapsed={elapsed/60:.1f}m", flush=True)

        current_step = step + 1
        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            write_ckpt(ckpt_path, snap_step)
            next_snap_idx += 1

    if not snapshots:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_ckpt(args.out, n_steps)


if __name__ == "__main__":
    main()
