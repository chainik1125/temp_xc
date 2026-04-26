"""10k-step training of the Han multi-distance H8 champion architecture
(``TXCBareMultiDistanceContrastiveAntidead``) with snapshots compatible
with frontier_sweep's ``--steerer han`` path.

H8 recipe: bare TXC + matryoshka H/L (h-prefix = d_sae/5) + multi-distance
InfoNCE (shifts {1, 2}) + anti-dead stack.

    uv run python -m experiments.em_features.run_training_han_champion \\
        --config experiments/em_features/config.yaml \\
        --out_prefix /root/em_features/checkpoints/qwen_l15_han_champ_10k \\
        --total_steps 10000 --snapshot_at 5000 10000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)
from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
)
from experiments.em_features.hf_upload import upload_if_enabled  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--snapshot_at", type=int, nargs="+", default=[5000, 10000])
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--aux_k", type=int, default=512)
    p.add_argument("--dead_threshold_tokens", type=int, default=640_000)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--alpha_contrastive", type=float, default=1.0)
    p.add_argument("--shifts", type=int, nargs="+", default=[1, 2])
    p.add_argument("--matryoshka_h_div", type=int, default=5,
                   help="h-prefix size = d_sae // matryoshka_h_div (H8 default = 5)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume_from", type=Path, default=None,
                   help="Resume from a snapshot ckpt (loads state_dict + start_step). Adam resets.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())

    print(f"Loading {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    buf_cfg = StreamingBufferConfig(
        layer=int(cfg["layer_txc"]),
        d_model=d_model,
        buffer_seqs=max(1, int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"])),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = StreamingActivationBuffer(hf_model, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    buffer.warmup()

    def sample_fn(n):
        return buffer.sample_txc_windows(n, args.T).float()

    max_shift = max(args.shifts)
    assert buf_cfg.chunk_len >= args.T + max_shift, \
        f"chunk_len={buf_cfg.chunk_len} < T+max_shift={args.T + max_shift}"

    def pair_sample_fn(batch_size):
        """Returns (B, 1+K, T, d) where K = len(shifts). Anchor at shift=0 first."""
        buf = buffer._buf  # (N_seqs, chunk_len, d_model)
        N, L, _ = buf.shape
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, L - args.T - max_shift, (batch_size,), device=buf.device)
        rng = torch.arange(args.T, device=buf.device)
        outs = []
        for s in (0, *args.shifts):
            pos = (off + s).unsqueeze(1) + rng.unsqueeze(0)
            w = buf[seq.unsqueeze(1).expand(-1, args.T), pos].float()
            outs.append(w)
        return torch.stack(outs, dim=1)

    torch.manual_seed(42)
    matryoshka_h_size = args.d_sae // args.matryoshka_h_div
    model = TXCBareMultiDistanceContrastiveAntidead(
        d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
        shifts=tuple(args.shifts),
        matryoshka_h_size=matryoshka_h_size,
        alpha=args.alpha_contrastive,
        aux_k=args.aux_k,
        dead_threshold_tokens=args.dead_threshold_tokens,
        auxk_alpha=args.auxk_alpha,
    ).to(args.device)
    start_step = 0
    if args.resume_from is not None and args.resume_from.exists():
        rckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        model.load_state_dict(rckpt["state_dict"])
        start_step = int(rckpt["config"].get("steps_trained", 0))
        print(f"  resumed from {args.resume_from} at step {start_step} (Adam reset, skipping b_dec init)", flush=True)
    else:
        x0 = sample_fn(args.batch_size)
        model.init_b_dec_geometric_median(x0)
    print(f"H8 champion: shifts={args.shifts}  h_size={matryoshka_h_size}  α_c={args.alpha_contrastive}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    snapshots = sorted(set(args.snapshot_at))
    if snapshots[-1] != args.total_steps:
        raise ValueError("last snapshot must equal --total_steps")

    history: list[dict] = []
    best_loss = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0
    while next_snap_idx < len(snapshots) and snapshots[next_snap_idx] <= start_step:
        next_snap_idx += 1

    for step in range(start_step, args.total_steps):
        x_train = pair_sample_fn(args.batch_size)
        loss, x_hat, z = model(x_train)

        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()

        loss_scalar = float(loss.detach())
        best_loss = min(best_loss, loss_scalar)

        current_step = step + 1
        if current_step % args.log_every == 0:
            with torch.no_grad():
                probe = sample_fn(2048).to(args.device).float()
                z_probe = model.encode(probe)
                fire_count = (z_probe > 0).sum(dim=0)
            entry = {
                "step": current_step,
                "loss": loss_scalar,
                "loss_auxk": float(getattr(model, "last_auxk_loss", torch.tensor(0.0)).item()),
                "n_dead": int((fire_count == 0).sum().item()),
                "n_features": args.d_sae,
                "max_fire": int(fire_count.max().item()),
                "elapsed_min": (time.time() - train_t0) / 60,
            }
            history.append(entry)
            print(f"[han_champ] step {current_step:>5}/{args.total_steps}  "
                  f"loss={loss_scalar:.1f}  auxk={entry['loss_auxk']:.4f}  "
                  f"dead={entry['n_dead']}/{args.d_sae} ({100*entry['n_dead']/args.d_sae:.1f}%)  "
                  f"elapsed={entry['elapsed_min']:.1f}m", flush=True)

        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            ckpt = {
                "state_dict": model.state_dict(),
                "config": {
                    "d_in": d_model, "d_sae": args.d_sae, "T": args.T,
                    "k": args.k, "aux_k": args.aux_k,
                    "shifts": list(args.shifts),
                    "matryoshka_h_size": matryoshka_h_size,
                    "alpha_contrastive": args.alpha_contrastive,
                    "auxk_alpha": args.auxk_alpha,
                    "dead_threshold_tokens": args.dead_threshold_tokens,
                    "subject_model": cfg["subject_model"],
                    "layer": int(cfg["layer_txc"]),
                    "steps_trained": snap_step,
                    "training_recipe": "han_multidistance_champion_h8",
                },
                "best_loss_so_far": best_loss,
            }
            torch.save(ckpt, ckpt_path)
            print(f"  snapshot saved: {ckpt_path} (step {snap_step}, best loss {best_loss:.2f})", flush=True)
            upload_if_enabled(ckpt_path, category="han_champ")
            next_snap_idx += 1

    meta_path = args.out_prefix.with_name(f"{args.out_prefix.name}_training.meta.json")
    with meta_path.open("w") as f:
        json.dump({
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "history": history,
        }, f, indent=2)
    print(f"wrote {meta_path}  (best loss={best_loss:.4f})", flush=True)


if __name__ == "__main__":
    main()
