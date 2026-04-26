"""30k-step TXC training with Bricken resampling + EMA-AuxK — the winning
recipe from the 5k ablation (46.8% dead at step 5k vs 87% baseline).

Snapshots at 5k / 10k / 20k / 30k for dead-fraction + frontier-impact
analysis. Saves checkpoints compatible with run_find_misalignment_features.py
so the em-features α-sweep can consume them.

    uv run python -m experiments.em_features.run_training_txc_bricken_auxk \
        --config experiments/em_features/config.yaml \
        --out_prefix experiments/em_features/checkpoints/qwen_l15_txc_brickenauxk \
        --total_steps 30000 --snapshot_at 5000 10000 20000 30000
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sae_day.sae import TemporalCrosscoder  # noqa: E402

from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
)
from experiments.em_features.gao_topk_training import (  # noqa: E402
    init_b_dec_geometric_median, remove_decoder_parallel_grad, _encode_pre_topk, _decode,
)
from experiments.em_features.dead_feature_resample import DeadFeatureResampler  # noqa: E402
from experiments.em_features.hf_upload import upload_if_enabled  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=30000)
    p.add_argument("--snapshot_at", type=int, nargs="+", default=[5000, 10000, 20000, 30000])
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bricken_resample_every", type=int, default=500)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--k_aux", type=int, default=512)
    # dead threshold scaled so AuxK engages by step ~500
    p.add_argument("--dead_token_threshold", type=int, default=640_000)
    p.add_argument("--auxk_ema_decay", type=float, default=0.99)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--resume_from", type=Path, default=None,
                   help="Resume from a snapshot ckpt (loads state_dict + start_step from config['steps_trained']). Adam resets.")
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

    torch.manual_seed(42)
    txc = TemporalCrosscoder(
        d_in=d_model, d_sae=args.d_sae, T=args.T, k_total=args.k_total,
    ).to(args.device)

    start_step = 0
    rckpt = None
    if args.resume_from is not None and args.resume_from.exists():
        rckpt = torch.load(args.resume_from, map_location=args.device, weights_only=False)
        txc.load_state_dict(rckpt["state_dict"])
        start_step = int(rckpt["config"].get("steps_trained", 0))
    else:
        # Geom-median b_dec init (the one uncontroversial win from han's stack).
        init_b_dec_geometric_median(txc, sample_fn, n=8192)
        print("b_dec geom-median initialized", flush=True)

    opt = torch.optim.Adam(txc.parameters(), lr=args.lr, betas=(0.9, 0.999))
    if rckpt is not None:
        if "optimizer_state" in rckpt:
            opt.load_state_dict(rckpt["optimizer_state"])
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam state restored)", flush=True)
        else:
            print(f"  resumed from {args.resume_from} at step {start_step} (Adam fresh — no optimizer_state in ckpt)", flush=True)

    resampler = DeadFeatureResampler(
        txc, resample_every=args.bricken_resample_every,
        min_fires=1, n_check=2048,
    )

    tokens_since_fired = torch.zeros(args.d_sae, dtype=torch.long, device=args.device)
    ema_main = None

    snapshots = sorted(set(args.snapshot_at))
    if snapshots[-1] != args.total_steps:
        raise ValueError(f"last snapshot must equal --total_steps")

    history: list[dict] = []
    best_loss = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0
    while next_snap_idx < len(snapshots) and snapshots[next_snap_idx] <= start_step:
        next_snap_idx += 1

    for step in range(start_step, args.total_steps):
        x = sample_fn(args.batch_size)

        # Forward (live) + main recon loss.
        pre = _encode_pre_topk(txc, x)
        topk_vals, topk_idx = pre.topk(args.k_total, dim=-1)
        z_live = torch.zeros_like(pre)
        z_live.scatter_(-1, topk_idx, topk_vals)
        x_hat_live = _decode(txc, z_live)
        loss_main = (x - x_hat_live).pow(2).sum(dim=-1).mean()

        # AuxK: top-k_aux dead features' reconstruction of residual.
        dead_mask = tokens_since_fired >= args.dead_token_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead > 0:
            pre_dead = pre.clone()
            pre_dead[:, ~dead_mask] = -float("inf")
            k_use = min(args.k_aux, n_dead)
            top_dead_vals, top_dead_idx = pre_dead.topk(k_use, dim=-1)
            top_dead_vals = torch.where(
                torch.isinf(top_dead_vals) | torch.isnan(top_dead_vals),
                torch.zeros_like(top_dead_vals), top_dead_vals,
            )
            z_aux = torch.zeros_like(pre)
            z_aux.scatter_(-1, top_dead_idx, top_dead_vals)
            x_aux = torch.einsum("bm,tmd->btd", z_aux, txc.W_dec)
            residual = x - x_hat_live.detach()
            loss_auxk = (residual - x_aux).pow(2).sum(dim=-1).mean()

            # EMA-normalized AuxK (stable across training).
            if ema_main is None:
                ema_main = float(loss_main.detach())
            else:
                ema_main = args.auxk_ema_decay * ema_main + (1 - args.auxk_ema_decay) * float(loss_main.detach())
            loss_auxk_norm = loss_auxk / (ema_main + 1e-8)
            loss_total = loss_main + args.auxk_alpha * loss_auxk_norm
        else:
            if ema_main is None:
                ema_main = float(loss_main.detach())
            else:
                ema_main = args.auxk_ema_decay * ema_main + (1 - args.auxk_ema_decay) * float(loss_main.detach())
            loss_auxk = torch.tensor(0.0, device=args.device)
            loss_total = loss_main

        opt.zero_grad(set_to_none=True)
        loss_total.backward()
        remove_decoder_parallel_grad(txc)
        opt.step()

        # Bookkeeping + decoder renormalize (every 100 steps, matches our earlier setting).
        with torch.no_grad():
            fired_this_batch = (z_live != 0).any(dim=0)
            tokens_since_fired += args.batch_size * args.T  # han's convention — real tokens
            tokens_since_fired[fired_this_batch] = 0
        if (step + 1) % 100 == 0:
            txc.normalize_decoder()

        # Bricken resample every resample_every steps.
        resampler.maybe_resample(step + 1, sample_fn)

        current_step = step + 1
        loss_scalar = float(loss_main.detach())
        best_loss = min(best_loss, loss_scalar)

        if current_step % args.log_every == 0:
            with torch.no_grad():
                probe = sample_fn(2048).to(args.device).float()
                z_probe = txc.encode(probe)
                fire_count = (z_probe != 0).sum(dim=0)
            entry = {
                "step": current_step,
                "loss": loss_scalar,
                "loss_auxk": float(loss_auxk.detach()),
                "loss_total": float(loss_total.detach()),
                "n_dead": int((fire_count == 0).sum().item()),
                "n_features": args.d_sae,
                "n_active_in_batch": int(fired_this_batch.sum().item()),
                "max_fire": int(fire_count.max().item()),
                "elapsed_min": (time.time() - train_t0) / 60,
                "n_resampled_so_far": sum(h.n_resampled for h in resampler.history),
                "ema_main": ema_main,
            }
            history.append(entry)
            print(f"[brickenauxk] step {current_step:>6}/{args.total_steps}  "
                  f"loss={loss_scalar:.1f}  auxk={entry['loss_auxk']:.4f}  "
                  f"dead={entry['n_dead']}/{args.d_sae} "
                  f"({100*entry['n_dead']/args.d_sae:.1f}%)  "
                  f"active={entry['n_active_in_batch']}  "
                  f"resampled={entry['n_resampled_so_far']}", flush=True)

        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            ckpt = {
                "state_dict": txc.state_dict(),
                "optimizer_state": opt.state_dict(),
                "config": {
                    "d_in": d_model, "d_sae": args.d_sae, "T": args.T,
                    "k_total": args.k_total,
                    "subject_model": cfg["subject_model"],
                    "layer": int(cfg["layer_txc"]),
                    "steps_trained": snap_step,
                    "training_recipe": "bricken_resample+ema_auxk+geom_median_b_dec",
                },
                "best_loss_so_far": best_loss,
                "cumulative_resamples": sum(h.n_resampled for h in resampler.history),
            }
            torch.save(ckpt, ckpt_path)
            print(f"  snapshot saved: {ckpt_path} (step {snap_step}, best loss {best_loss:.2f})", flush=True)
            upload_if_enabled(ckpt_path, category="txc")
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
