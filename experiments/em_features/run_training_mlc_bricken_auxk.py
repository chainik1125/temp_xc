"""MLC training with Bricken resampling + EMA-AuxK — same recipe as
``run_training_txc_bricken_auxk.py`` but using ``MultiLayerCrosscoder``
across the configured layer stack instead of a single-layer temporal
window.

    uv run python -m experiments.em_features.run_training_mlc_bricken_auxk \\
        --config experiments/em_features/config.yaml \\
        --out_prefix /root/em_features/checkpoints/qwen_mlc_brickenauxk_a32 \\
        --total_steps 10000 --snapshot_at 5000 10000
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

from sae_day.sae import MultiLayerCrosscoder  # noqa: E402

from experiments.em_features.streaming_buffer import (  # noqa: E402
    MultiLayerStreamingBuffer, MultiLayerBufferConfig, mixed_text_iter,
)
from experiments.em_features.gao_topk_training import (  # noqa: E402
    init_b_dec_geometric_median, remove_decoder_parallel_grad,
    _encode_pre_topk, _decode,
)
from experiments.em_features.dead_feature_resample import DeadFeatureResampler  # noqa: E402
from experiments.em_features.hf_upload import upload_if_enabled  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_prefix", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--snapshot_at", type=int, nargs="+", default=[5000, 10000])
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--bricken_resample_every", type=int, default=500)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--k_aux", type=int, default=512)
    p.add_argument("--dead_token_threshold", type=int, default=128_000)
    p.add_argument("--auxk_ema_decay", type=float, default=0.99)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
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
    layers = list(cfg["layers_mlc"])
    L = len(layers)

    mlc_block = cfg.get("mlc", {})
    buffer_seqs = int(mlc_block.get("buffer_seqs", 2000))
    buf_cfg = MultiLayerBufferConfig(
        layers=layers,
        d_model=d_model,
        buffer_seqs=buffer_seqs,
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = MultiLayerStreamingBuffer(hf_model, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    buffer.warmup()

    def sample_fn(n):
        return buffer.sample_mlc_batch(n).float()

    torch.manual_seed(42)
    mlc = MultiLayerCrosscoder(
        d_in=d_model, d_sae=args.d_sae, L=L, k_total=args.k_total,
    ).to(args.device)

    init_b_dec_geometric_median(mlc, sample_fn, n=8192)
    print("b_dec geom-median initialized", flush=True)

    opt = torch.optim.Adam(mlc.parameters(), lr=args.lr, betas=(0.9, 0.999))

    resampler = DeadFeatureResampler(
        mlc, resample_every=args.bricken_resample_every,
        min_fires=1, n_check=2048,
    )

    tokens_since_fired = torch.zeros(args.d_sae, dtype=torch.long, device=args.device)
    ema_main = None

    snapshots = sorted(set(args.snapshot_at))
    if snapshots[-1] != args.total_steps:
        raise ValueError("last snapshot must equal --total_steps")

    history: list[dict] = []
    best_loss = float("inf")
    train_t0 = time.time()
    next_snap_idx = 0

    for step in range(args.total_steps):
        x = sample_fn(args.batch_size)  # (B, L, d)

        pre = _encode_pre_topk(mlc, x)
        topk_vals, topk_idx = pre.topk(args.k_total, dim=-1)
        z_live = torch.zeros_like(pre)
        z_live.scatter_(-1, topk_idx, topk_vals)
        x_hat_live = _decode(mlc, z_live)
        loss_main = (x - x_hat_live).pow(2).sum(dim=-1).mean()

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
            x_aux = torch.einsum("bm,tmd->btd", z_aux, mlc.W_dec)
            residual = x - x_hat_live.detach()
            loss_auxk = (residual - x_aux).pow(2).sum(dim=-1).mean()

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
        remove_decoder_parallel_grad(mlc)
        opt.step()

        with torch.no_grad():
            fired_this_batch = (z_live != 0).any(dim=0)
            tokens_since_fired += args.batch_size * L
            tokens_since_fired[fired_this_batch] = 0
        if (step + 1) % 100 == 0:
            mlc.normalize_decoder()

        resampler.maybe_resample(step + 1, sample_fn)

        current_step = step + 1
        loss_scalar = float(loss_main.detach())
        best_loss = min(best_loss, loss_scalar)

        if current_step % args.log_every == 0:
            with torch.no_grad():
                probe = sample_fn(2048).to(args.device).float()
                z_probe = mlc.encode(probe)
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
            print(f"[mlc_brickenauxk] step {current_step:>6}/{args.total_steps}  "
                  f"loss={loss_scalar:.1f}  auxk={entry['loss_auxk']:.4f}  "
                  f"dead={entry['n_dead']}/{args.d_sae} "
                  f"({100*entry['n_dead']/args.d_sae:.1f}%)  "
                  f"active={entry['n_active_in_batch']}  "
                  f"resampled={entry['n_resampled_so_far']}", flush=True)

        while next_snap_idx < len(snapshots) and current_step >= snapshots[next_snap_idx]:
            snap_step = snapshots[next_snap_idx]
            ckpt_path = args.out_prefix.with_name(f"{args.out_prefix.name}_step{snap_step}.pt")
            ckpt = {
                "state_dict": mlc.state_dict(),
                "config": {
                    "d_in": d_model, "d_sae": args.d_sae, "L": L,
                    "layers": layers,
                    "k_total": args.k_total,
                    "subject_model": cfg["subject_model"],
                    "steps_trained": snap_step,
                    "training_recipe": "bricken_resample+ema_auxk+geom_median_b_dec",
                },
                "best_loss_so_far": best_loss,
                "cumulative_resamples": sum(h.n_resampled for h in resampler.history),
            }
            torch.save(ckpt, ckpt_path)
            print(f"  snapshot saved: {ckpt_path} (step {snap_step}, best loss {best_loss:.2f})", flush=True)
            upload_if_enabled(ckpt_path, category="mlc")
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
