"""Minimal-AuxK ablation: keep our default training (Adam β1=0.9 β2=0.999,
lr=3e-4, b_dec=0, no tied-init-override, no decoder-parallel grad removal),
add ONLY the AuxK loss.

Isolates whether AuxK by itself is the active ingredient, or whether it
only works bundled with the other Gao recipe components.

Appends results to dead_feature_experiment.json under key "minimal_auxk".

    uv run python -m experiments.em_features.test_minimal_auxk \
        --config experiments/em_features/config.yaml \
        --out_dir /root/em_features/results/dead_feature_experiment \
        --total_steps 20000 --dead_token_threshold 128000
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
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--dead_token_threshold", type=int, default=128000)  # ~500 steps
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--k_aux", type=int, default=512)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--key", default="minimal_auxk",
                   help="JSON key to store results under.")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def encode_pre(crosscoder, x):
    pre = torch.einsum("btd,tdm->bm", x, crosscoder.W_enc) + crosscoder.b_enc
    if getattr(crosscoder, "use_relu", True):
        pre = torch.relu(pre)
    return pre


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())

    print(f"Loading {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
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
    buffer = StreamingActivationBuffer(model, tok, text_iter, buf_cfg)
    print("warmup...", flush=True)
    buffer.warmup()

    torch.manual_seed(42)
    txc = TemporalCrosscoder(d_in=d_model, d_sae=args.d_sae,
                             T=args.T, k_total=args.k_total).to(args.device)
    # Default Adam, default lr — matches what NO_RESAMPLE used.
    optim = torch.optim.Adam(txc.parameters(), lr=args.lr)

    tokens_since_fired = torch.zeros(args.d_sae, dtype=torch.long, device=args.device)
    history: list[dict] = []
    train_t0 = time.time()

    for step in range(args.total_steps):
        x = buffer.sample_txc_windows(args.batch_size, args.T).float()
        pre = encode_pre(txc, x)
        topk_vals, topk_idx = pre.topk(args.k_total, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        x_hat = torch.einsum("bm,tmd->btd", z, txc.W_dec) + txc.b_dec
        loss_main = (x - x_hat).pow(2).sum(dim=-1).mean()

        dead_mask = tokens_since_fired >= args.dead_token_threshold
        n_dead_live = int(dead_mask.sum().item())
        if n_dead_live > 0:
            pre_dead = pre.clone()
            pre_dead[:, ~dead_mask] = -float("inf")
            k_use = min(args.k_aux, n_dead_live)
            top_dead_vals, top_dead_idx = pre_dead.topk(k_use, dim=-1)
            top_dead_vals = torch.where(
                torch.isinf(top_dead_vals) | torch.isnan(top_dead_vals),
                torch.zeros_like(top_dead_vals), top_dead_vals,
            )
            z_aux = torch.zeros_like(pre)
            z_aux.scatter_(-1, top_dead_idx, top_dead_vals)
            x_aux = torch.einsum("bm,tmd->btd", z_aux, txc.W_dec)
            residual = x - x_hat.detach()
            loss_auxk = (residual - x_aux).pow(2).sum(dim=-1).mean()
            loss_auxk_norm = loss_auxk / (loss_main.detach() + 1e-8)
            loss_total = loss_main + args.auxk_alpha * loss_auxk_norm
        else:
            loss_auxk = torch.tensor(0.0, device=args.device)
            loss_total = loss_main

        optim.zero_grad(set_to_none=True)
        loss_total.backward()
        optim.step()

        with torch.no_grad():
            fired = (z != 0).any(dim=0)
            tokens_since_fired += args.batch_size
            tokens_since_fired[fired] = 0

        if (step + 1) % 100 == 0:
            txc.normalize_decoder()

        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                probe = buffer.sample_txc_windows(2048, args.T).float()
                z_probe = txc.encode(probe)
                fire_count = (z_probe != 0).sum(dim=0)
            n_dead_probe = int((fire_count == 0).sum().item())
            history.append({
                "step": step + 1,
                "loss": float(loss_main.detach()),
                "loss_auxk": float(loss_auxk.detach()),
                "loss_total": float(loss_total.detach()),
                "n_dead": n_dead_probe,
                "n_features": args.d_sae,
                "n_active_in_batch": int(fired.sum().item()),
                "max_fire": int(fire_count.max().item()),
                "elapsed_min": (time.time() - train_t0) / 60,
                "n_resampled_so_far": 0,
            })
            print(f"[minimal_auxk] step {step+1:>6}/{args.total_steps}  "
                  f"main={history[-1]['loss']:.1f}  auxk={history[-1]['loss_auxk']:.2f}  "
                  f"dead={n_dead_probe}/{args.d_sae} ({100*n_dead_probe/args.d_sae:.1f}%)  "
                  f"active_in_batch={history[-1]['n_active_in_batch']}", flush=True)

    existing = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(existing.read_text()) if existing.exists() else {}
    data[args.key] = history
    data[f"{args.key}_args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with existing.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"appended {args.key} to {existing}")


if __name__ == "__main__":
    main()
