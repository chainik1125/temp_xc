"""5k-step test of han's TXCBareAntidead — the exact recipe used by the
H8 champion on the han branch, minus matryoshka and contrastive.

Key differences from our earlier Gao attempts:
  - AuxK denominator = variance of residual (not current main loss)
  - _normalize_decoder() every step (not every 100)
  - Grad clip
  - Standard Adam betas
  - Encoder tied to decoder AFTER unit-norm
  - Dead tracker increments by B·T (not B)
  - b_dec init = per-position geometric median, one-shot on first batch

Appends results to dead_feature_experiment.json under "han_antidead".
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

from experiments.em_features.architectures.txc_bare_antidead import TXCBareAntidead  # noqa: E402
from experiments.em_features.streaming_buffer import (  # noqa: E402
    StreamingActivationBuffer, StreamingBufferConfig, mixed_text_iter,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k", type=int, default=128)
    p.add_argument("--aux_k", type=int, default=512)
    # Han's default is 10M, but at 5k*256*5=6.4M total tokens that threshold
    # would never trigger. Scale down for short-run testing; for real training
    # use 10M as in the champion recipe.
    p.add_argument("--dead_threshold_tokens", type=int, default=640_000,
                   help="~500 steps worth of tokens at B=256 T=5.")
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--key", default="han_antidead")
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def probe_dead(txc: TXCBareAntidead, sample_fn, n_check: int = 2048) -> dict:
    x = sample_fn(n_check).to(next(txc.parameters()).device).float()
    z = txc.encode(x)
    fire_count = (z > 0).sum(dim=0)
    n_dead = int((fire_count == 0).sum().item())
    return {
        "n_dead": n_dead,
        "n_features": txc.d_sae,
        "max_fire": int(fire_count.max().item()),
        "median_fire": int(fire_count.median().item()),
    }


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

    def sample_fn(n):
        return buffer.sample_txc_windows(n, args.T).float()

    torch.manual_seed(42)
    txc = TXCBareAntidead(
        d_in=d_model, d_sae=args.d_sae, T=args.T, k=args.k,
        aux_k=args.aux_k,
        dead_threshold_tokens=args.dead_threshold_tokens,
        auxk_alpha=args.auxk_alpha,
    ).to(args.device)

    # Han's step 1: one-shot b_dec init on the first batch.
    x0 = sample_fn(args.batch_size)
    txc.init_b_dec_geometric_median(x0)
    print(f"b_dec initialized from geometric median of {args.batch_size} windows", flush=True)

    opt = torch.optim.Adam(txc.parameters(), lr=args.lr)
    history: list[dict] = []
    t0 = time.time()

    for step in range(args.total_steps):
        x = sample_fn(args.batch_size)
        loss, x_hat, z = txc(x)
        opt.zero_grad()
        loss.backward()
        # Han's ordering: grad projection → grad clip → optim step → normalize.
        txc.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(txc.parameters(), args.grad_clip)
        opt.step()
        txc._normalize_decoder()

        if (step + 1) % args.log_every == 0:
            probe = probe_dead(txc, sample_fn)
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "loss_auxk": float(txc.last_auxk_loss.item()),
                "loss_total": float(loss.detach()),  # han combines into single loss
                "n_dead": probe["n_dead"],
                "n_features": probe["n_features"],
                "n_active_in_batch": int((z > 0).any(dim=0).sum().item()),
                "max_fire": probe["max_fire"],
                "elapsed_min": (time.time() - t0) / 60,
                "n_resampled_so_far": 0,  # han's approach is resample-free
                "dead_tracker_count": int(txc.last_dead_count.item()),
            })
            print(f"[han_antidead] step {step+1:>5}/{args.total_steps}  "
                  f"loss={history[-1]['loss']:.1f}  "
                  f"auxk={history[-1]['loss_auxk']:.4f}  "
                  f"dead(probe)={probe['n_dead']}/{probe['n_features']} "
                  f"({100*probe['n_dead']/probe['n_features']:.1f}%)  "
                  f"dead(tracker)={history[-1]['dead_tracker_count']}  "
                  f"active_in_batch={history[-1]['n_active_in_batch']}  "
                  f"max_fire={probe['max_fire']}", flush=True)

    existing = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(existing.read_text()) if existing.exists() else {}
    data[args.key] = history
    data[f"{args.key}_args"] = {k: (str(v) if isinstance(v, Path) else v)
                                for k, v in vars(args).items()}
    with existing.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"appended {args.key} to {existing}")


if __name__ == "__main__":
    main()
