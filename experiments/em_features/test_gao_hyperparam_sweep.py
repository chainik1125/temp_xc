"""Three 5k-step conditions back-to-back to probe which Gao hyperparam is
actually the bottleneck.

  (A) Gao default ("gao_default"): α=1/32, per-step norm (current buggy v1).
  (B) EMA-normalized AuxK ("gao_ema_high_alpha"): α=0.25, EMA-of-main-loss
      normalization, k_aux=n_dead (cap), dead_threshold=12800 (50 steps).
  (C) Bricken + AuxK combined ("gao_ema_plus_bricken"): EMA-norm AuxK
      (α=1/32) + Bricken resample every 500 steps.

All conditions: d_sae=32k, T=5, k=128, 5000 steps, same streaming buffer
(so identical data distribution across conditions). Results appended to
dead_feature_experiment.json so the comparison plot picks them all up.
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
from experiments.em_features.gao_topk_training import train_gao_topk, GaoStats  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def history_to_dicts(history: list[GaoStats]) -> list[dict]:
    return [dict(step=s.step, loss=s.loss_main, loss_auxk=s.loss_auxk,
                 loss_total=s.loss_total, n_dead=s.n_dead,
                 n_features=s.n_features,
                 n_active_in_batch=s.n_active_this_batch,
                 max_fire=s.max_fire_count,
                 elapsed_min=s.elapsed_min,
                 n_resampled_so_far=s.extra.get("n_resampled_so_far", 0))
            for s in history]


def run_condition(name, sample_fn, args, d_model, **train_kwargs):
    print(f"\n================= {name} =================", flush=True)
    print(f"   kwargs: {train_kwargs}", flush=True)
    torch.manual_seed(42)
    txc = TemporalCrosscoder(
        d_in=d_model, d_sae=args.d_sae, T=args.T, k_total=args.k_total,
    ).to(args.device)
    history = train_gao_topk(
        txc, sample_fn,
        batch_size=args.batch_size,
        n_steps=args.total_steps,
        log_every=args.log_every,
        device=args.device,
        **train_kwargs,
    )
    return history_to_dicts(history)


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

    out_json = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(out_json.read_text()) if out_json.exists() else {}

    # --- (A) Gao default: per-step norm, k_aux=1792, threshold=128k ---
    data["gao_default"] = run_condition(
        "gao_default", sample_fn, args, d_model,
        auxk_alpha=1.0 / 32, k_aux=1792,
        dead_token_threshold=128000,
        auxk_norm="per_step",
    )
    with out_json.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[A] written")

    # --- (B) EMA-norm AuxK, higher α, aggressive threshold ---
    data["gao_ema_high_alpha"] = run_condition(
        "gao_ema_high_alpha", sample_fn, args, d_model,
        auxk_alpha=0.25, k_aux=None,           # k_aux=None → d_in/2 = 1792
        dead_token_threshold=12800,            # 50 steps
        auxk_norm="ema", auxk_ema_decay=0.99,
    )
    with out_json.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[B] written")

    # --- (C) EMA-norm AuxK + Bricken resample combo ---
    data["gao_ema_plus_bricken"] = run_condition(
        "gao_ema_plus_bricken", sample_fn, args, d_model,
        auxk_alpha=1.0 / 32, k_aux=1792,
        dead_token_threshold=12800,
        auxk_norm="ema", auxk_ema_decay=0.99,
        bricken_resample_every=500,
    )
    with out_json.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"[C] written")

    print("done all 3 conditions")


if __name__ == "__main__":
    main()
