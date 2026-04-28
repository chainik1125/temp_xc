"""Sweep AuxK coefficient α at the Bricken+EMA-AuxK recipe.

Fixed-recipe α sweep: each run is 5k steps, Bricken resample every 500,
EMA-normalized AuxK, geom-median b_dec init, T=5, d_sae=32k, k=128.
Only α varies.

  α ∈ {1/16, 1/8, 1/4, 1/2}  (baseline is 1/32)

Appends each run to dead_feature_experiment.json under keys
"brickenauxk_alpha_1_16", "..._1_8", etc.
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
    train_gao_topk, GaoStats,
)


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
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[1/16, 1/8, 1/4, 1/2])
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


def frac_key(alpha: float) -> str:
    """Stable key from an α value — tries to use the 1/N form."""
    for denom in (2, 4, 8, 16, 32, 64, 128):
        if abs(alpha - 1.0 / denom) < 1e-9:
            return f"1_{denom}"
    return f"{alpha:.4f}".replace(".", "_")


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
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

    out_json = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(out_json.read_text()) if out_json.exists() else {}

    for alpha in args.alphas:
        key = f"brickenauxk_alpha_{frac_key(alpha)}"
        if key in data:
            print(f"skip {key} — already present")
            continue
        print(f"\n=========== {key} (auxk_alpha={alpha}) ===========", flush=True)
        torch.manual_seed(42)
        txc = TemporalCrosscoder(
            d_in=d_model, d_sae=args.d_sae, T=args.T, k_total=args.k_total,
        ).to(args.device)
        history = train_gao_topk(
            txc, sample_fn,
            batch_size=args.batch_size,
            n_steps=args.total_steps,
            auxk_alpha=alpha,
            k_aux=512,
            dead_token_threshold=128_000,
            lr_base=3e-4,            # match our Bricken baseline, not Gao's 2e-4
            log_every=args.log_every,
            device=args.device,
            auxk_norm="ema",
            auxk_ema_decay=0.99,
            adam_betas=(0.9, 0.999),
            bricken_resample_every=500,
        )
        data[key] = history_to_dicts(history)
        data[f"{key}_args"] = {"auxk_alpha": alpha, "total_steps": args.total_steps}
        with out_json.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"[{key}] written")

    print("alpha sweep done")


if __name__ == "__main__":
    main()
