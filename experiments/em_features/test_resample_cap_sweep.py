"""Sweep Bricken max_resample_fraction with α=1/8 fixed.

Fixed α=1/8 (the best-loss α from the α sweep). Vary the per-call Bricken
resample cap: 0.5 (current default), 0.8, 1.0 (no cap).

  (1) resample_cap_0_5 — current default, produces 50% dead plateau
  (2) resample_cap_0_8 — more aggressive
  (3) resample_cap_1_0 — unlimited (re-random every dead feature found)

All else held constant: 5k steps, d_sae=32k, T=5, k=128, batch=256,
EMA AuxK, dead_threshold=128k, bricken_every=500, geom-median b_dec, etc.
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
from experiments.em_features.dead_feature_resample import DeadFeatureResampler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 8.0)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    p.add_argument("--fractions", type=float, nargs="+", default=[0.5, 0.8, 1.0])
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
        return buffer.sample_txc_windows(n, 5).float()

    out_json = args.out_dir / "dead_feature_experiment.json"
    data = json.loads(out_json.read_text()) if out_json.exists() else {}

    # Monkey-patch the resampler's default max_resample_fraction between runs.
    orig_init = DeadFeatureResampler.__init__

    for frac in args.fractions:
        # Key like "resample_cap_0_5"
        fkey = str(frac).replace(".", "_")
        key = f"resample_cap_{fkey}"
        if key in data:
            print(f"skip {key}")
            continue
        print(f"\n=========== {key} (frac={frac}) ===========", flush=True)

        def patched_init(self, crosscoder, *a, **kw):
            kw["max_resample_fraction"] = frac
            orig_init(self, crosscoder, *a, **kw)
        DeadFeatureResampler.__init__ = patched_init

        torch.manual_seed(42)
        txc = TemporalCrosscoder(
            d_in=d_model, d_sae=32768, T=5, k_total=128,
        ).to(args.device)
        history = train_gao_topk(
            txc, sample_fn,
            batch_size=256,
            n_steps=args.total_steps,
            auxk_alpha=args.auxk_alpha,
            k_aux=512,
            dead_token_threshold=128_000,
            lr_base=3e-4,
            log_every=args.log_every,
            device=args.device,
            auxk_norm="ema",
            auxk_ema_decay=0.99,
            adam_betas=(0.9, 0.999),
            bricken_resample_every=500,
        )
        data[key] = history_to_dicts(history)
        data[f"{key}_args"] = {
            "auxk_alpha": args.auxk_alpha,
            "max_resample_fraction": frac,
            "total_steps": args.total_steps,
        }
        with out_json.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"[{key}] written")

    DeadFeatureResampler.__init__ = orig_init
    print("resample-cap sweep done")


if __name__ == "__main__":
    main()
