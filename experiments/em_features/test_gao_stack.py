"""5k-step test of the Gao 2024 TopK stack on TemporalCrosscoder.

Runs one condition (Gao full stack) and appends to the existing
dead_feature_experiment.json from test_dead_feature_resample.py so the
plot can show all three curves (NO_RESAMPLE / WITH_RESAMPLE / GAO).

    uv run python -m experiments.em_features.test_gao_stack \
        --config experiments/em_features/config.yaml \
        --out_dir /root/em_features/results/dead_feature_experiment \
        --dead_token_threshold 5120
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
from experiments.em_features.gao_topk_training import train_gao_topk  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    # For a 5k-step test, 10M-token dead threshold is too high (total =
    # 5k*256 = 1.28M tokens). Scale down so AuxK fires from step 1.
    p.add_argument("--dead_token_threshold", type=int, default=5120)
    p.add_argument("--auxk_alpha", type=float, default=1.0 / 32.0)
    p.add_argument("--k_aux", type=int, default=512)
    p.add_argument("--lr_base", type=float, default=2e-4)
    p.add_argument("--log_every", type=int, default=500)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(args.config.read_text())

    print(f"Loading subject model: {cfg['subject_model']}", flush=True)
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

    def sample_fn(n):
        return buffer.sample_txc_windows(n, args.T).float()

    print("\n========== condition: GAO (tied init, geom-median b_dec, AuxK, β1=0 β2=0.9999, "
          f"lr_base={args.lr_base}, k_aux={args.k_aux}, dead_thr={args.dead_token_threshold}) ==========",
          flush=True)

    history = train_gao_topk(
        txc, sample_fn,
        batch_size=args.batch_size,
        n_steps=args.total_steps,
        auxk_alpha=args.auxk_alpha,
        k_aux=args.k_aux,
        dead_token_threshold=args.dead_token_threshold,
        lr_base=args.lr_base,
        log_every=args.log_every,
        device=args.device,
    )

    # Convert dataclasses → dicts so json.dump works.
    gao_log = [dict(step=s.step, loss=s.loss_main, loss_auxk=s.loss_auxk,
                    loss_total=s.loss_total, n_dead=s.n_dead,
                    n_features=s.n_features,
                    n_active_in_batch=s.n_active_this_batch,
                    max_fire=s.max_fire_count,
                    elapsed_min=s.elapsed_min,
                    n_resampled_so_far=0)  # plot script expects this key
               for s in history]

    existing = args.out_dir / "dead_feature_experiment.json"
    if existing.exists():
        data = json.loads(existing.read_text())
    else:
        data = {}
    data["gao"] = gao_log
    data["gao_args"] = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    with existing.open("w") as f:
        json.dump(data, f, indent=2)
    print(f"appended gao to {existing}")


if __name__ == "__main__":
    main()
