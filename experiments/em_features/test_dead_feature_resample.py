"""Short 5k-step training run that compares dead-feature rate with and
without periodic resampling. Runs two conditions sequentially on the same
streaming buffer (so identical data distribution), logs dead-feature rate
every 500 steps, produces a side-by-side plot + JSON summary.

    uv run python -m experiments.em_features.test_dead_feature_resample \
        --config experiments/em_features/config.yaml \
        --out_dir /root/em_features/results/dead_feature_experiment
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
from experiments.em_features.dead_feature_resample import DeadFeatureResampler  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out_dir", type=Path, required=True)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--measure_every", type=int, default=500)
    p.add_argument("--resample_every", type=int, default=500)
    p.add_argument("--d_sae", type=int, default=32768)
    p.add_argument("--T", type=int, default=5)
    p.add_argument("--k_total", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def train_one_condition(args, cfg, buffer, name, resampling):
    print(f"\n========== condition: {name}  (resampling={resampling}) ==========", flush=True)
    torch.manual_seed(42)
    txc = TemporalCrosscoder(d_in=int(cfg["d_model"]), d_sae=args.d_sae,
                             T=args.T, k_total=args.k_total).to(args.device)
    optim = torch.optim.Adam(txc.parameters(), lr=args.lr)
    resampler = DeadFeatureResampler(txc, resample_every=args.resample_every,
                                      min_fires=1, n_check=2048) if resampling else None

    def sample_fn(n):  # used by both the resampler and diagnostic
        return buffer.sample_txc_windows(n, args.T).float()

    log: list[dict] = []
    train_t0 = time.time()

    for step in range(args.total_steps):
        x = buffer.sample_txc_windows(args.batch_size, args.T).float()
        x_hat, z = txc(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % 100 == 0:
            txc.normalize_decoder()

        if resampler is not None:
            resampler.maybe_resample(step + 1, sample_fn)

        if (step + 1) % args.measure_every == 0:
            diag = DeadFeatureResampler(txc, n_check=2048).diagnostic(sample_fn)
            diag["step"] = step + 1
            diag["loss"] = float(loss.detach())
            diag["elapsed_min"] = (time.time() - train_t0) / 60
            diag["n_resampled_so_far"] = (
                sum(h.n_resampled for h in resampler.history) if resampler else 0
            )
            print(f"[{name}] step {step+1:>5}/{args.total_steps}  loss={diag['loss']:.1f}  "
                  f"dead={diag['n_dead']}/{diag['n_features']} "
                  f"({100*diag['n_dead']/diag['n_features']:.1f}%)  "
                  f"resampled_total={diag['n_resampled_so_far']}  "
                  f"max_fire={diag['max_fire']}", flush=True)
            log.append(diag)

    return log


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

    no_resample = train_one_condition(args, cfg, buffer, "NO_RESAMPLE", resampling=False)
    with_resample = train_one_condition(args, cfg, buffer, "WITH_RESAMPLE", resampling=True)

    out_json = args.out_dir / "dead_feature_experiment.json"
    with out_json.open("w") as f:
        json.dump({
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "no_resample": no_resample,
            "with_resample": with_resample,
        }, f, indent=2)
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
