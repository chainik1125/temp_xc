"""Train a TemporalCrosscoder on streaming Qwen-7B residual activations.

Run on a100_1 (H100 80 GB).

    uv run python -m experiments.em_features.run_training \
        --config experiments/em_features/config.yaml \
        --out experiments/em_features/checkpoints/qwen_l15_txc_t5_k128.pt
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
    StreamingActivationBuffer,
    StreamingBufferConfig,
    mixed_text_iter,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log_every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading subject model: {cfg['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"  # Training corpus; left-pad only matters for prompt_last pooling.

    model = AutoModelForCausalLM.from_pretrained(
        cfg["subject_model"],
        torch_dtype=torch.float16,
        device_map=args.device,
    ).eval()

    d_model = int(cfg["d_model"])
    assert model.config.hidden_size == d_model, (
        f"config d_model={d_model} != model hidden_size={model.config.hidden_size}"
    )

    buf_cfg = StreamingBufferConfig(
        layer=int(cfg["layer_txc"]),
        d_model=d_model,
        buffer_seqs=max(1, int(cfg["streaming"]["buffer_activations"]) // int(cfg["streaming"]["chunk_len"])),
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    print(f"Buffer: {buf_cfg.buffer_seqs} seqs × {buf_cfg.chunk_len} toks @ fp16 "
          f"≈ {buf_cfg.buffer_seqs*buf_cfg.chunk_len*d_model*2/1e9:.1f} GB VRAM", flush=True)

    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"])
    buffer = StreamingActivationBuffer(model, tok, text_iter, buf_cfg)

    print("Warming up buffer (one full Qwen forward pass over buffer_seqs)...", flush=True)
    t0 = time.time()
    buffer.warmup()
    print(f"  warmup done in {time.time()-t0:.1f}s", flush=True)

    txc_cfg = cfg["txc"]
    txc = TemporalCrosscoder(
        d_in=d_model,
        d_sae=int(txc_cfg["d_sae"]),
        T=int(txc_cfg["T"]),
        k_total=int(txc_cfg["k_total"]),
    ).to(args.device)

    optim = torch.optim.Adam(txc.parameters(), lr=float(txc_cfg["lr"]))
    n_steps = int(txc_cfg["steps"])
    batch_size = int(txc_cfg["batch_size"])
    T = int(txc_cfg["T"])

    loss_history: list[float] = []
    l0_history: list[tuple[int, float]] = []
    best = float("inf")
    train_t0 = time.time()

    for step in range(n_steps):
        x = buffer.sample_txc_windows(batch_size, T).float()  # (B, T, d_in)
        x_hat, z = txc(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if (step + 1) % 100 == 0:
            txc.normalize_decoder()
        loss_history.append(float(loss.detach()))
        best = min(best, loss_history[-1])
        if (step + 1) % args.log_every == 0:
            with torch.no_grad():
                l0 = (z != 0).float().sum(dim=-1).mean().item()
            l0_history.append((step + 1, l0))
            elapsed = time.time() - train_t0
            print(f"[train] step {step+1:>6}/{n_steps}  loss={loss_history[-1]:.6f}  "
                  f"L0={l0:.2f}  elapsed={elapsed/60:.1f}m", flush=True)

    ckpt = {
        "state_dict": txc.state_dict(),
        "config": {
            "d_in": d_model,
            "d_sae": int(txc_cfg["d_sae"]),
            "T": T,
            "k_total": int(txc_cfg["k_total"]),
            "subject_model": cfg["subject_model"],
            "layer": int(cfg["layer_txc"]),
        },
        "loss_history": loss_history,
        "l0_history": l0_history,
        "best_loss": best,
    }
    torch.save(ckpt, args.out)
    with (args.out.with_suffix(".meta.json")).open("w") as f:
        json.dump({k: v for k, v in ckpt.items() if k != "state_dict"}, f, indent=2)
    print(f"Saved {args.out}  (best train loss={best:.6f})", flush=True)


if __name__ == "__main__":
    main()
