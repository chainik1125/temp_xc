"""INSTRUMENTATION (writes nothing): l13-IT vs base-l12 stream A/B against
the upstream agentic_txc_02 training trace.

Why: CARD § 8 forces the pf arm onto `gemma_2_2b_it_l13_fineweb_24k128`
on the premise that "the T5 anchors are l13-IT-trained". Every upstream
training log on this pod instead records `subject_model
google/gemma-2-2b`, `anchor_layer 12`. That premise is listed PENDING
TEAM REVIEW in the card, and a wrong-stream grid cell is unusable
rather than merely slow — so it is worth ~20 minutes to settle before
spending grid hours.

Method: run the port at the upstream log's OWN config (T=5, seed 42,
k_win=500) on each candidate stream, logging loss on the upstream
cadence (every 200 steps), and compare early trace shape to
`txcdr-base:training_logs/agentic_txc_02__seed42.json`. Step 0 alone is
diagnostic: b_dec is geometric-median-initialised from the data, so the
initial loss is strongly stream-dependent (upstream step 0 = 91888.16).

NOT an experiment: no leaderboard row, no checkpoint, no manifest entry
is written, so training at T=5 here does NOT violate the "T5 = archived
anchor, never retrained" alias rule — nothing is minted under the
anchors' train_keys.

Usage:
  .venv/bin/python -m experiments.explorations.actmix_rlhf.probe_stream_ab \
      --datasource gemma_2_2b_base_l12_phase7 --steps 400
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from temp_bench.core.config import import_by_path, load_arch, load_datasource
from temp_bench.core.schemas import TrainingConfig
from temp_bench.core.trainer import _build_refill_source, _infer_d_in
from temp_bench.data.sequence_buffer import SequenceBuffer
from experiments.explorations.actmix_rlhf.cells import PF_ARCH, _pf_batch

UPSTREAM_LOG = Path("/workspace/caches/rlhf/txcdr-base/training_logs/"
                    "agentic_txc_02__seed42.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--T", type=int, default=5)       # the upstream log's T
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--log-every", type=int, default=200)
    args = ap.parse_args()

    cfg = TrainingConfig(n_steps=25_000, batch_size=_pf_batch(args.T),
                         arch_hparams_override={})
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_spec = load_datasource(args.datasource)
    d_in = _infer_d_in(data_spec)

    torch.manual_seed(args.seed)
    cls = import_by_path(load_arch(PF_ARCH).class_path)
    arch = cls(d_in=d_in, d_sae=18432, T=args.T, k_pos=100 * args.T)
    arch.to(device).train()

    refill = _build_refill_source(data_spec, seed=args.seed)
    it = SequenceBuffer(refill, seq_len=128, device=device, seed=args.seed)
    optim = torch.optim.Adam(arch.parameters(), lr=cfg.learning_rate)
    sched = (torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: min(1.0, (s + 1) / cfg.warmup_steps))
        if cfg.warmup_steps > 0 else None)

    print(f"[ab] datasource={args.datasource} d_in={d_in} T={args.T} "
          f"seed={args.seed} batch={cfg.batch_size} lr={cfg.learning_rate} "
          f"warmup={cfg.warmup_steps}")

    trace = []
    t0 = time.perf_counter()
    for step in range(args.steps + 1):
        arch.pre_step()
        optim.zero_grad(set_to_none=True)
        m = arch.train_step(it(cfg.batch_size))
        m["loss"].backward()
        optim.step()
        if sched is not None:
            sched.step()
        arch.post_step()
        if step % args.log_every == 0:
            loss = float(m["loss"].detach().item())
            l0 = float(m["l0"].detach().item())
            trace.append({"step": step, "loss": loss, "l0": l0})
            print(f"[ab] step={step:>5} loss={loss:.4f} l0={l0:.2f} "
                  f"({time.perf_counter() - t0:.0f}s)")

    up = json.loads(UPSTREAM_LOG.read_text())
    print(f"\n[ab] upstream (T={up['T']} k_win={up['k_win']} "
          f"subject={up['subject_model']} layer={up['anchor_layer']}):")
    print(f"{'step':>6} {'ours':>14} {'upstream':>14} {'ratio':>8}")
    for rec in trace:
        i = rec["step"] // 200
        if i < len(up["loss"]):
            u = up["loss"][i]
            print(f"{rec['step']:>6} {rec['loss']:>14.2f} {u:>14.2f} "
                  f"{rec['loss'] / u:>8.3f}")
    print(f"[ab] upstream l0 at those steps: "
          f"{[round(x, 1) for x in up['l0'][:len(trace)]]}")


if __name__ == "__main__":
    main()
