"""INSTRUMENTATION (writes nothing): which stream were the T5 anchors trained on?

The § 8 substrate question is a question about the ARCHIVED ANCHORS, so
ask them directly instead of inferring from metadata or from a fresh
training run. An SAE reconstructs its own training distribution far
better than a foreign one, so a pure forward pass over each candidate
stream is decisive and costs no training at all.

Reports, per stream: fraction of variance unexplained (FVU), MSE, and
mean L0 against the anchor's k_win. The anchors' native stream should
win all three, and the gap should not be subtle.

Writes nothing — no leaderboard row, no ckpt, no manifest.

Usage:
  .venv/bin/python -m experiments.explorations.actmix_rlhf.probe_anchor_stream
"""

from __future__ import annotations

import argparse

import torch

from temp_bench.core.config import import_by_path, load_arch, load_datasource
from temp_bench.core.trainer import _build_refill_source, _infer_d_in
from temp_bench.data.sequence_buffer import SequenceBuffer
from experiments.explorations.actmix_rlhf.cells import PF_ARCH

ANCHOR = ("/workspace/caches/rlhf/txcdr-base/ckpts/"
          "agentic_txc_02__seed{seed}.pt")
STREAMS = ["gemma_2_2b_base_l12_phase7", "gemma_2_2b_it_l13_fineweb_24k128"]


@torch.no_grad()
def score(arch, x):
    """FVU / MSE / L0 of the anchor's own full-scale reconstruction."""
    z = arch.encode(x)
    xhat = arch.decode(z)
    mse = torch.mean((xhat - x.float()) ** 2)
    var = torch.var(x.float(), unbiased=False)
    l0 = (z > 0).float().sum(dim=-1).mean()
    return {"fvu": float(mse / var), "mse": float(mse), "l0": float(l0),
            "var": float(var), "x_absmean": float(x.float().abs().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--T", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sd = torch.load(ANCHOR.format(seed=args.seed), map_location="cpu",
                    weights_only=False)

    print(f"[anchor] agentic_txc_02__seed{args.seed}.pt  T={args.T} "
          f"k_win=500  batch={args.batch}")
    rows = {}
    for ds_name in STREAMS:
        data_spec = load_datasource(ds_name)
        d_in = _infer_d_in(data_spec)
        cls = import_by_path(load_arch(PF_ARCH).class_path)
        arch = cls(d_in=d_in, d_sae=18432, T=args.T, k_pos=100 * args.T)
        missing, unexpected = arch.load_state_dict(sd, strict=False)
        missing = [k for k in missing if k not in
                   ("global_step", "converged_step")]
        if missing or unexpected:
            print(f"  [warn] missing={missing} unexpected={unexpected}")
        arch.to(device).eval()

        refill = _build_refill_source(data_spec, seed=0)
        it = SequenceBuffer(refill, seq_len=128, device=device, seed=0)
        x = it(args.batch)                      # (B, 128, d_in)
        # the arch consumes T-windows; take the leading window per sequence
        r = score(arch, x[:, :args.T, :])
        rows[ds_name] = r
        print(f"  {ds_name:38} FVU={r['fvu']:.4f}  MSE={r['mse']:.2f}  "
              f"L0={r['l0']:.1f}/500  act|x|={r['x_absmean']:.3f}")
        del arch, it
        torch.cuda.empty_cache()

    a, b = STREAMS
    print(f"\n[anchor] FVU ratio {a.split('_')[-1]} vs "
          f"{b.split('_')[-1]}: {rows[a]['fvu'] / rows[b]['fvu']:.3f} "
          f"(<1 ⇒ base-l12 is the anchor's native stream)")


if __name__ == "__main__":
    main()
