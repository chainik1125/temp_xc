"""Post-training sanity check for the trained TXC.

Computes on a held-out stream:

  - FVE   = 1 − mean‖x − x̂‖² / mean‖x‖²
  - mean L0 (non-zero latents per window)
  - KL divergence between Qwen's next-token distribution with clean vs.
    TXC-reconstructed residuals substituted at the training layer.

Exit criteria targets (from the plan): FVE ≥ 0.6, KL < 0.2.

    uv run python -m experiments.em_features.run_eval_recon \
        --config experiments/em_features/config.yaml \
        --ckpt  experiments/em_features/checkpoints/qwen_l15_txc_t5_k128.pt \
        --out   experiments/em_features/results/qwen_l15_txc/recon_eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
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
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n_eval_batches", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.ckpt, map_location=args.device)
    ccfg = ckpt["config"]
    txc = TemporalCrosscoder(
        d_in=ccfg["d_in"], d_sae=ccfg["d_sae"], T=ccfg["T"], k_total=ccfg["k_total"],
    )
    txc.load_state_dict(ckpt["state_dict"])
    txc.eval().to(args.device)

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
        buffer_seqs=500,
        chunk_len=int(cfg["streaming"]["chunk_len"]),
        refill_chunks=int(cfg["streaming"]["refill_chunks"]),
        device=args.device,
        dtype=torch.float16,
    )
    text_iter = mixed_text_iter(tok, cfg["streaming"]["corpus_mix"], seed=999)
    buf = StreamingActivationBuffer(model, tok, text_iter, buf_cfg)
    buf.warmup()

    # -- Reconstruction FVE + L0 ---------------------------------------------
    num = den = 0.0
    l0_sum = 0.0
    n_windows = 0
    T = ccfg["T"]
    for _ in range(args.n_eval_batches):
        x = buf.sample_txc_windows(args.batch_size, T).float()  # (B, T, d)
        x_hat, z = txc(x)
        num += (x - x_hat).pow(2).sum().item()
        den += x.pow(2).sum().item()
        l0_sum += (z != 0).float().sum(dim=-1).sum().item()
        n_windows += x.shape[0]
    fve = 1.0 - num / max(den, 1e-12)
    mean_l0 = l0_sum / max(n_windows, 1)

    # -- KL when substituting reconstructions at the training layer ----------
    # Pick a small fresh set of sequences, forward Qwen twice: once clean,
    # once with the layer-L residual replaced by the TXC's per-position
    # reconstruction of a T-window ending at each position.
    kl_sum = 0.0
    kl_count = 0
    # One chunk is enough for a coarse estimate.
    input_ids = buf._tokenize_batch(8)  # (8, chunk_len)
    clean_out = model(input_ids=input_ids, output_hidden_states=True, use_cache=False)
    clean_hs = clean_out.hidden_states[buf_cfg.layer + 1].float()  # (B, L, d)
    clean_logits = clean_out.logits.float()

    # Build a dense reconstructed-hs tensor position-by-position.
    # For position p, window = clean_hs[:, p-T+1:p+1, :] (left-pad clamped).
    B, L, d = clean_hs.shape
    recon_hs = clean_hs.clone()
    for p in range(T - 1, L):
        win = clean_hs[:, p - T + 1:p + 1, :]  # (B, T, d)
        recon, _ = txc(win)
        recon_hs[:, p, :] = recon[:, -1, :]

    def _hook(module, inputs, output):
        if isinstance(output, tuple):
            head, *rest = output
            return (recon_hs.to(head.dtype),) + tuple(rest)
        return recon_hs.to(output.dtype)

    layer_mod = model.model.layers[buf_cfg.layer]
    handle = layer_mod.register_forward_hook(_hook)
    try:
        patched_out = model(input_ids=input_ids, use_cache=False)
    finally:
        handle.remove()
    patched_logits = patched_out.logits.float()

    log_q = F.log_softmax(patched_logits, dim=-1)
    log_p = F.log_softmax(clean_logits, dim=-1)
    p = log_p.exp()
    kl_per_pos = (p * (log_p - log_q)).sum(dim=-1)  # (B, L)
    kl_sum = kl_per_pos.mean().item()
    kl_count = 1  # pooled across all positions

    summary = {
        "ckpt": str(args.ckpt),
        "fve": fve,
        "mean_l0": mean_l0,
        "k_total_configured": ccfg["k_total"],
        "kl_clean_vs_patched": kl_sum,
        "n_eval_windows": n_windows,
        "targets": {"fve": 0.6, "kl": 0.2},
    }
    with args.out.open("w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if fve < 0.6:
        print(f"WARNING: FVE {fve:.3f} < 0.6 target")
    if kl_sum > 0.2:
        print(f"WARNING: KL {kl_sum:.3f} > 0.2 target")


if __name__ == "__main__":
    main()
