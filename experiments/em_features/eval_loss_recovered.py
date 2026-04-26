"""Quick reconstruction-quality eval for the 4 architectures (SAE / TXC /
MLC / Han champion).

For each ckpt:
  - Sample N activation windows from the streaming buffer at the trained
    layer(s).
  - Forward through encoder+decoder.
  - Report:
        explained_variance = 1 - MSE(x_hat, x) / Var(x)
        mse_recon          = MSE(x_hat, x)
        mse_baseline       = Var(x)        (predict-the-mean baseline)

Higher EV ⇒ better reconstruction. Trivial mean-prediction gives EV=0.
A working SAE typically lands in the 0.6–0.95 range.

    uv run python -m experiments.em_features.eval_loss_recovered \\
        --config experiments/em_features/config.yaml \\
        --ckpts /root/em_features/checkpoints/qwen_l15_sae_arditi_k128_step10000.pt \\
                /root/em_features/checkpoints/qwen_l15_txc_brickenauxk_a32_10k_step10000.pt \\
                /root/em_features/checkpoints/qwen_l15_mlc_brickenauxk_a32_10k_step10000.pt \\
                /root/em_features/checkpoints/qwen_l15_han_champ_10k_step10000.pt \\
        --n_samples 16384
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR_SRC = REPO_ROOT / "experiments" / "separation_scaling" / "vendor" / "src"
for p in (str(VENDOR_SRC), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from sae_day.sae import TopKSAE, TemporalCrosscoder, MultiLayerCrosscoder  # noqa: E402

from experiments.em_features.streaming_buffer import (  # noqa: E402
    HookpointStreamingBuffer, HookpointBufferConfig,
    StreamingActivationBuffer, StreamingBufferConfig,
    MultiLayerStreamingBuffer, MultiLayerBufferConfig,
    mixed_text_iter,
)
from experiments.em_features.architectures.txc_bare_multidistance_contrastive_antidead import (  # noqa: E402
    TXCBareMultiDistanceContrastiveAntidead,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--ckpts", type=Path, nargs="+", required=True)
    p.add_argument("--n_samples", type=int, default=16384)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path, default=None)
    return p.parse_args()


def detect_kind(ckpt: dict) -> str:
    cfg = ckpt["config"]
    if "L" in cfg and "layers" in cfg:
        return "mlc"
    if "T" in cfg and "k_total" in cfg:
        return "txc"
    if "T" in cfg and "k" in cfg and "shifts" in cfg:
        return "han"
    if "T" in cfg and "k" in cfg:
        return "han"  # han also has T and k (no k_total). order matters
    return "sae"


def load_model(kind: str, cfg: dict, device: str):
    if kind == "sae":
        m = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k"])
    elif kind == "txc":
        m = TemporalCrosscoder(d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k_total=cfg["k_total"])
    elif kind == "mlc":
        m = MultiLayerCrosscoder(d_in=cfg["d_in"], d_sae=cfg["d_sae"], L=cfg["L"], k_total=cfg["k_total"])
    elif kind == "han":
        m = TXCBareMultiDistanceContrastiveAntidead(
            d_in=cfg["d_in"], d_sae=cfg["d_sae"], T=cfg["T"], k=cfg["k"],
            shifts=tuple(cfg.get("shifts", (1, 2))),
            matryoshka_h_size=cfg.get("matryoshka_h_size", cfg["d_sae"] // 5),
            alpha=cfg.get("alpha_contrastive", 1.0),
            aux_k=cfg.get("aux_k", 512),
            dead_threshold_tokens=cfg.get("dead_threshold_tokens", 640_000),
            auxk_alpha=cfg.get("auxk_alpha", 1.0 / 32.0),
        )
    else:
        raise ValueError(kind)
    return m.to(device)


def reconstruct(kind: str, m, x: torch.Tensor) -> torch.Tensor:
    """Return x_hat with same shape as x."""
    if kind == "sae":
        x_hat, _ = m(x)  # TopKSAE.forward returns (x_hat, z)
        return x_hat
    if kind == "txc":
        z = m.encode(x)  # (B, d_sae)
        return torch.einsum("bm,tmd->btd", z, m.W_dec) + m.b_dec
    if kind == "mlc":
        z = m.encode(x)  # (B, d_sae)
        return torch.einsum("bm,lmd->bld", z, m.W_dec) + m.b_dec
    if kind == "han":
        # Han model.forward returns (loss, x_hat, z); use the (B, T, d) reconstruction.
        loss, x_hat, z = m(x)
        return x_hat
    raise ValueError(kind)


def main():
    args = parse_args()
    cfg_yaml = yaml.safe_load(args.config.read_text())

    print(f"Loading subject model {cfg_yaml['subject_model']}", flush=True)
    tok = AutoTokenizer.from_pretrained(cfg_yaml["subject_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg_yaml["subject_model"], torch_dtype=torch.float16, device_map=args.device,
    ).eval()
    d_model = int(cfg_yaml["d_model"])

    # Build three buffers (one per layout). Activations are sampled fresh per ckpt
    # so each architecture sees the same statistical distribution but different draws.
    layer_txc = int(cfg_yaml["layer_txc"])
    layers_mlc = list(cfg_yaml["layers_mlc"])

    text_iter_for = lambda: mixed_text_iter(tok, cfg_yaml["streaming"]["corpus_mix"])

    print("Building HookpointStreamingBuffer (resid_post, single-layer)...", flush=True)
    sae_buf = HookpointStreamingBuffer(hf_model, tok, text_iter_for(),
        HookpointBufferConfig(
            hookpoint="resid_post", layer=layer_txc, d_model=d_model,
            buffer_seqs=int(cfg_yaml["streaming"]["buffer_activations"]) // int(cfg_yaml["streaming"]["chunk_len"]),
            chunk_len=int(cfg_yaml["streaming"]["chunk_len"]),
            refill_chunks=int(cfg_yaml["streaming"]["refill_chunks"]),
            device=args.device, dtype=torch.float16,
        ))
    sae_buf.warmup()

    print("Building TXC StreamingActivationBuffer...", flush=True)
    txc_buf = StreamingActivationBuffer(hf_model, tok, text_iter_for(),
        StreamingBufferConfig(
            layer=layer_txc, d_model=d_model,
            buffer_seqs=max(1, int(cfg_yaml["streaming"]["buffer_activations"]) // int(cfg_yaml["streaming"]["chunk_len"])),
            chunk_len=int(cfg_yaml["streaming"]["chunk_len"]),
            refill_chunks=int(cfg_yaml["streaming"]["refill_chunks"]),
            device=args.device, dtype=torch.float16,
        ))
    txc_buf.warmup()

    print("Building MultiLayerStreamingBuffer...", flush=True)
    mlc_buf = MultiLayerStreamingBuffer(hf_model, tok, text_iter_for(),
        MultiLayerBufferConfig(
            layers=layers_mlc, d_model=d_model,
            buffer_seqs=int(cfg_yaml.get("mlc", {}).get("buffer_seqs", 2000)),
            chunk_len=int(cfg_yaml["streaming"]["chunk_len"]),
            refill_chunks=int(cfg_yaml["streaming"]["refill_chunks"]),
            device=args.device, dtype=torch.float16,
        ))
    mlc_buf.warmup()

    results = []
    for ckpt_path in args.ckpts:
        print(f"\n=== {ckpt_path.name} ===", flush=True)
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        cfg = ckpt["config"]

        # Detect kind by config keys
        if "L" in cfg and "layers" in cfg:
            kind = "mlc"
        elif "shifts" in cfg:
            kind = "han"
        elif "T" in cfg:
            kind = "txc"
        else:
            kind = "sae"

        m = load_model(kind, cfg, args.device)
        m.load_state_dict(ckpt["state_dict"])
        m.eval()
        T = cfg.get("T", None)
        L = cfg.get("L", None)

        n_so_far = 0
        sse = 0.0  # sum of squared errors
        ssx = 0.0  # sum of squared x's (for variance baseline against mean=0)
        sx = torch.zeros(d_model, device=args.device, dtype=torch.float64)
        sxx = torch.zeros(d_model, device=args.device, dtype=torch.float64)
        n_elem = 0

        with torch.no_grad():
            while n_so_far < args.n_samples:
                bs = min(args.batch_size, args.n_samples - n_so_far)
                if kind == "sae":
                    x = sae_buf.sample_flat(bs).float()  # (B, d)
                elif kind in ("txc", "han"):
                    x = txc_buf.sample_txc_windows(bs, T).float()  # (B, T, d)
                elif kind == "mlc":
                    x = mlc_buf.sample_mlc_batch(bs).float()  # (B, L, d)

                x_hat = reconstruct(kind, m, x)
                err = (x_hat - x).pow(2).sum(dim=-1)  # sum over d_in
                sse += float(err.sum().item())
                ssx += float(x.pow(2).sum(dim=-1).sum().item())

                # Per-feature mean / var accumulation (for "predict-the-mean" baseline)
                flat = x.reshape(-1, d_model).double()
                sx += flat.sum(dim=0)
                sxx += (flat * flat).sum(dim=0)
                n_elem += flat.shape[0]

                n_so_far += bs

        # Reconstruction MSE (per-element-of-d_in)
        # err per sample summed over d_in; we want mean over (n_samples × extras × d_in)
        n_dim = T or L or 1
        mse_recon = sse / (n_so_far * n_dim * d_model)

        # Variance of x (per-element). Var = E[x²] - (E[x])²
        mean_x = sx / n_elem
        var_x = (sxx / n_elem) - mean_x.pow(2)
        var_total = float(var_x.sum().item())  # total variance over all d_in dims
        ev_zero_baseline = 1.0 - (sse / (n_so_far * n_dim * d_model)) / (ssx / (n_so_far * n_dim * d_model))
        ev_mean_baseline = 1.0 - mse_recon / (var_total / d_model)

        rec = {
            "ckpt": str(ckpt_path),
            "kind": kind,
            "n_samples": n_so_far,
            "mse_recon": mse_recon,
            "mse_zero_baseline": ssx / (n_so_far * n_dim * d_model),
            "var_per_element": var_total / d_model,
            "ev_vs_zero": ev_zero_baseline,
            "ev_vs_mean": ev_mean_baseline,
        }
        results.append(rec)
        print(f"  kind={kind}  n={n_so_far}  mse_recon={mse_recon:.4f}  "
              f"EV_vs_zero={ev_zero_baseline:.4f}  EV_vs_mean={ev_mean_baseline:.4f}", flush=True)

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2))
        print(f"\nwrote {args.out}")
    else:
        print("\n=== Summary ===")
        for r in results:
            name = Path(r["ckpt"]).stem
            print(f"{name:50s}  EV_vs_mean={r['ev_vs_mean']:.4f}  EV_vs_zero={r['ev_vs_zero']:.4f}")


if __name__ == "__main__":
    main()
