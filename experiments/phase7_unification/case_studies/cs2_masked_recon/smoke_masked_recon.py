"""CS2 smoke — held-out-position reconstruction MSE per arch.

Hypothesis (orientation log candidate #4): when a single token's L12
residual is held out (zeroed), window archs can reconstruct it
better than per-token archs because their encoder integrates over T
neighbouring positions and "fills in" the missing token using
context. Per-token archs see zero at that position and can only
reconstruct b_dec.

This is a structural-prior test: TXC's win is by construction (it
has T-1 unmasked neighbours in its encoder window), not by training
loss optimisation. The question is just how big the win is and
whether it's preserved across realistic activation distributions.

Smoke-test design:

  - Reuse the 150-sentence Q1.1 cache (50 valid 256-token passages
    are too long for memory at this scale; 150 short sentences from
    the steering concept set is fine for a smoke-level check).
  - For each position t with a valid token, hold out x[t] (set to
    0) and ask each arch's SAE to reconstruct at position t.
  - For per-token archs: encoder sees 0 at t, decoder produces
    a constant ~b_dec.
  - For window archs (right-edge attribution): the encoder window
    ending at t contains a 0 at the rightmost position but valid
    residuals at the other T-1 positions. Decoder right-edge slice
    is the reconstruction at t.
  - Compute MSE = mean((x_hat[t] - x[t])^2) per arch over all valid
    held-out positions.

Output:
  results/case_studies/cs2_masked_recon/smoke_masked_recon.{png,json}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR, MLC_LAYERS
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
    load_phase7_model_safe as _load_phase7_model,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, ANCHOR_LAYER, DEFAULT_D_IN,
)


ARCHS = [
    ("topk_sae", "per-token (k=500)"),
    ("tsae_paper_k20", "per-token (k=20)"),
    ("agentic_txc_02", "TXC matryoshka (T=5)"),
    ("phase5b_subseq_h8", "SubseqH8 (T_max=10)"),
]
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/cs2_masked_recon")
)
ACTS_CACHE = (CASE_STUDIES_DIR / "steering_magnitude" / "_l12_acts_cache.npz")


def _decode_full_window(sae, src_class: str, z: torch.Tensor, T: int) -> torch.Tensor:
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _recon_for_arch(arch_id: str, acts_l12: np.ndarray, attn: np.ndarray,
                    device: torch.device) -> dict:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        # MLC needs a multi-layer cache; skip in this smoke pass.
        return {"src_class": src_class, "skipped": "MLC needs multi-layer cache"}
    print(f"  loading {arch_id} ({src_class})...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    T = window_T(sae, src_class, meta)
    d_sae = _d_sae_of(sae, src_class)
    print(f"    T={T} d_sae={d_sae}")

    N, S, d_in = acts_l12.shape
    pos_idx = torch.arange(S, device=device)
    x_full = torch.from_numpy(acts_l12).float().to(device)              # (N, S, d_in)
    m_full = torch.from_numpy(attn.astype(np.float32)).to(device)        # (N, S)

    # Baseline reconstruction (no masking) at every position. Returns x_hat[t]
    # for each valid t.
    print(f"    computing baseline reconstruction (no mask)...")
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        flat = x_full.reshape(N * S, d_in)
        with torch.no_grad():
            if src_class == "TemporalMatryoshkaBatchTopKSAE":
                z = sae.encode(flat, use_threshold=True)
                if isinstance(z, tuple):
                    z = z[0]
            else:
                z = sae.encode(flat)
            x_hat_baseline = sae.decode(z).reshape(N, S, d_in)            # (N, S, d_in)
    else:
        # Window arch baseline.
        windows = x_full.unfold(1, T, 1).movedim(-1, 2).contiguous()       # (N, K, T, d_in)
        K = windows.shape[1]
        flat_w = windows.reshape(N * K, T, d_in)
        x_hat_baseline = torch.zeros_like(x_full)
        bs = 16
        for i in range(0, flat_w.shape[0], bs):
            j = min(i + bs, flat_w.shape[0])
            sub = flat_w[i:j]
            with torch.no_grad():
                z = sae.encode(sub)
                full_recon = _decode_full_window(sae, src_class, z, T)    # (B, T, d_in)
            # Right-edge slice maps to position window_index + T - 1.
            x_hat_R = full_recon[:, -1, :].reshape(j - i, d_in)
            # write into x_hat_baseline[:, T-1:T-1+K, :] at the right window indices
            # easier: accumulate inside the K loop using positional indexing.
            for k_local, idx in enumerate(range(i, j)):
                n_idx = idx // K
                k_idx = idx % K
                t_pos = k_idx + T - 1
                x_hat_baseline[n_idx, t_pos] = x_hat_R[k_local]
            del z, full_recon, x_hat_R

    # Held-out reconstruction. For per-token: zero out x[t], encode + decode at t.
    # For window: zero out x[t] in the window ending at t, encode + decode (right-edge).
    print(f"    computing held-out (mask-out) reconstruction...")
    x_hat_holdout = torch.zeros_like(x_full)
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        # Held-out per token = zero input -> encoder produces nonzero from b_dec? Check.
        zero = torch.zeros((1, d_in), device=device, dtype=x_full.dtype)
        with torch.no_grad():
            if src_class == "TemporalMatryoshkaBatchTopKSAE":
                z = sae.encode(zero, use_threshold=True)
                if isinstance(z, tuple):
                    z = z[0]
            else:
                z = sae.encode(zero)
            x_hat_zero = sae.decode(z).squeeze(0)                           # (d_in,)
        # Same constant prediction at every position.
        x_hat_holdout[:] = x_hat_zero.unsqueeze(0).unsqueeze(0)
    else:
        # Window arch: for each window ending at t (t in [T-1, S-1]), zero out
        # the rightmost position and reconstruct. The encoder still sees T-1
        # valid surroundings.
        windows = x_full.unfold(1, T, 1).movedim(-1, 2).contiguous()       # (N, K, T, d_in)
        K = windows.shape[1]
        windows_holdout = windows.clone()
        windows_holdout[:, :, -1, :] = 0.0                                  # zero right edge
        flat_w = windows_holdout.reshape(N * K, T, d_in)
        bs = 16
        for i in range(0, flat_w.shape[0], bs):
            j = min(i + bs, flat_w.shape[0])
            sub = flat_w[i:j]
            with torch.no_grad():
                z = sae.encode(sub)
                full_recon = _decode_full_window(sae, src_class, z, T)
            x_hat_R = full_recon[:, -1, :].reshape(j - i, d_in)
            for k_local, idx in enumerate(range(i, j)):
                n_idx = idx // K; k_idx = idx % K; t_pos = k_idx + T - 1
                x_hat_holdout[n_idx, t_pos] = x_hat_R[k_local]
            del z, full_recon, x_hat_R

    # MSE per position, masked to valid (attn=1, position >= T-1 for window).
    valid = (m_full > 0)
    if T > 1:
        valid = valid & (pos_idx.unsqueeze(0) >= T - 1)

    # Total per-position MSE.
    err_baseline = ((x_hat_baseline - x_full) ** 2).sum(dim=-1)             # (N, S)
    err_holdout = ((x_hat_holdout - x_full) ** 2).sum(dim=-1)
    var_signal = (x_full ** 2).sum(dim=-1)                                  # (N, S)

    base_mse = float(err_baseline[valid].mean().item())
    hold_mse = float(err_holdout[valid].mean().item())
    sig_var = float(var_signal[valid].mean().item())

    # Frac variance explained.
    base_frac = 1.0 - base_mse / max(sig_var, 1e-9)
    hold_frac = 1.0 - hold_mse / max(sig_var, 1e-9)

    del sae, x_hat_baseline, x_hat_holdout
    torch.cuda.empty_cache()

    return {
        "src_class": src_class, "T": T, "d_sae": d_sae,
        "n_valid_tokens": int(valid.sum().item()),
        "signal_variance": sig_var,
        "baseline_mse": base_mse, "baseline_frac_var_explained": base_frac,
        "holdout_mse": hold_mse, "holdout_frac_var_explained": hold_frac,
        "holdout_relative_to_baseline": hold_mse / max(base_mse, 1e-9),
    }


def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k20": "#d62728",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
    }
    label = {
        "topk_sae": "TopKSAE per-token (k=500)",
        "tsae_paper_k20": "T-SAE per-token (k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
    }
    archs = [a for a, _ in ARCHS if a in payload["archs"] and "skipped" not in payload["archs"][a]]
    x = np.arange(len(archs))
    width = 0.4

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    base = [payload["archs"][a]["baseline_frac_var_explained"] for a in archs]
    hold = [payload["archs"][a]["holdout_frac_var_explained"] for a in archs]
    colors = [palette[a] for a in archs]
    ax.bar(x - width / 2, base, width, label="baseline (no mask)",
           color=colors, alpha=0.85, edgecolor="black")
    ax.bar(x + width / 2, hold, width, label="held-out (mask one position)",
           color=colors, alpha=0.40, edgecolor="black", hatch="//")
    ax.set_xticks(x)
    ax.set_xticklabels([label[a] for a in archs], rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("fraction of L12 residual variance explained")
    ax.set_title(f"CS2 — held-out-position reconstruction quality\n"
                 f"({payload['n_valid_tokens_summary']} held-out positions, "
                 f"150-sentence concept probe)")
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    plt.tight_layout()
    save_figure(fig, str(out_path))
    plt.close(fig)


def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    if not ACTS_CACHE.exists():
        raise FileNotFoundError(f"missing {ACTS_CACHE}; run Q1.1 first")
    with np.load(ACTS_CACHE, allow_pickle=False) as z:
        acts_l12 = z["l12"]; attn = z["attn"]
    print(f"CS2 — loaded {acts_l12.shape} L12 acts (mean valid tok / row "
          f"{attn.sum(axis=1).mean():.1f})")

    device = torch.device("cuda")
    archs_data = {}
    for arch_id, _label in ARCHS:
        print(f"\n=== {arch_id} ===")
        archs_data[arch_id] = _recon_for_arch(arch_id, acts_l12, attn, device)
        d = archs_data[arch_id]
        if "skipped" in d:
            print(f"  skipped: {d['skipped']}")
            continue
        print(f"  signal var per token: {d['signal_variance']:.2f}")
        print(f"  baseline MSE: {d['baseline_mse']:.2f}  "
              f"frac_var_explained: {d['baseline_frac_var_explained']:.3f}")
        print(f"  holdout  MSE: {d['holdout_mse']:.2f}  "
              f"frac_var_explained: {d['holdout_frac_var_explained']:.3f}")
        print(f"  holdout/baseline ratio: {d['holdout_relative_to_baseline']:.2f}")

    # Summary table at end.
    print(f"\n  {'arch':<26}  {'baseline_FVE':>13}  {'holdout_FVE':>12}  {'ratio':>6}")
    for arch_id, d in archs_data.items():
        if "skipped" in d:
            continue
        print(f"  {arch_id:<26}  {d['baseline_frac_var_explained']:>13.3f}  "
              f"{d['holdout_frac_var_explained']:>12.3f}  "
              f"{d['holdout_relative_to_baseline']:>6.2f}")

    # n_valid summary across non-skipped archs.
    n_valid_max = max(
        (d["n_valid_tokens"] for d in archs_data.values() if "skipped" not in d),
        default=0,
    )
    payload = {
        "n_sentences": int(acts_l12.shape[0]),
        "max_len": int(acts_l12.shape[1]),
        "n_valid_tokens_summary": int(n_valid_max),
        "archs": archs_data,
    }
    json_path = OUT_SUBDIR / "smoke_masked_recon.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")
    png_path = OUT_SUBDIR / "smoke_masked_recon.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")


if __name__ == "__main__":
    main()
