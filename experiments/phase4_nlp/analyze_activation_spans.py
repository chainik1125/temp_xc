"""Phase 1b: Activation span distribution per architecture.

For each architecture's alive features, compute the distribution of
activation span lengths on held-out data. A "span" is a contiguous run
of tokens where a feature is active. The paper claims TFA pred codes
represent slow-moving contextual features with long spans. We test this
directly: TFA pred spans vs TFA novel vs TXCDR vs Stacked.

Output:
    results/analysis/activation_spans/
        span_histograms.png
        summary.json
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")

CKPT_DIR = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/activation_spans"
ACT_PATH = "/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy"
D_IN = 2304
D_SAE = 18432
T_WIN = 5
K = 100
LAYER = "resid_L25"
N_EVAL_SEQ = 500
ALIVE_THRESHOLD = 0.0001


def compute_spans(active_bool: torch.Tensor) -> list[int]:
    """Compute span lengths from a (n_tokens,) boolean array.

    A span is a maximal run of consecutive True values.
    Returns list of span lengths (can be empty).
    """
    active = active_bool.numpy().astype(np.int32)
    # Append 0 at start/end to catch edge runs
    padded = np.concatenate([[0], active, [0]])
    diff = np.diff(padded)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return (ends - starts).tolist()


@torch.no_grad()
def stacked_activations(eval_x: torch.Tensor) -> np.ndarray:
    """Return (N, seq_len, d_sae) bool array: feature i active at (seq, pos)?

    For Stacked, position t uses sae[t]. We use a 5-token sliding window;
    for token at absolute position p in seq, we use whichever window sees
    it at position t. For span analysis, we take activations from the
    ORIGINAL-position SAE (p mod T). Actually: Stacked only has 5 SAEs
    (one per window-position), so feature activation at absolute token p
    is only defined when we pick a window. We pick the window where p is
    at position t=0 — this gives one firing per absolute position.
    """
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = torch.stack([state[f"saes.{t}.W_enc"] for t in range(T_WIN)]).cuda()  # (T, d_sae, d_in)
    b_enc = torch.stack([state[f"saes.{t}.b_enc"] for t in range(T_WIN)]).cuda()  # (T, d_sae)
    b_dec = torch.stack([state[f"saes.{t}.b_dec"] for t in range(T_WIN)]).cuda()  # (T, d_in)

    N, seq_len, d_in = eval_x.shape
    # We'll use the t=0 SAE for all tokens to get a dense per-token activation
    # Because Stacked sub-SAEs see window-position-specific distributions,
    # choosing t=0 is a reasonable canonical choice (it matches how Andre does it).
    out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        B = x.shape[0]
        pre = (x - b_dec[0]) @ W_enc[0].T + b_enc[0]  # (B, seq_len, d_sae)
        _, idx = pre.topk(K, dim=-1)
        mask = torch.zeros_like(pre, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        active = mask & (pre > 0)
        out[s : s + B] = active.cpu().numpy()
    return out


@torch.no_grad()
def txcdr_activations(eval_x: torch.Tensor) -> np.ndarray:
    """Return (N, seq_len, d_sae) bool array.

    TXCDR uses shared-latent for a T-window. The natural "activation at
    token p" is the shared latent from the window starting at p (or any
    fixed window-alignment). We use the window starting at token p (so
    the token sits at window-position 0). Edge tokens (last T-1 positions)
    have no such window; we fall back to window starting at seq_len-T.
    """
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = state["W_enc"].cuda()  # (T, d_in, d_sae)
    b_enc = state["b_enc"].cuda()
    k_total = K * T_WIN

    N, seq_len, d_in = eval_x.shape
    out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        B = x.shape[0]
        # For each window-start position p in [0, seq_len - T], compute shared z
        # and assign to out[:, p, :]. Remaining tokens at end use last window.
        for p in range(seq_len - T_WIN + 1):
            window = x[:, p : p + T_WIN, :]  # (B, T, d_in)
            pre = torch.einsum("btd,tds->bs", window, W_enc) + b_enc
            _, idx = pre.topk(k_total, dim=-1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            active = mask & (pre > 0)
            out[s : s + B, p, :] = active.cpu().numpy()
        # For tail positions, copy the last-window activations
        for p in range(seq_len - T_WIN + 1, seq_len):
            out[s : s + B, p, :] = out[s : s + B, seq_len - T_WIN, :]
    return out


@torch.no_grad()
def tfa_activations(eval_x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return (novel_active, pred_active), both (N, seq_len, d_sae) bool."""
    from src.architectures._tfa_module import TemporalSAE

    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=True, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    # Scaling factor from a sample batch
    scaling_factor = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()

    N, seq_len, _ = eval_x.shape
    novel_out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    pred_out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = (eval_x[s : s + 16].cuda() * scaling_factor)
        B = x.shape[0]
        _, inter = model(x)
        novel_out[s : s + B] = (inter["novel_codes"] > 0).cpu().numpy()
        pred_out[s : s + B] = (inter["pred_codes"].abs() > 1e-6).cpu().numpy()
    return novel_out, pred_out


def aggregate_spans(active: np.ndarray, alive_mask: torch.Tensor,
                    max_features: int = 500) -> list[int]:
    """Collect span lengths across all sequences and alive features.

    active: (N, seq_len, d_sae) bool. alive_mask: (d_sae,) bool.
    Returns a flat list of span lengths.
    """
    alive_idx = torch.where(alive_mask)[0].numpy()
    # To keep this tractable, sample up to max_features features
    if len(alive_idx) > max_features:
        np.random.seed(0)
        alive_idx = np.random.choice(alive_idx, max_features, replace=False)

    all_spans = []
    N = active.shape[0]
    for fi in alive_idx:
        feat_active = active[:, :, fi]  # (N, seq_len)
        for seq_i in range(N):
            row = torch.from_numpy(feat_active[seq_i])
            all_spans.extend(compute_spans(row))
    return all_spans


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {N_EVAL_SEQ} eval sequences...")
    arr = np.load(ACT_PATH, mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-N_EVAL_SEQ:])).float()
    print(f"  eval_x: {list(eval_x.shape)}")

    # Load alive masks from previous analysis
    freqs = torch.load(
        "/home/elysium/temp_xc/results/analysis/decoder_alignment/firing_frequencies.pt",
    )
    alive_stacked = freqs["freq_stacked"] > ALIVE_THRESHOLD
    alive_txcdr = freqs["freq_txcdr"] > ALIVE_THRESHOLD
    alive_tfa_novel = freqs["freq_tfa_novel"] > ALIVE_THRESHOLD
    alive_tfa_pred = freqs["freq_tfa_pred"] > ALIVE_THRESHOLD

    print(f"\nAlive counts:")
    print(f"  Stacked:    {alive_stacked.sum().item()}")
    print(f"  TXCDR:      {alive_txcdr.sum().item()}")
    print(f"  TFA novel:  {alive_tfa_novel.sum().item()}")
    print(f"  TFA pred:   {alive_tfa_pred.sum().item()}")

    print("\nComputing per-architecture activation grids...")

    print("  Stacked...")
    active_stacked = stacked_activations(eval_x)
    torch.cuda.empty_cache()

    print("  TXCDR...")
    active_txcdr = txcdr_activations(eval_x)
    torch.cuda.empty_cache()

    print("  TFA (novel + pred)...")
    active_tfa_novel, active_tfa_pred = tfa_activations(eval_x)
    torch.cuda.empty_cache()

    print("\nComputing activation spans (sampling 500 alive features each)...")

    spans = {}
    print("  Stacked...")
    spans["stacked"] = aggregate_spans(active_stacked, alive_stacked)
    print(f"    {len(spans['stacked'])} spans, mean={np.mean(spans['stacked']):.2f}")

    print("  TXCDR...")
    spans["txcdr"] = aggregate_spans(active_txcdr, alive_txcdr)
    print(f"    {len(spans['txcdr'])} spans, mean={np.mean(spans['txcdr']):.2f}")

    print("  TFA novel...")
    spans["tfa_novel"] = aggregate_spans(active_tfa_novel, alive_tfa_novel)
    print(f"    {len(spans['tfa_novel'])} spans, mean={np.mean(spans['tfa_novel']):.2f}")

    print("  TFA pred...")
    spans["tfa_pred"] = aggregate_spans(active_tfa_pred, alive_tfa_pred)
    print(f"    {len(spans['tfa_pred'])} spans, mean={np.mean(spans['tfa_pred']):.2f}")

    # Summary statistics
    def stats(s):
        if not s:
            return {}
        a = np.array(s)
        return {
            "n": int(len(a)),
            "mean": float(a.mean()),
            "median": float(np.median(a)),
            "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)),
            "p90": float(np.percentile(a, 90)),
            "p99": float(np.percentile(a, 99)),
            "frac_longer_than_5":  float((a > 5).mean()),
            "frac_longer_than_10": float((a > 10).mean()),
            "frac_longer_than_20": float((a > 20).mean()),
        }

    summary = {name: stats(s) for name, s in spans.items()}

    print("\nSummary:")
    print(f"{'arch':12s} {'mean':>6s} {'median':>7s} {'p90':>5s} {'p99':>5s} "
          f"{'>5':>6s} {'>10':>6s} {'>20':>6s}")
    for name, st in summary.items():
        print(f"{name:12s} {st['mean']:>6.2f} {st['median']:>7.1f} {st['p90']:>5.1f} "
              f"{st['p99']:>5.1f} {st['frac_longer_than_5']:>6.3f} "
              f"{st['frac_longer_than_10']:>6.3f} {st['frac_longer_than_20']:>6.3f}")

    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Linear scale histogram (short spans)
    bins = np.arange(1, 30)
    for name, s in spans.items():
        if not s:
            continue
        axes[0].hist(s, bins=bins, alpha=0.5, label=name, density=True)
    axes[0].set_xlabel("span length (tokens)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Activation span distribution (linear)")
    axes[0].legend()
    axes[0].set_xlim(0.5, 30)

    # Log-log complementary CDF (shows tail)
    for name, s in spans.items():
        if not s:
            continue
        a = np.sort(np.array(s))
        ccdf = 1.0 - np.arange(len(a)) / len(a)
        axes[1].loglog(a, ccdf, label=name)
    axes[1].set_xlabel("span length")
    axes[1].set_ylabel("P(span >= x)")
    axes[1].set_title("Tail of span distribution (log-log)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/span_histograms.png", dpi=120)
    print(f"\n  -> {OUT_DIR}/span_histograms.png")


if __name__ == "__main__":
    main()
