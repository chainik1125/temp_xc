"""High-span feature comparison: does TXCDR extract temporal features
substantially different from what TFA and Stacked extract?

Steps:
  1. Per-feature mean-span: for each alive feature in each architecture,
     compute the mean length of its activation bursts across 500 eval seqs.
  2. Identify top-100 high-span features per architecture (the "temporal"
     subset of each).
  3. Alignment on the temporal subset: compute pairwise decoder cosine sim
     matrices restricted to each architecture's top-span features. Compare
     to a random-subset control. If aligned-above-random => temporal
     features overlap architecturally; if not => each arch finds DIFFERENT
     temporal features.
  4. Activation co-firing: for each TXCDR high-span feature, find its best
     activation-correlated partner in TFA-pred's high-span set (and in
     Stacked's). Compare correlation distribution against shuffled/random
     pairs.

Outputs: results/analysis/high_span_comparison/
    per_feature_span.pt
    high_span_alignment.json
    cofiring_distribution.png
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
ACT_PATH = "/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/resid_L25.npy"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/high_span_comparison"
ALIGN_DIR = "/home/elysium/temp_xc/results/analysis/decoder_alignment"
D_IN = 2304
D_SAE = 18432
T_WIN = 5
K = 100
LAYER = "resid_L25"
N_EVAL = 500
N_TOP_SPAN = 100
ALIVE_THRESHOLD = 0.0001
# Pad/BOS are not of interest — exclude them from span computation
PAD_ID = 0
BOS_ID = 2


def unit_norm_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.clip(norms, 1e-8, None)


def mean_span_per_feature(active: np.ndarray, mask: np.ndarray,
                          chunk: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized per-feature mean activation-span length.

    active: (N, seq_len, d_sae) bool
    mask: (N, seq_len) bool — only count activations at content positions
    Returns (mean_span, span_count) each (d_sae,).

    Uses: mean_span = total_active_positions / number_of_runs
    where number_of_runs is the count of (0 -> 1) transitions, computed
    by np.diff along the sequence axis with zero-padding at the boundaries.
    """
    N, seq_len, d_sae = active.shape
    mean_spans = np.zeros(d_sae, dtype=np.float32)
    span_counts = np.zeros(d_sae, dtype=np.int64)
    mask_bool = mask.astype(bool)

    # Chunk over features to keep memory bounded
    for start in range(0, d_sae, chunk):
        end = min(start + chunk, d_sae)
        # (N, seq_len, C) bool AND mask per (N, seq_len)
        feat = active[:, :, start:end] & mask_bool[:, :, None]  # (N, seq_len, C)
        # Pad with False on both ends of axis=1
        pad = np.zeros((N, 1, end - start), dtype=np.int8)
        padded = np.concatenate([pad, feat.astype(np.int8), pad], axis=1)
        diff = np.diff(padded, axis=1)  # (N, seq_len + 1, C)
        # Count 0→1 transitions = number of runs
        n_runs = (diff == 1).sum(axis=(0, 1))  # (C,)
        # Total active positions
        total_active = feat.sum(axis=(0, 1))  # (C,)
        span_counts[start:end] = n_runs
        mean_spans[start:end] = np.where(
            n_runs > 0, total_active / np.maximum(n_runs, 1), 0.0
        ).astype(np.float32)
        print(f"    span {end}/{d_sae}...", flush=True)
    return mean_spans, span_counts


@torch.no_grad()
def stacked_activations(eval_x: torch.Tensor) -> np.ndarray:
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = state["saes.0.W_enc"].cuda()
    b_enc = state["saes.0.b_enc"].cuda()
    b_dec = state["saes.0.b_dec"].cuda()
    N, seq_len, _ = eval_x.shape
    out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        pre = (x - b_dec) @ W_enc.T + b_enc
        _, idx = pre.topk(K, dim=-1)
        mask = torch.zeros_like(pre, dtype=torch.bool)
        mask.scatter_(-1, idx, True)
        out[s : s + x.shape[0]] = (mask & (pre > 0)).cpu().numpy()
    return out


@torch.no_grad()
def txcdr_activations(eval_x: torch.Tensor) -> np.ndarray:
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_enc = state["W_enc"].cuda()
    b_enc = state["b_enc"].cuda()
    k_total = K * T_WIN
    N, seq_len, _ = eval_x.shape
    out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda()
        B = x.shape[0]
        for w in range(seq_len - T_WIN + 1):
            window = x[:, w : w + T_WIN, :]
            pre = torch.einsum("btd,tds->bs", window, W_enc) + b_enc
            _, idx = pre.topk(k_total, dim=-1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            active = (mask & (pre > 0)).cpu().numpy()
            out[s : s + B, w, :] = active
        for w in range(seq_len - T_WIN + 1, seq_len):
            out[s : s + B, w, :] = out[s : s + B, seq_len - T_WIN, :]
    return out


@torch.no_grad()
def tfa_pred_activations(eval_x: torch.Tensor) -> np.ndarray:
    from src.bench.architectures._tfa_module import TemporalSAE
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
    scale = math.sqrt(D_IN) / eval_x[:16].norm(dim=-1).mean().item()
    N, seq_len, _ = eval_x.shape
    out = np.zeros((N, seq_len, D_SAE), dtype=bool)
    for s in range(0, N, 16):
        x = eval_x[s : s + 16].cuda() * scale
        _, inter = model(x)
        out[s : s + x.shape[0]] = (inter["pred_codes"].abs() > 1e-6).cpu().numpy()
    return out


def best_match_alignment(
    a: np.ndarray, b: np.ndarray, batch: int = 256
) -> np.ndarray:
    """For each row in a, max |cos| with any row in b. Both must be unit-normed."""
    a_t = torch.from_numpy(a).float().cuda()
    b_t = torch.from_numpy(b).float().cuda()
    out = np.zeros(a.shape[0], dtype=np.float32)
    for s in range(0, a.shape[0], batch):
        sims = (a_t[s : s + batch] @ b_t.T).abs()
        out[s : s + batch] = sims.max(dim=-1).values.cpu().numpy()
    return out


def load_decoders() -> dict[str, np.ndarray]:
    out = {}
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W = torch.stack([state[f"saes.{t}.W_dec"] for t in range(T_WIN)]).mean(dim=0).T
    out["stacked"] = unit_norm_rows(W.numpy())
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    out["txcdr"] = unit_norm_rows(state["W_dec"].mean(dim=1).numpy())
    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    out["tfa_pred"] = unit_norm_rows(state["D"].numpy())  # shared D; use for pred subset
    return out


def cofiring_correlations(
    active_a: np.ndarray, active_b: np.ndarray,
    features_a: list[int], features_b: list[int],
) -> np.ndarray:
    """Pearson correlation matrix (|features_a|, |features_b|) of binary firing.

    Computes over all (seq, token) positions, flattened to 1D.
    """
    # Flatten
    N, T = active_a.shape[:2]
    # Extract just selected feature columns
    sub_a = active_a[:, :, features_a].reshape(N * T, -1).astype(np.float32)
    sub_b = active_b[:, :, features_b].reshape(N * T, -1).astype(np.float32)
    # Center
    sub_a -= sub_a.mean(axis=0)
    sub_b -= sub_b.mean(axis=0)
    # Denominator: norms along positions-axis
    na = np.linalg.norm(sub_a, axis=0) + 1e-8
    nb = np.linalg.norm(sub_b, axis=0) + 1e-8
    corr = (sub_a.T @ sub_b) / (na[:, None] * nb[None, :])
    return corr  # (nA, nB)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading eval data...")
    arr = np.load(ACT_PATH, mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-N_EVAL:])).float()
    tok_path = ACT_PATH.replace(f"{LAYER}.npy", "token_ids.npy")
    eval_tok = np.array(np.load(tok_path, mmap_mode="r")[-N_EVAL:])
    content_mask = (eval_tok != PAD_ID) & (eval_tok != BOS_ID)  # (N, seq_len)
    print(f"  eval_x: {list(eval_x.shape)}")

    print("\n[1/4] Computing activation grids...")
    print("  Stacked...")
    act_stacked = stacked_activations(eval_x)
    torch.cuda.empty_cache()
    print("  TXCDR...")
    act_txcdr = txcdr_activations(eval_x)
    torch.cuda.empty_cache()
    print("  TFA pred...")
    act_tfa = tfa_pred_activations(eval_x)
    torch.cuda.empty_cache()

    print("\n[2/4] Per-feature mean span (this takes ~5-10 min per arch)...")
    # Only compute on alive features; others are 0
    freqs = torch.load(f"{ALIGN_DIR}/firing_frequencies.pt")
    alive_stacked = (freqs["freq_stacked"] > ALIVE_THRESHOLD).numpy()
    alive_txcdr = (freqs["freq_txcdr"] > ALIVE_THRESHOLD).numpy()
    alive_tfa = (freqs["freq_tfa_pred"] > ALIVE_THRESHOLD).numpy()

    print("  Stacked spans...")
    span_stacked, count_stacked = mean_span_per_feature(act_stacked, content_mask)
    print("  TXCDR spans...")
    span_txcdr, count_txcdr = mean_span_per_feature(act_txcdr, content_mask)
    print("  TFA-pred spans...")
    span_tfa, count_tfa = mean_span_per_feature(act_tfa, content_mask)

    torch.save({
        "span_stacked": torch.from_numpy(span_stacked),
        "count_stacked": torch.from_numpy(count_stacked),
        "span_txcdr": torch.from_numpy(span_txcdr),
        "count_txcdr": torch.from_numpy(count_txcdr),
        "span_tfa_pred": torch.from_numpy(span_tfa),
        "count_tfa_pred": torch.from_numpy(count_tfa),
    }, f"{OUT_DIR}/per_feature_span.pt")

    # Require at least 20 spans to have a reliable estimate
    MIN_COUNT = 20
    reliable_stacked = alive_stacked & (count_stacked >= MIN_COUNT)
    reliable_txcdr = alive_txcdr & (count_txcdr >= MIN_COUNT)
    reliable_tfa = alive_tfa & (count_tfa >= MIN_COUNT)

    print(f"\n  Reliable features (alive + >={MIN_COUNT} spans):")
    print(f"    Stacked:  {reliable_stacked.sum()}")
    print(f"    TXCDR:    {reliable_txcdr.sum()}")
    print(f"    TFA-pred: {reliable_tfa.sum()}")

    # Top-N by span length, restricted to reliable features
    def top_n_span(span, reliable, n):
        idx = np.where(reliable)[0]
        sorted_idx = idx[np.argsort(-span[idx])]
        return sorted_idx[:n]

    top_stacked = top_n_span(span_stacked, reliable_stacked, N_TOP_SPAN)
    top_txcdr = top_n_span(span_txcdr, reliable_txcdr, N_TOP_SPAN)
    top_tfa = top_n_span(span_tfa, reliable_tfa, N_TOP_SPAN)

    print(f"\n  Top-{N_TOP_SPAN} span stats:")
    print(f"    Stacked   mean={span_stacked[top_stacked].mean():.2f}  max={span_stacked[top_stacked].max():.2f}")
    print(f"    TXCDR     mean={span_txcdr[top_txcdr].mean():.2f}  max={span_txcdr[top_txcdr].max():.2f}")
    print(f"    TFA-pred  mean={span_tfa[top_tfa].mean():.2f}  max={span_tfa[top_tfa].max():.2f}")

    print("\n[3/4] Decoder alignment on high-span subsets...")
    D = load_decoders()

    # For each pair, compute alignment of (arch A high-span) features vs
    # (arch B high-span) features. Control = random reliable features.
    def run_alignment_pair(arch_a, arch_b, top_a, top_b, reliable_a, reliable_b):
        Da = D[arch_a][top_a]
        Db = D[arch_b][top_b]
        align_highspan = best_match_alignment(Da, Db)
        # Control: random subset of reliable features
        idx_a = np.where(reliable_a)[0]
        idx_b = np.where(reliable_b)[0]
        rng = np.random.RandomState(42)
        rand_a = rng.choice(idx_a, min(N_TOP_SPAN, len(idx_a)), replace=False)
        rand_b = rng.choice(idx_b, min(N_TOP_SPAN, len(idx_b)), replace=False)
        align_random = best_match_alignment(D[arch_a][rand_a], D[arch_b][rand_b])
        return {
            "highspan_median": float(np.median(align_highspan)),
            "highspan_p90": float(np.percentile(align_highspan, 90)),
            "highspan_fracge0.3": float((align_highspan >= 0.3).mean()),
            "highspan_fracge0.5": float((align_highspan >= 0.5).mean()),
            "random_median": float(np.median(align_random)),
            "random_p90": float(np.percentile(align_random, 90)),
            "random_fracge0.3": float((align_random >= 0.3).mean()),
            "random_fracge0.5": float((align_random >= 0.5).mean()),
            "highspan_values": align_highspan.tolist(),
            "random_values": align_random.tolist(),
        }

    alignment_pairs = {
        "txcdr_to_tfa":     run_alignment_pair("txcdr",    "tfa_pred", top_txcdr,   top_tfa,     reliable_txcdr,   reliable_tfa),
        "txcdr_to_stacked": run_alignment_pair("txcdr",    "stacked",  top_txcdr,   top_stacked, reliable_txcdr,   reliable_stacked),
        "tfa_to_stacked":   run_alignment_pair("tfa_pred", "stacked",  top_tfa,     top_stacked, reliable_tfa,     reliable_stacked),
        "tfa_to_txcdr":     run_alignment_pair("tfa_pred", "txcdr",    top_tfa,     top_txcdr,   reliable_tfa,     reliable_txcdr),
        "stacked_to_txcdr": run_alignment_pair("stacked",  "txcdr",    top_stacked, top_txcdr,   reliable_stacked, reliable_txcdr),
        "stacked_to_tfa":   run_alignment_pair("stacked",  "tfa_pred", top_stacked, top_tfa,     reliable_stacked, reliable_tfa),
    }

    print("\n  Pair (arch_a → arch_b): median_highspan / median_random  (>=0.5: highspan / random)")
    for pair, stats in alignment_pairs.items():
        print(f"    {pair:22s}  {stats['highspan_median']:.3f} / {stats['random_median']:.3f}  "
              f"({stats['highspan_fracge0.5']:.3f} / {stats['random_fracge0.5']:.3f})")

    print("\n[4/4] Activation co-firing between TXCDR and TFA-pred high-span features...")
    # For each high-span TXCDR feature, find best-correlated TFA-pred high-span feature
    corr_txcdr_tfa = cofiring_correlations(act_txcdr, act_tfa, list(top_txcdr), list(top_tfa))
    best_per_txcdr = np.abs(corr_txcdr_tfa).max(axis=1)
    corr_txcdr_stacked = cofiring_correlations(act_txcdr, act_stacked, list(top_txcdr), list(top_stacked))
    best_per_txcdr_to_stacked = np.abs(corr_txcdr_stacked).max(axis=1)
    # Control: random high-span features vs random others
    rng = np.random.RandomState(0)
    rand_txcdr = np.where(reliable_txcdr)[0]
    rand_tfa = np.where(reliable_tfa)[0]
    rand_pick_txcdr = rng.choice(rand_txcdr, N_TOP_SPAN, replace=False)
    rand_pick_tfa = rng.choice(rand_tfa, N_TOP_SPAN, replace=False)
    corr_random = cofiring_correlations(act_txcdr, act_tfa, list(rand_pick_txcdr), list(rand_pick_tfa))
    best_random = np.abs(corr_random).max(axis=1)

    print(f"  |corr|(TXCDR-highspan → best TFA-pred highspan):  median={np.median(best_per_txcdr):.3f}")
    print(f"  |corr|(TXCDR-highspan → best Stacked highspan):    median={np.median(best_per_txcdr_to_stacked):.3f}")
    print(f"  |corr|(random TXCDR → best random TFA, control):   median={np.median(best_random):.3f}")

    summary = {
        "n_top_span": N_TOP_SPAN,
        "min_spans_required": MIN_COUNT,
        "n_reliable": {
            "stacked": int(reliable_stacked.sum()),
            "txcdr": int(reliable_txcdr.sum()),
            "tfa_pred": int(reliable_tfa.sum()),
        },
        "top_span_mean": {
            "stacked": float(span_stacked[top_stacked].mean()),
            "txcdr":   float(span_txcdr[top_txcdr].mean()),
            "tfa_pred": float(span_tfa[top_tfa].mean()),
        },
        "top_span_max": {
            "stacked": float(span_stacked[top_stacked].max()),
            "txcdr":   float(span_txcdr[top_txcdr].max()),
            "tfa_pred": float(span_tfa[top_tfa].max()),
        },
        "alignment": {k: {kk: vv for kk, vv in v.items() if not kk.endswith("_values")}
                       for k, v in alignment_pairs.items()},
        "cofiring": {
            "txcdr_highspan_best_tfa_median":     float(np.median(best_per_txcdr)),
            "txcdr_highspan_best_tfa_p90":        float(np.percentile(best_per_txcdr, 90)),
            "txcdr_highspan_best_stacked_median": float(np.median(best_per_txcdr_to_stacked)),
            "txcdr_highspan_best_stacked_p90":    float(np.percentile(best_per_txcdr_to_stacked, 90)),
            "random_ctrl_median":                 float(np.median(best_random)),
            "random_ctrl_p90":                    float(np.percentile(best_random, 90)),
        },
        "feature_indices": {
            "top_span_stacked": top_stacked.tolist(),
            "top_span_txcdr":   top_txcdr.tolist(),
            "top_span_tfa_pred": top_tfa.tolist(),
        },
    }
    with open(f"{OUT_DIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plots
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # Panel 1: alignment histogram, high-span vs random, for the TXCDR↔TFA pair
    for key, label, color in [
        ("txcdr_to_tfa", "TXCDR high-span → TFA-pred high-span", "tab:red"),
        ("tfa_to_txcdr", "TFA-pred high-span → TXCDR high-span", "tab:orange"),
        ("txcdr_to_stacked", "TXCDR high-span → Stacked high-span", "tab:blue"),
    ]:
        v = alignment_pairs[key]["highspan_values"]
        axes[0].hist(v, bins=30, alpha=0.45, label=label, color=color, density=True)
    axes[0].axvline(alignment_pairs["txcdr_to_tfa"]["random_median"], color="gray",
                     linestyle="--", alpha=0.7, label="random-ctrl median")
    axes[0].set_xlabel("best decoder cos-sim with target arch")
    axes[0].set_ylabel("density")
    axes[0].set_title("Decoder alignment on HIGH-SPAN subsets")
    axes[0].legend(fontsize=8)

    # Panel 2: co-firing correlation distribution
    axes[1].hist(best_per_txcdr, bins=30, alpha=0.6, label="TXCDR high-span → TFA-pred",
                  color="tab:red", density=True)
    axes[1].hist(best_per_txcdr_to_stacked, bins=30, alpha=0.6,
                  label="TXCDR high-span → Stacked", color="tab:blue", density=True)
    axes[1].hist(best_random, bins=30, alpha=0.3, label="random-random control",
                  color="gray", density=True)
    axes[1].set_xlabel("|Pearson corr| of binary firing (best per TXCDR feature)")
    axes[1].set_ylabel("density")
    axes[1].set_title("Activation co-firing of high-span features")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/high_span_comparison.png", dpi=140)
    print(f"\n  -> {OUT_DIR}/high_span_comparison.png")


if __name__ == "__main__":
    main()
