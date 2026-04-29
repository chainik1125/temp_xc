#!/usr/bin/env python3
"""Stage 3: SAE feature-space Difference-of-Means for backtracking.

Loads `fnlp/Llama-Scope` (LXR-8x at the configured layer) via sae_lens, encodes
the cached L10 activations in batches, and computes per-feature means over D₊
(positions 13..8 tokens before each backtracking event) and over D (every
position in the `<think>` region). The "backtracking direction" in feature
space is then:

    Δ_j  =  mean(z_j | r ∈ D₊)  −  mean(z_j | r ∈ D)

The top-K features by `|Δ_j|` are the SAE analogue of the paper's raw-DoM
direction `v`.

Outputs (in DECOMPOSE_DIR):
    feature_stats.npz        per-feature: mean_plus, mean_all, delta, n_active_plus
    top_features.json        top-50 by |Δ_j| with metadata for Stage 4
    raw_dom.fp16.npy         raw activation-space DoM vector (for the paper baseline)
    decompose_meta.json      configuration + n_d_plus / n_d_all token counts

Memory: streams encoded features one batch at a time; never materialises the
full (total_tokens, n_sae_features) tensor.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    ANCHOR_LAYER,
    CACHE_DIR,
    DECOMPOSE_DIR,
    LABELS_DIR,
    PUBLIC_SAE_CONFIG,
    PUBLIC_SAE_REPO,
    ensure_dirs,
)


# ── public-SAE loader ───────────────────────────────────────────────────────
def load_public_sae(layer: int, device: str):
    """Load a Llama-Scope residual SAE at `layer` via sae_lens.

    sae_lens pretrained registry uses release id `llama_scope_lxr_8x` and
    sae_id `l{layer}r_8x` (lowercase). If the pinned sae_lens version
    doesn't recognise the release, raise with an actionable message.
    """
    try:
        from sae_lens import SAE
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "sae_lens is required. The repo pins sae-lens>=6.35; "
            f"original error: {e}"
        )
    release = "llama_scope_lxr_8x"
    sae_id = f"l{layer}r_8x"
    print(f"[decompose] SAE.from_pretrained(release={release!r}, sae_id={sae_id!r})")
    sae, cfg_dict, sparsity = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, cfg_dict


# ── label assembly ─────────────────────────────────────────────────────────
def _build_masks(
    trace_ids: list[str], offsets: np.ndarray, labels_path: Path
) -> tuple[np.ndarray, np.ndarray]:
    """Return (mask_plus, mask_all) bool arrays of length total_tokens."""
    total = int(offsets[-1])
    mask_plus = np.zeros(total, dtype=bool)
    mask_all = np.zeros(total, dtype=bool)
    id_to_idx = {tid: i for i, tid in enumerate(trace_ids)}
    n_traces_with_events = 0
    n_events = 0
    with labels_path.open() as f:
        for line in f:
            lab = json.loads(line)
            i = id_to_idx.get(lab["trace_id"])
            if i is None:
                continue
            base = int(offsets[i])
            seq_len = int(offsets[i + 1] - offsets[i])
            for p in lab["d_plus_positions"]:
                if 0 <= p < seq_len:
                    mask_plus[base + p] = True
            think_lo = lab["think_lo"]
            think_hi = lab["think_hi"]
            for p in range(think_lo, min(think_hi, seq_len)):
                mask_all[base + p] = True
            if lab["n_events"] > 0:
                n_traces_with_events += 1
                n_events += lab["n_events"]
    print(
        f"[decompose] masks: |D_+|={mask_plus.sum()} |D|={mask_all.sum()} "
        f"events={n_events} traces_with_events={n_traces_with_events}"
    )
    return mask_plus, mask_all


# ── feature-space DoM via streamed encoding ─────────────────────────────────
def streamed_feature_stats(
    sae,
    activations: np.memmap,
    mask_plus: np.ndarray,
    mask_all: np.ndarray,
    batch: int,
    device: str,
) -> dict[str, np.ndarray]:
    """Streamed first + second moments per feature on D_+ and D.

    Returns a dict with keys mean_plus, mean_all, var_plus, var_all (each
    np.float32 of length d_sae). Variances use the unbiased (n-1) denominator.
    """
    d_sae = int(sae.cfg.d_sae if hasattr(sae.cfg, "d_sae") else sae.W_dec.shape[0])
    sum_plus = torch.zeros(d_sae, dtype=torch.float64, device=device)
    sum_all = torch.zeros(d_sae, dtype=torch.float64, device=device)
    sumsq_plus = torch.zeros(d_sae, dtype=torch.float64, device=device)
    sumsq_all = torch.zeros(d_sae, dtype=torch.float64, device=device)
    n_plus = int(mask_plus.sum())
    n_all = int(mask_all.sum())

    n = int(activations.shape[0])
    for s in range(0, n, batch):
        e = min(n, s + batch)
        x = torch.from_numpy(np.asarray(activations[s:e], dtype=np.float32)).to(device)
        with torch.no_grad():
            z = sae.encode(x).to(torch.float64)
        mp = torch.from_numpy(mask_plus[s:e]).to(device)
        ma = torch.from_numpy(mask_all[s:e]).to(device)
        if mp.any():
            zp = z[mp]
            sum_plus += zp.sum(dim=0)
            sumsq_plus += (zp * zp).sum(dim=0)
        if ma.any():
            za = z[ma]
            sum_all += za.sum(dim=0)
            sumsq_all += (za * za).sum(dim=0)

    def _moments(_sum, _sumsq, _n):
        if _n == 0:
            return np.zeros(d_sae, dtype=np.float32), np.zeros(d_sae, dtype=np.float32)
        mean = _sum / _n
        if _n > 1:
            var = (_sumsq - _n * mean * mean) / (_n - 1)
            var = torch.clamp(var, min=0.0)
        else:
            var = torch.zeros_like(_sum)
        return mean.cpu().numpy().astype(np.float32), var.cpu().numpy().astype(np.float32)

    mean_plus, var_plus = _moments(sum_plus, sumsq_plus, n_plus)
    mean_all, var_all = _moments(sum_all, sumsq_all, n_all)
    return {
        "mean_plus": mean_plus,
        "mean_all": mean_all,
        "var_plus": var_plus,
        "var_all": var_all,
    }


def welch_tstat(mean_plus, mean_all, var_plus, var_all, n_plus, n_all):
    """Welch's two-sample t-statistic per feature. SE floor avoids div-by-0
    on features that never fire (zero variance) — those get t=0."""
    se = np.sqrt(var_plus / max(n_plus, 1) + var_all / max(n_all, 1))
    tstat = np.where(se > 1e-12, (mean_plus - mean_all) / np.maximum(se, 1e-12), 0.0)
    return tstat.astype(np.float32)


def streamed_active_count_in_plus(
    sae,
    activations: np.memmap,
    mask_plus: np.ndarray,
    batch: int,
    device: str,
) -> np.ndarray:
    """Per-feature: number of D_+ positions where the feature fires (z>0)."""
    d_sae = int(sae.cfg.d_sae if hasattr(sae.cfg, "d_sae") else sae.W_dec.shape[0])
    n_active = torch.zeros(d_sae, dtype=torch.int64, device=device)
    n = int(activations.shape[0])
    for s in range(0, n, batch):
        e = min(n, s + batch)
        mp = mask_plus[s:e]
        if not mp.any():
            continue
        x = torch.from_numpy(np.asarray(activations[s:e][mp], dtype=np.float32)).to(device)
        with torch.no_grad():
            z = sae.encode(x)
        n_active += (z > 0).sum(dim=0).to(torch.int64)
    return n_active.cpu().numpy().astype(np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--rank-by",
        choices=("tstat", "delta", "ratio"),
        default="tstat",
        help="ranking metric for top_features.json. tstat = Welch two-sample (default; combines effect size and noise). delta = raw mean(D_+) − mean(D), favours always-active features. ratio = mean(D_+) / max(mean(D), eps), favours selective features but unstable for sparse features.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_stats = DECOMPOSE_DIR / "feature_stats.npz"
    out_top = DECOMPOSE_DIR / "top_features.json"
    out_raw = DECOMPOSE_DIR / "raw_dom.fp16.npy"
    out_meta = DECOMPOSE_DIR / "decompose_meta.json"
    if out_stats.exists() and not args.force:
        print(f"[decompose] {out_stats} exists; use --force to rebuild")
        return

    act_path = CACHE_DIR / "activations.fp16.npy"
    off_path = CACHE_DIR / "offsets.npy"
    ids_path = CACHE_DIR / "trace_ids.json"
    labels_path = LABELS_DIR / "labels.jsonl"
    for p in (act_path, off_path, ids_path, labels_path):
        if not p.exists():
            raise SystemExit(f"missing {p}; run earlier stages first")

    activations = np.load(act_path, mmap_mode="r")
    offsets = np.load(off_path)
    trace_ids = json.loads(ids_path.read_text())
    n_tokens, d_model = activations.shape
    print(f"[decompose] activations {activations.shape} {activations.dtype}; offsets[-1]={offsets[-1]}")
    assert n_tokens == int(offsets[-1])

    mask_plus, mask_all = _build_masks(trace_ids, offsets, labels_path)

    # Raw activation-space DoM (for the paper baseline in Stage 4).
    sum_plus = np.zeros(d_model, dtype=np.float64)
    sum_all = np.zeros(d_model, dtype=np.float64)
    n_plus = int(mask_plus.sum())
    n_all = int(mask_all.sum())
    BATCH_R = 16384
    for s in range(0, n_tokens, BATCH_R):
        e = min(n_tokens, s + BATCH_R)
        a = np.asarray(activations[s:e], dtype=np.float32)
        if mask_plus[s:e].any():
            sum_plus += a[mask_plus[s:e]].sum(axis=0)
        if mask_all[s:e].any():
            sum_all += a[mask_all[s:e]].sum(axis=0)
    mean_plus_raw = sum_plus / max(n_plus, 1)
    mean_all_raw = sum_all / max(n_all, 1)
    raw_dom = (mean_plus_raw - mean_all_raw).astype(np.float16)
    np.save(out_raw, raw_dom)
    print(f"[decompose] raw_dom |v|={np.linalg.norm(raw_dom.astype(np.float32)):.3f} → {out_raw}")

    # Feature-space DoM via Llama-Scope.
    print(f"[decompose] loading Llama-Scope at layer {args.layer} on {args.device}…")
    sae, sae_cfg = load_public_sae(args.layer, args.device)
    t0 = time.time()
    stats = streamed_feature_stats(
        sae, activations, mask_plus, mask_all, args.batch, args.device
    )
    mean_plus = stats["mean_plus"]
    mean_all = stats["mean_all"]
    var_plus = stats["var_plus"]
    var_all = stats["var_all"]
    delta = mean_plus - mean_all
    tstat = welch_tstat(mean_plus, mean_all, var_plus, var_all, n_plus, n_all)
    ratio = mean_plus / np.maximum(mean_all, 1e-6)
    n_active_plus = streamed_active_count_in_plus(sae, activations, mask_plus, args.batch, args.device)
    print(f"[decompose] feature stats in {time.time()-t0:.1f}s; d_sae={delta.shape[0]}")

    np.savez(
        out_stats,
        mean_plus=mean_plus,
        mean_all=mean_all,
        var_plus=var_plus,
        var_all=var_all,
        delta=delta,
        tstat=tstat,
        ratio=ratio,
        n_active_plus=n_active_plus,
    )

    # Rank features by the selected metric. Default is t-stat (combines effect
    # size and noise); the smoke run at N=20 showed rank-by-|delta| favours
    # always-active features (e.g. "general thinking" feat_10750), while the
    # actually-steerable backtracking feature (feat_27749) ranked 3rd by |delta|
    # but 1st by both ratio and t-stat.
    if args.rank_by == "tstat":
        score = tstat
    elif args.rank_by == "delta":
        score = delta
    elif args.rank_by == "ratio":
        # Cap ratio when mean_all is tiny — these are usually noise spikes.
        score = np.where(mean_all > 0.01, ratio, 0.0)
    else:
        raise ValueError(args.rank_by)
    top_idx = np.argsort(-score)[: args.top_k]
    W_dec = sae.W_dec.detach().cpu().numpy()  # (d_sae, d_model) for Llama-Scope
    if W_dec.shape != (delta.shape[0], d_model):  # be defensive across versions
        # Some SAE-Lens variants store W_dec as (d_model, d_sae); reshape.
        if W_dec.shape == (d_model, delta.shape[0]):
            W_dec = W_dec.T
        else:
            raise RuntimeError(f"unexpected W_dec shape {W_dec.shape}")
    top_records = []
    for rank, j in enumerate(top_idx.tolist()):
        top_records.append(
            {
                "rank": rank,
                "feature_idx": int(j),
                "delta": float(delta[j]),
                "tstat": float(tstat[j]),
                "ratio": float(ratio[j]),
                "mean_plus": float(mean_plus[j]),
                "mean_all": float(mean_all[j]),
                "var_plus": float(var_plus[j]),
                "var_all": float(var_all[j]),
                "n_active_plus": int(n_active_plus[j]),
                "decoder_norm": float(np.linalg.norm(W_dec[j])),
            }
        )
    out_top.write_text(json.dumps(top_records, indent=2))
    print(f"[decompose] top-{args.top_k} (rank_by={args.rank_by}) → {out_top}")

    meta = {
        "layer": args.layer,
        "sae_release": "llama_scope_lxr_8x",
        "sae_id": f"l{args.layer}r_8x",
        "sae_repo": PUBLIC_SAE_REPO,
        "sae_config": PUBLIC_SAE_CONFIG,
        "rank_by": args.rank_by,
        "n_tokens": int(n_tokens),
        "n_d_plus": int(n_plus),
        "n_d_all": int(n_all),
        "d_model": int(d_model),
        "d_sae": int(delta.shape[0]),
    }
    out_meta.write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
