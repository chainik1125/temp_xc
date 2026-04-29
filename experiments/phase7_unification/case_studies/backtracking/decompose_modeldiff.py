#!/usr/bin/env python3
"""Wang/persona-vector style model-diffing decomposition for backtracking.

Where `decompose_backtracking.py` ranks SAE features by their selectivity at
backtracking-precursor positions *within one model* (DeepSeek-R1-Distill),
this script ranks features by the *cross-model* difference at the same
positions — the SAE-feature analogue of the Ward et al. base-vs-distilled
comparison and the Wang et al. persona-vector recipe.

Procedure:
  1. Encode both the distilled-model cache (`cache_l10/`) and the
     base-Llama-3.1-8B cache (`cache_l10_base/`) through the same
     Llama-Scope SAE.
  2. For every feature j:
        Δ_j(model_diff) = mean(z_distilled[D, j]) − mean(z_base[D, j])
     across positions D (default: every think-region position).
  3. Optionally restrict D to the offset window D_+ (positions 13..8
     tokens before backtracking events) via --positions plus.
  4. Rank features by |Δ| or t-stat (--rank-by). The top features are
     "what the distilled model represents at L10 that the base model
     doesn't" — candidate backtracking-mediating features.

The result format mirrors decompose_backtracking.py so the existing
intervene / evaluate / plot pipeline reads it via --decompose-suffix
modeldiff.

Run from repo root (after both caches exist):

    python -m experiments.phase7_unification.case_studies.backtracking.decompose_modeldiff
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
    LABELS_DIR,
    PUBLIC_SAE_CONFIG,
    PUBLIC_SAE_REPO,
    RESULTS_DIR,
    ensure_dirs,
)
from experiments.phase7_unification.case_studies.backtracking.decompose_backtracking import (  # noqa: E402
    load_public_sae,
)


def _build_masks(trace_ids: list[str], offsets: np.ndarray, labels_path: Path,
                 positions: str) -> np.ndarray:
    """Return a boolean mask of length total_tokens.

    positions = "all"   → every think-region token in every trace
    positions = "plus"  → positions 13..8 before backtracking events (D_+)
    """
    total = int(offsets[-1])
    mask = np.zeros(total, dtype=bool)
    id_to_idx = {tid: i for i, tid in enumerate(trace_ids)}
    with labels_path.open() as f:
        for line in f:
            lab = json.loads(line)
            i = id_to_idx.get(lab["trace_id"])
            if i is None:
                continue
            base = int(offsets[i])
            seq_len = int(offsets[i + 1] - offsets[i])
            if positions == "plus":
                for p in lab["d_plus_positions"]:
                    if 0 <= p < seq_len:
                        mask[base + p] = True
            else:  # all
                lo = lab["think_lo"]
                hi = min(lab["think_hi"], seq_len)
                mask[base + lo : base + hi] = True
    return mask


def _streamed_means(sae, activations: np.memmap, mask: np.ndarray,
                    batch: int, device: str) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean and unbiased variance under `mask`."""
    d_sae = int(sae.cfg.d_sae if hasattr(sae.cfg, "d_sae") else sae.W_dec.shape[0])
    s = torch.zeros(d_sae, dtype=torch.float64, device=device)
    s2 = torch.zeros(d_sae, dtype=torch.float64, device=device)
    n_pts = int(mask.sum())
    n_total = int(activations.shape[0])
    for i in range(0, n_total, batch):
        e = min(n_total, i + batch)
        mb = mask[i:e]
        if not mb.any():
            continue
        x = torch.from_numpy(np.asarray(activations[i:e][mb], dtype=np.float32)).to(device)
        with torch.no_grad():
            z = sae.encode(x).to(torch.float64)
        s += z.sum(dim=0)
        s2 += (z * z).sum(dim=0)
    if n_pts == 0:
        return (
            np.zeros(d_sae, dtype=np.float32),
            np.zeros(d_sae, dtype=np.float32),
            0,
        )
    mean = s / n_pts
    var = (s2 - n_pts * mean * mean) / max(n_pts - 1, 1)
    var = torch.clamp(var, min=0.0)
    return (
        mean.cpu().numpy().astype(np.float32),
        var.cpu().numpy().astype(np.float32),
        n_pts,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=ANCHOR_LAYER)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--rank-by", choices=("tstat", "delta"), default="tstat")
    parser.add_argument("--positions", choices=("all", "plus"), default="all",
                        help="all = every think-region position (Wang/persona style); plus = D_+ only (offset window)")
    parser.add_argument("--distilled-cache-suffix", default="", help="distilled-model cache_l10<_suffix>/")
    parser.add_argument("--base-cache-suffix", default="base", help="base-model cache_l10_<suffix>/ (default 'base')")
    parser.add_argument("--decompose-suffix", default="modeldiff")
    parser.add_argument("--sae-release", default="llama_scope_lxr_8x")
    parser.add_argument("--sae-id", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    decompose_dir = RESULTS_DIR / f"decompose_{args.decompose_suffix}"
    decompose_dir.mkdir(parents=True, exist_ok=True)
    out_stats = decompose_dir / "feature_stats.npz"
    out_top = decompose_dir / "top_features.json"
    out_meta = decompose_dir / "decompose_meta.json"
    if out_stats.exists() and not args.force:
        print(f"[modeldiff] {out_stats} exists; use --force to rebuild")
        return

    # Load both caches.
    cache_dist = RESULTS_DIR / (f"cache_l10_{args.distilled_cache_suffix}" if args.distilled_cache_suffix else "cache_l10")
    cache_base = RESULTS_DIR / f"cache_l10_{args.base_cache_suffix}"
    if not (cache_dist / "activations.fp16.npy").exists():
        raise SystemExit(f"missing distilled cache at {cache_dist}")
    if not (cache_base / "activations.fp16.npy").exists():
        raise SystemExit(f"missing base cache at {cache_base}; run build_act_cache_backtracking with --model llama-3.1-8b --cache-suffix base")

    act_dist = np.load(cache_dist / "activations.fp16.npy", mmap_mode="r")
    act_base = np.load(cache_base / "activations.fp16.npy", mmap_mode="r")
    off_dist = np.load(cache_dist / "offsets.npy")
    off_base = np.load(cache_base / "offsets.npy")
    ids_dist = json.loads((cache_dist / "trace_ids.json").read_text())
    ids_base = json.loads((cache_base / "trace_ids.json").read_text())
    labels_path = LABELS_DIR / "labels.jsonl"

    if act_dist.shape != act_base.shape:
        raise SystemExit(
            f"shape mismatch: distilled {act_dist.shape} vs base {act_base.shape} — "
            "the two caches must share trace_ids and per-trace seq lengths"
        )
    if ids_dist != ids_base:
        raise SystemExit("trace_ids mismatch between distilled and base caches")
    if not np.array_equal(off_dist, off_base):
        raise SystemExit("offsets mismatch between distilled and base caches")

    print(f"[modeldiff] activations {act_dist.shape}; total tokens {int(off_dist[-1])}")

    mask = _build_masks(ids_dist, off_dist, labels_path, args.positions)
    print(f"[modeldiff] mask positions={args.positions}; |D|={int(mask.sum())}")

    sae_id = args.sae_id or f"l{args.layer}r_8x"
    print(f"[modeldiff] loading Llama-Scope (release={args.sae_release}, sae_id={sae_id})")
    sae, _ = load_public_sae(args.layer, args.device, release=args.sae_release, sae_id=sae_id)

    t0 = time.time()
    mean_d, var_d, n_d = _streamed_means(sae, act_dist, mask, args.batch, args.device)
    mean_b, var_b, n_b = _streamed_means(sae, act_base, mask, args.batch, args.device)
    delta = mean_d - mean_b
    se = np.sqrt(var_d / max(n_d, 1) + var_b / max(n_b, 1))
    tstat = np.where(se > 1e-12, delta / np.maximum(se, 1e-12), 0.0).astype(np.float32)
    print(f"[modeldiff] feature stats in {time.time()-t0:.1f}s; d_sae={delta.shape[0]}")

    np.savez(out_stats, mean_distilled=mean_d, mean_base=mean_b,
             var_distilled=var_d, var_base=var_b,
             delta=delta, tstat=tstat)

    score = tstat if args.rank_by == "tstat" else delta
    top_idx = np.argsort(-score)[: args.top_k]

    W_dec = sae.W_dec.detach().cpu().numpy()
    if W_dec.shape[0] != delta.shape[0]:
        if W_dec.shape == (act_dist.shape[1], delta.shape[0]):
            W_dec = W_dec.T
        else:
            raise RuntimeError(f"unexpected W_dec shape {W_dec.shape}")
    top_records = []
    for rank, j in enumerate(top_idx.tolist()):
        top_records.append({
            "rank": rank,
            "feature_idx": int(j),
            "delta": float(delta[j]),
            "tstat": float(tstat[j]),
            "mean_distilled": float(mean_d[j]),
            "mean_base": float(mean_b[j]),
            "decoder_norm": float(np.linalg.norm(W_dec[j])),
        })
    out_top.write_text(json.dumps(top_records, indent=2))

    out_meta.write_text(json.dumps({
        "layer": args.layer,
        "sae_release": args.sae_release,
        "sae_id": sae_id,
        "sae_repo": PUBLIC_SAE_REPO,
        "sae_config": PUBLIC_SAE_CONFIG,
        "decompose_suffix": args.decompose_suffix,
        "rank_by": args.rank_by,
        "positions": args.positions,
        "n_positions": int(mask.sum()),
        "n_distilled_traces": len(ids_dist),
        "n_base_traces": len(ids_base),
        "d_sae": int(delta.shape[0]),
        "kind": "modeldiff",
    }, indent=2))
    print(f"[modeldiff] top-{args.top_k} → {out_top}")

    # Synthesize a `raw_dom`-equivalent vector for compatibility with
    # intervene_backtracking's "raw_dom" mode: the cross-model mean
    # difference in *raw activation space* (not feature space). This is
    # the activation-level analogue of Wang's model-diff direction.
    print("[modeldiff] computing raw activation-space model-diff vector …")
    sum_d = np.zeros(act_dist.shape[1], dtype=np.float64)
    sum_b = np.zeros(act_dist.shape[1], dtype=np.float64)
    n = int(act_dist.shape[0])
    BATCH_R = 16384
    for s in range(0, n, BATCH_R):
        e = min(n, s + BATCH_R)
        m = mask[s:e]
        if not m.any():
            continue
        sum_d += np.asarray(act_dist[s:e][m], dtype=np.float32).sum(axis=0)
        sum_b += np.asarray(act_base[s:e][m], dtype=np.float32).sum(axis=0)
    mean_act_d = sum_d / max(int(mask.sum()), 1)
    mean_act_b = sum_b / max(int(mask.sum()), 1)
    raw_dom = (mean_act_d - mean_act_b).astype(np.float16)
    np.save(decompose_dir / "raw_dom.fp16.npy", raw_dom)
    print(f"[modeldiff] raw model-diff |v|={np.linalg.norm(raw_dom.astype(np.float32)):.3f}")


if __name__ == "__main__":
    main()
