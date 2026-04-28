"""Per-arch decomposition of HH-RLHF chosen/rejected activations (case study C.i).

For each Stage 1 / Stage 2 arch:

  1. Load cache from `data/cached_hh_rlhf/{chosen,rejected}.npz` (built
     by build_hh_rlhf_cache.py — fp16 L12 residuals + response masks).
  2. Load Phase 7 ckpt via `_load_phase7_model`.
  3. Encode each example into per-token / per-window features via the
     arch-uniform `_arch_utils.encode_per_position`. Window archs slide
     T-tokens with stride 1 and attribute the latent to the right edge.
  4. Aggregate per example: average activation over the RESPONSE tokens
     only (response_mask), giving `(N, d_sae)` per side.
  5. Per-feature stats: `mean_chosen`, `mean_rejected`, `diff = rejected
     - chosen` (paper's ranking metric, App B.1 Table 8).
  6. Length-spurious correlation: per top-K feature, compute Pearson r
     between per-example `(rejected_act - chosen_act)` and per-example
     `(rejected_response_len - chosen_response_len)`. Paper observes
     that semantically-relevant features have low |r| with length
     while spurious features (legal/formal language, transition words)
     have high |r|.

Outputs per arch (under `experiments/phase7_unification/results/case_studies/hh_rlhf/<arch_id>/`):

  feature_acts.npz:
    chosen_per_example   (N, d_sae) fp32 — mean activation over response tokens
    rejected_per_example (N, d_sae) fp32
    valid_mask           (N,)  bool — examples with both response_len > 0
  feature_stats.json:
    arch_id, src_class, n_examples, n_valid,
    mean_chosen, mean_rejected, diff,                           # full d_sae lists
    top_k_indices, top_k_diff, top_k_length_pearson_r,
    top_k_length_pearson_p, top_k_mean_chosen, top_k_mean_rejected,
    paper_t_test (rejected_len vs chosen_len on this subset).

Run:
    .venv/bin/python -m experiments.phase7_unification.case_studies.hh_rlhf.decompose_hh_rlhf
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import (
    OUT_DIR, banner,
)
from experiments.phase7_unification.case_studies._arch_utils import (
    load_phase7_model_safe as _load_phase7_model,
)
from experiments.phase7_unification.case_studies._paths import (
    HH_RLHF_CACHE_DIR, CASE_STUDIES_DIR, STAGE_1_ARCHS,
)
from experiments.phase7_unification.case_studies._arch_utils import (
    encode_per_position, window_T, _d_sae_of, MLC_CLASSES,
)


def _aggregate_per_example(
    model, src_class: str, acts: np.ndarray, mask: np.ndarray,
    *, T: int, batch_size: int,
) -> np.ndarray:
    """Encode all `(N, S, d_in)` activations and average each example's
    per-position features over the True positions of `mask`. Returns
    `(N, d_sae)` fp32 (zeros for examples with no True positions —
    caller filters via valid mask).
    """
    device = next(model.parameters()).device
    N, S, d_in = acts.shape
    d_sae = _d_sae_of(model, src_class)
    out = np.zeros((N, d_sae), dtype=np.float32)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        x = torch.from_numpy(acts[start:end]).float().to(device)
        z = encode_per_position(model, src_class, x, T=T)            # (B, S, d_sae)
        m = torch.from_numpy(mask[start:end]).to(device).float().unsqueeze(-1)
        sums = (z * m).sum(dim=1)                                    # (B, d_sae)
        counts = m.sum(dim=1).clamp(min=1)                           # (B, 1)
        out[start:end] = (sums / counts).cpu().numpy()
        del z, x, m, sums, counts
    return out


def decompose_one_arch(
    arch_id: str,
    *,
    batch_size: int = 16,
    top_k: int = 50,
) -> dict | None:
    """Per-arch HH-RLHF decomposition. Returns the feature-stats dict
    that was saved (or None if the arch needs MLC-style multilayer cache
    which we don't build in C.i v1).
    """
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    if not ckpt_path.exists() or not log_path.exists():
        print(f"  [skip] {arch_id}: ckpt or log missing")
        return None
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        print(
            f"  [skip] {arch_id}: MLC arch needs (N, S, n_layers, d_in) cache; "
            f"current HH-RLHF cache is L12-only. Add MLC cache build as follow-up."
        )
        return None

    device = torch.device("cuda")
    model, _ = _load_phase7_model(meta, ckpt_path, device)
    T = window_T(model, src_class, meta)
    d_sae = _d_sae_of(model, src_class)
    print(f"  src_class={src_class}  T={T}  d_sae={d_sae}  k_pos={meta.get('k_pos')}")

    chosen_npz = np.load(HH_RLHF_CACHE_DIR / "chosen.npz")
    rejected_npz = np.load(HH_RLHF_CACHE_DIR / "rejected.npz")
    chosen_acts = chosen_npz["acts"]
    rejected_acts = rejected_npz["acts"]
    chosen_mask = chosen_npz["response_mask"]
    rejected_mask = rejected_npz["response_mask"]
    chosen_len = chosen_npz["response_len"].astype(np.float64)
    rejected_len = rejected_npz["response_len"].astype(np.float64)
    N, S, d_in = chosen_acts.shape
    print(f"  cache: N={N} S={S} d_in={d_in}")

    t0 = time.time()
    print("  encoding chosen...", flush=True)
    chosen_per_ex = _aggregate_per_example(
        model, src_class, chosen_acts, chosen_mask,
        T=T, batch_size=batch_size,
    )
    print(f"    chosen done in {time.time() - t0:.1f}s")

    t0 = time.time()
    print("  encoding rejected...", flush=True)
    rejected_per_ex = _aggregate_per_example(
        model, src_class, rejected_acts, rejected_mask,
        T=T, batch_size=batch_size,
    )
    print(f"    rejected done in {time.time() - t0:.1f}s")

    valid = (chosen_len > 0) & (rejected_len > 0)
    n_valid = int(valid.sum())
    print(f"  N_valid = {n_valid}/{N} (excluded examples with response_len==0 either side)")

    mean_chosen = chosen_per_ex[valid].mean(axis=0)                  # (d_sae,)
    mean_rejected = rejected_per_ex[valid].mean(axis=0)
    diff = mean_rejected - mean_chosen
    abs_diff = np.abs(diff)
    top_idx = np.argsort(-abs_diff)[:top_k]

    # Per top-feature length-spurious correlation.
    from scipy.stats import pearsonr, ttest_rel
    length_diff = rejected_len[valid] - chosen_len[valid]
    pearson_r = np.zeros(top_k, dtype=np.float64)
    pearson_p = np.zeros(top_k, dtype=np.float64)
    for ki, j in enumerate(top_idx):
        feat_diff = rejected_per_ex[valid, j] - chosen_per_ex[valid, j]
        if feat_diff.std() < 1e-8:
            pearson_r[ki] = 0.0
            pearson_p[ki] = 1.0
        else:
            r, p = pearsonr(feat_diff, length_diff)
            pearson_r[ki] = float(r)
            pearson_p[ki] = float(p)

    t_stat, t_p = ttest_rel(rejected_len[valid], chosen_len[valid])

    out_dir = CASE_STUDIES_DIR / "hh_rlhf" / arch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "feature_acts.npz",
        chosen_per_example=chosen_per_ex,
        rejected_per_example=rejected_per_ex,
        valid_mask=valid,
    )

    feature_stats = {
        "arch_id": arch_id,
        "src_class": src_class,
        "T": T,
        "d_sae": int(d_sae),
        "k_pos": meta.get("k_pos"),
        "k_win": meta.get("k_win"),
        "n_examples": int(N),
        "n_valid": n_valid,
        "mean_chosen": mean_chosen.tolist(),
        "mean_rejected": mean_rejected.tolist(),
        "diff": diff.tolist(),
        "top_k": int(top_k),
        "top_k_indices": top_idx.tolist(),
        "top_k_diff": diff[top_idx].tolist(),
        "top_k_mean_chosen": mean_chosen[top_idx].tolist(),
        "top_k_mean_rejected": mean_rejected[top_idx].tolist(),
        "top_k_length_pearson_r": pearson_r.tolist(),
        "top_k_length_pearson_p": pearson_p.tolist(),
        "paper_response_len_t_test": {
            "rejected_mean": float(rejected_len[valid].mean()),
            "chosen_mean": float(chosen_len[valid].mean()),
            "diff_mean": float(length_diff.mean()),
            "t_stat": float(t_stat),
            "p_val": float(t_p),
        },
    }
    with open(out_dir / "feature_stats.json", "w") as f:
        json.dump(feature_stats, f)

    # ── headline print: top-10 features
    print("  top-10 features by |rejected - chosen|:")
    print(f"    {'rank':>4s} {'feat':>6s} {'diff':>8s} {'mean_chosen':>12s} {'mean_rejected':>14s} {'pearson_r':>10s} {'pearson_p':>10s}")
    for ki, j in enumerate(top_idx[:10]):
        print(
            f"    {ki:4d} {j:6d} {diff[j]:+8.4f} {mean_chosen[j]:12.4f} "
            f"{mean_rejected[j]:14.4f} {pearson_r[ki]:+10.3f} {pearson_p[ki]:10.2e}"
        )

    del model
    torch.cuda.empty_cache()
    gc.collect()
    return feature_stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS),
                    help="arch_ids to decompose (default: Stage 1)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--top-k", type=int, default=50)
    args = ap.parse_args()
    banner(__file__)
    for arch_id in args.archs:
        print(f"\n=== {arch_id} ===")
        decompose_one_arch(arch_id, batch_size=args.batch_size, top_k=args.top_k)


if __name__ == "__main__":
    main()
