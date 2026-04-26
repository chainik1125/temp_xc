"""Phase 7 sparse-probing runner — S-parameterized aggregation.

Replaces Phase 5 / Phase 5B's lp / mp / full_window split with a single
S-parameterized sliding mean-pool. For each (arch, task, S, k_feat) cell:

  1. Slide T-window across the S-token tail (T=1 = per-token, no slide).
  2. Drop the first T−1 windows (per-example: enforce the kept-window
     left-edge ≥ T−1 in the example's effective tail).
  3. Per-example effective_tail = min(seq_len, S). Skip examples where
     effective_tail < 2T−1 (no valid windows for this T,S combo).
  4. Hard-skip the entire (T, S) cell if S < 2T−1 (no example, however
     long, would have a valid window).
  5. Mean the kept per-window z's into a single (d_sae,) per-example
     representation; pass to top-k-by-class-sep + L1 LR (Phase 5
     SAEBench protocol via `sae_probe_metrics`).
  6. Apply FLIP convention on `winogrande_correct_completion` and
     `wsc_coreference` (Phase 5 carryover).

Reuses Phase 5 helpers (`_slide_windows`, `_encode_per_token`,
`_load_task_cache`, `top_k_by_class_sep`, `sae_probe_metrics`).

Run examples (from repo root):

    .venv/bin/python -m experiments.phase7_unification.run_probing_phase7 \\
        --run_ids txcdr_t5__seed42 --S 128 --k_feat 5 20

    # Headline pass: every run in the index, S in {128, 20}, k_feat in {5, 20}:
    .venv/bin/python -m experiments.phase7_unification.run_probing_phase7 --headline
"""
from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from experiments.phase5_downstream_utility.probing.run_probing import (
    _slide_windows,
    _encode_per_token,
    _load_task_cache,
    _split_indices_for_dataset,
    top_k_by_class_sep,
    sae_probe_metrics,
    VAL_FRAC,
)
from experiments.phase7_unification._paths import (
    PROBE_CACHE_DIR, INDEX_PATH, PROBING_PATH, OUT_DIR,
    ANCHOR_LAYER, MLC_LAYERS, SUBJECT_MODEL,
    DEFAULT_D_IN, DEFAULT_D_SAE, banner,
)


HEADLINE_S = (128, 64, 20)   # 128 = methodological headline (full context, no
                             #       boundary asymmetry choice, full T-sweep
                             #       headroom — at T=32 still 66 kept windows).
                             # 64  = S-sensitivity ablation (cheap; if rankings
                             #       agree with S=128 we can quote the smaller
                             #       number with a footnote, if they disagree
                             #       that's itself a finding).
                             # 20  = Phase 5 carryover for cross-phase continuity.
HEADLINE_K_FEAT = (5, 20)
ABLATION_K_FEAT = (1, 2)
FLIP_TASKS = frozenset({"winogrande_correct_completion", "wsc_coreference"})


def flipped_auc(auc: float, task_name: str) -> float:
    if task_name in FLIP_TASKS:
        return max(auc, 1.0 - auc)
    return auc


# ════════════════════════════════════════════════════════════════════
# S-parameterized aggregation — per-example correct
# ════════════════════════════════════════════════════════════════════


def aggregate_s(z_full: np.ndarray, last_idx: np.ndarray,
                T: int, S: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised per-example mean-pool with S-parameterised effective tail.

    Args:
      z_full:   (N, K, d_sae) where K = LAST_N (T=1) or LAST_N - T + 1 (T>1).
      last_idx: (N,) int — position of last real token in the LAST_N=128 tail.
      T:        window size (1 = per-token).
      S:        tail length to evaluate over.

    Returns:
      (z_mean, valid)  where
        z_mean: (N_valid, d_sae) — examples with at least one kept window.
        valid:  (N,) bool — which examples contributed.
    """
    N, K, _ = z_full.shape
    last_idx_t = torch.as_tensor(last_idx, dtype=torch.long)
    # effective_tail per example = min(S, last_idx + 1)
    effective = torch.minimum(torch.full_like(last_idx_t, S), last_idx_t + 1)
    # For T=1: mask on per-token positions [first_real, last_real]
    #   first_real = last_idx - effective + 1; last_real = last_idx
    # For T>1: mask on window-left-edges [lo, hi]
    #   lo = last_idx - effective + T     (= first_real + T - 1)
    #   hi = last_idx - T + 1
    if T == 1:
        lo = (last_idx_t - effective + 1).clamp(min=0)
        hi = last_idx_t.clamp(max=K - 1)
    else:
        lo = (last_idx_t - effective + T).clamp(min=0)
        hi = (last_idx_t - T + 1).clamp(max=K - 1)
    k_grid = torch.arange(K).view(1, K)                 # (1, K)
    mask = (k_grid >= lo.view(N, 1)) & (k_grid <= hi.view(N, 1))  # (N, K) bool
    counts = mask.sum(dim=1)                             # (N,)
    valid = counts > 0
    if not valid.any():
        return np.zeros((0, z_full.shape[-1]), dtype=z_full.dtype), valid.numpy()
    # Numpy mean with mask:
    mask_np = mask.numpy().astype(z_full.dtype)
    counts_np = counts.numpy().astype(z_full.dtype).clip(min=1)
    summed = (z_full * mask_np[:, :, None]).sum(axis=1)  # (N, d_sae)
    z_mean = summed / counts_np[:, None]
    return z_mean[valid.numpy()], valid.numpy()


def cell_is_valid(T: int, S: int) -> bool:
    """Hard skip if no example length could ever yield a valid window."""
    if T == 1:
        return S >= 1
    return S >= 2 * T - 1


# ════════════════════════════════════════════════════════════════════
# Model loading — dispatched off canonical_archs.json src_class
# ════════════════════════════════════════════════════════════════════


def _load_phase7_model(meta: dict, ckpt_path: Path, device) -> tuple:
    """Build the Phase 7 model class and load state_dict from a Phase 7 ckpt.

    Returns (model, src_class). meta must include `src_class` (Phase 7
    training driver bakes this in).
    """
    src_class = meta["src_class"]
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    cast_sd = {k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
                for k, v in sd.items()}
    d_in = int(meta.get("d_in", DEFAULT_D_IN))
    d_sae = int(meta.get("d_sae", DEFAULT_D_SAE))

    if src_class == "TopKSAE":
        from src.architectures.topk_sae import TopKSAE
        model = TopKSAE(d_in, d_sae, k=int(meta["k_pos"])).to(device)
    elif src_class == "TemporalMatryoshkaBatchTopKSAE":
        from src.architectures.tsae_paper import TemporalMatryoshkaBatchTopKSAE
        # Must match train_phase7.train_tsae_paper EXACTLY: 2-group [0.2, 0.8]
        # split, sum=d_sae. The earlier 4-group [d_sae//8, d_sae//4, d_sae//2,
        # d_sae] convention here was wrong (sum=34560 != d_sae=18432, would
        # trip the assert in TemporalMatryoshkaBatchTopKSAE.__init__).
        # Prefer reading group_sizes from meta if present (added in
        # _meta_from_arch for defensive reproducibility); fall back to the
        # trainer's [0.2, 0.8] convention.
        if meta.get("group_sizes"):
            group_sizes = list(meta["group_sizes"])
        else:
            group_sizes = [int(0.2 * d_sae), d_sae - int(0.2 * d_sae)]
        assert sum(group_sizes) == d_sae, (
            f"group_sizes {group_sizes} sum={sum(group_sizes)} != d_sae={d_sae}"
        )
        model = TemporalMatryoshkaBatchTopKSAE(
            activation_dim=d_in, dict_size=d_sae,
            k=int(meta["k_pos"]), group_sizes=group_sizes,
        ).to(device)
    elif src_class == "MultiLayerCrosscoder":
        from src.architectures.mlc import MultiLayerCrosscoder
        model = MultiLayerCrosscoder(
            d_in, d_sae, n_layers=int(meta.get("n_layers", len(MLC_LAYERS))),
            k=int(meta["k_win"]),
        ).to(device)
    elif src_class == "MLCContrastive":
        from src.architectures.mlc_contrastive import MLCContrastive
        model = MLCContrastive(
            d_in, d_sae, n_layers=int(meta.get("n_layers", len(MLC_LAYERS))),
            k=int(meta["k_win"]), h=int(d_sae * 0.2),
        ).to(device)
    elif src_class == "MLCContrastiveMultiscale":
        from src.architectures.mlc_contrastive_multiscale import MLCContrastiveMultiscale
        model = MLCContrastiveMultiscale(
            d_in, d_sae, n_layers=int(meta.get("n_layers", len(MLC_LAYERS))),
            k=int(meta["k_win"]),
            gamma=float(meta.get("gamma") if meta.get("gamma") is not None else 0.5),
            h=int(d_sae * 0.2),
        ).to(device)
    elif src_class == "TemporalSAE":
        from src.architectures._tfa_module import TemporalSAE
        model = TemporalSAE(
            dimin=d_in, width=d_sae, n_heads=4, sae_diff_type="topk",
            kval_topk=int(meta["k_win"]), tied_weights=True,
            n_attn_layers=1, bottleneck_factor=4, use_pos_encoding=False,
            max_seq_len=128,
        ).to(device)
    elif src_class == "MatryoshkaTXCDRContrastiveMultiscale":
        from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
            MatryoshkaTXCDRContrastiveMultiscale,
        )
        model = MatryoshkaTXCDRContrastiveMultiscale(
            d_in, d_sae, T=int(meta["T"]), k=int(meta["k_win"]),
            n_contr_scales=int(meta.get("n_scales") if meta.get("n_scales") is not None else 3),
            gamma=float(meta.get("gamma") if meta.get("gamma") is not None else 0.5),
        ).to(device)
    elif src_class == "TXCBareAntidead":
        from src.architectures.txc_bare_antidead import TXCBareAntidead
        model = TXCBareAntidead(d_in, d_sae, int(meta["T"]), int(meta["k_win"])).to(device)
    elif src_class == "SubseqTXCBareAntidead":
        from src.architectures.phase5b_subseq_sampling_txcdr import SubseqTXCBareAntidead
        model = SubseqTXCBareAntidead(
            d_in, d_sae, T_max=int(meta["T_max"]), k=int(meta["k_win"]),
            t_sample=int(meta["t_sample"]), contiguous=False,
        ).to(device)
    elif src_class == "SubseqH8":
        from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
        T_max = int(meta["T_max"])
        raw_shifts = (1, max(1, T_max // 4), max(1, T_max // 2))
        shifts = tuple(sorted(set(s for s in raw_shifts if 1 <= s <= T_max - 1)))
        model = SubseqH8(
            d_in, d_sae, T_max=T_max, k=int(meta["k_win"]),
            t_sample=int(meta["t_sample"]), contiguous=False,
            shifts=shifts, weights=None,
            matryoshka_h_size=int(d_sae * 0.2), alpha=float(meta.get("alpha") if meta.get("alpha") is not None else 1.0),
        ).to(device)
    elif src_class == "TemporalCrosscoder":
        from src.architectures.crosscoder import TemporalCrosscoder
        model = TemporalCrosscoder(d_in, d_sae, int(meta["T"]), int(meta["k_win"])).to(device)
    elif src_class == "TXCBareMultiDistanceContrastiveAntidead":
        from src.architectures.txc_bare_multidistance_contrastive_antidead import (
            TXCBareMultiDistanceContrastiveAntidead,
        )
        shifts = tuple(meta.get("shifts") or (1,))
        model = TXCBareMultiDistanceContrastiveAntidead(
            d_in, d_sae, int(meta["T"]), int(meta["k_win"]),
            shifts=shifts, weights=None,
            matryoshka_h_size=int(d_sae * 0.2),
            alpha=float(meta.get("alpha") if meta.get("alpha") is not None else 1.0),
        ).to(device)
    else:
        raise ValueError(f"unknown src_class={src_class}")

    model.load_state_dict(cast_sd, strict=False)
    model.eval()
    return model, src_class


# ════════════════════════════════════════════════════════════════════
# Probe-time encoders — per arch family
# ════════════════════════════════════════════════════════════════════


def _encode_per_token_z(model, anchor: np.ndarray, device) -> np.ndarray:
    """Per-token SAE encoder. anchor: (N, S=128, d). Returns (N, S, d_sae).
    Phase 5's `_encode_per_token` flattens (N, S, d) → (N*S, d) and unflattens.
    """
    N, S, d = anchor.shape
    flat = anchor.reshape(N * S, d)
    # Use shape-preserving wrapper: encode_fn expects (B, T, d) but per-token
    # SAEs handle (B, d) by ignoring T. We feed (B, 1, d) to be safe.
    z_flat = _encode_per_token(model.encode, flat[:, None, :], device)  # (N*S, d_sae)
    return z_flat.reshape(N, S, -1)


def _encode_window_z(model, anchor: np.ndarray, T: int, device) -> np.ndarray:
    """Window-arch encoder. anchor: (N, S=128, d). Returns (N, S-T+1, d_sae).
    """
    wins = _slide_windows(anchor, T)              # (N, K, T, d)
    N, K, _, d = wins.shape
    flat = wins.reshape(N * K, T, d)
    z = _encode_per_token(model.encode, flat, device)   # (N*K, d_sae)
    return z.reshape(N, K, -1)


def _encode_mlc_per_token_z(model, mlc_tail: np.ndarray, device) -> np.ndarray:
    """MLC encoder over per-token multi-layer. mlc_tail: (N, S=128, n_layers=5, d).
    Returns (N, S, d_sae).
    """
    N, S, n_lay, d = mlc_tail.shape
    flat = mlc_tail.reshape(N * S, n_lay, d)
    z = _encode_per_token(model.encode, flat, device)
    return z.reshape(N, S, -1)


def encode_for_S(model, src_class: str, meta: dict,
                 task_cache: dict, split: str, device) -> tuple[np.ndarray, np.ndarray]:
    """Build the (N, K, d_sae) encoded tensor + last_idx vector for a task split.

    Returns (z_full, last_idx) where z_full's K depends on the arch:
      - per-token / MLC: K = 128 (= LAST_N).
      - window archs:     K = 128 - T_eff + 1.
    """
    last_idx = task_cache[f"{split}_last_idx"]
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        anchor = task_cache[f"anchor_{split}"]            # (N, 128, d)
        return _encode_per_token_z(model, anchor, device), last_idx
    if src_class in {"MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale"}:
        mlc_tail = task_cache[f"mlc_tail_{split}"]        # (N, 128, 5, d)
        return _encode_mlc_per_token_z(model, mlc_tail, device), last_idx
    # Window archs — read T from meta.
    if "T" in meta and meta["T"] is not None:
        T_eff = int(meta["T"])
    elif "T_max" in meta and meta["T_max"] is not None:
        T_eff = int(meta["T_max"])
    else:
        raise ValueError(f"no T or T_max in meta for src_class={src_class}")
    anchor = task_cache[f"anchor_{split}"]                # (N, 128, d)
    return _encode_window_z(model, anchor, T_eff, device), last_idx


def get_T_for_aggregation(src_class: str, meta: dict) -> int:
    """The T value to use in aggregate_s. Per-token / MLC = 1; window = T or T_max."""
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE",
                     "MultiLayerCrosscoder", "MLCContrastive", "MLCContrastiveMultiscale"}:
        return 1
    if "T" in meta and meta["T"] is not None:
        return int(meta["T"])
    if "T_max" in meta and meta["T_max"] is not None:
        return int(meta["T_max"])
    raise ValueError(f"no T for {src_class}")


# ════════════════════════════════════════════════════════════════════
# Main loop
# ════════════════════════════════════════════════════════════════════


def _iter_index(run_ids: list[str] | None):
    if not INDEX_PATH.exists():
        print(f"[probe] no training index at {INDEX_PATH}")
        return
    with INDEX_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if run_ids and row["run_id"] not in run_ids:
                continue
            yield row["run_id"], Path(row["ckpt"]), row


def _load_task_cache_p7(task_dir: Path) -> dict:
    """Phase 7 task cache loader: includes mlc_tail (N, 128, 5, d) for MLC archs.

    Cast mlc_tail to fp32 — the .npz stores fp16 to save disk, but Phase 7's
    fp32 model parameters can't einsum against fp16 input ("expected Half
    but found Float" RuntimeError). Phase 5's _load_task_cache already
    casts the anchor cache to fp32; this matches that convention for the
    new mlc_tail field.
    """
    base = _load_task_cache(task_dir)
    mlc_tail_path = task_dir / "acts_mlc_tail.npz"
    if mlc_tail_path.exists():
        with np.load(mlc_tail_path) as z:
            base["mlc_tail_train"] = z["train_acts"].astype(np.float32)
            base["mlc_tail_test"] = z["test_acts"].astype(np.float32)
    return base


def run_probing(run_ids: list[str] | None = None,
                task_names: list[str] | None = None,
                S_values: tuple[int, ...] = HEADLINE_S,
                k_values: tuple[int, ...] = HEADLINE_K_FEAT,
                limit_archs: int | None = None) -> None:
    PROBING_PATH.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    task_dirs = [d for d in sorted(PROBE_CACHE_DIR.iterdir()) if d.is_dir()]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "meta.json").exists()
    ]
    print(f"[probe] {len(task_dirs)} tasks, S={S_values}, k_feat={k_values}")

    n_done = 0
    with PROBING_PATH.open("a") as out_f:
        for run_id, ckpt_path, meta in _iter_index(run_ids):
            if limit_archs is not None and n_done >= limit_archs:
                break
            n_done += 1
            if not ckpt_path.exists():
                print(f"  {run_id}: ckpt missing {ckpt_path}")
                continue
            try:
                model, src_class = _load_phase7_model(meta, ckpt_path, device)
            except Exception as e:
                print(f"  {run_id}: load FAIL {type(e).__name__}: {e}")
                continue
            T_eff = get_T_for_aggregation(src_class, meta)
            print(f"  {run_id} src_class={src_class} T={T_eff}")

            for task_dir in task_dirs:
                task_name = task_dir.name
                tc = _load_task_cache_p7(task_dir)
                ytr = tc["train_labels"]; yte = tc["test_labels"]
                try:
                    z_train_full, last_train = encode_for_S(
                        model, src_class, meta, tc, "train", device)
                    z_test_full, last_test = encode_for_S(
                        model, src_class, meta, tc, "test", device)
                except Exception as e:
                    print(f"    {task_name}: encode FAIL {type(e).__name__}: {e}")
                    del tc; gc.collect(); continue

                for S in S_values:
                    if not cell_is_valid(T_eff, S):
                        # Log skipped cell so it shows up in the index instead
                        # of being silently absent.
                        out_f.write(json.dumps({
                            "run_id": run_id, "src_class": src_class,
                            "task_name": task_name,
                            "S": int(S), "T": int(T_eff),
                            "skipped": True, "reason": "S_lt_2T_minus_1",
                            **{k: meta.get(k) for k in (
                                "arch_id", "row", "group", "seed",
                                "k_pos", "k_win",
                            )},
                        }) + "\n")
                        continue
                    Z_tr, valid_tr = aggregate_s(z_train_full, last_train, T_eff, S)
                    Z_te, valid_te = aggregate_s(z_test_full, last_test, T_eff, S)
                    n_drop_tr = int((~valid_tr).sum()); n_drop_te = int((~valid_te).sum())
                    if Z_tr.shape[0] == 0 or Z_te.shape[0] == 0:
                        out_f.write(json.dumps({
                            "run_id": run_id, "src_class": src_class,
                            "task_name": task_name, "S": int(S), "T": int(T_eff),
                            "skipped": True, "reason": "no_valid_examples_after_filter",
                            "n_drop_train": n_drop_tr, "n_drop_test": n_drop_te,
                            **{k: meta.get(k) for k in (
                                "arch_id", "row", "group", "seed", "k_pos", "k_win",
                            )},
                        }) + "\n")
                        continue
                    ytr_v = ytr[valid_tr]; yte_v = yte[valid_te]
                    for k_feat in k_values:
                        auc, acc = sae_probe_metrics(Z_tr, ytr_v, Z_te, yte_v, k_feat)
                        auc_flip = flipped_auc(auc, task_name)
                        row = {
                            "run_id": run_id, "src_class": src_class,
                            "arch_id": meta.get("arch_id"),
                            "row": meta.get("row"), "group": meta.get("group"),
                            "task_name": task_name,
                            "dataset_key": tc["meta"]["dataset_key"],
                            "S": int(S), "T": int(T_eff),
                            "k_feat": int(k_feat),
                            "test_auc": float(auc),
                            "test_auc_flip": float(auc_flip),
                            "test_acc": float(acc),
                            "n_train_eff": int(Z_tr.shape[0]),
                            "n_test_eff": int(Z_te.shape[0]),
                            "n_drop_train": n_drop_tr, "n_drop_test": n_drop_te,
                            **{k: meta.get(k) for k in (
                                "T_max", "t_sample", "n_layers",
                                "shifts", "alpha", "gamma", "n_scales",
                                "seed", "k_pos", "k_win",
                            ) if meta.get(k) is not None},
                        }
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                    print(
                        f"    {task_name:40s} S={S:3d} T={T_eff:2d} "
                        f"auc@5={sae_probe_metrics(Z_tr, ytr_v, Z_te, yte_v, 5)[0]:.4f} "
                        f"n_kept={Z_tr.shape[0]}/{ytr.size}"
                    )
                del tc
                torch.cuda.empty_cache()  # avoid GPU fragmentation across tasks
                gc.collect()
            del model
            torch.cuda.empty_cache(); gc.collect()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_ids", nargs="+", default=None)
    p.add_argument("--task_names", nargs="+", default=None)
    p.add_argument("--S", nargs="+", type=int, default=list(HEADLINE_S))
    p.add_argument("--k_feat", nargs="+", type=int, default=list(HEADLINE_K_FEAT))
    p.add_argument("--headline", action="store_true",
                   help="run S in {128,20} × k_feat in {5,20} on every run in index")
    p.add_argument("--limit_archs", type=int, default=None)
    args = p.parse_args()
    banner(__file__)
    if args.headline:
        run_probing(S_values=HEADLINE_S, k_values=HEADLINE_K_FEAT,
                    limit_archs=args.limit_archs)
    else:
        run_probing(
            run_ids=args.run_ids, task_names=args.task_names,
            S_values=tuple(args.S), k_values=tuple(args.k_feat),
            limit_archs=args.limit_archs,
        )


if __name__ == "__main__":
    main()
