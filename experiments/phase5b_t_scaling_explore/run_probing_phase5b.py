"""Phase 5B sparse-probing runner.

Standalone runner that:
  - reuses Phase 5's probe cache (read-only) + helpers (_slide_windows,
    _window_at_last, _encode_per_token, top_k_by_class_sep,
    sae_probe_metrics, _load_task_cache, _split_indices_for_dataset).
  - dispatches on Phase 5B arch ids (prefix `phase5b_`) to the correct
    model constructor + probe-time encoder.
  - writes to Phase 5B's own probing_results.jsonl (never to Phase 5's).

Invocation (from repo root):

    .venv/bin/python -m experiments.phase5b_t_scaling_explore.run_probing_phase5b \
        --run_ids phase5b_strided_track2__seed42 \
        --aggregations last_position mean_pool
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

# Reuse Phase 5 helpers (read-only from that module).
from experiments.phase5_downstream_utility.probing.run_probing import (
    _slide_windows,
    _window_at_last,
    _last_token,
    _encode_per_token,
    _load_task_cache,
    _split_indices_for_dataset,
    top_k_by_class_sep,
    sae_probe_metrics,
    VAL_FRAC,
)
from experiments.phase5b_t_scaling_explore._paths import (
    OUT_DIR, CKPT_DIR, INDEX_PATH, PROBING_PATH, PROBE_CACHE,
    DEFAULT_D_SAE,
)


K_VALUES = [1, 2, 5, 20]
SUPPORTED_AGGS = ("last_position", "mean_pool")

# Phase 5 convention: tasks with arbitrary label polarity get max(AUC, 1-AUC).
# These are cross-token coreference tasks where label-1 is "completion correct"
# but the SAE may have learned the opposite polarity.
FLIP_TASKS = frozenset({"winogrande_correct_completion", "wsc_coreference"})


def flipped_auc(auc: float, task_name: str) -> float:
    """Apply the Phase 5 FLIP convention to per-task AUC."""
    if task_name in FLIP_TASKS:
        return max(auc, 1.0 - auc)
    return auc


# ─────────────────────────────────────────── model loading


def _load_phase5b_model(run_id: str, ckpt_path: Path, device, meta: dict):
    """Build the Phase 5B model class and load state_dict."""
    arch = meta["arch"]
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    # Ckpts saved as fp16; cast to fp32 on load.
    cast_sd = {
        k: (v.float() if torch.is_tensor(v) and v.dtype == torch.float16 else v)
        for k, v in sd.items()
    }
    d_in = 2304
    d_sae = int(meta.get("d_sae", DEFAULT_D_SAE))

    if arch == "strided_track2":
        from src.architectures.phase5b_strided_txcdr import StridedTXCBareAntidead
        T_eff = int(meta["T_eff"])
        stride = int(meta["stride"])
        k = int(meta["k_win"])
        model = StridedTXCBareAntidead(d_in, d_sae, T_eff=T_eff, k=k, stride=stride).to(device)

    elif arch == "strided_h8":
        from src.architectures.phase5b_strided_txcdr import StridedH8
        T_eff = int(meta["T_eff"])
        stride = int(meta["stride"])
        k = int(meta["k_win"])
        mh = meta.get("matryoshka_h_size") or int(d_sae * 0.2)
        model = StridedH8(
            d_in, d_sae, T_eff=T_eff, k=k, stride=stride,
            matryoshka_h_size=mh,
            alpha=meta.get("alpha", 1.0),
        ).to(device)

    elif arch == "subseq_track2":
        from src.architectures.phase5b_subseq_sampling_txcdr import SubseqTXCBareAntidead
        T_max = int(meta["T_max"])
        t_sample = int(meta["t_sample"])
        k = int(meta["k_win"])
        model = SubseqTXCBareAntidead(
            d_in, d_sae, T_max=T_max, k=k, t_sample=t_sample,
            contiguous=bool(meta.get("contiguous", False)),
        ).to(device)

    elif arch == "subseq_h8":
        from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
        T_max = int(meta["T_max"])
        t_sample = int(meta["t_sample"])
        k = int(meta["k_win"])
        mh = meta.get("matryoshka_h_size") or int(d_sae * 0.2)
        model = SubseqH8(
            d_in, d_sae, T_max=T_max, k=k, t_sample=t_sample,
            contiguous=bool(meta.get("contiguous", False)),
            matryoshka_h_size=mh,
            alpha=meta.get("alpha", 1.0),
        ).to(device)

    elif arch == "token_subseq":
        from src.architectures.phase5b_token_subseq_sae import TokenSubseqSAE
        L_max = int(meta.get("L_max", 128))
        t_sample = int(meta["t_sample"])
        k = int(meta["k_pos"])
        pos_mode = meta.get("pos_mode")
        if pos_mode is None:
            # Legacy: derive from use_pos.
            pos_mode = "sinusoidal" if meta.get("use_pos") else "none"
        model = TokenSubseqSAE(
            d_in, d_sae, L_max=L_max, k=k, t_sample=t_sample,
            pos_mode=pos_mode,
        ).to(device)

    elif arch == "paired_track2":
        from src.architectures.txc_bare_antidead import TXCBareAntidead
        T_paired = int(meta["T_paired"])
        k = int(meta["k_win"])
        model = TXCBareAntidead(d_in, d_sae, T_paired, k=k).to(device)

    elif arch == "paired_h8":
        from src.architectures.txc_bare_multidistance_contrastive_antidead import (
            TXCBareMultiDistanceContrastiveAntidead,
        )
        T_paired = int(meta["T_paired"])
        k = int(meta["k_win"])
        mh = meta.get("matryoshka_h_size") or int(d_sae * 0.2)
        model = TXCBareMultiDistanceContrastiveAntidead(
            d_in, d_sae, T_paired, k=k,
            matryoshka_h_size=mh,
            alpha=meta.get("alpha", 1.0),
        ).to(device)

    elif arch == "subset_encoder":
        from src.architectures.phase5b_subset_encoder_txc import SubsetEncoderTXC
        T_window = int(meta["T_window"])
        t = int(meta["t"])
        k = int(meta["k_win"])
        model = SubsetEncoderTXC(
            d_in, d_sae, T_window=T_window, t=t, k=k,
            probe_strategy=meta.get("probe_strategy", "random_K_subsets"),
            probe_K=int(meta.get("probe_K", 16)),
        ).to(device)

    elif arch == "pps_matryoshka":
        from src.architectures.phase5b_per_pos_scale_matryoshka import (
            PerPosScaleMatryoshkaTXC,
        )
        T = int(meta["T"])
        k = int(meta["k_win"])
        n_scales = int(meta["n_scales"])
        alpha = float(meta.get("alpha", 0.0))
        shifts = tuple(meta.get("contr_shifts", (1,)))
        model = PerPosScaleMatryoshkaTXC(
            d_in, d_sae, T=T, k=k, n_scales=n_scales, alpha=alpha,
            contr_shifts=shifts,
        ).to(device)

    else:
        raise ValueError(f"Unknown phase5b arch: {arch}")

    model.load_state_dict(cast_sd, strict=False)
    model.eval()
    return model, arch


# ─────────────────────────────────────────── probe-time encoders


def _encode_full_window_to_mean(enc_slide_fn: Callable[[np.ndarray], np.ndarray],
                                 Z_full_window_shape: tuple[int, int, int],
                                 ) -> np.ndarray:
    """Given a full_window encoding that produced (N, K*d_sae) tensor,
    reshape to (N, K, d_sae) and mean over K. Convenience wrapper."""
    N, Kd_sae = Z_full_window_shape[0], Z_full_window_shape[1]
    return None  # placeholder — we inline this in the actual dispatch


def _encode_txc_like(model, anchor, last_idx, T, device, aggregation: str):
    """Standard TXCDR-style probe: (B, T, d) -> (B, d_sae).
    For last_position: window at last real token.
    For mean_pool: slide T-window across tail-20, mean across K slides.
    """
    if aggregation == "last_position":
        X = _window_at_last(anchor, last_idx, T)
        return _encode_per_token(model.encode, X, device)
    wins = _slide_windows(anchor, T)          # (N, K, T, d)
    N, K, _, d = wins.shape
    flat = wins.reshape(N * K, T, d)
    z = _encode_per_token(model.encode, flat, device)   # (N*K, d_sae)
    z = z.reshape(N, K, -1)
    return z.mean(axis=1)


def _encode_strided(model, anchor, last_idx, T_eff: int, stride: int,
                     device, aggregation: str):
    """Strided probe:
      - last_position: take last T_eff*stride tokens, stride-sample to T_eff, encode.
      - mean_pool: slide span=T_eff*stride window across tail-20, stride-sample
                   each slide, encode, mean.
    """
    span = T_eff * stride
    if aggregation == "last_position":
        # (N, span, d) ending at last real token
        X_span = _window_at_last(anchor, last_idx, span)           # (N, span, d)
        # Stride-sample every `stride`-th position, keeping last token aligned
        positions = np.arange(span - 1 - (T_eff - 1) * stride, span, stride)
        X = X_span[:, positions, :]                                # (N, T_eff, d)
        return _encode_per_token(model.encode, X, device)
    # mean_pool over tail-20 with span-sized windows, stride-sampled then encoded
    N, LN, d = anchor.shape
    if LN < span:
        # Can only do last_position; fall back.
        return _encode_strided(model, anchor, last_idx, T_eff, stride, device, "last_position")
    K = LN - span + 1
    positions_within_span = np.arange(span - 1 - (T_eff - 1) * stride, span, stride)
    # Slide: for each k in [0, K), take anchor[:, k:k+span, :], stride-sample.
    # Vectorized: slides = _slide_windows(anchor, span) -> (N, K, span, d)
    slides = _slide_windows(anchor, span)                          # (N, K, span, d)
    strided = slides[:, :, positions_within_span, :]               # (N, K, T_eff, d)
    flat = strided.reshape(N * K, T_eff, d)
    z = _encode_per_token(model.encode, flat, device)              # (N*K, d_sae)
    z = z.reshape(N, K, -1)
    return z.mean(axis=1)


def _encode_paired(model, anchor, last_idx, T_paired: int,
                    device, aggregation: str):
    """Pair-summed probe:
      - last_position: take last 2*T_paired tokens, pair-sum adjacent, encode.
      - mean_pool: slide 2*T_paired span across tail-20, pair-sum, encode, mean.
    """
    span = 2 * T_paired
    if aggregation == "last_position":
        X_span = _window_at_last(anchor, last_idx, span)           # (N, span, d)
        # Pair-sum: reshape to (N, T_paired, 2, d), sum over inner 2.
        N, _, d = X_span.shape
        X_paired = X_span.reshape(N, T_paired, 2, d).sum(axis=2)   # (N, T_paired, d)
        return _encode_per_token(model.encode, X_paired, device)
    N, LN, d = anchor.shape
    if LN < span:
        return _encode_paired(model, anchor, last_idx, T_paired, device, "last_position")
    K = LN - span + 1
    slides = _slide_windows(anchor, span)                          # (N, K, span, d)
    paired = slides.reshape(N, K, T_paired, 2, d).sum(axis=3)      # (N, K, T_paired, d)
    flat = paired.reshape(N * K, T_paired, d)
    z = _encode_per_token(model.encode, flat, device)              # (N*K, d_sae)
    z = z.reshape(N, K, -1)
    return z.mean(axis=1)


def _encode_token_subseq(model, anchor, last_idx, t_sample: int,
                          device, aggregation: str):
    """Token-level encoder with sparse sum:
      - last_position: encode the last t_sample tokens each, sum z's.
      - mean_pool: slide t_sample-window across tail-20, sum z's per slide, mean.

    TokenSubseqSAE.encode((B, T, d)) already sums per-token z's.
    """
    return _encode_txc_like(model, anchor, last_idx, t_sample, device, aggregation)


def _encode_for_probe_phase5b(model, arch: str, meta: dict,
                                anchor: np.ndarray, last_idx: np.ndarray,
                                device, aggregation: str):
    if aggregation not in SUPPORTED_AGGS:
        raise ValueError(f"Phase 5B supports {SUPPORTED_AGGS}, got {aggregation}")
    if arch == "strided_track2" or arch == "strided_h8":
        return _encode_strided(
            model, anchor, last_idx,
            T_eff=int(meta["T_eff"]), stride=int(meta["stride"]),
            device=device, aggregation=aggregation,
        )
    if arch == "subseq_track2" or arch == "subseq_h8":
        T_max = int(meta["T_max"])
        # At probe time we use FULL T_max (no subsampling) per plan.
        return _encode_txc_like(model, anchor, last_idx, T_max, device, aggregation)
    if arch == "token_subseq":
        t_sample = int(meta["t_sample"])
        return _encode_token_subseq(model, anchor, last_idx, t_sample, device, aggregation)
    if arch == "paired_track2" or arch == "paired_h8":
        T_paired = int(meta["T_paired"])
        return _encode_paired(model, anchor, last_idx, T_paired, device, aggregation)
    if arch == "pps_matryoshka":
        T = int(meta["T"])
        return _encode_txc_like(model, anchor, last_idx, T, device, aggregation)
    if arch == "subset_encoder":
        # Model's encode handles T_window padding/truncation + probe strategy.
        T_window = int(meta["T_window"])
        return _encode_txc_like(model, anchor, last_idx, T_window, device, aggregation)
    raise ValueError(f"No probe encoder for arch={arch}")


# ─────────────────────────────────────────── main loop


def _iter_index(run_ids: list[str] | None):
    """Walk the Phase 5B training index; yield (run_id, ckpt_path, meta)."""
    if not INDEX_PATH.exists():
        return
    with INDEX_PATH.open() as f:
        for line in f:
            row = json.loads(line)
            if run_ids and row["run_id"] not in run_ids:
                continue
            ckpt = Path(row["ckpt"])
            yield row["run_id"], ckpt, row


def run_probing_phase5b(run_ids: list[str] | None = None,
                         task_names: list[str] | None = None,
                         aggregations: list[str] | None = None,
                         k_values: list[int] | None = None):
    aggregations = aggregations or list(SUPPORTED_AGGS)
    for a in aggregations:
        assert a in SUPPORTED_AGGS, f"unknown agg {a}"
    k_values = k_values or K_VALUES
    device = torch.device("cuda")
    PROBING_PATH.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = [d for d in sorted(PROBE_CACHE.iterdir()) if d.is_dir()]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "meta.json").exists()
    ]
    print(f"[phase5b probe] {len(task_dirs)} tasks, aggs={aggregations}")

    # Outer loop: models (one model load covers all tasks × aggregations).
    import gc
    with PROBING_PATH.open("a") as out_f:
        for run_id, ckpt_path, meta in _iter_index(run_ids):
            if not ckpt_path.exists():
                print(f"  {run_id}: ckpt missing {ckpt_path}")
                continue
            try:
                model, arch = _load_phase5b_model(run_id, ckpt_path, device, meta)
            except Exception as e:
                print(f"  {run_id}: load FAIL {type(e).__name__}: {e}")
                continue
            print(f"  {run_id} arch={arch}")

            for task_dir in task_dirs:
                task_name = task_dir.name
                tc = _load_task_cache(task_dir)
                dkey = tc["meta"]["dataset_key"]
                ytr = tc["train_labels"]
                yte = tc["test_labels"]

                for aggregation in aggregations:
                    t0 = time.time()
                    try:
                        Z_tr = _encode_for_probe_phase5b(
                            model, arch, meta,
                            tc["anchor_train"], tc["train_last_idx"],
                            device, aggregation,
                        )
                        Z_te = _encode_for_probe_phase5b(
                            model, arch, meta,
                            tc["anchor_test"], tc["test_last_idx"],
                            device, aggregation,
                        )
                    except Exception as e:
                        print(f"    {task_name} {aggregation}: encode FAIL {type(e).__name__}: {e}")
                        continue

                    enc_t = time.time() - t0
                    for k_feat in k_values:
                        auc, acc = sae_probe_metrics(Z_tr, ytr, Z_te, yte, k_feat)
                        # Apply Phase 5 FLIP convention for cross-token tasks.
                        auc_flip = flipped_auc(auc, task_name)
                        row = {
                            "run_id": run_id, "arch": arch,
                            "task_name": task_name, "dataset_key": dkey,
                            "aggregation": aggregation,
                            "k_feat": int(k_feat),
                            "test_auc": float(auc),
                            "test_auc_flip": float(auc_flip),
                            "test_acc": float(acc),
                            "n_train": int(ytr.size), "n_test": int(yte.size),
                            "elapsed_s_encode": float(enc_t),
                            **{k: meta.get(k) for k in (
                                "T", "T_eff", "T_max", "T_paired", "stride",
                                "t_sample", "n_scales", "alpha", "contiguous",
                                "use_pos", "pos_mode",
                                "seed", "k_pos", "k_win",
                            ) if meta.get(k) is not None},
                        }
                        out_f.write(json.dumps(row) + "\n")
                        out_f.flush()
                    print(
                        f"    {task_name:40s} {aggregation:13s} "
                        f"auc@5={sae_probe_metrics(Z_tr, ytr, Z_te, yte, 5)[0]:.4f} "
                        f"enc_s={enc_t:.1f}"
                    )
                del tc
                gc.collect()
            del model
            torch.cuda.empty_cache()
            gc.collect()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_ids", nargs="+", default=None,
                    help="Phase 5B run_ids to probe. Default: all in training_index.jsonl")
    p.add_argument("--task_names", nargs="+", default=None,
                    help="Subset of tasks to probe (default: all 36)")
    p.add_argument("--aggregations", nargs="+",
                    choices=list(SUPPORTED_AGGS), default=list(SUPPORTED_AGGS))
    p.add_argument("--k_values", nargs="+", type=int, default=K_VALUES)
    args = p.parse_args()
    run_probing_phase5b(
        run_ids=args.run_ids,
        task_names=args.task_names,
        aggregations=args.aggregations,
        k_values=args.k_values,
    )


if __name__ == "__main__":
    main()
