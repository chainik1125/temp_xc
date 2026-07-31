"""Evaluate temporal pooling of one shared TopK SAE on C7 backtracking.

This is deliberately an eval-only experiment.  It loads the exact 20k-step
TopK SAE used by the matched C7 panel, encodes the same final five positions
given to TXC, pools aligned SAE feature IDs across time, and reruns the locked
GroupKFold sparse-probe protocol.

The old baseline sliced the SAE input to ``T=1`` before encoding.  This script
must reproduce that row with the ``last`` arm before any pooled result is
trusted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors.torch import load_file


S_GRID = (1, 2, 4, 8, 16, 32)
PRIMARY_ARMS = ("mean", "max")
TOP20_ARMS = ("mean", "max", "recency")
RECENCY_WEIGHTS = np.asarray([1, 2, 4, 8, 16], dtype=np.float32) / 31.0


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def sha256_file(path: Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_suffix_key(state: dict[str, torch.Tensor], suffix: str) -> str:
    if suffix in state:
        return suffix
    matches = [key for key in state if key.endswith("." + suffix)]
    if len(matches) != 1:
        raise KeyError(f"expected one checkpoint key ending in {suffix!r}; got {matches}")
    return matches[0]


def load_encoder(checkpoint: Path, device: torch.device) -> tuple[torch.Tensor, ...]:
    state = load_file(str(checkpoint), device="cpu")
    tensors = tuple(
        state[_unique_suffix_key(state, key)].to(device=device, dtype=torch.float32)
        for key in ("W_enc", "b_enc", "b_dec")
    )
    w_enc, b_enc, b_dec = tensors
    if w_enc.ndim != 2 or b_enc.shape != (w_enc.shape[0],):
        raise ValueError("invalid TopK SAE encoder shapes")
    if b_dec.shape != (w_enc.shape[1],):
        raise ValueError("decoder bias does not match encoder input width")
    return tensors


def encode_topk(
    x: torch.Tensor,
    w_enc: torch.Tensor,
    b_enc: torch.Tensor,
    b_dec: torch.Tensor,
    *,
    k: int,
) -> torch.Tensor:
    """Paper-v1 TopK->ReLU encoder, accepting ``(B,T,d_in)``."""
    batch, steps, d_in = x.shape
    flat = x.reshape(batch * steps, d_in)
    pre = (flat - b_dec) @ w_enc.T + b_enc
    values, indices = pre.topk(k, dim=-1)
    z = torch.zeros_like(pre)
    z.scatter_(-1, indices, torch.relu(values))
    return z.reshape(batch, steps, w_enc.shape[0])


def pool_dense_codes(z: torch.Tensor) -> dict[str, torch.Tensor]:
    """Pre-registered label-blind temporal pools over aligned SAE IDs."""
    if z.ndim != 3 or z.shape[1] != 5:
        raise ValueError(f"expected (B,5,d_sae) codes; got {tuple(z.shape)}")
    weights = torch.as_tensor(RECENCY_WEIGHTS, device=z.device, dtype=z.dtype)
    return {
        "last": z[:, -1],
        "first": z[:, 0],
        "mean": z.mean(dim=1),
        "max": z.amax(dim=1),
        "recency": (z * weights.view(1, 5, 1)).sum(dim=1),
        "reverse_recency": (z * weights.flip(0).view(1, 5, 1)).sum(dim=1),
        **{f"position_{index}": z[:, index] for index in range(5)},
    }


def truncate_topk(features: np.ndarray, k: int, chunk_rows: int = 512) -> np.ndarray:
    """Keep the largest positive ``k`` entries per row without a huge argpartition copy."""
    if features.ndim != 2:
        raise ValueError("features must be a matrix")
    if not 0 < k <= features.shape[1]:
        raise ValueError("k must be between 1 and the feature width")
    out = np.zeros(features.shape, dtype=np.float32)
    for start in range(0, len(features), chunk_rows):
        stop = min(start + chunk_rows, len(features))
        block = np.asarray(features[start:stop], dtype=np.float32)
        indices = np.argpartition(block, -k, axis=1)[:, -k:]
        rows = np.arange(stop - start)[:, None]
        out[start:stop][rows, indices] = block[rows, indices]
    return out


def feature_summary(features: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    positive = labels == 1
    mean_pos = features[positive].mean(axis=0)
    mean_neg = features[~positive].mean(axis=0)
    signed = mean_pos - mean_neg
    order = np.argsort(np.abs(signed))[-32:][::-1]
    support = np.count_nonzero(features, axis=1)
    return {
        "mean_support": float(support.mean()),
        "median_support": float(np.median(support)),
        "max_support": int(support.max()),
        "top_feature_ids_abs_diff": [int(value) for value in order],
        "top_feature_signed_diffs": [float(signed[value]) for value in order],
        "steering_feature_id_signed_positive": int(np.argmax(signed)),
        "steering_feature_selectivity": float(signed.max()),
    }


def probe_features(
    features: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray,
    *,
    s_grid: tuple[int, ...] = S_GRID,
    n_folds: int = 5,
    c: float = 1.0,
    random_state: int = 42,
) -> dict[str, Any]:
    """Exact C7 train-fold selection + L1-logistic GroupKFold protocol."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    splits = list(GroupKFold(n_splits=n_folds).split(features, labels, groups=question_ids))
    fold_scores: dict[str, list[float]] = {str(s): [] for s in s_grid}
    for train_index, test_index in splits:
        x_train, x_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]
        mean_pos = x_train[y_train == 1].mean(axis=0)
        mean_neg = x_train[y_train == 0].mean(axis=0)
        mean_difference = np.abs(mean_pos - mean_neg)
        for s in s_grid:
            selected = np.argsort(mean_difference)[-s:]
            classifier = LogisticRegression(
                penalty="l1",
                C=c,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )
            classifier.fit(x_train[:, selected], y_train)
            probability = classifier.predict_proba(x_test[:, selected])[:, 1]
            fold_scores[str(s)].append(float(average_precision_score(y_test, probability)))
    return {
        "pr_auc": {key: float(np.mean(values)) for key, values in fold_scores.items()},
        "fold_pr_auc": fold_scores,
    }


def _open_pool(path: Path, shape: tuple[int, int]) -> np.memmap:
    return np.memmap(path, mode="r", dtype=np.float32, shape=shape)


def encode_pool_memmaps(
    acts: np.ndarray,
    *,
    encoder: tuple[torch.Tensor, ...],
    scratch: Path,
    batch_size: int,
    k: int,
    heartbeat_path: Path,
) -> tuple[list[str], tuple[int, int]]:
    """Encode once and materialize each pool to a resumable local memmap."""
    scratch.mkdir(parents=True, exist_ok=True)
    w_enc, b_enc, b_dec = encoder
    n_rows = acts.shape[0]
    shape = (n_rows, int(w_enc.shape[0]))
    names = list(pool_dense_codes(torch.zeros(1, 5, 1)).keys())
    maps = {
        name: np.memmap(scratch / f"{name}.f32", mode="w+", dtype=np.float32, shape=shape)
        for name in names
    }
    started = time.time()
    with torch.inference_mode():
        for start in range(0, n_rows, batch_size):
            stop = min(start + batch_size, n_rows)
            batch = torch.from_numpy(np.asarray(acts[start:stop, -5:, :])).to(w_enc.device)
            z = encode_topk(batch, w_enc, b_enc, b_dec, k=k)
            for name, pooled in pool_dense_codes(z).items():
                maps[name][start:stop] = pooled.detach().cpu().numpy()
            if start == 0 or stop == n_rows or (start // batch_size) % 10 == 0:
                atomic_json(
                    heartbeat_path,
                    {
                        "phase": "encoding",
                        "rows_done": stop,
                        "rows_total": n_rows,
                        "elapsed_seconds": time.time() - started,
                        "updated_epoch": time.time(),
                    },
                )
            del batch, z
    for mmap in maps.values():
        mmap.flush()
    del maps
    (scratch / "ENCODING_COMPLETE").write_text("complete\n")
    return names, shape


def render_plot(payload: dict[str, Any], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    s_grid = payload["protocol"]["S_grid"]
    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    references = payload["references"]["models"]
    for name, style in (("txc_base", "--"), ("txc_pro", "-."), ("topk_sae_last", ":")):
        values = [references[name]["pr_auc"][str(s)] for s in s_grid]
        axis.plot(s_grid, values, style, linewidth=2, label=name.replace("_", " "))
    for name, marker in (("mean", "o"), ("max", "s"), ("mean_top20", "^"), ("max_top20", "v")):
        if name not in payload["arms"]:
            continue
        values = [payload["arms"][name]["pr_auc"][str(s)] for s in s_grid]
        axis.plot(s_grid, values, marker=marker, linewidth=2, label=f"shared SAE {name}")
    axis.set_xlabel("Probe feature budget S")
    axis.set_ylabel("GroupKFold PR-AUC")
    axis.set_xticks(s_grid)
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir = args.output_dir.resolve()
    scratch = output_dir / "scratch"
    output_dir.mkdir(parents=True, exist_ok=True)
    reference_path = Path(__file__).with_name("reference_20k.json")
    reference = json.loads(reference_path.read_text())
    checkpoint_cfg = json.loads(args.checkpoint_config.read_text())
    with np.load(args.sentence_acts, allow_pickle=True) as archive:
        acts = archive["X"]
        labels = archive["is_bt"].astype(np.int64)
        keys = archive["keys"].astype(str)
    question_ids = np.asarray([key.split("|")[0] for key in keys], dtype=object)
    if acts.ndim != 3 or acts.shape[1] != 6:
        raise ValueError(f"expected cached C7 activations shaped (N,6,d); got {acts.shape}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    encoder = load_encoder(args.checkpoint, device)
    names, shape = encode_pool_memmaps(
        acts,
        encoder=encoder,
        scratch=scratch,
        batch_size=args.batch_size,
        k=args.k,
        heartbeat_path=output_dir / "heartbeat.json",
    )
    del acts, encoder

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "hypothesis": "A shared SAE pooled over the same five-token window can match or beat TXC.",
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": sha256_file(args.checkpoint),
            "config": checkpoint_cfg,
        },
        "data": {
            "path": str(args.sentence_acts),
            "sha256": sha256_file(args.sentence_acts),
            "n_sentences": int(len(labels)),
            "n_positive": int(labels.sum()),
            "positive_rate": float(labels.mean()),
            "n_questions": int(len(np.unique(question_ids))),
            "cached_window": 6,
            "evaluated_window": 5,
        },
        "protocol": {
            "S_grid": list(S_GRID),
            "n_folds": 5,
            "probe": "L1 logistic, C=1, train-fold abs(mean_pos-mean_neg) selection",
            "groups": "question_id",
            "sae_k_per_token": args.k,
            "pooled_union_support_upper_bound": args.k * 5,
            "primary_arms": list(PRIMARY_ARMS),
            "recency_weights_oldest_to_newest": RECENCY_WEIGHTS.tolist(),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "references": reference,
        "arms": {},
    }

    for index, name in enumerate(names):
        features = _open_pool(scratch / f"{name}.f32", shape)
        evaluated = np.asarray(features)
        arm = probe_features(evaluated, labels, question_ids)
        arm.update(feature_summary(evaluated, labels))
        result["arms"][name] = arm
        del features, evaluated
        atomic_json(output_dir / "raw_results.json", result)
        atomic_json(
            output_dir / "heartbeat.json",
            {
                "phase": "probing",
                "arms_done": index + 1,
                "arms_total": len(names) + len(TOP20_ARMS),
                "last_arm": name,
                "elapsed_seconds": time.time() - started,
                "updated_epoch": time.time(),
            },
        )

    for name in TOP20_ARMS:
        features = _open_pool(scratch / f"{name}.f32", shape)
        truncated = truncate_topk(features, args.k)
        arm_name = f"{name}_top{args.k}"
        arm = probe_features(truncated, labels, question_ids)
        arm.update(feature_summary(truncated, labels))
        result["arms"][arm_name] = arm
        del features, truncated
        atomic_json(output_dir / "raw_results.json", result)

    published_last = reference["models"]["topk_sae_last"]["pr_auc"]
    reproduced_last = result["arms"]["last"]["pr_auc"]
    max_abs_error = max(abs(reproduced_last[str(s)] - published_last[str(s)]) for s in S_GRID)
    result["validation"] = {
        "last_token_max_abs_error_vs_existing": float(max_abs_error),
        "last_token_reproduced": bool(max_abs_error <= args.reproduction_tolerance),
        "reproduction_tolerance": args.reproduction_tolerance,
        "permutation_invariant_arms": ["mean", "max"],
    }
    txc_base = reference["models"]["txc_base"]["pr_auc"]
    txc_pro = reference["models"]["txc_pro"]["pr_auc"]
    result["decision"] = {
        "primary_S": 8,
        "best_primary_arm": max(PRIMARY_ARMS, key=lambda arm: result["arms"][arm]["pr_auc"]["8"]),
        "beats_txc_base_S8": any(
            result["arms"][arm]["pr_auc"]["8"] > txc_base["8"] for arm in PRIMARY_ARMS
        ),
        "beats_txc_pro_S8": any(
            result["arms"][arm]["pr_auc"]["8"] > txc_pro["8"] for arm in PRIMARY_ARMS
        ),
        "beats_txc_base_all_S": any(
            all(result["arms"][arm]["pr_auc"][str(s)] > txc_base[str(s)] for s in S_GRID)
            for arm in PRIMARY_ARMS
        ),
        "beats_txc_pro_all_S": any(
            all(result["arms"][arm]["pr_auc"][str(s)] > txc_pro[str(s)] for s in S_GRID)
            for arm in PRIMARY_ARMS
        ),
        "proceed_to_judged_steering": bool(
            result["validation"]["last_token_reproduced"]
            and any(result["arms"][arm]["pr_auc"]["8"] >= txc_pro["8"] for arm in PRIMARY_ARMS)
        ),
    }
    result["elapsed_seconds"] = time.time() - started
    atomic_json(output_dir / "raw_results.json", result)
    render_plot(result, output_dir / "comparison.png")
    atomic_json(
        output_dir / "heartbeat.json",
        {"phase": "complete", "elapsed_seconds": result["elapsed_seconds"], "updated_epoch": time.time()},
    )
    if not args.keep_scratch:
        for path in scratch.glob("*.f32"):
            path.unlink()
        (scratch / "ENCODING_COMPLETE").unlink(missing_ok=True)
        scratch.rmdir()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-config", type=Path, required=True)
    parser.add_argument("--sentence-acts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--reproduction-tolerance", type=float, default=2e-3)
    parser.add_argument("--keep-scratch", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate(args)
    print(json.dumps({"validation": result["validation"], "decision": result["decision"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
