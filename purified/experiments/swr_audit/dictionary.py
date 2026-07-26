"""Actual-dictionary C7 order-control audit for TXC-base and TopK-SAE.

This is distinct from the supervised residual SWR upper bound. It loads only
the frozen encoder tensors from the exact C7 checkpoints, caches sparse codes,
then fits a grouped sparse probe on ordered codes. The same fitted probe and
feature set are applied to shuffled, reversed, and circularly shifted codes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from experiments.swr_audit.run import _shuffle_rows, c7_groups, trailing_window


PROTOCOL_VERSION = "0.1.0"
EXPECTED_CACHE = "fb2a74be884e512a"
EXPECTED_DATASOURCE = "llama_3_1_8b_base_l10_ward_nousmirror"
EXPECTED_ARCH = {
    "08fe3af07682fab4": "txc_base",
    "f437e623fabc37ec": "topk_sae",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_encoder(checkpoint_dir: Path, train_key: str, device: str) -> dict:
    from safetensors import safe_open

    config = json.loads((checkpoint_dir / "config.json").read_text())
    expected_arch = EXPECTED_ARCH[train_key]
    checks = {
        "arch": config.get("arch") == expected_arch,
        "act_cache_key": config.get("act_cache_key") == EXPECTED_CACHE,
        "datasource": config.get("datasource") == EXPECTED_DATASOURCE,
    }
    if not all(checks.values()):
        raise ValueError(f"checkpoint provenance mismatch: {checks}, config={config}")
    path = checkpoint_dir / "model.safetensors"
    with safe_open(path, framework="pt", device="cpu") as handle:
        state = {
            "W_enc": handle.get_tensor("W_enc").to(device),
            "b_enc": handle.get_tensor("b_enc").to(device),
        }
        if expected_arch == "topk_sae":
            state["b_dec"] = handle.get_tensor("b_dec").to(device)
    return {"config": config, "state": state, "path": path}


def _sparse_from_topk(
    values: torch.Tensor, indices: torch.Tensor, n_features: int
) -> sparse.csr_matrix:
    values_np = values.float().cpu().numpy()
    indices_np = indices.int().cpu().numpy()
    rows: list[np.ndarray] = []
    data: list[np.ndarray] = []
    indptr = [0]
    for row_values, row_indices in zip(values_np, indices_np):
        keep = row_values > 0
        rows.append(row_indices[keep])
        data.append(row_values[keep])
        indptr.append(indptr[-1] + int(keep.sum()))
    all_indices = np.concatenate(rows).astype(np.int32, copy=False)
    all_data = np.concatenate(data).astype(np.float32, copy=False)
    return sparse.csr_matrix(
        (all_data, all_indices, np.asarray(indptr, dtype=np.int64)),
        shape=(len(values_np), n_features),
    )


@torch.no_grad()
def encode_txc_batch(
    x: torch.Tensor, state: dict[str, torch.Tensor], k_pos: int = 20
) -> sparse.csr_matrix:
    weight, bias = state["W_enc"], state["b_enc"]
    if x.shape[1] != weight.shape[0]:
        raise ValueError(f"TXC expects T={weight.shape[0]}, got {x.shape[1]}")
    pre = torch.einsum("btd,tds->bs", x.to(weight.dtype), weight) + bias
    values, indices = pre.topk(min(k_pos * x.shape[1], weight.shape[-1]), dim=-1)
    return _sparse_from_topk(torch.relu(values), indices, weight.shape[-1])


@torch.no_grad()
def encode_topk_pool_batch(
    x: torch.Tensor, state: dict[str, torch.Tensor], k_pos: int = 20
) -> sparse.csr_matrix:
    weight, bias, decoder_bias = state["W_enc"], state["b_enc"], state["b_dec"]
    batch, window, width = x.shape
    flat = x.reshape(batch * window, width).to(weight.dtype)
    pre = (flat - decoder_bias) @ weight.T + bias
    values, indices = pre.topk(k_pos, dim=-1)
    dense = torch.zeros_like(pre)
    dense.scatter_(1, indices, torch.relu(values))
    pooled = dense.reshape(batch, window, -1).amax(dim=1)
    pooled_values, pooled_indices = pooled.topk(
        min(k_pos * window, weight.shape[0]), dim=-1
    )
    return _sparse_from_topk(pooled_values, pooled_indices, weight.shape[0])


def encode_dataset(
    x: np.ndarray,
    *,
    arch: str,
    state: dict[str, torch.Tensor],
    batch_size: int,
    device: str,
) -> sparse.csr_matrix:
    chunks = []
    encoder = encode_txc_batch if arch == "txc_base" else encode_topk_pool_batch
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start : start + batch_size]).to(device)
        chunks.append(encoder(batch, state))
    return sparse.vstack(chunks, format="csr")


def _metrics(y: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    return {
        "pr_auc": float(average_precision_score(y, probabilities)),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
    }


def grouped_fixed_probe(
    ordered: sparse.csr_matrix,
    controls: dict[str, sparse.csr_matrix],
    y: np.ndarray,
    groups: np.ndarray,
    *,
    folds: int,
    s_grid: list[int],
    seed: int,
) -> list[dict]:
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    rows = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(ordered, y, groups)):
        pos = ordered[train_idx[y[train_idx] == 1]].mean(axis=0).A1
        neg = ordered[train_idx[y[train_idx] == 0]].mean(axis=0).A1
        ranking = np.argsort(np.abs(pos - neg))
        for n_features in s_grid:
            selected = ranking[-n_features:]
            x_train = ordered[train_idx][:, selected].toarray()
            x_test = ordered[test_idx][:, selected].toarray()
            classifier = LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=seed + fold,
            ).fit(x_train, y[train_idx])
            ordered_probability = classifier.predict_proba(x_test)[:, 1]
            control_scores = {
                name: _metrics(
                    y[test_idx],
                    classifier.predict_proba(matrix[test_idx][:, selected].toarray())[:, 1],
                )
                for name, matrix in controls.items()
            }
            ordered_score = _metrics(y[test_idx], ordered_probability)
            rows.append(
                {
                    "fold": fold,
                    "n_features": n_features,
                    "n_train": int(len(train_idx)),
                    "n_test": int(len(test_idx)),
                    "test_positive_rate": float(y[test_idx].mean()),
                    "ordered": ordered_score,
                    "controls": control_scores,
                    "fixed_probe_order_gap_pr_auc": {
                        name: float(ordered_score["pr_auc"] - score["pr_auc"])
                        for name, score in control_scores.items()
                    },
                }
            )
    return rows


def _whole_group_subsample(
    groups: np.ndarray, max_rows: int, seed: int
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    keep = []
    for group in rng.permutation(np.unique(groups)):
        keep.extend(np.flatnonzero(groups == group).tolist())
        if len(keep) >= max_rows:
            break
    return np.asarray(sorted(keep), dtype=np.int64)


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--train-key", choices=sorted(EXPECTED_ARCH), required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--s-grid", default="8,16,32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    arch = EXPECTED_ARCH[args.train_key]
    checkpoint_dir = args.checkpoint_root / args.train_key
    loaded = _load_encoder(checkpoint_dir, args.train_key, args.device)
    actual_checksum = _sha256(loaded["path"])
    if actual_checksum != args.checkpoint_sha256:
        raise ValueError(
            f"checkpoint checksum mismatch: {actual_checksum} != {args.checkpoint_sha256}"
        )

    with np.load(args.artifact, allow_pickle=True) as payload:
        x = payload["X"].astype(np.float32, copy=False)
        y = payload["is_bt"].astype(np.int64, copy=False)
        groups = c7_groups(payload["keys"])
    x = trailing_window(x, 5)
    if args.max_rows is not None and args.max_rows < len(x):
        keep = _whole_group_subsample(groups, args.max_rows, args.seed)
        x, y, groups = x[keep], y[keep], groups[keep]

    args.code_dir.mkdir(parents=True, exist_ok=True)
    matrices = {}
    for name in ("ordered", "shuffle", "reverse", "circular"):
        path = args.code_dir / f"{args.train_key}_{name}.npz"
        if path.exists():
            matrices[name] = sparse.load_npz(path).tocsr()
            if matrices[name].shape[0] != len(x):
                raise ValueError(
                    f"cached code row mismatch at {path}: "
                    f"{matrices[name].shape[0]} != {len(x)}"
                )
        elif arch == "topk_sae" and name != "ordered":
            matrices[name] = matrices["ordered"]
            sparse.save_npz(path, matrices[name])
        else:
            condition = (
                x
                if name == "ordered"
                else _shuffle_rows(x, args.seed + 1000, name)
            )
            matrices[name] = encode_dataset(
                condition,
                arch=arch,
                state=loaded["state"],
                batch_size=args.batch_size,
                device=args.device,
            )
            sparse.save_npz(path, matrices[name])

    rows = grouped_fixed_probe(
        matrices.pop("ordered"),
        matrices,
        y,
        groups,
        folds=args.folds,
        s_grid=_csv_ints(args.s_grid),
        seed=args.seed,
    )
    metadata = {
        "record_type": "metadata",
        "protocol_version": PROTOCOL_VERSION,
        "interpretation": "actual frozen unsupervised dictionary with supervised sparse probe",
        "arch": arch,
        "train_key": args.train_key,
        "checkpoint_sha256": actual_checksum,
        "act_cache_key": loaded["config"]["act_cache_key"],
        "input_offsets": [-12, -11, -10, -9, -8],
        "n_rows": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(json.dumps(metadata, sort_keys=True), flush=True)
    for row in rows:
        print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
