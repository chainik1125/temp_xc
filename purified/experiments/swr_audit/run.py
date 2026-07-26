"""Event-conditioned shared-window recoverability (SWR) audit.

The first supported cohort is the C7 backtracking sentence cache. The audit
asks whether an order-sensitive, rank-limited bottleneck recovers labels that
neither the best single offset nor a capacity-matched permutation-trained
bottleneck can recover. All preprocessing is fit inside the outer train fold.

The current C7 artifact contains only the six pre-sentence offsets ``-13..-8``.
It can test windows up to ``T=6``; testing ``T=10`` or ``T=20`` requires a new
activation extraction with a wider, explicitly aligned offset range.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from torch import nn


PROTOCOL_VERSION = "0.1.0"


def c7_groups(keys: np.ndarray) -> np.ndarray:
    """Return the question/prompt id from ``<qid>|<trace>|<sentence>`` keys."""
    return np.asarray([str(key).split("|", 1)[0] for key in keys], dtype=object)


def trailing_window(x: np.ndarray, window: int) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"expected (n, T, d), got {x.shape}")
    if not 1 <= window <= x.shape[1]:
        raise ValueError(f"window must be in [1, {x.shape[1]}], got {window}")
    return x[:, -window:, :]


@dataclass
class FoldPreprocessor:
    """Per-offset centering, optional token RMS control, then shared PCA."""

    normalization: str
    pca_dim: int
    seed: int
    pca_sample_tokens: int = 50_000

    def fit(self, x: np.ndarray) -> "FoldPreprocessor":
        if self.normalization not in {"raw", "token_rms"}:
            raise ValueError(f"unknown normalization {self.normalization!r}")
        work = self._normalize(x)
        self.offset_mean_ = work.mean(axis=0, dtype=np.float64).astype(np.float32)
        flat = (work - self.offset_mean_[None]).reshape(-1, work.shape[-1])
        if len(flat) > self.pca_sample_tokens:
            rng = np.random.default_rng(self.seed)
            flat = flat[rng.choice(len(flat), self.pca_sample_tokens, replace=False)]
        n_components = min(self.pca_dim, flat.shape[0] - 1, flat.shape[1])
        if n_components < 1:
            raise ValueError("not enough train-fold tokens to fit PCA")
        self.pca_ = PCA(
            n_components=n_components,
            whiten=True,
            svd_solver="randomized",
            random_state=self.seed,
        ).fit(flat)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        work = self._normalize(x)
        centered = work - self.offset_mean_[None]
        shape = centered.shape
        return self.pca_.transform(centered.reshape(-1, shape[-1])).reshape(
            shape[0], shape[1], -1
        ).astype(np.float32, copy=False)

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        work = np.asarray(x, dtype=np.float32)
        if self.normalization == "token_rms":
            rms = np.sqrt(np.mean(np.square(work), axis=-1, keepdims=True))
            work = work / np.maximum(rms, 1e-6)
        return work


class TemporalBottleneck(nn.Module):
    """Rank-limited nonlinear readout with one weight matrix per offset."""

    def __init__(self, window: int, d_in: int, rank: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(window, d_in, rank))
        self.bias = nn.Parameter(torch.zeros(rank))
        self.head = nn.Linear(rank, 1)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(torch.einsum("btp,tpr->br", x, self.weight) + self.bias)
        return self.head(hidden).squeeze(-1)


class MeanPoolBottleneck(nn.Module):
    """DeepSets readout: shared token map, mean aggregation, then head."""

    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.token_projection = nn.Linear(d_in, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = torch.relu(self.token_projection(x)).mean(dim=1)
        return self.head(hidden).squeeze(-1)


@dataclass
class FitResult:
    model: nn.Module
    calibrator: LogisticRegression | None
    best_epoch: int
    best_val_ap: float


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _shuffle_rows(x: np.ndarray, seed: int, mode: str) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, window, _ = x.shape
    if window == 1:
        return x.copy()
    if mode == "shuffle":
        order = np.stack([rng.permutation(window) for _ in range(n)])
    elif mode == "reverse":
        order = np.broadcast_to(np.arange(window - 1, -1, -1), (n, window))
    elif mode == "circular":
        shifts = rng.integers(1, window, size=n)
        base = np.arange(window)
        order = np.stack([np.roll(base, int(shift)) for shift in shifts])
    else:
        raise ValueError(f"unknown order control {mode!r}")
    return np.take_along_axis(x, order[:, :, None], axis=1)


@torch.no_grad()
def _logits(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    out: list[np.ndarray] = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        out.append(model(xb).float().cpu().numpy())
    return np.concatenate(out)


def _permutation_average_logits(
    model: nn.Module,
    x: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    repeats: int = 8,
) -> np.ndarray:
    """Monte Carlo symmetrization used to select the permutation null."""
    values = [
        _logits(
            model,
            _shuffle_rows(x, seed + repeat, "shuffle"),
            device=device,
            batch_size=batch_size,
        )
        for repeat in range(repeats)
    ]
    return np.mean(values, axis=0)


def fit_bottleneck(
    x_fit: np.ndarray,
    y_fit: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    rank: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    permute_train: bool = False,
    model_kind: str = "ordered",
    hidden: int | None = None,
) -> FitResult:
    """Fit an ordered or exactly invariant bottleneck."""
    _seed_everything(seed)
    torch_device = torch.device(device)
    if model_kind == "ordered":
        model: nn.Module = TemporalBottleneck(x_fit.shape[1], x_fit.shape[2], rank)
    elif model_kind == "mean_pool":
        model = MeanPoolBottleneck(x_fit.shape[2], hidden or rank)
    else:
        raise ValueError(f"unknown model_kind {model_kind!r}")
    model = model.to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    n_pos = max(int(y_fit.sum()), 1)
    pos_weight = torch.tensor((len(y_fit) - n_pos) / n_pos, device=torch_device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_val_ap = -math.inf
    best_epoch = 0
    stale = 0

    for epoch in range(epochs):
        model.train()
        order = rng.permutation(len(x_fit))
        if permute_train:
            epoch_x = _shuffle_rows(x_fit, seed + 2 * epoch, "shuffle")
            epoch_x_pair = _shuffle_rows(x_fit, seed + 2 * epoch + 1, "shuffle")
        else:
            epoch_x = x_fit
            epoch_x_pair = None
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            xb = torch.from_numpy(epoch_x[idx]).to(torch_device)
            yb = torch.from_numpy(y_fit[idx].astype(np.float32)).to(torch_device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if epoch_x_pair is not None:
                xb_pair = torch.from_numpy(epoch_x_pair[idx]).to(torch_device)
                paired_logits = model(xb_pair)
                loss = 0.5 * (loss + loss_fn(paired_logits, yb))
                loss = loss + 5.0 * torch.mean((logits - paired_logits) ** 2)
            loss.backward()
            optimizer.step()
        if permute_train:
            val_logits = _permutation_average_logits(
                model,
                x_val,
                device=torch_device,
                batch_size=batch_size,
                seed=seed + 50_000 + 10 * epoch,
            )
        else:
            val_logits = _logits(model, x_val, device=torch_device, batch_size=batch_size)
        val_ap = float(average_precision_score(y_val, val_logits))
        if val_ap > best_val_ap + 1e-5:
            best_val_ap = val_ap
            best_epoch = epoch + 1
            best_state = {k: value.detach().cpu().clone() for k, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 12:
            break

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    if permute_train:
        val_logits = _permutation_average_logits(
            model,
            x_val,
            device=torch_device,
            batch_size=batch_size,
            seed=seed + 60_000,
        )
    else:
        val_logits = _logits(model, x_val, device=torch_device, batch_size=batch_size)
    calibrator: LogisticRegression | None = None
    if len(np.unique(y_val)) == 2:
        calibrator = LogisticRegression(C=1e3, solver="lbfgs").fit(
            val_logits[:, None], y_val
        )
    return FitResult(model, calibrator, best_epoch, best_val_ap)


def score_model(
    fit: FitResult,
    x: np.ndarray,
    y: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> dict[str, float]:
    logits = _logits(fit.model, x, device=torch.device(device), batch_size=batch_size)
    if fit.calibrator is None:
        probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    else:
        probs = fit.calibrator.predict_proba(logits[:, None])[:, 1]
    return {
        "pr_auc": float(average_precision_score(y, probs)),
        "roc_auc": float(roc_auc_score(y, probs)),
        "log_loss": float(log_loss(y, probs, labels=[0, 1])),
    }


def _inner_split(
    train_idx: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    inner_fit, inner_val = next(
        splitter.split(train_idx, y[train_idx], groups=groups[train_idx])
    )
    return train_idx[inner_fit], train_idx[inner_val]


def run_c7_fold(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    *,
    fold: int,
    window: int,
    normalization: str,
    pca_dim: int,
    rank: int,
    seed: int,
    device: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pca_sample_tokens: int,
    artifact_offsets: tuple[int, ...] | None = None,
) -> dict:
    x = trailing_window(x, window)
    if artifact_offsets is None:
        window_offsets = tuple(range(-window, 0))
    else:
        if len(artifact_offsets) < window:
            raise ValueError(
                f"artifact has {len(artifact_offsets)} offsets but T={window} was requested"
            )
        window_offsets = artifact_offsets[-window:]
    pre = FoldPreprocessor(normalization, pca_dim, seed + fold, pca_sample_tokens)
    pre.fit(x[train_idx])
    z = pre.transform(x)
    fit_idx, val_idx = _inner_split(train_idx, y, groups, seed + fold)
    common = dict(
        rank=rank,
        device=device,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    ordered = fit_bottleneck(
        z[fit_idx], y[fit_idx], z[val_idx], y[val_idx], seed=seed + 100 * fold, **common
    )
    invariant_matched = fit_bottleneck(
        z[fit_idx],
        y[fit_idx],
        z[val_idx],
        y[val_idx],
        seed=seed + 100 * fold + 1,
        model_kind="mean_pool",
        hidden=window * rank,
        **common,
    )
    invariant_rank = fit_bottleneck(
        z[fit_idx],
        y[fit_idx],
        z[val_idx],
        y[val_idx],
        seed=seed + 100 * fold + 2,
        model_kind="mean_pool",
        hidden=rank,
        **common,
    )
    permuted = fit_bottleneck(
        z[fit_idx],
        y[fit_idx],
        z[val_idx],
        y[val_idx],
        seed=seed + 100 * fold + 3,
        permute_train=True,
        **common,
    )

    best_offset = 0
    best_token: FitResult | None = None
    for offset in range(window):
        candidate = fit_bottleneck(
            z[fit_idx, offset : offset + 1],
            y[fit_idx],
            z[val_idx, offset : offset + 1],
            y[val_idx],
            seed=seed + 100 * fold + 20 + offset,
            **common,
        )
        if best_token is None or candidate.best_val_ap > best_token.best_val_ap:
            best_token = candidate
            best_offset = offset
    assert best_token is not None

    ordered_score = score_model(ordered, z[test_idx], y[test_idx], device=device, batch_size=batch_size)
    invariant_matched_score = score_model(
        invariant_matched, z[test_idx], y[test_idx], device=device, batch_size=batch_size
    )
    invariant_rank_score = score_model(
        invariant_rank, z[test_idx], y[test_idx], device=device, batch_size=batch_size
    )
    permuted_score = score_model(permuted, z[test_idx], y[test_idx], device=device, batch_size=batch_size)
    token_score = score_model(
        best_token,
        z[test_idx, best_offset : best_offset + 1],
        y[test_idx],
        device=device,
        batch_size=batch_size,
    )
    controls = {
        mode: score_model(
            ordered,
            _shuffle_rows(z[test_idx], seed + 10_000 + fold, mode),
            y[test_idx],
            device=device,
            batch_size=batch_size,
        )
        for mode in ("shuffle", "reverse", "circular")
    }
    return {
        "fold": fold,
        "window": window,
        "normalization": normalization,
        "pca_dim_actual": int(z.shape[-1]),
        "rank": rank,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "test_positive_rate": float(y[test_idx].mean()),
        "window_offsets": list(window_offsets),
        "best_offset_relative": int(window_offsets[best_offset]),
        "ordered": ordered_score,
        "mean_pool_param_matched": invariant_matched_score,
        "mean_pool_same_rank": invariant_rank_score,
        "permutation_trained": permuted_score,
        "best_token": token_score,
        "controls": controls,
        "parameter_counts": {
            "ordered": _parameter_count(ordered.model),
            "mean_pool_param_matched": _parameter_count(invariant_matched.model),
            "mean_pool_same_rank": _parameter_count(invariant_rank.model),
            "permutation_trained": _parameter_count(permuted.model),
            "best_token": _parameter_count(best_token.model),
        },
        "swr_pr_auc": float(
            ordered_score["pr_auc"]
            - max(
                invariant_matched_score["pr_auc"],
                invariant_rank_score["pr_auc"],
                token_score["pr_auc"],
            )
        ),
        "swr_pr_auc_param_matched": float(
            ordered_score["pr_auc"]
            - max(invariant_matched_score["pr_auc"], token_score["pr_auc"])
        ),
        "order_gap_pr_auc": float(
            ordered_score["pr_auc"] - controls["shuffle"]["pr_auc"]
        ),
        "best_epochs": {
            "ordered": ordered.best_epoch,
            "mean_pool_param_matched": invariant_matched.best_epoch,
            "mean_pool_same_rank": invariant_rank.best_epoch,
            "permutation_trained": permuted.best_epoch,
            "best_token": best_token.best_epoch,
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--windows", default="1,3,5,6")
    parser.add_argument(
        "--artifact-offsets",
        default="-13,-12,-11,-10,-9,-8",
        help="Comma-separated offsets represented by the artifact's T axis.",
    )
    parser.add_argument("--normalizations", default="raw,token_rms")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--rank", type=int, default=20)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--pca-sample-tokens", type=int, default=50_000)
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    with np.load(args.artifact, allow_pickle=True) as payload:
        x = payload["X"].astype(np.float32, copy=False)
        y = payload["is_bt"].astype(np.int64, copy=False)
        keys = payload["keys"]
    groups = c7_groups(keys)
    artifact_offsets = tuple(_csv_ints(args.artifact_offsets))
    if len(artifact_offsets) != x.shape[1]:
        raise ValueError(
            f"--artifact-offsets has {len(artifact_offsets)} entries but X has T={x.shape[1]}"
        )
    if args.max_rows is not None and args.max_rows < len(x):
        rng = np.random.default_rng(args.seed)
        chosen_groups = rng.permutation(np.unique(groups))
        keep: list[int] = []
        for group in chosen_groups:
            keep.extend(np.flatnonzero(groups == group).tolist())
            if len(keep) >= args.max_rows:
                break
        keep = sorted(keep[: args.max_rows])
        x, y, groups = x[keep], y[keep], groups[keep]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "record_type": "metadata",
        "protocol_version": PROTOCOL_VERSION,
        "artifact": str(args.artifact.resolve()),
        "artifact_sha256": sha256(args.artifact),
        "n_rows": int(len(x)),
        "n_groups": int(len(np.unique(groups))),
        "positive_rate": float(y.mean()),
        "interpretation": (
            "supervised residual upper-bound; does not show that an unsupervised "
            "SAE/TXC dictionary recovered the mechanism"
        ),
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    with args.output.open("w") as handle:
        handle.write(json.dumps(metadata, sort_keys=True) + "\n")

    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    splits = list(splitter.split(x, y, groups))
    for window in _csv_ints(args.windows):
        for normalization in args.normalizations.split(","):
            for fold, (train_idx, test_idx) in enumerate(splits):
                row = run_c7_fold(
                    x,
                    y,
                    groups,
                    train_idx,
                    test_idx,
                    fold=fold,
                    window=window,
                    normalization=normalization,
                    pca_dim=args.pca_dim,
                    rank=args.rank,
                    seed=args.seed,
                    device=args.device,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    pca_sample_tokens=args.pca_sample_tokens,
                    artifact_offsets=artifact_offsets,
                )
                with args.output.open("a") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
