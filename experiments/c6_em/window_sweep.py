"""Paper-matched TXC window-size sweep for C6 medical detection.

This runner starts from the exact seed-42 C6 TXC-base recipe and changes
only ``T``.  It deliberately uses the v1 paper trainer rather than the v2
``WindowBuffer`` training path.

Training:

    python -m experiments.c6_em.window_sweep --T 1

Evaluate the published T=5 checkpoint without retraining:

    python -m experiments.c6_em.window_sweep --T 5 \
        --checkpoint /path/to/88a4ddf6819d8057/model.safetensors
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from temp_bench import runner
from temp_bench.config import act_cache_dir, load_arch
from temp_bench.schemas import TrainingConfig

COMPONENT = "c6"
ARCH = "txc_base"
DATASOURCE = "qwen_2_5_7b_instruct_medical_l15_resid_post"
EVAL_PROTOCOL_VERSION = "3.0.0"
PAPER_TRAIN_KEY = "88a4ddf6819d8057"
PAPER_PR_AUC_S16 = 0.5419601387297858
PAPER_PR_AUC_SHUFFLED_S16 = 0.6011688044779191

S_GRID = (1, 2, 4, 8, 16, 32)
N_FOLDS = 5
PROBE_SEED = 42


def training_cfg(T: int) -> TrainingConfig:
    """The paper C6 TXC recipe, with T as the sole experimental axis."""
    return TrainingConfig(
        n_steps=25_000,
        batch_size=1_024,
        learning_rate=3e-4,
        optimizer="adam",
        warmup_steps=1_000,
        precision="bf16",
        bricken_enabled=True,
        bricken_resample_every=500,
        bricken_min_fires=1,
        bricken_n_check=2_048,
        bricken_max_resample_fraction=0.5,
        ema_auxk_alpha=1.0 / 8.0,
        dead_threshold_tokens=128_000,
        arch_hparams_override={"T": int(T)},
    )


def _resolve_class(class_path: str):
    module_path, class_name = class_path.split(":", 1)
    mod = __import__(module_path, fromlist=[class_name])
    return getattr(mod, class_name)


def _instantiate_txc(
    *,
    T: int,
    state_dict: dict[str, torch.Tensor],
) -> torch.nn.Module:
    spec = load_arch(ARCH, component=COMPONENT)
    hparams = {**spec.hparams, "T": int(T)}
    # These are the paper's per-cell Bricken/AuxK overrides.
    hparams["auxk_alpha"] = 1.0 / 8.0
    hparams["dead_threshold_tokens"] = 128_000
    model = _resolve_class(spec.class_path)(d_in=3_584, **hparams)
    model.load_state_dict(state_dict, strict=True)
    if torch.cuda.is_available():
        model = model.cuda()
    return model.eval()


def _encode(model, windows: torch.Tensor, *, batch_size: int = 1_024) -> np.ndarray:
    device = next(model.parameters()).device
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            z = model.encode(windows[i : i + batch_size].to(device)).abs()
            if z.ndim == 3 and z.shape[1] > 1:
                z = z.amax(dim=1)
            elif z.ndim == 3:
                z = z.squeeze(1)
            chunks.append(z.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _probe(feats: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[int, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold

    splits = list(GroupKFold(N_FOLDS).split(feats, labels, groups=groups))
    out: dict[int, float] = {}
    for S in S_GRID:
        fold_scores: list[float] = []
        for train_idx, test_idx in splits:
            X_train = feats[train_idx]
            y_train = labels[train_idx]
            mean_pos = X_train[y_train == 1].mean(axis=0)
            mean_neg = X_train[y_train == 0].mean(axis=0)
            top = np.argsort(np.abs(mean_pos - mean_neg))[-S:]
            clf = LogisticRegression(
                penalty="l1",
                C=1.0,
                solver="liblinear",
                max_iter=2_000,
                random_state=PROBE_SEED,
            )
            clf.fit(X_train[:, top], y_train)
            score = clf.predict_proba(feats[test_idx][:, top])[:, 1]
            fold_scores.append(float(average_precision_score(labels[test_idx], score)))
        out[S] = float(np.mean(fold_scores))
    return out


def _window_cohort(T: int) -> tuple[torch.Tensor, np.ndarray, np.ndarray, int]:
    cohort_dir = Path(
        os.environ.get(
            "TEMP_BENCH_EM_COHORT_DIR",
            "/workspace/conv_depth_caches/em_medical",
        )
    )
    acts = np.load(cohort_dir / "hs16.npy", mmap_mode="r")
    lens = np.load(cohort_dir / "lens.npy")
    labels = np.load(cohort_dir / "labels.npy")
    qids = np.load(cohort_dir / "qids.npy")

    rows: list[tuple[int, int]] = []
    window_labels: list[int] = []
    window_qids: list[int] = []
    for rollout_idx, length in enumerate(lens):
        for start in range(max(int(length) - T + 1, 0)):
            rows.append((rollout_idx, start))
            window_labels.append(int(labels[rollout_idx]))
            window_qids.append(int(qids[rollout_idx]))

    windows = torch.empty((len(rows), T, acts.shape[-1]), dtype=torch.float32)
    for out_idx, (rollout_idx, start) in enumerate(rows):
        windows[out_idx] = torch.from_numpy(
            np.asarray(acts[rollout_idx, start : start + T], dtype=np.float32)
        )
    return (
        windows,
        np.asarray(window_labels, dtype=np.int64),
        np.asarray(window_qids, dtype=np.int64),
        len(lens),
    )


def evaluate_state(
    *,
    T: int,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Run the paper protocol-3 sparse-probe detection + shuffle control."""
    model = _instantiate_txc(T=T, state_dict=state_dict)
    windows, labels, groups, n_rollouts = _window_cohort(T)

    metrics: dict[str, float] = {
        "n_sent": float(len(windows)),
        "positive_rate": float(labels.mean()),
        "n_rollouts": float(n_rollouts),
        "n_folds": float(N_FOLDS),
        "l0_per_window": float(25 * T),
        "l0_per_token": 25.0,
    }
    ordered = _probe(_encode(model, windows), labels, groups)
    for S, value in ordered.items():
        metrics[f"pr_auc_S{S}"] = value

    if T == 1:
        shuffled = ordered
    else:
        generator = torch.Generator().manual_seed(PROBE_SEED)
        perms = torch.argsort(torch.rand(len(windows), T, generator=generator), dim=1)
        index = perms.unsqueeze(-1).expand(-1, -1, windows.shape[-1])
        shuffled_windows = torch.gather(windows, 1, index)
        shuffled = _probe(_encode(model, shuffled_windows), labels, groups)
    for S, value in shuffled.items():
        metrics[f"pr_auc_shuffled_S{S}"] = value
        metrics[f"shuffle_gap_S{S}"] = ordered[S] - value
    return metrics


def make_eval_fn(T: int):
    def eval_fn(*, model, eval_cfg, component):
        del model, component
        metrics = evaluate_state(T=T, state_dict=eval_cfg["_state_dict"])
        return metrics, "pr_auc_S16"

    return eval_fn


def _load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(path), device="cpu")


def _write_output(payload: dict) -> None:
    output = os.environ.get("TEMP_BENCH_EM_WINDOW_OUTPUT")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, required=True, choices=(1, 2, 4, 5, 6))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--force-train", action="store_true")
    args = parser.parse_args(argv)
    if args.seed != 42:
        raise ValueError("Reviewer window sweep is pinned to the headline seed 42")

    if args.checkpoint:
        metrics = evaluate_state(T=args.T, state_dict=_load_checkpoint(args.checkpoint))
        payload = {
            "T": args.T,
            "seed": args.seed,
            "arch": ARCH,
            "paper_checkpoint": str(args.checkpoint),
            "paper_train_key": PAPER_TRAIN_KEY if args.T == 5 else None,
            "metrics": metrics,
        }
        if args.T == 5:
            payload["paper_gate_abs_error"] = {
                "ordered_S16": abs(metrics["pr_auc_S16"] - PAPER_PR_AUC_S16),
                "shuffled_S16": abs(
                    metrics["pr_auc_shuffled_S16"] - PAPER_PR_AUC_SHUFFLED_S16
                ),
            }
        _write_output(payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return

    # The activation cache is provisioned by the RunPod launcher. Avoid an
    # accidental multi-GB rebuild under an unexpected key.
    from temp_bench.config import compute_act_cache_key, load_datasource

    cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    cache_dir = act_cache_dir(cache_key)
    specs = cache_dir / "layer_specs.json"
    if not specs.exists():
        raise FileNotFoundError(f"paper activation cache not provisioned at {cache_dir}")

    from experiments.c6_em.train import my_train_fn

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=ARCH,
        seed=args.seed,
        datasource_name=DATASOURCE,
        training_cfg=training_cfg(args.T),
        eval_cfg={
            "arch_T": args.T,
            "window_sweep": True,
            "cohort": "paper-stage4-medical",
            "S_grid": list(S_GRID),
            "shuffle_seed": PROBE_SEED,
        },
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn,
        eval_fn=make_eval_fn(args.T),
        force_train=args.force_train,
        agent="codex-em-paper-window-sweep",
    )
    payload = {
        "T": args.T,
        "seed": args.seed,
        "arch": ARCH,
        "train_key": result.train_key,
        "eval_key": result.eval_key,
        "cached": result.cached,
        "metrics": result.metrics,
    }
    _write_output(payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
