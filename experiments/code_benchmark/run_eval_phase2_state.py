"""Phase 2 — program-state linear probes.

For each architecture, linearly probe (ridge for continuous, logistic for
categorical) each program-state label from the encoded features. Report
global R² / AUC and stratified by (a) bracket-depth bucket and (b) scope
nesting bucket.

Outputs::

    results/phase2_probes_<arch>.json
    results/phase2_summary.json
    plots/phase2_r2_by_arch.png
    plots/phase2_stratified.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from code_pipeline.python_code import SubjectModelConfig, load_cache  # noqa: E402
from code_pipeline.eval_utils import (  # noqa: E402
    FIELD_NAMES,
    build_model_from_checkpoint,
    bucketize,
    encode_mlc_per_token,
    encode_sae_per_token,
    encode_txc_per_window,
    gather_labels,
    labels_for_sources,
)


CONTINUOUS = ["bracket_depth", "indent_spaces", "scope_nesting", "distance_to_header"]
CATEGORICAL = ["scope_kind"]
BINARY = ["has_await"]


def resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    return torch.device(spec)


# ---------------------------------------------------------------------------
# Probes
# ---------------------------------------------------------------------------


def ridge_r2(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    ridge: float = 1e-3,
) -> float:
    x_mean = X_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xc = X_train - x_mean
    yc = y_train - y_mean
    xtx = xc.T @ xc
    xty = xc.T @ yc
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.solve(xtx, xty)
    pred = (X_test - x_mean) @ beta + y_mean
    sse = ((y_test - pred) ** 2).sum()
    sst = ((y_test - y_test.mean()) ** 2).sum() + 1e-12
    return float(1.0 - sse / sst)


def logistic_auc(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    C: float = 1.0,
    multiclass: bool = False,
) -> float:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    # Newer sklearn (>=1.5) removed the explicit ``multi_class`` kwarg —
    # multinomial vs ovr is auto-detected from the label cardinality.
    lr = LogisticRegression(C=C, solver="lbfgs", max_iter=1000)
    lr.fit(X_train, y_train)
    if multiclass:
        # one-vs-rest AUC (macro)
        probs = lr.predict_proba(X_test)
        try:
            return float(roc_auc_score(y_test, probs, multi_class="ovr", average="macro"))
        except ValueError:
            return float("nan")
    else:
        probs = lr.predict_proba(X_test)[:, 1]
        try:
            return float(roc_auc_score(y_test, probs))
        except ValueError:
            return float("nan")


# ---------------------------------------------------------------------------
# Probe every field for one architecture
# ---------------------------------------------------------------------------


def probe_all_fields(
    latents: np.ndarray,
    labels_1d: dict[str, np.ndarray],
    n_train_windows: int,
    ridge: float,
    stratify_buckets: dict[str, list[int]],
    seed: int = 42,
) -> dict:
    rng = np.random.default_rng(seed)
    n = latents.shape[0]
    if n_train_windows >= n:
        n_train_windows = int(0.8 * n)
    perm = rng.permutation(n)
    tr = perm[:n_train_windows]
    te = perm[n_train_windows:]
    results: dict = {"overall": {}, "stratified": {}}

    # overall
    for field in CONTINUOUS:
        y = labels_1d[field].astype(np.float32)
        # drop samples with padding label (-1)
        mask = y >= 0
        if mask.sum() < 100:
            continue
        Xtr, ytr = latents[tr][mask[tr]], y[tr][mask[tr]]
        Xte, yte = latents[te][mask[te]], y[te][mask[te]]
        if ytr.shape[0] < 50 or yte.shape[0] < 50:
            continue
        r2 = ridge_r2(Xtr, ytr[:, None], Xte, yte[:, None], ridge=ridge)
        results["overall"][field] = {"metric": "r2", "value": r2,
                                     "n_train": int(ytr.shape[0]),
                                     "n_test": int(yte.shape[0])}
    for field in CATEGORICAL:
        y = labels_1d[field].astype(np.int64)
        mask = y >= 0
        classes = np.unique(y[mask])
        if len(classes) < 2:
            continue
        Xtr, ytr = latents[tr][mask[tr]], y[tr][mask[tr]]
        Xte, yte = latents[te][mask[te]], y[te][mask[te]]
        if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
            continue
        auc = logistic_auc(Xtr, ytr, Xte, yte, multiclass=True)
        results["overall"][field] = {"metric": "auc", "value": auc,
                                     "n_train": int(ytr.shape[0]),
                                     "n_test": int(yte.shape[0])}
    for field in BINARY:
        y = labels_1d[field].astype(np.int64)
        mask = y >= 0
        if mask.sum() < 50 or np.unique(y[mask]).size < 2:
            continue
        Xtr, ytr = latents[tr][mask[tr]], y[tr][mask[tr]]
        Xte, yte = latents[te][mask[te]], y[te][mask[te]]
        if np.unique(ytr).size < 2 or np.unique(yte).size < 2:
            continue
        auc = logistic_auc(Xtr, ytr, Xte, yte, multiclass=False)
        results["overall"][field] = {"metric": "auc", "value": auc,
                                     "n_train": int(ytr.shape[0]),
                                     "n_test": int(yte.shape[0])}

    # stratified
    for strat_key, thresholds in stratify_buckets.items():
        if strat_key not in labels_1d:
            continue
        buckets = bucketize(labels_1d[strat_key], thresholds)
        results["stratified"][strat_key] = {}
        for b in range(len(thresholds)):
            bucket_mask = buckets == b
            if bucket_mask.sum() < 200:
                continue
            for field in CONTINUOUS:
                y = labels_1d[field].astype(np.float32)
                lbl_mask = y >= 0
                mask = bucket_mask & lbl_mask
                if mask.sum() < 100:
                    continue
                idx_tr = tr[mask[tr]]
                idx_te = te[mask[te]]
                if idx_tr.size < 50 or idx_te.size < 50:
                    continue
                r2 = ridge_r2(latents[idx_tr], y[idx_tr][:, None],
                               latents[idx_te], y[idx_te][:, None],
                               ridge=ridge)
                results["stratified"][strat_key].setdefault(
                    f"bucket_{b}", {})[field] = {
                        "metric": "r2", "value": r2,
                        "n_train": int(idx_tr.size), "n_test": int(idx_te.size),
                }
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_overall(
    summary: dict,
    out_path: Path,
) -> None:
    import matplotlib.pyplot as plt
    archs = list(summary.keys())
    fields = sorted({f for a in archs for f in summary[a]["overall"]})
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(fields)), 4))
    x = np.arange(len(fields))
    width = 0.8 / max(1, len(archs))
    for i, a in enumerate(archs):
        vals = []
        for f in fields:
            entry = summary[a]["overall"].get(f)
            vals.append(entry["value"] if entry else np.nan)
        ax.bar(x + i * width, vals, width, label=a)
    ax.set_xticks(x + width * (len(archs) - 1) / 2)
    ax.set_xticklabels(fields, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("probe R² / AUC")
    ax.set_title("Phase 2 — program-state probe quality")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def plot_stratified(summary: dict, out_path: Path, strat_key: str) -> None:
    import matplotlib.pyplot as plt
    archs = list(summary.keys())
    any_strat = next((summary[a]["stratified"].get(strat_key, {}) for a in archs), {})
    bucket_keys = sorted(any_strat.keys())
    if not bucket_keys:
        return
    fields = sorted({f for b in bucket_keys for f in any_strat.get(b, {})})
    fig, axes = plt.subplots(1, len(fields), figsize=(4 * max(1, len(fields)), 4),
                              squeeze=False)
    for j, f in enumerate(fields):
        ax = axes[0, j]
        for a in archs:
            vals = []
            for b in bucket_keys:
                entry = summary[a]["stratified"].get(strat_key, {}).get(b, {}).get(f)
                vals.append(entry["value"] if entry else np.nan)
            ax.plot(bucket_keys, vals, marker="o", label=a)
        ax.set_title(f"{f} (stratified by {strat_key})")
        ax.set_ylabel("R²")
        ax.set_xticks(range(len(bucket_keys)))
        ax.set_xticklabels(bucket_keys, rotation=30, ha="right", fontsize=8)
        if j == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=HERE / "config.yaml")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(args.device or cfg.get("device", "auto"))
    seed = int(cfg.get("seed", 42))
    ph2 = cfg["phase2_state"]

    cache_root = HERE / cfg.get("cache_root", "cache")
    checkpoint_root = HERE / cfg.get("checkpoint_root", "checkpoints")
    results_root = HERE / cfg.get("output_root", "results")
    plot_root = HERE / cfg.get("plot_root", "plots")

    subject_cfg = SubjectModelConfig.from_dict(cfg["subject_model"])
    layers = subject_cfg.required_layers()
    _, sources, acts_by_layer, _ = load_cache(cache_root, layers)
    split = torch.load(cache_root / "split.pt")
    eval_idx = split["eval_idx"]
    # use eval slice for probing (the train split was used to train the SAEs)
    n_total = ph2["n_train_windows"] + ph2["n_eval_windows"]
    # ensure we have enough chunks; if not, use what we have
    needed_chunks = max(1, n_total // split.get("_tokens_per_chunk", 128))
    eval_idx = eval_idx[: max(needed_chunks, 16)]
    acts_anchor = acts_by_layer[subject_cfg.anchor_layer][eval_idx].float()
    acts_mlc = {L: acts_by_layer[L][eval_idx].float() for L in subject_cfg.mlc_layers}
    sources_eval = [sources[i] for i in eval_idx.tolist()]
    labels_nt = labels_for_sources(sources_eval)

    stratify_buckets = {
        "bracket_depth": cfg["coarse_eval"]["category_buckets"]["bracket_depth"],
    }
    summary: dict = {}
    for arch in cfg["architectures"]:
        name = arch["name"]
        if args.only and name != args.only:
            continue
        ckpt_path = checkpoint_root / f"{name}.pt"
        if not ckpt_path.exists():
            print(f"[phase2] skip {name}: missing checkpoint")
            continue
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model, family = build_model_from_checkpoint(ckpt, subject_cfg.d_model)
        if family == "topk":
            latents, _, c_idx, t_idx = encode_sae_per_token(model, acts_anchor, device)
        elif family == "txc":
            latents, _, c_idx, t_idx = encode_txc_per_window(model, acts_anchor, device)
        elif family == "mlxc":
            latents, _, c_idx, t_idx = encode_mlc_per_token(
                model, acts_mlc, subject_cfg.mlc_layers, device)
        else:
            raise ValueError(family)
        lbl = gather_labels(labels_nt, c_idx, t_idx)
        result = probe_all_fields(
            latents, lbl,
            n_train_windows=ph2["n_train_windows"],
            ridge=ph2["ridge_lambda"],
            stratify_buckets=stratify_buckets,
            seed=seed,
        )
        summary[name] = result
        print(f"[phase2] {name}: overall={result['overall']}")
        with (results_root / f"phase2_probes_{name}.json").open("w") as f:
            json.dump(result, f, indent=2)

    results_root.mkdir(parents=True, exist_ok=True)
    with (results_root / "phase2_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    plot_overall(summary, plot_root / "phase2_r2_by_arch.png")
    plot_stratified(summary, plot_root / "phase2_stratified_bracket_depth.png",
                    strat_key="bracket_depth")
    print("[phase2] done")


if __name__ == "__main__":
    main()
