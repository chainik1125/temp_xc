"""§3 — Sentence-level backtracking detection via sparse linear probes.

Reads the mined npz files (which already contain `pos_act` / `neg_act`
matrices = top-32 feature activations per labeled sentence), builds a
common (sentence_key → feature_matrix) dataset across all 5 (+1 appendix)
architectures, then fits sparse logistic-regression probes for
|S| ∈ {1, 2, 4, 8, 16, 32}.

5-fold CV is grouped by `question_id` to prevent within-question leakage.
Reports AUC and F1 per (arch × |S| × fold). Aggregates and saves:
  - probe_results.parquet (long-form rows)
  - detection_headline.png (AUC vs |S| per arch)
  - wilcoxon_table.csv (TXC vs each baseline at headline |S|, paired
    Wilcoxon across folds, Holm-Bonferroni corrected)

Out of scope for this script (deferred to follow-up):
  - Raw-residual baseline. Requires re-capturing residuals at sentence
    tokens (feasible but adds GPU + 10 min). Note as appendix work.
  - Aggregation modes (last/mean/max/full_window) for non-temporal arches.
    The mined `pos_act` for SAE/TSAE/TFA is ALREADY arch-specific
    (window-mean for TXC family; pred+novel mean for TSAE/TFA; window-mean
    for MLC). We use these as-is — that's the apples-to-apples comparison.
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import GroupKFold

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.detection.probe")


# Same arch list used in the headline plot (plus TXC-H8 for the appendix).
ARCH_FILES = [
    ("TXC",        "txc__resid_L10__k16__s42.npz"),
    ("TXC-H8",     "txc_h8__resid_L10__k16__s42.npz"),
    ("SAE",        "topk_sae__ln1_L10__k64__s42.npz"),
    ("TSAE-paper", "tsae__resid_L10__k32__s42.npz"),
    ("TFA",        "tfa__resid_L10__k32__s42.npz"),
    ("MLC",        "mlc__resid_L10__k32__s42.npz"),
]

# From plot/headline_steering.py
HEADLINE_LABELS = {"TXC", "SAE", "TSAE-paper", "TFA", "MLC"}
ARCH_PALETTE = {
    "TXC":        "#1f4e79",
    "SAE":        "#e07b00",
    "TSAE-paper": "#c83e80",
    "TFA":        "#7f3f98",
    "MLC":        "#2ca02c",
    "TXC-H8":     "#5b9bd5",
}

S_VALUES = [1, 2, 4, 8, 16, 32]
N_FOLDS = 5
HEADLINE_S = 8


def load_arch_data(features_dir: Path, npz_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y, qids, top_feat_ids) for one architecture.

    X: (n_sent, K=32) — feature activations for the top-K features
       (K = top_k_features from mining config).
    y: (n_sent,) — 0/1 backtracking label.
    qids: (n_sent,) — question_id per sentence (for GroupKFold).
    top_feat_ids: (K,) — feature ids from the architecture.
    """
    z = np.load(features_dir / npz_name, allow_pickle=True)
    pos_keys = z["sentence_keys_pos"]
    neg_keys = z["sentence_keys_neg"]
    pos_act = z["pos_act"]
    neg_act = z["neg_act"]
    top = z["top_features"]
    keys = np.concatenate([pos_keys, neg_keys])
    X = np.concatenate([pos_act, neg_act], axis=0)
    y = np.concatenate([np.ones(len(pos_keys), dtype=int),
                        np.zeros(len(neg_keys), dtype=int)])
    # qid is the part before the first '|' in the sentence key.
    qids = np.array([k.split("|", 1)[0] for k in keys])
    return X, y, qids, top


def align_to_common(per_arch: dict[str, dict]) -> dict[str, dict]:
    """Restrict each arch's matrix to the intersection of sentence keys."""
    common = None
    for arch, d in per_arch.items():
        keys = set(zip(d["qids"].tolist(), d["sent_idx"].tolist()))  # type: ignore
        common = keys if common is None else common & keys
    log.info("[align] common sentences: %d", len(common))
    out = {}
    for arch, d in per_arch.items():
        keys = list(zip(d["qids"].tolist(), d["sent_idx"].tolist()))
        idx = np.array([i for i, k in enumerate(keys) if k in common])
        out[arch] = {
            "X": d["X"][idx],
            "y": d["y"][idx],
            "qids": d["qids"][idx],
        }
    return out


def fit_probes(X: np.ndarray, y: np.ndarray, qids: np.ndarray) -> list[dict]:
    """For each |S| in S_VALUES, fit logreg with top-S features (selected per fold
    via mean-difference between classes on the training fold) and report AUC + F1
    via 5-fold CV grouped by qid."""
    rows = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for S in S_VALUES:
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups=qids)):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            # Per-fold feature selection: top-S by |mu_pos - mu_neg|
            mu_pos = X_tr[y_tr == 1].mean(axis=0) if (y_tr == 1).any() else np.zeros(X.shape[1])
            mu_neg = X_tr[y_tr == 0].mean(axis=0) if (y_tr == 0).any() else np.zeros(X.shape[1])
            score = np.abs(mu_pos - mu_neg)
            top = np.argsort(-score)[:S]
            clf = LogisticRegression(max_iter=200, solver="liblinear")
            try:
                clf.fit(X_tr[:, top], y_tr)
                proba = clf.predict_proba(X_te[:, top])[:, 1]
                pred = clf.predict(X_te[:, top])
                auc = float(roc_auc_score(y_te, proba)) if len(set(y_te)) > 1 else float("nan")
                # PR-AUC (average precision) is the right metric for our
                # 12%-positive class. F1 at threshold 0.5 is included for
                # back-compat but reads as catastrophe due to class imbalance.
                ap = float(average_precision_score(y_te, proba)) if len(set(y_te)) > 1 else float("nan")
                f1 = float(f1_score(y_te, pred, zero_division=0))
            except Exception as e:
                log.warning("fit failed S=%d fold=%d: %s", S, fold, e)
                auc, ap, f1 = float("nan"), float("nan"), float("nan")
            rows.append({"S": S, "fold": fold, "auc": auc, "pr_auc": ap, "f1": f1,
                         "n_train": len(train_idx), "n_test": len(test_idx)})
    return rows


def render_headline(df: pd.DataFrame, out_path: Path, label_filter: set | None = None):
    # Three panels now: ROC-AUC + PR-AUC + F1@0.5. PR-AUC is the right
    # primary metric for our 12%-positive class; F1@0.5 retained for
    # back-compat with the earlier draft.
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
    arches = sorted(df["arch"].unique())
    if label_filter:
        arches = [a for a in arches if a in label_filter]
    metric_specs = [
        ("auc", "ROC-AUC", None),
        ("pr_auc", "PR-AUC (avg precision)\n[primary metric — class is ~12% positive]", None),
        ("f1", "F1 @ threshold=0.5\n(thresholded; reads low due to class imbalance)", None),
    ]
    for (metric, ylabel, _), ax in zip(metric_specs, axes):
        for arch in arches:
            sub = df[df["arch"] == arch]
            agg = sub.groupby("S")[metric].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["S"], agg["mean"], yerr=agg["std"],
                        marker="o", linewidth=1.6, markersize=4,
                        label=arch, color=ARCH_PALETTE.get(arch, "#888"),
                        capsize=2)
        ax.set_xscale("log", base=2)
        ax.set_xticks(S_VALUES); ax.set_xticklabels(S_VALUES)
        ax.set_xlabel("|S| (number of features)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Backtracking sentence detection: 5-fold grouped CV (group=question)",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    log.info("[saved] %s", out_path)
    plt.close(fig)


def wilcoxon_vs_baselines(df: pd.DataFrame, ref_arch: str, out_path: Path,
                          metric: str = "auc") -> None:
    from scipy.stats import wilcoxon
    rows = []
    sub = df[df["S"] == HEADLINE_S]
    ref = sub[sub["arch"] == ref_arch].sort_values("fold")[metric].values
    others = sorted({a for a in sub["arch"].unique() if a != ref_arch})
    pvals = []
    for other in others:
        cmp = sub[sub["arch"] == other].sort_values("fold")[metric].values
        if len(ref) == len(cmp) and len(ref) >= 2:
            try:
                stat, p = wilcoxon(ref, cmp, alternative="two-sided",
                                   zero_method="wilcox", correction=False)
                rows.append({"comparison": f"{ref_arch} vs {other}",
                             "metric": metric, "S": HEADLINE_S,
                             "n_folds": len(ref), "wilcoxon_W": float(stat),
                             "p_raw": float(p),
                             f"ref_mean_{metric}": float(np.mean(ref)),
                             f"other_mean_{metric}": float(np.mean(cmp))})
                pvals.append(p)
            except Exception as e:
                log.warning("wilcoxon failed %s vs %s: %s", ref_arch, other, e)
    # Holm-Bonferroni across the family
    if pvals:
        order = np.argsort(pvals)
        m = len(pvals)
        adj = [None] * m
        for rank, i in enumerate(order):
            adj[i] = min(1.0, pvals[i] * (m - rank))
        for r, p_adj in zip(rows, adj):
            r["p_holm"] = float(p_adj)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["comparison", "metric", "S", "n_folds", "wilcoxon_W", "p_raw", "p_holm",
                  f"ref_mean_{metric}", f"other_mean_{metric}"]
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    log.info("[saved] %s", out_path)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--features-dir", type=Path,
                   default=Path("results/ward_backtracking_txc/features"))
    p.add_argument("--out", type=Path,
                   default=Path("results/ward_backtracking_txc/detection"))
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)

    # Load all archs, aligned by sentence_key intersection.
    per_arch_full = {}
    for label, npz_name in ARCH_FILES:
        path = args.features_dir / npz_name
        if not path.exists():
            log.warning("[skip] %s: %s missing", label, path)
            continue
        X, y, qids, top = load_arch_data(args.features_dir, npz_name)
        # Use the sentence key directly (qid|trace_idx|s_idx); we don't need to align by index.
        z = np.load(path, allow_pickle=True)
        keys_full = np.concatenate([z["sentence_keys_pos"], z["sentence_keys_neg"]])
        per_arch_full[label] = {"X": X, "y": y, "qids": qids, "keys": keys_full}
        log.info("[%-12s] X=%s y_pos=%d y_neg=%d", label, X.shape, int(y.sum()), int((1-y).sum()))

    # Align by intersection of sentence keys
    common_keys = set.intersection(*[set(d["keys"].tolist()) for d in per_arch_full.values()])
    log.info("[align] common sentences: %d", len(common_keys))
    per_arch = {}
    for label, d in per_arch_full.items():
        idx = np.array([i for i, k in enumerate(d["keys"]) if k in common_keys])
        per_arch[label] = {
            "X": d["X"][idx],
            "y": d["y"][idx],
            "qids": d["qids"][idx],
        }

    # Fit probes per arch
    all_rows = []
    for arch, d in per_arch.items():
        log.info("[fit] %s", arch)
        rows = fit_probes(d["X"], d["y"], d["qids"])
        for r in rows:
            r["arch"] = arch
            all_rows.append(r)

    df = pd.DataFrame(all_rows)
    pq = args.out / "probe_results.parquet"
    df.to_parquet(pq, compression="snappy")
    log.info("[saved] %s", pq)

    # Headline figure (5 archs)
    render_headline(df, args.out / "detection_headline.png",
                    label_filter=HEADLINE_LABELS)
    # Appendix figure (6 archs)
    render_headline(df, args.out / "detection_appendix.png", label_filter=None)

    # Wilcoxon TXC vs each baseline at |S|=8 — both metrics
    wilcoxon_vs_baselines(df, ref_arch="TXC",
                          out_path=args.out / "wilcoxon_detection_table.csv",
                          metric="auc")
    wilcoxon_vs_baselines(df, ref_arch="TXC",
                          out_path=args.out / "wilcoxon_detection_table_pr_auc.csv",
                          metric="pr_auc")

    # Summary table — mean AUC + PR-AUC + F1 per (arch, |S|)
    summary = df.groupby(["arch", "S"])[["auc", "pr_auc", "f1"]].mean().reset_index()
    summary.to_csv(args.out / "summary_auc_f1.csv", index=False)
    log.info("[saved] %s", args.out / "summary_auc_f1.csv")
    log.info("\n=== summary ===\n%s", summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
