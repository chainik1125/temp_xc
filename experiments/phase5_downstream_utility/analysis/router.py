"""TXC/MLC task router — Experiment (iv).

Post-hoc analysis of per-task AUCs from agentic_txc_02 + agentic_mlc_08.
Produces three numbers per aggregation:

  1. Best individual : max(mean_auc(agentic_txc_02), mean_auc(agentic_mlc_08))
  2. Oracle router   : mean over tasks of max(auc_txc[task], auc_mlc[task])
  3. Learned router  : k-fold CV LogisticRegression predicting winner-arch
                       from task-metadata features; effective AUC = mean of
                       (per-held-out-task auc of the predicted arch).

If (3) > (1), the "complementary archs" story concretizes.

Also reports per-source win breakdown + writes results to
`results/router_results.json` and `results/plots/router_*.png`.

Usage:
    .venv/bin/python experiments/phase5_downstream_utility/analysis/router.py
"""

from __future__ import annotations

import json
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning,
                        module="sklearn.linear_model._logistic")

REPO = Path("/workspace/temp_xc")
sys.path.insert(0, str(REPO))

from src.plotting.save_figure import save_figure  # noqa: E402

JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
RESULTS = REPO / "experiments/phase5_downstream_utility/results"
PLOTS = RESULTS / "plots"

ARCH_A = "agentic_txc_02"
ARCH_B = "agentic_mlc_08"


def load_last_write(agg: str, seed: int = 42, k_feat: int = 5) -> dict[str, dict[str, float]]:
    """{arch: {task: test_auc}} using last-write-wins per (arch, task)."""
    by_key: dict[tuple[str, str], float] = {}
    with JSONL.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("aggregation") != agg or r.get("k_feat") != k_feat:
                continue
            rid = r["run_id"]
            if not rid.endswith(f"__seed{seed}"):
                continue
            auc = r.get("test_auc")
            if auc is None:
                continue
            arch = rid.rsplit(f"__seed{seed}", 1)[0]
            by_key[(arch, r["task_name"])] = float(auc)
    per_arch: dict[str, dict[str, float]] = defaultdict(dict)
    for (arch, task), auc in by_key.items():
        per_arch[arch][task] = auc
    return per_arch


def task_source(task: str) -> str:
    """Dataset source prefix (one of 7 sources)."""
    if task.startswith("ag_news"):
        return "ag_news"
    if task.startswith("amazon_reviews"):
        return "amazon_reviews"
    if task.startswith("bias_in_bios"):
        return "bias_in_bios"
    if task.startswith("europarl"):
        return "europarl"
    if task.startswith("github_code"):
        return "github_code"
    if task.startswith("winogrande"):
        return "winogrande"
    if task.startswith("wsc"):
        return "wsc"
    raise ValueError(f"unknown source for task {task}")


def bias_in_bios_set(task: str) -> int:
    if "set1" in task:
        return 1
    if "set2" in task:
        return 2
    if "set3" in task:
        return 3
    return 0


def load_task_meta(task: str) -> dict:
    p = (REPO / "experiments/phase5_downstream_utility/results/probe_cache"
         / task / "meta.json")
    return json.loads(p.read_text())


def task_features(task: str) -> np.ndarray:
    """Dataset-source one-hot + bias_in_bios set + class-balance proxy.

    Feature layout (10):
      [0:7]  one-hot source (ag_news, amazon_reviews, bias_in_bios,
             europarl, github_code, winogrande, wsc)
      [7]    bias_in_bios set index (0..3; 0 for non-bios tasks)
      [8]    |train_pos_frac - 0.5|  (class imbalance, ∈ [0, 0.5])
      [9]    train_pos_frac           (raw polarity)
    """
    sources = ["ag_news", "amazon_reviews", "bias_in_bios",
               "europarl", "github_code", "winogrande", "wsc"]
    src = task_source(task)
    feat = np.zeros(len(sources) + 3, dtype=np.float32)
    feat[sources.index(src)] = 1.0
    feat[len(sources)] = float(bias_in_bios_set(task))
    meta = load_task_meta(task)
    pos = float(meta.get("train_pos_frac", 0.5))
    feat[len(sources) + 1] = abs(pos - 0.5)
    feat[len(sources) + 2] = pos
    return feat


def run(agg: str) -> dict:
    per_arch = load_last_write(agg)
    if ARCH_A not in per_arch or ARCH_B not in per_arch:
        raise RuntimeError(f"missing {ARCH_A} or {ARCH_B} at {agg}")
    auc_a = per_arch[ARCH_A]
    auc_b = per_arch[ARCH_B]
    tasks = sorted(set(auc_a.keys()) & set(auc_b.keys()))
    assert len(tasks) == 36, f"expected 36 tasks, got {len(tasks)}"

    # Per-task arrays
    x = np.stack([task_features(t) for t in tasks])
    a = np.array([auc_a[t] for t in tasks])
    b = np.array([auc_b[t] for t in tasks])

    # 1. Best individual
    mean_a = float(a.mean())
    mean_b = float(b.mean())
    best_individual = max(mean_a, mean_b)
    best_arch = ARCH_A if mean_a > mean_b else ARCH_B

    # 2. Oracle router
    oracle = float(np.maximum(a, b).mean())

    # Per-task winner vector (1 if A wins, 0 if B wins; tie → whoever's
    # higher mean, for deterministic target)
    winner = (a > b).astype(int)  # 1 = TXC, 0 = MLC
    wins_a = int(winner.sum())
    wins_b = int((1 - winner).sum())

    # 3. Learned router: leave-one-out CV
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut, KFold

    n = len(tasks)
    # Use LeaveOneOut since n=36 and source-stratified 6-fold has
    # only 1 bias_in_bios class per fold which breaks fit on binary targets.
    loo = LeaveOneOut()
    loo_preds = np.zeros(n, dtype=int)
    loo_probs = np.zeros(n)
    for tr, te in loo.split(x):
        # Skip folds where training set is degenerate (all one class)
        if len(set(winner[tr])) < 2:
            # fall back to majority-class prediction
            loo_preds[te[0]] = int(winner[tr].mean() > 0.5)
            loo_probs[te[0]] = float(winner[tr].mean())
            continue
        clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000)
        clf.fit(x[tr], winner[tr])
        loo_preds[te[0]] = clf.predict(x[te])[0]
        loo_probs[te[0]] = clf.predict_proba(x[te])[0, 1]

    # Effective AUC: for each task, use the AUC of the predicted arch
    effective = np.where(loo_preds == 1, a, b)
    learned_router = float(effective.mean())
    cv_acc = float((loo_preds == winner).mean())

    # 6-fold CV as sanity check (source-stratified is fragile at n=36)
    kf = KFold(n_splits=6, shuffle=True, random_state=0)
    kf_preds = np.zeros(n, dtype=int)
    for tr, te in kf.split(x):
        if len(set(winner[tr])) < 2:
            kf_preds[te] = int(winner[tr].mean() > 0.5)
            continue
        clf = LogisticRegression(C=1.0, penalty="l2", max_iter=1000)
        clf.fit(x[tr], winner[tr])
        kf_preds[te] = clf.predict(x[te])
    kf_effective = np.where(kf_preds == 1, a, b)
    kf_learned_router = float(kf_effective.mean())
    kf_acc = float((kf_preds == winner).mean())

    # Per-source win breakdown
    by_source: dict[str, dict] = defaultdict(lambda: {"a": 0, "b": 0, "n": 0})
    for i, t in enumerate(tasks):
        s = task_source(t)
        by_source[s]["n"] += 1
        if winner[i] == 1:
            by_source[s]["a"] += 1
        else:
            by_source[s]["b"] += 1
    by_source = dict(by_source)

    return {
        "aggregation": agg,
        "n_tasks": n,
        "best_individual": best_individual,
        "best_arch": best_arch,
        "mean_txc": mean_a,
        "mean_mlc": mean_b,
        "oracle_router": oracle,
        "learned_router_loo": learned_router,
        "loo_cv_accuracy": cv_acc,
        "learned_router_6fold": kf_learned_router,
        "kfold6_cv_accuracy": kf_acc,
        "txc_wins": wins_a,
        "mlc_wins": wins_b,
        "by_source": by_source,
        "tasks": tasks,
        "auc_txc_per_task": {t: float(auc_a[t]) for t in tasks},
        "auc_mlc_per_task": {t: float(auc_b[t]) for t in tasks},
        "winner_per_task": {t: ARCH_A if winner[i] == 1 else ARCH_B
                             for i, t in enumerate(tasks)},
        "loo_pred_per_task": {t: ARCH_A if loo_preds[i] == 1 else ARCH_B
                               for i, t in enumerate(tasks)},
    }


def plot_winners(result: dict, path: Path) -> None:
    """Horizontal bar: per-task AUC gap (TXC - MLC), colored by winner."""
    tasks = result["tasks"]
    a = np.array([result["auc_txc_per_task"][t] for t in tasks])
    b = np.array([result["auc_mlc_per_task"][t] for t in tasks])
    diff = a - b
    order = np.argsort(diff)
    tasks_s = [tasks[i] for i in order]
    diff_s = diff[order]

    fig, ax = plt.subplots(figsize=(10, 0.25 * len(tasks) + 1.5))
    colors = ["tab:green" if d > 0 else "tab:blue" for d in diff_s]
    ax.barh(range(len(tasks_s)), diff_s, color=colors)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_yticks(range(len(tasks_s)))
    ax.set_yticklabels(tasks_s, fontsize=7)
    ax.set_xlabel("TXC auc − MLC auc  (positive → TXC wins)")
    ax.set_title(f"Per-task winner — {result['aggregation']}")
    fig.tight_layout()
    save_figure(fig, str(path))
    plt.close(fig)


def plot_summary(results: dict, path: Path) -> None:
    """Grouped bar: best_individual / oracle / learned at both aggregations."""
    aggs = list(results.keys())
    labels = ["best individual", "oracle", "learned (LOO)", "learned (6-fold)"]
    keys = ["best_individual", "oracle_router",
            "learned_router_loo", "learned_router_6fold"]
    vals = np.array([[results[a][k] for k in keys] for a in aggs])

    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.2
    x = np.arange(len(labels))
    for i, agg in enumerate(aggs):
        ax.bar(x + i * width, vals[i], width=width, label=agg)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.75, max(1.0, vals.max() + 0.01))
    ax.set_ylabel("mean AUC")
    ax.set_title("TXC/MLC router — best individual vs oracle vs learned")
    ax.legend()
    for i, agg in enumerate(aggs):
        for j, v in enumerate(vals[i]):
            ax.text(j + i * width, v + 0.002, f"{v:.4f}",
                    ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    save_figure(fig, str(path))
    plt.close(fig)


def main() -> None:
    PLOTS.mkdir(parents=True, exist_ok=True)
    results = {}
    for agg in ["last_position", "mean_pool"]:
        r = run(agg)
        results[agg] = r
        print(f"\n=== {agg} ===")
        print(f"  mean(TXC)          = {r['mean_txc']:.4f}")
        print(f"  mean(MLC)          = {r['mean_mlc']:.4f}")
        print(f"  best individual    = {r['best_individual']:.4f}  ({r['best_arch']})")
        print(f"  oracle router      = {r['oracle_router']:.4f}  (ceiling)")
        print(f"  learned (LOO)      = {r['learned_router_loo']:.4f}  (acc {r['loo_cv_accuracy']:.2%})")
        print(f"  learned (6-fold)   = {r['learned_router_6fold']:.4f}  (acc {r['kfold6_cv_accuracy']:.2%})")
        print(f"  wins: TXC={r['txc_wins']:2d}  MLC={r['mlc_wins']:2d}")
        print(f"  per-source wins:")
        for s, d in r["by_source"].items():
            print(f"    {s:<15s}  n={d['n']:2d}  TXC={d['a']}  MLC={d['b']}")

        # Plot per-task winner bar
        plot_winners(r, PLOTS / f"router_per_task_{agg}.png")

    plot_summary(results, PLOTS / "router_summary.png")

    out = RESULTS / "router_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
