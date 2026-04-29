"""Phase 7 T-sweep — txcdr_t<T> + phase57_partB_h8_bare_multidistance_t<T>
2-seed mean ± σ across {1, 42}.

Companion to `build_leaderboard_2seed.py`. Reads
`results/probing_results.jsonl` and emits two T-sweep plots
(barebones TXCDR + hill-climbed H8 multidistance) at k_feat ∈ {5, 20}.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.build_tsweep_2seed
"""
from __future__ import annotations

import json
import os
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

os.environ.setdefault("TQDM_DISABLE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.phase7_unification._paths import OUT_DIR, PLOTS_DIR
from experiments.phase7_unification.task_sets import HEADLINE as PAPER


PROBING_PATH = OUT_DIR / "probing_results.jsonl"
SEEDS = (1, 42)
S_FILTER = 32

# Two T-sweep families
BAREBONES_RE = re.compile(r"^txcdr_t(\d+)$")
HILLCLIMB_RE = re.compile(r"^phase57_partB_h8_bare_multidistance_t(\d+)$")


def save_figure(fig, path: str, dpi: int = 150, thumb_max_width: int = 288, thumb_dpi: int = 48):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    w_in, _ = fig.get_size_inches()
    thumb_dpi_eff = min(thumb_dpi, int(thumb_max_width / max(w_in, 0.1)))
    fig.savefig(p.with_suffix(".thumb.png"), dpi=thumb_dpi_eff, bbox_inches="tight")


DEFAULT_SUBJECT_MODEL = "google/gemma-2-2b"


def _row_subject(r: dict) -> str:
    return r.get("subject_model") or DEFAULT_SUBJECT_MODEL


def load_tsweep_data(task_set=PAPER,
                     subject_model: str = DEFAULT_SUBJECT_MODEL):
    """Return dict[(family, T, k_feat, seed)] -> list of test_auc_flip across tasks.

    Filters to `task_set` (default PAPER) and `subject_model`
    (default BASE — google/gemma-2-2b). Backwards-compat: rows without
    a subject_model field are treated as BASE.
    """
    out = defaultdict(list)
    with PROBING_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line: continue
            r = json.loads(line)
            if r.get("S") != S_FILTER: continue
            if r.get("seed") not in SEEDS: continue
            if r.get("k_feat") not in (5, 20): continue
            if r.get("task_name") not in task_set: continue
            if "skipped" in r: continue
            if _row_subject(r) != subject_model: continue
            arch = r.get("arch_id") or ""
            m_b = BAREBONES_RE.match(arch)
            m_h = HILLCLIMB_RE.match(arch)
            if m_b:
                family, T = "barebones", int(m_b.group(1))
            elif m_h:
                family, T = "hillclimb", int(m_h.group(1))
            else:
                continue
            out[(family, T, r["k_feat"], r["seed"])].append(r.get("test_auc_flip", r["test_auc"]))
    return out


def summarise(d):
    """Returns {(family, T, k_feat) -> {"mean": x, "sd_seeds": y, "n_seeds": n, "tasks": [n_per_seed]}}"""
    by_cell = defaultdict(dict)
    for (family, T, kf, seed), aucs in d.items():
        by_cell[(family, T, kf)][seed] = aucs
    out = {}
    for (family, T, kf), seedmap in by_cell.items():
        per_seed_means = []
        n_per_seed = []
        for seed, aucs in seedmap.items():
            per_seed_means.append(float(np.mean(aucs)))
            n_per_seed.append(len(aucs))
        if not per_seed_means:
            continue
        out[(family, T, kf)] = {
            "mean": float(np.mean(per_seed_means)),
            "sd_seeds": float(np.std(per_seed_means, ddof=1)) if len(per_seed_means) > 1 else 0.0,
            "n_seeds": len(per_seed_means),
            "tasks_per_seed": tuple(n_per_seed),
        }
    return out


def make_plot(summ, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    for ax, kf in zip(axes, (5, 20)):
        for family, color, label in [
            ("barebones", "#4472c4", "barebones TXCDR"),
            ("hillclimb", "#c00000", "H8 multidistance (hill-climb)"),
        ]:
            cells = sorted([(T, summ[(family, T, kf)])
                            for T in range(1, 50)
                            if (family, T, kf) in summ])
            if not cells:
                continue
            Ts = [T for T, _ in cells]
            ms = [c["mean"] for _, c in cells]
            sds = [c["sd_seeds"] for _, c in cells]
            ns = [c["n_seeds"] for _, c in cells]
            ax.errorbar(Ts, ms, yerr=sds, marker="o", color=color, label=label,
                        capsize=3, alpha=0.85)
            # Annotate cells with n_seeds=1 (no error bars meaningful)
            for T, m, n in zip(Ts, ms, ns):
                if n == 1:
                    ax.plot(T, m, marker="x", color=color, mew=2, markersize=8, alpha=0.6)
        ax.set_xlabel("T (window size)")
        ax.set_ylabel(f"mean test_auc_flip (k_feat={kf})")
        ax.set_title(f"k_feat = {kf}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower left", fontsize=9)
        ax.set_xticks([3, 5, 8, 10, 12, 16, 20, 24])

    fig.suptitle("Phase 7 T-sweep — 2-seed mean ± σ_seeds across {1, 42}, "
                 "PAPER task set",
                 fontsize=11, weight="bold")
    out_path = out_dir / "phase7_tsweep_2seed.png"
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"\nWrote {out_path}")


def print_table(summ):
    print("=" * 95)
    print("Phase 7 T-sweep — 2-seed mean ± σ across {1, 42}")
    print("=" * 95)
    for kf in (5, 20):
        print()
        print(f"k_feat = {kf}")
        print(f"  {'family':14s}  {'T':>3s}  {'mean':>8s}  {'σ_seeds':>9s}  {'n_seeds':>9s}  {'tasks':>10s}")
        rows = sorted([(family, T, c) for (family, T, k), c in summ.items() if k == kf])
        for family, T, c in rows:
            print(f"  {family:14s}  {T:>3d}  {c['mean']:>8.4f}  {c['sd_seeds']:>9.4f}  "
                  f"{c['n_seeds']:>9d}  {str(c['tasks_per_seed']):>10s}")


def main():
    d = load_tsweep_data()
    summ = summarise(d)
    print_table(summ)
    make_plot(summ, PLOTS_DIR)


if __name__ == "__main__":
    main()
