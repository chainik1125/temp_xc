"""Render Figures 2-6 of the paper from the leaderboard.

Run via:

    python run.py render-figures

OR directly:

    python -m experiments.render_paper_figures

Reads ``purified/results/leaderboard.jsonl`` and produces PDFs in
``purified/docs/figs/``. Each render function targets one paper
figure and is a thin compose of pandas + matplotlib.

PORT STATUS: skeleton. Concrete figure renderers awaiting port from
``origin/final:purified/experiments/c*/analysis.py`` (which had per-
component figure render code, e.g. ``c3_probing/analysis.py``).

For each figure, the legacy renderer:
- queries the leaderboard,
- filters to canonical train_keys (per § 12 + § 15 of decisions.md),
- aggregates by (arch, k_feat) or (arch, seed, organism) as appropriate,
- writes the figure to docs/figs/<name>.{png,pdf}.

The shape contract for each is::

    def render_<name>() -> Path:
        ...  # writes to docs/figs/<name>.pdf
        return path_to_pdf
"""

from __future__ import annotations

from pathlib import Path


def render_synthetic_overview() -> Path:
    """Fig 2 — § 4 synthetic results overview.

    Two-panel figure: gAUC + eAUC vs k_pos for each arch, on the
    coupling bench (toy_coupled_K10_M20_d256) and the denoising bench
    (toy_markov_n20_d40_noisy). Aggregates across seeds (mean ± std).
    """
    import json
    from collections import defaultdict
    import matplotlib.pyplot as plt
    import numpy as np

    from temp_bench.core.cache import iter_leaderboard

    # Bench → list of arch results indexed by k_pos
    by_bench: dict[str, dict[str, dict[int, list[dict]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    # Archs removed from the active registry — historical leaderboard rows
    # remain (audit trail) but are excluded from figures.
    deprecated_archs = {"txc_pro", "tfa", "tfa_pos"}
    # The synthetic d_sae was changed 40 → 20 on 2026-06-01. Older rows
    # were the over-dictionary regime; new rows are the scarce-dictionary
    # regime that the headline figure now describes.
    d_sae_cutover_ts = "2026-05-31T22:30:00Z"

    n_rows = 0
    for row in iter_leaderboard():
        if row.experiment != "synthetic":
            continue
        if row.evaluator_protocol_version != "1.2.0":      # matches SyntheticRecovery
            continue                                       # filter older-protocol cells
        if row.eval_cfg.get("smoke", False):
            continue
        if row.arch in deprecated_archs:
            continue
        if getattr(row, "ts", "") < d_sae_cutover_ts:
            continue                                       # filter historical d_sae=40 rows
        k_pos = row.eval_cfg.get("k_pos")
        if k_pos is None:
            # Some rows may have k_pos in training_cfg.arch_hparams_override instead
            ovr = row.training_cfg.arch_hparams_override or {}
            k_pos = ovr.get("k_pos")
        if k_pos is None:
            continue
        by_bench[row.datasource][row.arch][int(k_pos)].append(row.metrics)
        n_rows += 1

    print(f"[render_synthetic_overview] {n_rows} canonical rows aggregated")
    if n_rows == 0:
        raise RuntimeError(
            "No § 4 rows found. Run `python run.py reproduce synthetic` first."
        )

    # Plot
    benches = sorted(by_bench.keys())
    fig, axes = plt.subplots(2, len(benches), figsize=(6 * len(benches), 8),
                             squeeze=False, sharex=False)

    arch_color = {
        "txc_base":    "#1f77b4",
        "topk_sae":    "#d62728",
        "stacked_sae": "#9467bd",
        "tsae":        "#ff7f0e",
        "mlc":         "#7f7f7f",
        "sae_arditi":  "#8c564b",
    }
    arch_marker = {
        "txc_base": "o", "topk_sae": "^",
        "stacked_sae": "v", "tsae": "D", "mlc": "*",
    }

    for col, bench in enumerate(benches):
        bench_label = bench.replace("toy_", "").replace("_", " ")
        archs_here = sorted(by_bench[bench].keys())
        for metric_row, metric_name in enumerate(["gauc", "eauc"]):
            ax = axes[metric_row, col]
            for arch in archs_here:
                kdata = by_bench[bench][arch]
                ks = sorted(kdata.keys())
                if not ks:
                    continue
                means, stds = [], []
                for k in ks:
                    vals = [m.get(metric_name, float("nan")) for m in kdata[k]]
                    vals = [v for v in vals if not np.isnan(v)]
                    if vals:
                        means.append(np.mean(vals))
                        stds.append(np.std(vals) if len(vals) > 1 else 0.0)
                    else:
                        means.append(float("nan")); stds.append(0)
                means = np.array(means); stds = np.array(stds)
                ax.errorbar(
                    ks, means, yerr=stds,
                    color=arch_color.get(arch, "#444"),
                    marker=arch_marker.get(arch, "o"),
                    label=arch, linewidth=2, capsize=3,
                )
            ax.set_xscale("log")
            ax.set_xticks([1, 2, 5, 10, 20])
            ax.set_xticklabels(["1", "2", "5", "10", "20"])
            ax.set_ylim(0, 1.05)
            ax.set_xlabel("k_pos (sparsity)")
            ax.set_ylabel(metric_name.upper())
            ax.set_title(
                f"{bench_label} — "
                f"{'global' if metric_name == 'gauc' else 'local'} recovery"
            )
            ax.grid(True, alpha=0.3)
            if metric_row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=9, ncol=2)

    fig.suptitle(
        "Fig 2 — § 4 Synthetic feature recovery: "
        "TXC dictionaries align with global features; "
        "per-token SAEs align with local features",
        fontsize=12,
    )
    fig.tight_layout()

    out_dir = Path("docs/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig2_synthetic_overview_v2.pdf"
    png = out_dir / "fig2_synthetic_overview_v2.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_synthetic_overview] wrote {pdf} + {png}")
    return pdf


def render_sparse_probing() -> Path:
    """Fig 3 — § 5.1 probing AUC vs k_feat."""
    raise NotImplementedError(
        "Port from origin/final:purified/experiments/c3_probing/analysis.py "
        "(auc_by_k log-x plot with TXC-base T-sweep)."
    )


def render_backtracking() -> Path:
    """Fig 4 — § 5.2 detection + inducement."""
    raise NotImplementedError(
        "Port from origin/final:purified/experiments/c7_backtracking/"
        "analysis.py."
    )


def render_em() -> Path:
    """Fig 5 — § 5.3 emergent misalignment coherence × alignment Pareto."""
    raise NotImplementedError(
        "Port from origin/final:purified/experiments/c6_em/analysis.py "
        "(c6_pareto_*_clustered.png Pareto scatter)."
    )


def render_rlhf() -> Path:
    """Fig 6 — § 5.4 HH-RLHF preference decomposition."""
    raise NotImplementedError(
        "Port from origin/final:purified/experiments/c5_steering/analysis.py."
    )


def render_all() -> list[Path]:
    """Render every paper figure. Errors per-figure don't abort the rest."""
    results: list[Path] = []
    for fn in (
        render_synthetic_overview,
        render_sparse_probing,
        render_backtracking,
        render_em,
        render_rlhf,
    ):
        try:
            p = fn()
            print(f"[render] {fn.__name__}: {p}")
            results.append(p)
        except NotImplementedError as e:
            print(f"[render] {fn.__name__}: PENDING PORT — {e}")
        except Exception as e:
            print(f"[render] {fn.__name__}: FAILED — {type(e).__name__}: {e}")
    return results


if __name__ == "__main__":
    render_all()
