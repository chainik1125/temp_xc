"""Render Figures 2-6 of the paper from the leaderboard.

Run via:

    python run.py render-figures

OR directly:

    python -m experiments.render_paper_figures

Reads ``purified/results/leaderboard.jsonl`` and produces PDFs in
``purified/docs/aniket/figs/``. Each render function targets one paper
figure and is a thin compose of pandas + matplotlib.

PORT STATUS: skeleton. Concrete figure renderers awaiting port from
``origin/final:purified/experiments/c*/analysis.py`` (which had per-
component figure render code, e.g. ``c3_probing/analysis.py``).

For each figure, the legacy renderer:
- queries the leaderboard,
- filters to canonical train_keys (per § 12 + § 15 of decisions.md),
- aggregates by (arch, k_feat) or (arch, seed, organism) as appropriate,
- writes the figure to docs/aniket/figs/<name>.{png,pdf}.

The shape contract for each is::

    def render_<name>() -> Path:
        ...  # writes to docs/aniket/figs/<name>.pdf
        return path_to_pdf
"""

from __future__ import annotations

from pathlib import Path


def render_synthetic_overview() -> Path:
    """Fig 2 — § 4 synthetic results overview."""
    raise NotImplementedError(
        "Port from origin/final:purified/experiments/c2_synthetic_coupled/"
        "plot_headline.py + plot_setup_a.py. Produces 2-panel figure "
        "(denoising + coupling)."
    )


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
