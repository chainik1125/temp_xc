"""Per-component results binding — leaderboard.jsonl → docs/components/cN.md.

This module is the **only** sanctioned pathway from raw cell results to
the prose Results section of a component writeup. Agents do **not**
hand-type numbers into ``cN.md`` between the AUTO-RESULTS markers;
instead, each component owns an ``analysis.py`` that:

1. Queries ``leaderboard.jsonl`` via :func:`query_leaderboard`
2. Computes summary numbers + writes aggregate plots to
   ``experiments/cN_*/plots/``
3. Returns an :class:`AnalysisResult` whose ``markdown`` field is the
   content that goes between ``<!-- BEGIN AUTO-RESULTS -->`` and
   ``<!-- END AUTO-RESULTS -->`` in ``docs/components/cN.md``.

Then :func:`render` (or :func:`render_all`) is invoked from a runner /
post-cell hook / ``make report`` and atomically rewrites the marked
section. Hand-edits between markers are forbidden — see
PROTOCOL.md § 7 *Results live in state*.

Why: hand-typed numbers drift from the leaderboard. Hand-typed numbers
also can't be re-rendered when a seed lands or a metric is fixed. By
sourcing every paper-relevant number from the leaderboard via a deter-
ministic ``analysis.py`` we keep results coherent with state and the
re-render is idempotent.
"""

from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from temp_bench.cache import _read_jsonl, leaderboard_path
from temp_bench.config import purified_root
from temp_bench.schemas import LeaderboardRow

# Public marker syntax. Agents copy these literally into cN.md once.
BEGIN_MARKER = "<!-- BEGIN AUTO-RESULTS -->"
END_MARKER = "<!-- END AUTO-RESULTS -->"

COMPONENTS = ("c1", "c2", "c3", "c4", "c5", "c6", "c7")


@dataclass
class AnalysisResult:
    """Output of a component's ``analysis.py:run_analysis()``.

    Attributes:
        markdown: Content placed between the AUTO-RESULTS markers in
            ``docs/components/cN.md``. Should be self-contained markdown:
            tables, image links, narrative one-liners. Numbers must be
            derived from leaderboard rows, not hardcoded.
        results: Structured numbers dumped to
            ``experiments/cN_*/results.json``. Keys are agent-defined
            (e.g. ``{"txc_pro": {"k20": {"mean_auc": 0.91, "n_seeds": 3}}}``).
            This file is consumed by the paper-rendering pipeline (the
            outline / NeurIPS draft pulls from here, not from cN.md).
        plot_paths: Aggregate plots written by ``analysis.py``, relative
            to ``purified/``. Recorded for the rendering layer; the .md
            references them via relative paths.
    """

    markdown: str
    results: dict[str, Any] = field(default_factory=dict)
    plot_paths: list[Path] = field(default_factory=list)


# ── Leaderboard query ───────────────────────────────────────────────────


def query_leaderboard(
    *,
    component: str | None = None,
    arch: str | None = None,
    seed: int | None = None,
    datasource: str | None = None,
) -> list[LeaderboardRow]:
    """Read leaderboard.jsonl and return rows matching all given filters.

    Each filter is ANDed; passing ``None`` skips the filter. Rows are
    schema-validated as they're loaded; a malformed row aborts (matching
    the append-time guarantee — corrupt state is never silently
    tolerated).
    """
    rows: list[LeaderboardRow] = []
    for raw in _read_jsonl(leaderboard_path()):
        row = LeaderboardRow(**raw)
        if component is not None and row.component != component:
            continue
        if arch is not None and row.arch != arch:
            continue
        if seed is not None and row.seed != seed:
            continue
        if datasource is not None and row.datasource != datasource:
            continue
        rows.append(row)
    return rows


# ── Component → analysis.py resolution ──────────────────────────────────


def _experiment_dir(component: str) -> Path | None:
    """Find ``experiments/<component>_*/`` for a given component.

    Returns ``None`` if no experiment directory exists yet (agent hasn't
    started the component). Returns the path otherwise. Raises if more
    than one matches — the convention is one experiments-dir per cN.
    """
    root = purified_root() / "experiments"
    if not root.exists():
        return None
    candidates = sorted(p for p in root.glob(f"{component}_*") if p.is_dir())
    if len(candidates) == 0:
        return None
    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple experiment dirs match {component}_*: {candidates}. "
            "Convention is one dir per component (e.g. c1_synthetic_topk)."
        )
    return candidates[0]


def _analysis_module_name(exp_dir: Path) -> str:
    return f"experiments.{exp_dir.name}.analysis"


# ── Marker-based atomic .md rewrite ─────────────────────────────────────


_MARKER_RE = re.compile(
    re.escape(BEGIN_MARKER) + r"(.*?)" + re.escape(END_MARKER),
    flags=re.DOTALL,
)


def _component_md(component: str) -> Path:
    return purified_root() / "docs" / "components" / f"{component}.md"


def _replace_auto_results(md_path: Path, new_inner: str) -> None:
    """Rewrite the content between BEGIN/END AUTO-RESULTS markers.

    Idempotent: passing the same ``new_inner`` twice produces the same
    file. Raises if markers are missing — every cN.md is required to
    have them (test_report enforces).
    """
    text = md_path.read_text()
    if BEGIN_MARKER not in text or END_MARKER not in text:
        raise RuntimeError(
            f"{md_path} is missing AUTO-RESULTS markers. Add\n"
            f"  {BEGIN_MARKER}\n  ...\n  {END_MARKER}\n"
            "around the Results section once. Then never hand-edit between "
            "them — temp_bench.report.render() owns that block."
        )
    block = f"{BEGIN_MARKER}\n{new_inner.strip()}\n{END_MARKER}"
    new_text = _MARKER_RE.sub(lambda _m: block, text, count=1)
    if new_text == text:
        return  # nothing changed
    tmp = md_path.with_suffix(md_path.suffix + ".tmp")
    tmp.write_text(new_text)
    tmp.replace(md_path)


# ── Public render API ───────────────────────────────────────────────────


def render(component: str, *, missing_ok: bool = False) -> AnalysisResult | None:
    """Render one component's Results section + results.json.

    Steps:
      1. Resolve ``experiments/<component>_*/analysis.py``.
      2. Import it and call ``run_analysis() -> AnalysisResult``.
      3. Atomically rewrite the AUTO-RESULTS block in
         ``docs/components/<component>.md``.
      4. Write ``experiments/<component>_*/results.json``.

    Returns the :class:`AnalysisResult`, or ``None`` if ``missing_ok``
    is True and the experiment dir / analysis.py doesn't exist yet.
    """
    if component not in COMPONENTS:
        raise ValueError(f"Unknown component {component!r}; expected one of {COMPONENTS}.")

    exp_dir = _experiment_dir(component)
    if exp_dir is None:
        if missing_ok:
            return None
        raise FileNotFoundError(
            f"No experiments/{component}_*/ directory. Create one before rendering."
        )

    mod_name = _analysis_module_name(exp_dir)
    try:
        mod = importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if missing_ok:
            return None
        raise FileNotFoundError(
            f"{exp_dir}/analysis.py not found. Copy "
            f"experiments/_analysis_template.py and implement run_analysis()."
        ) from e
    importlib.reload(mod)  # always pick up latest analysis logic

    if not hasattr(mod, "run_analysis"):
        raise AttributeError(
            f"{mod_name}.run_analysis not defined. See _analysis_template.py."
        )

    result: AnalysisResult = mod.run_analysis()
    if not isinstance(result, AnalysisResult):
        raise TypeError(
            f"{mod_name}.run_analysis() returned {type(result).__name__}; "
            "must return temp_bench.report.AnalysisResult."
        )

    # Write results.json
    (exp_dir / "results.json").write_text(
        json.dumps(result.results, indent=2, sort_keys=True, default=str)
    )

    # Rewrite AUTO-RESULTS block in cN.md
    _replace_auto_results(_component_md(component), result.markdown)

    return result


def render_all(*, missing_ok: bool = True) -> dict[str, AnalysisResult | None]:
    """Render every component listed in :data:`COMPONENTS`.

    Default is ``missing_ok=True`` so partial rollouts (only c1+c2 have
    analysis.py yet) don't error. Returns ``{component: result_or_None}``.
    """
    out: dict[str, AnalysisResult | None] = {}
    for c in COMPONENTS:
        out[c] = render(c, missing_ok=missing_ok)
    return out


# ── Contract check (used by tests + CI) ─────────────────────────────────


def check_markers(components: Iterable[str] = COMPONENTS) -> list[str]:
    """Return a list of cN.md paths that are missing the markers.

    Empty list = contract holds. Used by tests/test_report.py and any
    pre-commit hook that wants to enforce the rule.
    """
    missing: list[str] = []
    for c in components:
        path = _component_md(c)
        if not path.exists():
            missing.append(str(path) + " (file missing)")
            continue
        text = path.read_text()
        if BEGIN_MARKER not in text or END_MARKER not in text:
            missing.append(str(path))
    return missing
