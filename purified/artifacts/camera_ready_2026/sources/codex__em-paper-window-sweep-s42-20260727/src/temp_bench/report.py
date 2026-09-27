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
from temp_bench.config import (
    compute_act_cache_key,
    compute_train_key,
    load_arch,
    load_datasource,
    purified_root,
)
from temp_bench.schemas import LeaderboardRow, TrainingConfig

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
            to ````. Recorded for the rendering layer; the .md
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


# ── Canonical train_key filter ───────────────────────────────────────────


def canonical_train_keys(
    *,
    component: str,
    archs: Iterable[str],
    seeds: Iterable[int],
    datasource_names: Iterable[str],
    training_cfg: TrainingConfig | None = None,
) -> set[str]:
    """Compute the set of canonical ``train_key`` values for a sweep.

    Use this in a component's ``analysis.py`` to filter stale leaderboard
    rows out of AUTO-RESULTS. When a paper-wide default changes (e.g.,
    ``TrainingConfig.batch_size`` was raised from 256 to 1024 on
    2026-05-04), prior cells stay in ``leaderboard.jsonl`` for diff
    comparison but their ``train_key`` no longer matches the canonical
    config. This helper returns only the keys that match the supplied
    ``training_cfg`` (or the current ``TrainingConfig()`` defaults if
    omitted), so filtering is a one-line set membership check::

        valid = canonical_train_keys(
            component="c5",
            archs=["txc_base", "txc_pro", "tsae_paper"],
            seeds=(1, 2, 42),
            datasource_names=["gemma_2_2b_it_l13_fineweb_24k128"],
        )
        rows = [r for r in query_leaderboard(component="c5")
                if r.train_key in valid]

    Args:
        component: Component name (``"c1"``…``"c7"``). Propagated to
            :func:`temp_bench.config.load_arch` so any
            ``per_component_hparams`` overrides (e.g. C7's ``d_sae=32768``)
            are applied.
        archs: Architecture names from ``configs/locked_archs.yaml``.
            Names that aren't registered or whose class can't be imported
            are silently skipped — so you can pass the union of archs the
            component might care about without per-arch try/except.
        seeds: Seed values to canonicalize.
        datasource_names: Datasource names from
            ``configs/datasources.yaml``. Multiple are allowed (e.g., a
            primary plus a mirror); unknown names are silently skipped.
        training_cfg: :class:`TrainingConfig` to canonicalize against.
            Defaults to ``TrainingConfig()`` (the paper-wide canonical
            defaults). Pass an explicit instance only when filtering for
            an alternate canonical config (e.g., a component that runs
            with ``bricken_enabled=True``).

    Returns:
        Set of 16-char ``train_key`` strings. Empty if every (arch,
        datasource) pair was unresolvable.
    """
    cfg = training_cfg if training_cfg is not None else TrainingConfig()

    valid: set[str] = set()
    for ds_name in datasource_names:
        try:
            ds = load_datasource(ds_name)
        except KeyError:
            continue
        ack = compute_act_cache_key(ds)
        for arch_name in archs:
            try:
                spec = load_arch(arch_name, component=component)
            except (KeyError, ImportError):
                continue
            for seed in seeds:
                valid.add(compute_train_key(
                    arch=spec,
                    seed=int(seed),
                    training_cfg=cfg,
                    act_cache_key=ack,
                ))
    return valid


# ── Component → analysis.py resolution ──────────────────────────────────


def _experiment_dir(component: str) -> Path | None:
    """Find the canonical ``experiments/<component>_*/`` directory.

    Returns ``None`` if no experiment directory exists yet (agent hasn't
    started the component). When multiple directories match (e.g. an IT
    lead dir plus auxiliary BASE / sub-driver dirs), prefer the dir whose
    name has the fewest ``_``-separated segments — by convention the
    canonical lead is the shortest (e.g. ``c3_probing`` over
    ``c3_probing_base`` or ``c3_probing_tfa_baseline``). Sub-drivers
    that don't expose ``analysis.py`` are skipped automatically.
    """
    root = purified_root() / "experiments"
    if not root.exists():
        return None
    candidates = sorted(p for p in root.glob(f"{component}_*") if p.is_dir())
    if len(candidates) == 0:
        return None
    # Prefer dirs that actually expose analysis.py (the rendering hook).
    with_analysis = [p for p in candidates if (p / "analysis.py").exists()]
    if with_analysis:
        candidates = with_analysis
    # Tiebreak: shortest segment count, then lexicographic. The canonical
    # IT-side lead always wins under both rules.
    candidates.sort(key=lambda p: (len(p.name.split("_")), p.name))
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
