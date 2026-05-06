"""Tests for ``temp_bench.report`` — results-binding contract.

Three things matter:

1. **Marker contract**: every shipped ``docs/components/cN.md`` has the
   ``BEGIN_MARKER`` + ``END_MARKER`` pair. CI fails otherwise.
2. **Render writes the right file regions**: given a fake leaderboard
   and a fake ``analysis.py``, ``render(component=...)`` rewrites the
   AUTO-RESULTS block + writes ``results.json``.
3. **Idempotency**: rendering the same analysis twice produces an
   identical .md file.
"""

from __future__ import annotations

import importlib
import json
import shutil
import sys
import tempfile
from pathlib import Path

import pytest


def test_all_components_have_auto_results_markers():
    """Contract check — runs against the real ``docs/components/`` dir."""
    from temp_bench.report import COMPONENTS, check_markers

    missing = check_markers(COMPONENTS)
    assert missing == [], (
        f"These cN.md files are missing AUTO-RESULTS markers: {missing}. "
        "Add `<!-- BEGIN AUTO-RESULTS -->` and `<!-- END AUTO-RESULTS -->` "
        "around the Results section."
    )


@pytest.fixture
def tmp_temp_bench_root(monkeypatch):
    """Stage a tmp purified/ for render() tests.

    Copies configs/ + a minimal docs/components/c1.md with markers and
    sets up an experiments/c1_synthetic_topk/analysis.py module so
    importlib can import it.
    """
    real_root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        shutil.copytree(real_root / "configs", tmp / "configs")
        (tmp / "results").mkdir()
        (tmp / "results" / "leaderboard.jsonl").touch()

        # Minimal docs/components/c1.md with markers
        (tmp / "docs" / "components").mkdir(parents=True)
        (tmp / "docs" / "components" / "c1.md").write_text(
            "## Results\n\n"
            "<!-- BEGIN AUTO-RESULTS -->\nplaceholder\n<!-- END AUTO-RESULTS -->\n"
            "\n## Caveats\n\nfoo\n"
        )

        # Minimal experiments/c1_synthetic_topk/analysis.py
        exp_dir = tmp / "experiments" / "c1_synthetic_topk"
        exp_dir.mkdir(parents=True)
        (tmp / "experiments" / "__init__.py").touch()
        (exp_dir / "__init__.py").touch()
        (exp_dir / "analysis.py").write_text(
            "from temp_bench.report import AnalysisResult\n"
            "def run_analysis():\n"
            "    return AnalysisResult(\n"
            "        markdown='| arch | mean |\\n|---|---:|\\n| txc_pro | 0.91 |',\n"
            "        results={'txc_pro': {'mean_auc': 0.91}},\n"
            "    )\n"
        )

        monkeypatch.setenv("TEMP_BENCH_ROOT", str(tmp))
        # Add the tmp root to sys.path so `import experiments.c1_synthetic_topk.analysis` works
        sys.path.insert(0, str(tmp))
        # Bust caches
        from temp_bench.config import _load_archs_yaml, _load_datasources_yaml
        _load_archs_yaml.cache_clear()
        _load_datasources_yaml.cache_clear()

        yield tmp

        # Cleanup sys.path + module cache (the parent `experiments`
        # package caches the path of the first tmp; without this the
        # second test sees a stale namespace).
        sys.path.remove(str(tmp))
        for mod in list(sys.modules):
            if mod == "experiments" or mod.startswith("experiments."):
                del sys.modules[mod]


def test_render_writes_results_json_and_rewrites_auto_block(tmp_temp_bench_root):
    from temp_bench import report

    result = report.render("c1")
    assert result is not None
    assert result.results == {"txc_pro": {"mean_auc": 0.91}}

    # results.json was written
    rj = (tmp_temp_bench_root / "experiments" / "c1_synthetic_topk" / "results.json")
    assert rj.exists()
    assert json.loads(rj.read_text()) == {"txc_pro": {"mean_auc": 0.91}}

    # cN.md AUTO-RESULTS block was rewritten
    md = (tmp_temp_bench_root / "docs" / "components" / "c1.md").read_text()
    assert "txc_pro" in md
    assert "0.91" in md
    # Markers preserved
    assert "<!-- BEGIN AUTO-RESULTS -->" in md
    assert "<!-- END AUTO-RESULTS -->" in md
    # The rest of the file (Caveats) is untouched
    assert "## Caveats" in md
    assert "foo" in md


def test_render_is_idempotent(tmp_temp_bench_root):
    from temp_bench import report

    report.render("c1")
    md1 = (tmp_temp_bench_root / "docs" / "components" / "c1.md").read_text()
    rj1 = (tmp_temp_bench_root / "experiments" / "c1_synthetic_topk" / "results.json").read_text()

    report.render("c1")
    md2 = (tmp_temp_bench_root / "docs" / "components" / "c1.md").read_text()
    rj2 = (tmp_temp_bench_root / "experiments" / "c1_synthetic_topk" / "results.json").read_text()

    assert md1 == md2, "render() should be idempotent on .md"
    assert rj1 == rj2, "render() should be idempotent on results.json"


def test_render_missing_analysis_with_missing_ok_returns_none(tmp_temp_bench_root):
    """If experiments/c2_*/analysis.py doesn't exist, missing_ok=True returns None."""
    from temp_bench import report

    # c2 dir + analysis.py do not exist in the fixture
    out = report.render("c2", missing_ok=True)
    assert out is None


def test_render_missing_analysis_without_missing_ok_raises(tmp_temp_bench_root):
    from temp_bench import report

    with pytest.raises(FileNotFoundError):
        report.render("c2", missing_ok=False)


def test_render_unknown_component_raises():
    from temp_bench import report

    with pytest.raises(ValueError, match="Unknown component"):
        report.render("c99")


def test_render_md_without_markers_raises(tmp_temp_bench_root):
    """If cN.md is hand-stripped of markers, render() raises."""
    from temp_bench import report

    md = tmp_temp_bench_root / "docs" / "components" / "c1.md"
    md.write_text("## Results\n\nhand-typed numbers!\n")  # no markers

    with pytest.raises(RuntimeError, match="missing AUTO-RESULTS markers"):
        report.render("c1")


def test_query_leaderboard_filters(tmp_temp_bench_root):
    """Smoke test: write a couple rows, query by component + arch."""
    from temp_bench.cache import append_leaderboard
    from temp_bench.report import query_leaderboard
    from temp_bench.schemas import LeaderboardRow

    base = dict(
        eval_key="0" * 16,
        train_key="1" * 16,
        act_cache_key="2" * 16,
        arch_version="1.0.0",
        seed=1,
        datasource="toy_markov_n20_d40",
        eval_protocol_version="1.0.0",
        eval_cfg={"k_feat": 5},
        metrics={"nmse": 0.1},
        primary_metric="nmse",
        agent="test",
        ts="2026-05-03T22:00:00Z",
    )
    append_leaderboard(LeaderboardRow(**{**base, "component": "c1", "arch": "topk_sae"}))
    append_leaderboard(LeaderboardRow(
        **{**base, "component": "c2", "arch": "txc_base", "eval_key": "3" * 16}
    ))

    rows_c1 = query_leaderboard(component="c1")
    assert len(rows_c1) == 1
    assert rows_c1[0].arch == "topk_sae"

    rows_txc = query_leaderboard(arch="txc_base")
    assert len(rows_txc) == 1
    assert rows_txc[0].component == "c2"

    rows_none = query_leaderboard(arch="nonexistent")
    assert rows_none == []


def test_canonical_train_keys_matches_runner(tmp_temp_bench_root):
    """The keys returned by canonical_train_keys must equal what
    runner.run_cell would write for the same inputs."""
    from temp_bench.config import (
        compute_act_cache_key,
        compute_train_key,
        load_arch,
        load_datasource,
    )
    from temp_bench.report import canonical_train_keys
    from temp_bench.schemas import TrainingConfig

    archs = ("topk_sae", "txc_base")
    seeds = (1, 42)
    ds_name = "toy_markov_n20_d40"

    direct: set[str] = set()
    ds = load_datasource(ds_name)
    ack = compute_act_cache_key(ds)
    for arch in archs:
        spec = load_arch(arch, component="c1")
        for seed in seeds:
            direct.add(compute_train_key(
                arch=spec, seed=seed,
                training_cfg=TrainingConfig(), act_cache_key=ack,
            ))

    via_helper = canonical_train_keys(
        component="c1",
        archs=archs,
        seeds=seeds,
        datasource_names=[ds_name],
    )
    assert via_helper == direct
    assert len(via_helper) == len(archs) * len(seeds)


def test_canonical_train_keys_excludes_alternate_cfg(tmp_temp_bench_root):
    """A train_key computed under a different TrainingConfig (e.g., the
    pre-2026-05-04 batch=256) must NOT appear in the canonical set."""
    from temp_bench.config import (
        compute_act_cache_key,
        compute_train_key,
        load_arch,
        load_datasource,
    )
    from temp_bench.report import canonical_train_keys
    from temp_bench.schemas import TrainingConfig

    archs = ("topk_sae",)
    seeds = (42,)
    ds_name = "toy_markov_n20_d40"

    spec = load_arch("topk_sae", component="c1")
    ack = compute_act_cache_key(load_datasource(ds_name))
    old_key = compute_train_key(
        arch=spec, seed=42,
        training_cfg=TrainingConfig(batch_size=256, n_steps=30_000,
                                    plateau_early_stop=True),
        act_cache_key=ack,
    )
    canonical = canonical_train_keys(
        component="c1", archs=archs, seeds=seeds,
        datasource_names=[ds_name],
    )
    assert old_key not in canonical
    assert len(canonical) == 1


def test_canonical_train_keys_skips_unknown_archs_and_datasources(tmp_temp_bench_root):
    """Unknown archs / datasources are silently skipped (no exception)."""
    from temp_bench.report import canonical_train_keys

    keys = canonical_train_keys(
        component="c1",
        archs=("topk_sae", "nonexistent_arch"),
        seeds=(42,),
        datasource_names=["toy_markov_n20_d40", "nonexistent_ds"],
    )
    # Only the (topk_sae, toy_markov_n20_d40) pair should produce a key
    assert len(keys) == 1


def test_canonical_train_keys_respects_explicit_training_cfg(tmp_temp_bench_root):
    """Passing an explicit TrainingConfig (e.g., bricken-on for C6) yields
    a different key than the default."""
    from temp_bench.report import canonical_train_keys
    from temp_bench.schemas import TrainingConfig

    default_keys = canonical_train_keys(
        component="c1",
        archs=("topk_sae",),
        seeds=(42,),
        datasource_names=["toy_markov_n20_d40"],
    )
    bricken_keys = canonical_train_keys(
        component="c1",
        archs=("topk_sae",),
        seeds=(42,),
        datasource_names=["toy_markov_n20_d40"],
        training_cfg=TrainingConfig(bricken_enabled=True),
    )
    assert default_keys != bricken_keys
    assert len(default_keys) == len(bricken_keys) == 1
