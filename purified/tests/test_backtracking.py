"""C7 backtracking helpers — pure-Python tests (no GPU, no HF download).

Exercises the deterministic / data-only helpers in
:mod:`temp_bench.case_studies.backtracking`. The GPU-touching paths
(``run_arch_evaluation``, ``extract_labeled_sentence_acts``,
``load_reasoning_lm``) are integration-tested via
``experiments/c7_backtracking/smoke.py``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from temp_bench.case_studies.backtracking import (
    DEFAULT_CUT_FRACTION,
    DEFAULT_MAGNITUDE_GRID,
    DEFAULT_PR_AUC_S_GRID,
    SONNET_JUDGE_MODEL,
    JudgeOutput,
    SonnetBacktrackingJudge,
    answers_match,
    compute_delta_gc,
    compute_pr_auc_at_S,
    cut25_token_position,
    extract_boxed,
    load_cohort_from_parquet,
    load_stage_a,
    parse_judge_reply,
    split_pos_neg,
)


# ── Constants & defaults ──────────────────────────────────────────────


def test_defaults_match_spec():
    """Spec values match docs/components/c7.md."""
    assert DEFAULT_CUT_FRACTION == 0.25
    assert DEFAULT_PR_AUC_S_GRID == (1, 2, 4, 8, 16, 32)
    assert len(DEFAULT_MAGNITUDE_GRID) == 25
    assert min(DEFAULT_MAGNITUDE_GRID) == -16
    assert max(DEFAULT_MAGNITUDE_GRID) == 16
    assert 0 in DEFAULT_MAGNITUDE_GRID
    assert SONNET_JUDGE_MODEL == "claude-sonnet-4-6"


def test_cut25_token_position():
    assert cut25_token_position(list(range(100))) == 25
    assert cut25_token_position(list(range(100)), fraction=0.5) == 50
    assert cut25_token_position([]) == 0


# ── extract_boxed ──────────────────────────────────────────────────────


def test_extract_boxed_simple():
    assert extract_boxed("answer is \\boxed{42}") == "42"


def test_extract_boxed_nested():
    """Nested braces (e.g. \\frac{1}{2}) parse correctly."""
    assert extract_boxed("result \\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"


def test_extract_boxed_takes_last():
    """Multiple boxed → take the last (final answer convention)."""
    assert extract_boxed("first \\boxed{1} then \\boxed{2}") == "2"


def test_extract_boxed_none_on_empty():
    assert extract_boxed("") is None
    assert extract_boxed("no boxed here") is None


# ── answers_match ──────────────────────────────────────────────────────


def test_answers_match_exact():
    assert answers_match("42", "42")
    assert not answers_match("42", "43")


def test_answers_match_latex_normalisation():
    """\\dfrac and \\frac should match (paper-typical)."""
    assert answers_match("\\dfrac{1}{2}", "\\frac{1}{2}")


def test_answers_match_none():
    assert not answers_match(None, "42")


# ── parse_judge_reply ─────────────────────────────────────────────────


def test_parse_judge_reply_valid():
    assert parse_judge_reply("COUNT: 3\nNOTES: blah") == 3
    assert parse_judge_reply("COUNT: 0\nNOTES: none") == 0
    assert parse_judge_reply("COUNT: 12") == 12


def test_parse_judge_reply_negative_signal():
    """-1 sentinel for parse failure (matches Aniket's grade_backtracking.py)."""
    assert parse_judge_reply("garbage") == -1
    assert parse_judge_reply("") == -1


# ── Cohort + stage A loaders ──────────────────────────────────────────


def test_load_cohort_from_parquet_shape():
    """Aniket's cohort qids are read deterministically and total 61."""
    pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    cohort = load_cohort_from_parquet()
    assert len(cohort.truly_wrong) == 31
    assert len(cohort.originally_correct) == 30
    assert cohort.source == "aniket_parquet"
    # Qids look like MATH-500 unique_ids
    assert all(q.startswith("test/") for q in cohort.truly_wrong)


def test_load_stage_a_shape():
    """Stage A artifacts load + 300 of each."""
    sa = load_stage_a()
    assert len(sa.prompts) == 300
    assert len(sa.traces) == 300
    assert len(sa.sentence_labels) == 300
    assert "base" in sa.dom_vectors
    assert "reasoning" in sa.dom_vectors


# ── Judge persistence ─────────────────────────────────────────────────


def test_sonnet_judge_persistence_writes_jsonl(tmp_path: Path):
    """A persisted JudgeOutput round-trips through judge_outputs.jsonl."""
    judge = SonnetBacktrackingJudge(workspace=tmp_path)
    out = JudgeOutput(
        transcript_id="qid-x",
        magnitude=-8.0,
        arch="topk_sae",
        seed=42,
        judge_id="judge-stub",
        judge_model="model-stub",
        prompt_hash="abc123",
        label=2,
        raw="COUNT: 2\nNOTES: stub",
        ts="2026-05-04T00:00:00Z",
    )
    judge._persist(out)
    rows = [json.loads(l) for l in (tmp_path / "judge_outputs.jsonl").open()]
    assert len(rows) == 1
    assert rows[0]["transcript_id"] == "qid-x"
    assert rows[0]["magnitude"] == -8.0
    assert rows[0]["label"] == 2


def test_sonnet_judge_existing_keys_resume(tmp_path: Path):
    """Existing-key set lets a resumed sweep skip already-judged rows."""
    judge = SonnetBacktrackingJudge(workspace=tmp_path)
    judge._persist(JudgeOutput(
        transcript_id="q1", magnitude=0.0, arch="topk_sae", seed=42,
        judge_id="j", judge_model="m", prompt_hash="h", label=1,
        raw="COUNT: 1", ts="t",
    ))
    judge._persist(JudgeOutput(
        transcript_id="q2", magnitude=4.0, arch="tsae_paper", seed=42,
        judge_id="j", judge_model="m", prompt_hash="h", label=0,
        raw="COUNT: 0", ts="t",
    ))
    keys = judge.existing_keys()
    assert ("q1", 0.0, "topk_sae", 42) in keys
    assert ("q2", 4.0, "tsae_paper", 42) in keys
    assert ("q3", 0.0, "topk_sae", 42) not in keys


# ── compute_delta_gc ──────────────────────────────────────────────────


def test_delta_gc_baseline_correction():
    """Δgc subtracts per-(arch, qid) value at mag=0 from non-zero mags."""
    rows = [
        # arch=A, qid=q1: gc(0)=2, gc(+8)=5 → delta=+3
        {"arch": "A", "transcript_id": "q1", "seed": 1, "magnitude": 0.0, "label": 2},
        {"arch": "A", "transcript_id": "q1", "seed": 1, "magnitude": 8.0, "label": 5},
        # arch=A, qid=q2: gc(0)=1, gc(+8)=2 → delta=+1
        {"arch": "A", "transcript_id": "q2", "seed": 1, "magnitude": 0.0, "label": 1},
        {"arch": "A", "transcript_id": "q2", "seed": 1, "magnitude": 8.0, "label": 2},
    ]
    out = compute_delta_gc(rows)
    # mean delta over q1, q2 at mag=+8 = (3 + 1) / 2 = 2.0
    assert out["by_arch_mag"][("A", 8.0)] == pytest.approx(2.0)
    assert out["peak"]["A"] == (8.0, 2.0)


def test_delta_gc_skips_negative_labels():
    """label=-1 (parse failure) rows are excluded from aggregation."""
    rows = [
        {"arch": "A", "transcript_id": "q1", "seed": 1, "magnitude": 0.0, "label": 2},
        {"arch": "A", "transcript_id": "q1", "seed": 1, "magnitude": 8.0, "label": -1},
        {"arch": "A", "transcript_id": "q1", "seed": 1, "magnitude": 8.0, "label": 4},
    ]
    out = compute_delta_gc(rows)
    # gc(8) is mean of [4] (skipping -1) = 4 → delta = 4 - 2 = 2
    assert out["by_arch_mag"][("A", 8.0)] == pytest.approx(2.0)


# ── compute_pr_auc_at_S ───────────────────────────────────────────────


def test_pr_auc_returns_value_per_S():
    """PR-AUC dict has one entry per S; values in [0, 1]."""
    rng = np.random.default_rng(0)
    n = 200
    d = 64
    feats = rng.normal(size=(n, d)).astype(np.float32)
    # Inject signal: feature 0 is high for positives.
    labels = (rng.random(n) < 0.2).astype(int)
    feats[labels == 1, 0] += 3.0
    qids = np.array([f"q{i // 4}" for i in range(n)], dtype=object)  # 4 sentences/q
    out = compute_pr_auc_at_S(feats, labels, qids, S_grid=(1, 4, 16))
    assert set(out.keys()) == {1, 4, 16}
    for v in out.values():
        assert 0.0 <= v <= 1.0
    # Strong injected signal at feature 0 should make S=1 recover well.
    assert out[1] > 0.5


# ── split_pos_neg ─────────────────────────────────────────────────────


def test_split_pos_neg_shapes():
    sa = {
        "X": np.zeros((10, 6, 4), dtype=np.float32),
        "is_bt": np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1], dtype=bool),
        "keys": np.array([f"k{i}" for i in range(10)], dtype=object),
    }
    pn = split_pos_neg(sa)
    assert pn["pos"].shape == (4, 6, 4)
    assert pn["neg"].shape == (6, 6, 4)
