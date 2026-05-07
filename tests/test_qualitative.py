"""Tests for ``temp_bench.eval.qualitative`` — pure helpers.

Covers the deterministic/data-only paths:
- ``pick_top_features_by_var``
- ``gather_top_contexts``
- ``pareto_frontier``
- ``persist_judge_record``
- ``_normalize_verdict``
- ``load_concat_corpus`` (uses on-disk JSONs)

The Anthropic-API + GPU paths (``call_judges``, ``encode_concat_corpus``,
``top_256_semantic``) are integration-tested via ``--smoke`` runs.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from temp_bench.eval.qualitative import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_N_CONTEXTS,
    DEFAULT_N_FEATURES,
    JUDGE_PROMPTS,
    LABEL_PROMPT,
    _judge_outputs_path,
    _normalize_verdict,
    gather_top_contexts,
    load_concat_corpus,
    pareto_frontier,
    persist_judge_record,
    pick_top_features_by_var,
)


# ── Constants match c4.md ─────────────────────────────────────────────────


def test_constants_match_spec():
    assert DEFAULT_N_FEATURES == 256
    assert DEFAULT_N_CONTEXTS == 10
    assert DEFAULT_CONTEXT_WINDOW == 20
    assert len(JUDGE_PROMPTS) == 2  # 2-judge majority per c4.md


# ── pick_top_features_by_var ──────────────────────────────────────────────


def test_pick_top_features_basic():
    rng = np.random.default_rng(0)
    z = rng.normal(size=(100, 50)).astype(np.float32)
    # Boost variance on features [3, 7, 11]
    z[:, [3, 7, 11]] *= 5.0
    top = pick_top_features_by_var(z, 3)
    assert set(top) == {3, 7, 11}, f"top-3 should be the boosted features; got {top}"


def test_pick_top_features_returns_int64():
    z = np.zeros((10, 5), dtype=np.float32)
    z[:, 2] = np.arange(10, dtype=np.float32)  # only feat 2 has variance
    top = pick_top_features_by_var(z, 3)
    assert top.dtype == np.int64
    # Sorted descending by var → highest-var feature first
    assert top[0] == 2


def test_pick_top_features_n_too_large():
    z = np.zeros((5, 3), dtype=np.float32)
    z[:, 0] = [1, 2, 3, 4, 5]
    z[:, 1] = [0, 0, 0, 0, 1]
    z[:, 2] = [0, 0, 0, 0, 0]
    top = pick_top_features_by_var(z, 100)  # n > d_sae OK; returns d_sae
    assert len(top) == 3


def test_pick_top_features_bad_shape():
    with pytest.raises(ValueError, match="(n_tokens, d_sae)"):
        pick_top_features_by_var(np.zeros(10), 3)


# ── gather_top_contexts ───────────────────────────────────────────────────


class _FakeTokenizer:
    """Just decodes integers to strings for testing."""
    def decode(self, ids):
        if isinstance(ids, list):
            return " ".join(f"tok{i}" for i in ids)
        return f"tok{ids[0]}" if hasattr(ids, '__iter__') else f"tok{ids}"


def test_gather_top_contexts_basic():
    token_ids = list(range(100))
    tok = _FakeTokenizer()
    z_col = np.zeros(100)
    z_col[10] = 5.0  # peak at position 10
    z_col[20] = 3.0
    z_col[50] = 1.0
    contexts = gather_top_contexts(token_ids, tok, z_col, n_ctx=3, win=20)
    assert len(contexts) == 3
    # Top context at position 10 (highest)
    assert contexts[0]["position"] == 10
    assert contexts[0]["strength"] == 5.0
    # Each context has the marker «»
    for c in contexts:
        assert "«" in c["text"]
        assert "»" in c["text"]


def test_gather_top_contexts_dead_feature():
    token_ids = list(range(100))
    tok = _FakeTokenizer()
    z_col = np.zeros(100)  # all zeros — dead feature
    contexts = gather_top_contexts(token_ids, tok, z_col)
    assert contexts == []


def test_gather_top_contexts_n_ctx_capped():
    token_ids = list(range(100))
    tok = _FakeTokenizer()
    z_col = np.zeros(100)
    z_col[5] = 1.0  # only 1 active position
    contexts = gather_top_contexts(token_ids, tok, z_col, n_ctx=10)
    # Cannot return more contexts than active positions
    assert len(contexts) == 1


def test_gather_top_contexts_window_at_boundary():
    """Position near the start/end of the sequence — window clamps."""
    token_ids = list(range(50))
    tok = _FakeTokenizer()
    z_col = np.zeros(50)
    z_col[2] = 1.0   # near start (pos 2 with win=20 → lo=0)
    z_col[48] = 0.5  # near end (pos 48 with win=20 → hi=50)
    contexts = gather_top_contexts(token_ids, tok, z_col, n_ctx=2, win=20)
    assert len(contexts) == 2
    # Both should have valid text (no negative slice)
    for c in contexts:
        assert isinstance(c["text"], str)
        assert "«" in c["text"]


# ── pareto_frontier ──────────────────────────────────────────────────────


def test_pareto_frontier_empty():
    assert pareto_frontier([]) == []


def test_pareto_frontier_singleton():
    result = pareto_frontier([("a", 0.5, 50)])
    assert len(result) == 1
    assert result[0][0] == "a"


def test_pareto_frontier_dominates():
    points = [
        ("a", 0.7, 100),  # dominates b, c
        ("b", 0.6, 80),
        ("c", 0.5, 60),
    ]
    result = pareto_frontier(points)
    assert len(result) == 1
    assert result[0][0] == "a"


def test_pareto_frontier_multiple_non_dominated():
    points = [
        ("a", 0.9, 50),   # strong AUC, weak SEMANTIC
        ("b", 0.7, 100),  # mid AUC, strong SEMANTIC
        ("c", 0.5, 30),   # dominated
        ("d", 0.6, 80),   # dominated by b
    ]
    result = pareto_frontier(points)
    labels = sorted(p[0] for p in result)
    assert labels == ["a", "b"], f"frontier should be a + b; got {labels}"


# ── _normalize_verdict ───────────────────────────────────────────────────


def test_normalize_verdict_clean():
    assert _normalize_verdict("SEMANTIC") == "SEMANTIC"
    assert _normalize_verdict("SYNTACTIC") == "SYNTACTIC"
    assert _normalize_verdict("semantic") == "SEMANTIC"  # case-insensitive
    assert _normalize_verdict("  SEMANTIC  ") == "SEMANTIC"  # whitespace


def test_normalize_verdict_with_extra():
    assert _normalize_verdict("SEMANTIC (concept)") == "SEMANTIC"
    assert _normalize_verdict("My answer is SEMANTIC.") == "SEMANTIC"


def test_normalize_verdict_unknown():
    assert _normalize_verdict("") == "UNKNOWN"
    assert _normalize_verdict("BLAH") == "UNKNOWN"
    assert _normalize_verdict(None) == "UNKNOWN"


# ── persist_judge_record ─────────────────────────────────────────────────


def test_persist_judge_record_appends(tmp_path, monkeypatch):
    # Redirect purified_root to tmp_path so jsonl lands in test dir
    import temp_bench.eval.qualitative as q
    monkeypatch.setattr(q, "purified_root", lambda: tmp_path)

    eval_key = "abcdef0123456789"
    persist_judge_record(
        eval_key,
        feature_id=42, context_id=0, judge_id=1,
        label="capitalised verbs", raw_response="SEMANTIC",
        verdict="SEMANTIC", judge_model="claude-haiku-test",
        prompt="dummy prompt 1",
    )
    persist_judge_record(
        eval_key,
        feature_id=42, context_id=0, judge_id=2,
        label="capitalised verbs", raw_response="SYNTACTIC",
        verdict="SYNTACTIC", judge_model="claude-haiku-test",
        prompt="dummy prompt 2",
    )

    p = tmp_path / "results" / "runs" / eval_key / "judge_outputs.jsonl"
    assert p.exists()
    lines = p.read_text().strip().splitlines()
    assert len(lines) == 2
    rec1 = json.loads(lines[0])
    assert rec1["feature_id"] == 42
    assert rec1["judge_id"] == 1
    assert rec1["verdict"] == "SEMANTIC"
    assert rec1["prompt_hash"]   # 16-char sha256 truncation
    assert "ts" in rec1


# ── load_concat_corpus ───────────────────────────────────────────────────


def test_load_concat_corpus_real():
    """Uses the real on-disk JSONs ported from wasteland."""
    d = load_concat_corpus("concat_A")
    assert "token_ids" in d
    assert "provenance" in d
    assert isinstance(d["token_ids"], list)
    assert len(d["token_ids"]) > 0
    assert isinstance(d["provenance"], list)


def test_load_concat_corpus_missing():
    with pytest.raises(FileNotFoundError):
        load_concat_corpus("nonexistent_corpus_blah")
