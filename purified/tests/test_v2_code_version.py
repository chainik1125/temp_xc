"""Code-version capture contract."""

from __future__ import annotations

import os

import pytest

from temp_bench.core.code_version import (
    capture,
    commit_sha,
    diff_hash,
    is_dirty,
)


def test_commit_sha_format() -> None:
    sha = commit_sha()
    assert isinstance(sha, str)
    assert len(sha) == 40
    assert all(c in "0123456789abcdef" for c in sha)


def test_is_dirty_returns_bool() -> None:
    assert isinstance(is_dirty(), bool)


def test_diff_hash_consistent_with_dirty() -> None:
    if is_dirty():
        h = diff_hash()
        assert h is not None
        assert len(h) == 64
    else:
        assert diff_hash() is None


def test_capture_refuses_dirty_by_default(monkeypatch) -> None:
    monkeypatch.delenv("TEMP_BENCH_ALLOW_DIRTY", raising=False)
    if not is_dirty():
        pytest.skip("working tree is clean; capture would succeed")
    with pytest.raises(RuntimeError, match="dirty working tree"):
        capture()


def test_capture_accepts_env_override(monkeypatch) -> None:
    monkeypatch.setenv("TEMP_BENCH_ALLOW_DIRTY", "1")
    cv = capture()
    assert cv.commit_sha
    assert isinstance(cv.dirty, bool)


def test_capture_accepts_kwarg_override() -> None:
    cv = capture(allow_dirty=True)
    assert cv.commit_sha
    if cv.dirty:
        assert cv.diff_sha256 is not None
