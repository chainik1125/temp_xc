"""Token resolution tests.

Validates the unified resolution chain works the same way regardless
of whether the canonical store is at /workspace/.tokens (RunPod) or
~/.tokens (local).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture
def tmp_tokens_dir(monkeypatch):
    """Point TEMP_BENCH_TOKENS_DIR at a clean tmp dir; clear lru_cache."""
    with tempfile.TemporaryDirectory() as td:
        monkeypatch.setenv("TEMP_BENCH_TOKENS_DIR", td)
        # Clear any cached default
        from temp_bench.utils.tokens import tokens_dir
        tokens_dir.cache_clear()
        # Clear all relevant env vars so file paths are exercised
        for v in ("HF_TOKEN", "ANTHROPIC_API_KEY", "GH_TOKEN"):
            monkeypatch.delenv(v, raising=False)
        yield Path(td)


def test_tokens_dir_respects_override(tmp_tokens_dir):
    from temp_bench.utils.tokens import tokens_dir
    assert tokens_dir() == tmp_tokens_dir


def test_get_token_reads_from_file(tmp_tokens_dir):
    from temp_bench.utils.tokens import get_token

    (tmp_tokens_dir / "hf_token").write_text("hf_test_token_value\n")
    assert get_token("hf") == "hf_test_token_value"

    (tmp_tokens_dir / "anthropic_key").write_text("sk-ant-test\n")
    assert get_token("anthropic") == "sk-ant-test"

    (tmp_tokens_dir / "gh_token").write_text("ghp_test\n")
    assert get_token("gh") == "ghp_test"


def test_get_token_env_var_takes_precedence(tmp_tokens_dir, monkeypatch):
    from temp_bench.utils.tokens import get_token

    (tmp_tokens_dir / "hf_token").write_text("from_file")
    monkeypatch.setenv("HF_TOKEN", "from_env")
    assert get_token("hf") == "from_env", "env var must override .tokens/ file"


def test_get_token_returns_none_when_unset(tmp_tokens_dir):
    from temp_bench.utils.tokens import get_token
    # Anthropic — no fallback for this kind
    assert get_token("anthropic") is None


def test_get_token_unknown_kind_raises(tmp_tokens_dir):
    from temp_bench.utils.tokens import get_token
    with pytest.raises(ValueError, match="unknown token kind"):
        get_token("not_a_real_kind")


def test_require_token_raises_clear_error(tmp_tokens_dir):
    from temp_bench.utils.tokens import require_token
    with pytest.raises(RuntimeError, match="No anthropic token found"):
        require_token("anthropic")


def test_hf_falls_back_to_huggingface_default(tmp_tokens_dir, monkeypatch, tmp_path):
    """HF specifically falls back to ~/.cache/huggingface/token (legacy)."""
    from temp_bench.utils.tokens import get_token, tokens_dir

    # Redirect HOME so we don't touch the user's real cache
    fake_home = tmp_path
    monkeypatch.setenv("HOME", str(fake_home))
    hf_default = fake_home / ".cache" / "huggingface" / "token"
    hf_default.parent.mkdir(parents=True)
    hf_default.write_text("from_huggingface_default\n")

    assert get_token("hf") == "from_huggingface_default"


def test_hf_tokens_dir_beats_huggingface_default(tmp_tokens_dir, monkeypatch, tmp_path):
    """When both ~/.tokens/hf_token and ~/.cache/huggingface/token exist,
    the canonical .tokens/ wins."""
    from temp_bench.utils.tokens import get_token

    monkeypatch.setenv("HOME", str(tmp_path))
    hf_default = tmp_path / ".cache" / "huggingface" / "token"
    hf_default.parent.mkdir(parents=True)
    hf_default.write_text("from_huggingface_default")

    (tmp_tokens_dir / "hf_token").write_text("from_canonical")

    assert get_token("hf") == "from_canonical"


def test_token_status_reports_resolution(tmp_tokens_dir, monkeypatch):
    from temp_bench.utils.tokens import token_status

    (tmp_tokens_dir / "hf_token").write_text("x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "y")

    s = token_status()
    assert s["hf"]["resolved_from"] == str(tmp_tokens_dir / "hf_token")
    assert s["anthropic"]["resolved_from"] == "$ANTHROPIC_API_KEY"
    assert s["gh"]["resolved_from"] is None
