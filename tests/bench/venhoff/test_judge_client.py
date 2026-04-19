"""Smoke tests for the judge client dispatch + import paths.

Runs without network — only validates that `make_judge` routes to the
right class and that the classes instantiate with a mock client. Real
API calls are exercised only in the live smoke run, not in CI.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


def _ensure_stub(module_name: str, attr_name: str) -> None:
    """Install a MagicMock-backed stub for a missing SDK module.

    We don't want to hard-require the anthropic/openai SDKs just to run
    router tests. If the real module is importable, leave it alone.
    """
    if importlib.util.find_spec(module_name) is not None:
        return
    mod = types.ModuleType(module_name)
    setattr(mod, attr_name, MagicMock())
    sys.modules[module_name] = mod


_ensure_stub("anthropic", "AsyncAnthropic")
_ensure_stub("openai", "AsyncOpenAI")

from src.bench.venhoff.judge_client import (  # noqa: E402
    GPT_4O,
    HAIKU_4_5,
    AnthropicJudge,
    OpenAIJudge,
    make_judge,
)


def test_make_judge_routes_claude_to_anthropic():
    with patch("anthropic.AsyncAnthropic", return_value=MagicMock()):
        judge = make_judge(HAIKU_4_5)
    assert isinstance(judge, AnthropicJudge)
    assert judge.model == HAIKU_4_5


def test_make_judge_routes_gpt_to_openai():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        judge = make_judge(GPT_4O)
    assert isinstance(judge, OpenAIJudge)
    assert judge.model == GPT_4O


def test_make_judge_rejects_unknown_prefix():
    with pytest.raises(ValueError, match="unknown judge model"):
        make_judge("llama-3.1-70b")


def test_anthropic_judge_default_model():
    with patch("anthropic.AsyncAnthropic", return_value=MagicMock()):
        judge = AnthropicJudge()
    assert judge.model == HAIKU_4_5
    assert judge.n_calls == 0
    assert judge.n_errors == 0


def test_openai_judge_default_model():
    with patch("openai.AsyncOpenAI", return_value=MagicMock()):
        judge = OpenAIJudge()
    assert judge.model == GPT_4O
