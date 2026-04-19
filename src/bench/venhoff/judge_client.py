"""Async judge client for the Venhoff taxonomy pipeline.

Two backends behind a uniform `.call(system, user) -> str` coroutine:
  - `AnthropicJudge`: Claude (default: Haiku 4.5). Primary judge for the
    run; see `docs/aniket/experiments/venhoff_eval/plan.md § 2`.
  - `OpenAIJudge`: GPT-4o, used only for the 100-sentence bridge drift
    check at smoke time per Q5 lock-in (drift threshold 0.5/10).

The Anthropic path is a direct lift of `ClaudeAPIBackend` from
`temporal_crosscoders/NLP/autointerp.py` — same semaphore + exp-backoff
behavior, same async pattern. Extracting it here lets Venhoff taxonomy
code use it without pulling autointerp's ~1k-line module.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Protocol

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.judge")

HAIKU_4_5 = "claude-haiku-4-5-20251001"
GPT_4O = "gpt-4o"


class Judge(Protocol):
    """Uniform judge interface."""

    model: str
    n_calls: int
    n_errors: int

    async def call(self, system: str, user: str) -> str: ...


class AnthropicJudge:
    """Async Anthropic Messages API wrapper with semaphore + retry."""

    def __init__(
        self,
        model: str = HAIKU_4_5,
        max_tokens: int = 512,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ):
        import anthropic

        self._client = anthropic.AsyncAnthropic()
        self.model = model
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_errors = 0

    async def call(self, system: str, user: str) -> str:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.messages.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        system=system,
                        messages=[{"role": "user", "content": user}],
                    )
                    self.n_calls += 1
                    return resp.content[0].text
            except Exception as e:
                self.n_errors += 1
                if attempt == self.max_retries - 1:
                    log.error("[error] anthropic_judge_failed | attempts=%d | exc=%s", self.max_retries, e)
                    return ""
                log.warning("[warn] anthropic_judge_retry | attempt=%d | exc=%s | delay=%.1fs", attempt + 1, e, delay)
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60.0)
        return ""


class OpenAIJudge:
    """Async OpenAI Chat Completions wrapper — bridge-drift use only.

    Kept intentionally minimal: this is not on the hot path. The only
    reason it exists is the Q5 lock-in bridge run comparing Haiku 4.5
    against GPT-4o on 100 sampled sentences at smoke time.
    """

    def __init__(
        self,
        model: str = GPT_4O,
        max_tokens: int = 512,
        max_concurrent: int = 5,
        max_retries: int = 3,
    ):
        from openai import AsyncOpenAI

        self._client = AsyncOpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self._sem = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.n_calls = 0
        self.n_errors = 0

    async def call(self, system: str, user: str) -> str:
        delay = 1.0
        for attempt in range(self.max_retries):
            try:
                async with self._sem:
                    resp = await self._client.chat.completions.create(
                        model=self.model,
                        max_tokens=self.max_tokens,
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                    )
                    self.n_calls += 1
                    return resp.choices[0].message.content or ""
            except Exception as e:
                self.n_errors += 1
                if attempt == self.max_retries - 1:
                    log.error("[error] openai_judge_failed | attempts=%d | exc=%s", self.max_retries, e)
                    return ""
                log.warning("[warn] openai_judge_retry | attempt=%d | exc=%s | delay=%.1fs", attempt + 1, e, delay)
                await asyncio.sleep(delay + random.uniform(0, 0.5))
                delay = min(delay * 2, 60.0)
        return ""


def make_judge(model: str = HAIKU_4_5, **kwargs) -> Judge:
    """Dispatch by model name prefix.

    Anthropic model ids start with `claude-`; OpenAI's `gpt-4o` is the
    only OpenAI model we use here (bridge only).
    """
    if model.startswith("claude-"):
        return AnthropicJudge(model=model, **kwargs)
    if model.startswith("gpt-"):
        return OpenAIJudge(model=model, **kwargs)
    raise ValueError(f"unknown judge model: {model!r}. Use a claude-* or gpt-* id.")
