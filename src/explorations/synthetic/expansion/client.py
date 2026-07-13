"""Claude API client for the expansion loop: role routing + a hard spend cap.

Three fixed roles (briefing: "Bulk labeling on Haiku; Sonnet only for
validation/adjudication; Opus only for hypotheses + the skeptic"):

    bulk     -> claude-haiku-4-5-20251001
    validate -> claude-sonnet-5
    think    -> claude-opus-4-8

Every call is metered against a persistent JSON meter (default cap $25/cycle).
The meter is checked BEFORE each request and raises :class:`SpendCapExceeded`
at the cap — the caller reports partial results, never exceeds it. A per-call
audit line goes to ``spend_log.jsonl`` next to the meter file.

Prices are USD/MTok estimates pinned here so the meter is deterministic; they
err on the published list prices.
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROLES = {
    "bulk": "claude-haiku-4-5-20251001",
    "validate": "claude-sonnet-5",
    "think": "claude-opus-4-8",
}

# USD per MTok (input, output) — meter estimates, pinned for determinism.
PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-5": (3.0, 15.0),
    "claude-opus-4-8": (5.0, 25.0),
}

# Claude 5-family models reject the `temperature` param ("deprecated for this
# model"); only send it where supported.
_SUPPORTS_TEMPERATURE = {"claude-haiku-4-5-20251001"}

DEFAULT_CAP_USD = 25.0
_REPO = Path(__file__).resolve().parents[4]
DEFAULT_METER = _REPO / "experiments/explorations/synthetic/expansion/results/spend.json"

_RETRIABLE = (429, 500, 502, 503, 529)


class SpendCapExceeded(RuntimeError):
    """Raised when a call would run past the per-cycle cost cap."""


class Meter:
    """Thread-safe, file-persistent spend meter (survives process restarts)."""

    def __init__(self, path: Path | str = DEFAULT_METER, cap_usd: float = DEFAULT_CAP_USD):
        self.path = Path(path)
        self.log_path = self.path.with_name("spend_log.jsonl")
        self.cap = float(cap_usd)
        self._lock = threading.Lock()
        if self.path.exists():
            self._state = json.loads(self.path.read_text())
        else:
            self._state = {"cap_usd": self.cap, "spent_usd": 0.0, "n_calls": 0, "by_model": {}}
        self._state["cap_usd"] = self.cap

    @property
    def spent(self) -> float:
        return float(self._state["spent_usd"])

    def check(self):
        if self.spent >= self.cap:
            raise SpendCapExceeded(f"spend ${self.spent:.2f} >= cap ${self.cap:.2f}")

    def add(self, model: str, in_tok: int, out_tok: int, tag: str = ""):
        pin, pout = PRICES[model]
        usd = (in_tok * pin + out_tok * pout) / 1e6
        with self._lock:
            self._state["spent_usd"] = self.spent + usd
            self._state["n_calls"] += 1
            bm = self._state["by_model"].setdefault(model, {"in": 0, "out": 0, "usd": 0.0})
            bm["in"] += in_tok
            bm["out"] += out_tok
            bm["usd"] += usd
            self._flush()
            with self.log_path.open("a") as f:
                f.write(json.dumps({"ts": time.time(), "model": model, "in": in_tok,
                                    "out": out_tok, "usd": round(usd, 6), "tag": tag}) + "\n")
        return usd

    def _flush(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._state, indent=2))
        os.replace(tmp, self.path)

    def summary(self) -> dict:
        return dict(self._state)


class Judge:
    """Metered Claude caller with retries; per-role model routing."""

    def __init__(self, meter: Meter | None = None, max_retries: int = 6):
        import anthropic  # deferred so the pure-math modules never need it

        self._anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.meter = meter if meter is not None else Meter()
        self.max_retries = max_retries

    def call(self, role: str, system: str, user: str, *, max_tokens: int = 1024,
             temperature: float = 0.0, tag: str = "") -> str:
        model = ROLES[role]
        last = None
        for attempt in range(self.max_retries):
            self.meter.check()
            kw = {"temperature": temperature} if model in _SUPPORTS_TEMPERATURE else {}
            try:
                r = self.client.messages.create(
                    model=model, max_tokens=max_tokens, system=system,
                    messages=[{"role": "user", "content": user}], **kw)
                self.meter.add(model, r.usage.input_tokens, r.usage.output_tokens, tag=tag)
                return "".join(b.text for b in r.content if b.type == "text")
            except (self._anthropic.RateLimitError, self._anthropic.APIConnectionError) as e:
                last = e
            except self._anthropic.APIStatusError as e:
                if e.status_code not in _RETRIABLE:
                    raise
                last = e
            time.sleep(min(60.0, 2.0 ** attempt + random.random()))
        raise RuntimeError(f"call failed after {self.max_retries} retries: {last}")

    def call_many(self, role: str, jobs: list[dict], *, workers: int = 8,
                  tag: str = "") -> list[str | None]:
        """Run jobs (each: system, user, max_tokens?) concurrently, order-preserving.

        A job that still fails after retries (or hits the cap) yields ``None`` —
        the caller decides whether partial coverage is acceptable.
        """
        out: list[str | None] = [None] * len(jobs)

        def one(i: int):
            j = jobs[i]
            try:
                out[i] = self.call(role, j["system"], j["user"],
                                   max_tokens=j.get("max_tokens", 1024),
                                   temperature=j.get("temperature", 0.0),
                                   tag=j.get("tag", tag))
            except SpendCapExceeded:
                raise
            except Exception as e:  # noqa: BLE001 — record and continue; coverage reported
                print(f"[call_many] job {i} failed: {type(e).__name__}: {e}")

        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, i) for i in range(len(jobs))]
            for f in futs:
                try:
                    f.result()
                except SpendCapExceeded:
                    for g in futs:
                        g.cancel()
                    raise
        return out
