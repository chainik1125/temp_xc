#!/usr/bin/env python3
"""Stage 5b': Sonnet-graded coherence per generation.

Sends every generation in `intervene/generations.jsonl` to Claude Sonnet
with the standard 0–3 coherence rubric (verbatim from T-SAE paper §B.2 and
the existing `case_studies/steering/grade_with_sonnet.py`):

    0 = completely incoherent (e.g. very few or no English words)
    1 = somewhat coherent (e.g. highly repetitive)
    2 = mostly coherent
    3 = fully coherent

Concurrent ThreadPoolExecutor with `--workers 5` and `max_retries=12` to stay
under the 50 req/min rate limit. Output `intervene/coherence_grades.jsonl`,
one row per source row with the same identifying keys plus `coherence_grade`
and `coherence_raw` (the model's literal reply for audit).

Idempotent: skips already-graded rows by (mode, target, magnitude, prompt_id)
so a second run only fills in new generations.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    INTERVENE_DIR,
    RESULTS_DIR,
    ensure_dirs,
)


COHERENCE_PROMPT = (
    "You are an LLM expert. Given the following text, grade its linguistic "
    "coherence. Only respond with one of the grades: 0 = completely incoherent "
    "(e.g. very few or no English words), 1 = somewhat coherent (e.g. highly "
    "repetitive), 2 = mostly coherent, or 3 = fully coherent. Respond with the "
    "digit only.\n\nText: {text}\n\nGrade:"
)

_DIGIT_RE = re.compile(r"^[\s\D]*([0-3])")


def _parse_grade(text: str) -> int | None:
    if not text:
        return None
    m = _DIGIT_RE.search(text)
    if m:
        return int(m.group(1))
    for ch in text:
        if ch in "0123":
            return int(ch)
    return None


def _load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    for candidate in (
        Path.home() / ".tokens/anthropic_key",
        Path("/root/.tokens/anthropic_key"),
        Path("/workspace/.tokens/anthropic_key"),
    ):
        if candidate.exists():
            return candidate.read_text().strip()
    raise RuntimeError(
        "no Anthropic API key found: set $ANTHROPIC_API_KEY or place the key "
        "in ~/.tokens/anthropic_key, /root/.tokens/anthropic_key, or "
        "/workspace/.tokens/anthropic_key"
    )


def _row_key(row: dict) -> tuple:
    return (row["mode"], row["target"], float(row["magnitude"]), row["prompt_id"])


def _load_done(out_path: Path) -> dict[tuple, dict]:
    if not out_path.exists():
        return {}
    out: dict[tuple, dict] = {}
    with out_path.open() as f:
        for line in f:
            r = json.loads(line)
            out[_row_key(r)] = r
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-chars", type=int, default=4000, help="truncate generation before sending to grader")
    parser.add_argument("--intervene-suffix", default="", help="read from intervene<_suffix>/")
    parser.add_argument("--force", action="store_true", help="ignore existing grades and re-grade everything")
    args = parser.parse_args()

    ensure_dirs()
    intervene_dir = (
        RESULTS_DIR / (f"intervene_{args.intervene_suffix}" if args.intervene_suffix else "intervene")
    )
    intervene_dir.mkdir(parents=True, exist_ok=True)
    gens_path = intervene_dir / "generations.jsonl"
    out_path = intervene_dir / "coherence_grades.jsonl"
    if not gens_path.exists():
        raise SystemExit(f"missing {gens_path}; run Stage 4 first")

    if args.force and out_path.exists():
        out_path.unlink()
    done = _load_done(out_path)

    rows = [json.loads(line) for line in gens_path.open()]
    todo = [r for r in rows if _row_key(r) not in done]
    print(f"[grade_coherence] {len(rows)} total, {len(done)} already graded, {len(todo)} todo")
    if not todo:
        return

    from anthropic import Anthropic

    client = Anthropic(api_key=_load_api_key(), max_retries=12)

    def _grade(row: dict) -> dict:
        text = row["generation"]
        if len(text) > args.max_chars:
            text = text[: args.max_chars]
        prompt = COHERENCE_PROMPT.format(text=text)
        msg = client.messages.create(
            model=args.model,
            max_tokens=10,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = msg.content[0].text.strip()
        return {
            "mode": row["mode"],
            "target": row["target"],
            "magnitude": float(row["magnitude"]),
            "prompt_id": row["prompt_id"],
            "category": row.get("category", ""),
            "feature_idx": row.get("feature_idx"),
            "coherence_grade": _parse_grade(raw),
            "coherence_raw": raw,
        }

    t0 = time.time()
    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(_grade, r): r for r in todo}
        for i, fut in enumerate(as_completed(futures)):
            try:
                grade = fut.result()
            except Exception as e:
                src = futures[fut]
                print(f"  [error] {_row_key(src)}: {e}")
                continue
            fout.write(json.dumps(grade) + "\n")
            fout.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                rate = (i + 1) / max(time.time() - t0, 1e-3)
                print(f"  [{i+1:>4}/{len(todo)}] {rate:.1f} req/s")

    print(f"[grade_coherence] wrote {out_path} in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
