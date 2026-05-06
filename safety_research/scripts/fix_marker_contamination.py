"""
Fix the >>>/<<< marker-leakage in existing autointerp explanations.

Two operations per arm:
  1. Globally rewrite top_texts to use the new [FOCUS]...[/FOCUS] tags
     instead of >>>...<<<. The decoded text doesn't change; only the
     wrapping changes.
  2. For every record whose explanation matches the contamination regex
     (literal markers, "between markers", "angle brackets", "highlighted
     window", etc.) re-prompt Haiku 4.5 with the updated tags + a
     stricter system prompt and replace the explanation in place.

Usage:
  /home/cs29824/.venv/bin/python safety_research/scripts/fix_marker_contamination.py
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

NLP_DIR = Path("/home/cs29824/andre/temp_xc/temporal_crosscoders/NLP")
SAFETY_DIR = Path("/home/cs29824/andre/temp_xc/safety_research")
sys.path.insert(0, str(NLP_DIR))
load_dotenv(SAFETY_DIR / ".env")

from autointerp import ClaudeAPIBackend  # noqa: E402

# Reuse the new-pipeline system prompt and prompt builder so this fix is
# truly identical to a fresh run.
sys.path.insert(0, str(SAFETY_DIR / "scripts"))
from run_autointerp import SYSTEM, _build_user_prompt, parse_response  # noqa: E402

ARMS = ["sae", "tsae", "txc"]
EXPLAIN_MODEL = "claude-haiku-4-5-20251001"
MAX_CONCURRENT = 1
MAX_RETRIES = 3

# Strict marker-leakage regex: looks for explicit references to our
# bracketing rather than legitimate uses of words like "delimiter".
CONTAM_PATTERNS = [
    re.compile(r">>>"),
    re.compile(r"<<<"),
    re.compile(r"between\s+markers", re.I),
    re.compile(r"highlight(ed)?\s+window", re.I),
    re.compile(r"window\s+markers", re.I),
    re.compile(r"special\s+markers", re.I),
    re.compile(r"marker\s+tokens", re.I),
    re.compile(r"between\s+the\s+markers", re.I),
    re.compile(r"delimited\s+contexts", re.I),
    re.compile(r"within\s+the\s+highlighted", re.I),
    re.compile(r"the\s+highlighted\s+(span|region|portion|context)", re.I),
    re.compile(r"angle\s+brackets", re.I),
    re.compile(r"enclosed\s+(in|by|within)\s+(triple|the)\s", re.I),
    re.compile(r"\[FOCUS\]", re.I),
    re.compile(r"\[/FOCUS\]", re.I),
]


def is_contaminated(explanation: str) -> bool:
    if not explanation:
        return False
    return any(p.search(explanation) for p in CONTAM_PATTERNS)


def rewrite_text_markers(s: str) -> str:
    """Replace legacy >>>/<<< markers with [FOCUS]/[/FOCUS]."""
    return s.replace(">>>", "[FOCUS]").replace("<<<", "[/FOCUS]")


async def reexplain(backend: ClaudeAPIBackend, rec: dict) -> dict:
    user = _build_user_prompt(rec)
    text = await backend.call(SYSTEM, user)
    if not text:
        return rec  # leave the old explanation if Haiku failed
    parsed = parse_response(text)
    rec["explanation"] = parsed["explanation"]
    rec["safety"] = parsed["safety"]
    rec["raw"] = parsed["raw"]
    return rec


def fix_arm(arm: str, backend: ClaudeAPIBackend) -> dict:
    path = SAFETY_DIR / "results" / "autointerp" / arm / "explanations.jsonl"
    print(f"\n=== {arm} ===")
    print(f"  reading {path}")
    recs = [json.loads(l) for l in open(path)]
    n_total = len(recs)

    # Step 1: globally rewrite top_texts markers in-place.
    n_rewrote_top_texts = 0
    for r in recs:
        new_top = [rewrite_text_markers(t) for t in r.get("top_texts", [])]
        if new_top != r.get("top_texts"):
            n_rewrote_top_texts += 1
        r["top_texts"] = new_top
    print(f"  rewrote markers in top_texts of {n_rewrote_top_texts}/{n_total} records")

    # Step 2: re-explain every contaminated record.
    contaminated_idx = [i for i, r in enumerate(recs)
                        if is_contaminated(r.get("explanation", ""))]
    print(f"  {len(contaminated_idx)}/{n_total} records flagged as contaminated; re-prompting…")

    if not contaminated_idx:
        with open(path, "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        return {"arm": arm, "rewrote": n_rewrote_top_texts, "reexplained": 0}

    async def go() -> None:
        sem = asyncio.Semaphore(MAX_CONCURRENT)

        async def one(i: int) -> None:
            async with sem:
                recs[i] = await reexplain(backend, recs[i])

        tasks = [asyncio.create_task(one(i)) for i in contaminated_idx]
        for fut in tqdm(asyncio.as_completed(tasks), total=len(tasks),
                        desc=f"reexplain {arm}"):
            await fut

    asyncio.run(go())

    # Re-check after the run — ideally zero contamination remains.
    still_contaminated = sum(1 for r in recs
                             if is_contaminated(r.get("explanation", "")))
    print(f"  after re-prompt: {still_contaminated}/{n_total} still flagged")

    with open(path, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {path}")

    return {"arm": arm, "rewrote": n_rewrote_top_texts,
            "reexplained": len(contaminated_idx),
            "still_contaminated": still_contaminated}


def main() -> None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY missing — source safety_research/.env")
    backend = ClaudeAPIBackend(model=EXPLAIN_MODEL,
                               max_concurrent=MAX_CONCURRENT,
                               max_retries=MAX_RETRIES)
    t0 = time.time()
    summaries = [fix_arm(arm, backend) for arm in ARMS]
    elapsed = time.time() - t0

    print(f"\nDONE in {elapsed:.0f}s "
          f"({backend.n_calls} successful calls, {backend.n_errors} retry hits)")
    for s in summaries:
        print(f"  {s['arm']:6s}  rewrote_top_texts={s['rewrote']}  "
              f"reexplained={s['reexplained']}  "
              f"still_contaminated={s.get('still_contaminated', '?')}")


if __name__ == "__main__":
    main()
