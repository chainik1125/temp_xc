"""debug_normalize.py — figure out why the Ġ→space normalization fix
isn't landing.

Prints:
  1. Where src.bench.venhoff.responses is loaded from (live source or
     an installed-copy in site-packages that would explain why a
     git pull didn't update behaviour).
  2. Whether the `_normalize_byte_level_bpe` function exists.
  3. A direct call on a real trace: U+0120 count before/after, ASCII
     space count after, sample of the normalized output.
  4. Full chain: extract_thinking_process → split_into_sentences, with
     sentence count.

Usage (from repo root, inside the venv):
    python scripts/debug_normalize.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

TRACE_PATH = (
    _REPO_ROOT
    / "results/venhoff_eval"
    / "deepseek-r1-distill-llama-8b_math500-test_n500_L6_seed42"
    / "traces.json"
)


def main() -> int:
    # ── 1. Module provenance ──────────────────────────────────────
    import src.bench.venhoff.responses as r

    print("== module provenance ==")
    print(f"  source file:       {r.__file__}")
    print(f"  has _normalize:    {hasattr(r, '_normalize_byte_level_bpe')}")
    src_lines = inspect.getsource(r.extract_thinking_process).splitlines()
    print("  extract_thinking_process (first 6 lines):")
    for line in src_lines[:6]:
        print(f"    {line}")
    print()

    if not TRACE_PATH.exists():
        print(f"[error] no traces at {TRACE_PATH}")
        return 2

    d = json.load(TRACE_PATH.open())
    fr = d[0]["full_response"]

    # ── 2. Direct normalize test ──────────────────────────────────
    print("== direct _normalize_byte_level_bpe test ==")
    print(f"  before — U+0120 count: {fr.count(chr(0x120))}")
    print(f"  before — ASCII space count: {fr.count(' ')}")

    if not hasattr(r, "_normalize_byte_level_bpe"):
        print("  [skipping — function is not defined on the imported module]")
    else:
        normed = r._normalize_byte_level_bpe(fr)
        print(f"  after  — U+0120 count: {normed.count(chr(0x120))}")
        print(f"  after  — ASCII space count: {normed.count(' ')}")
        print(f"  after  — first 200 chars: {normed[:200]!r}")
    print()

    # ── 3. Full pipeline ──────────────────────────────────────────
    print("== full pipeline (extract → split) ==")
    extracted = r.extract_thinking_process(fr)
    print(f"  extracted — U+0120 count: {extracted.count(chr(0x120))}")
    print(f"  extracted — ASCII space count: {extracted.count(' ')}")
    print(f"  extracted — first 200 chars: {extracted[:200]!r}")

    from src.bench.venhoff.tokenization import split_into_sentences
    sentences = split_into_sentences(extracted)
    print(f"  split_into_sentences → {len(sentences)} sentences")
    for i, s in enumerate(sentences[:3]):
        print(f"    [{i}] {s[:80]!r}")
    print()

    # ── 4. Verdict ────────────────────────────────────────────────
    if extracted.count(chr(0x120)) > 0:
        print("[verdict] extracted text STILL contains U+0120 — normalization did not run")
        print("          likely the venv has a cached copy of responses.py;")
        print("          run: uv pip install -e . --reinstall && retry")
    elif len(sentences) == 0:
        print("[verdict] normalization worked but splitter returns 0 — splitter bug")
    else:
        print(f"[verdict] pipeline clean — {len(sentences)} sentences produced.")
        print("          safe to relaunch runpod_venhoff_paper_run.sh")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
