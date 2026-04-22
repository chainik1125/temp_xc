"""inspect_traces.py — diagnose why Phase 1 (collect_activations) is failing.

Usage (from repo root, inside the .venv):
    python scripts/inspect_traces.py

Or point at a different traces.json:
    python scripts/inspect_traces.py path/to/traces.json

Reports:
  - file type + size, number of traces, keys present
  - full_response and thinking_process field lengths + previews
  - whether <think>...</think> tags are present
  - what extract_thinking_process() returns
  - what split_into_sentences() returns
  - final verdict: regenerate Phase 0 | fix extract | fix splitter | schema mismatch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the repo root is on sys.path so `from src.bench...` works even
# when the script is invoked without PYTHONPATH set.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_PATH = Path(
    "results/venhoff_eval/"
    "deepseek-r1-distill-llama-8b_math500-test_n500_L6_seed42/"
    "traces.json"
)


def main(path: Path) -> int:
    if not path.exists():
        print(f"[error] no such file: {path}")
        return 2

    size_mb = path.stat().st_size / 1024 / 1024
    print(f"file:      {path}")
    print(f"size:      {size_mb:.2f} MB")

    with path.open() as f:
        d = json.load(f)

    print(f"json type: {type(d).__name__}")

    if not isinstance(d, list):
        print(f"[verdict] unexpected JSON root type: {type(d).__name__}")
        return 0

    print(f"n_traces:  {len(d)}")
    if not d:
        print("[verdict] traces list is EMPTY — regenerate Phase 0")
        return 0

    first = d[0]
    print(f"keys[0]:   {list(first.keys())}")
    print()

    # ── full_response diagnostics ──────────────────────────────────
    fr = first.get("full_response", "")
    if not isinstance(fr, str):
        print(f"[warn] full_response is {type(fr).__name__}, not str")
        fr = ""
    n_open = fr.count("<think>")
    n_close = fr.count("</think>")
    print(f"full_response[0]: len={len(fr)} chars")
    print(f"  <think> count:  {n_open}")
    print(f"  </think> count: {n_close}")
    print(f"  first 300 chars: {fr[:300]!r}")
    print(f"  last 200 chars:  {fr[-200:]!r}")
    # Count whitespace vs SentencePiece markers vs underscores.
    print("  character composition:")
    print(f"    ASCII space (0x20):          {fr.count(chr(0x20))}")
    print(f"    tab (0x09):                  {fr.count(chr(0x09))}")
    print(f"    newline (0x0a):              {fr.count(chr(0x0a))}")
    print(f"    non-breaking space (0xa0):   {fr.count(chr(0xa0))}")
    print(f"    literal underscore '_':      {fr.count('_')}")
    print(f"    SP marker U+2581 '▁':        {fr.count(chr(0x2581))}")
    print(f"  whitespace-split token count:  {len(fr.split())}")
    print()

    # ── thinking_process diagnostics ───────────────────────────────
    tp = first.get("thinking_process", "")
    if isinstance(tp, str):
        print(f"thinking_process[0]: len={len(tp)} chars")
        print(f"  first 300 chars: {tp[:300]!r}")
        print()

    # ── extract + split pipeline ───────────────────────────────────
    try:
        from src.bench.venhoff.responses import extract_thinking_process
        from src.bench.venhoff.tokenization import (
            split_into_sentences,
            sentence_token_span,
            get_char_to_token_map,
        )
    except ImportError as e:
        print(f"[warn] cannot import Phase 1 helpers ({e}); install/activate venv first")
        return 1

    extracted = extract_thinking_process(fr)
    print(f"extract_thinking_process(full_response): len={len(extracted)}")
    print(f"  first 300 chars: {extracted[:300]!r}")
    print()

    sentences = split_into_sentences(extracted)
    print(f"split_into_sentences(extracted): {len(sentences)} sentences")
    for i, s in enumerate(sentences[:3]):
        print(f"  [{i}] {s[:80]!r}")
    print()

    # ── test sentence_token_span on the first few sentences ───────
    # This is what Phase 1 uses downstream — if spans are None, Phase 1
    # collects no activations.
    print("sentence_token_span trial (uses HF tokenizer — loads on demand):")
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
        )
        char_to_token = get_char_to_token_map(fr, tokenizer)
        print(f"  char_to_token len: {len(char_to_token)} (full_response chars: {len(fr)})")
        span_hits = 0
        for i, s in enumerate(sentences[:5]):
            span = sentence_token_span(s, fr, char_to_token)
            if span is None:
                print(f"  [{i}] span=None  sentence={s[:60]!r}")
            else:
                span_hits += 1
                print(f"  [{i}] span={span}  sentence={s[:60]!r}")
        print(f"  {span_hits}/5 sentences produced a non-None span")
    except Exception as e:
        print(f"  [warn] could not run token-span check: {e}")
    print()

    # ── aggregate across all traces ────────────────────────────────
    empty_fr = sum(1 for t in d if not t.get("full_response"))
    empty_tp = sum(1 for t in d if not t.get("thinking_process"))
    no_think_tags = sum(
        1 for t in d
        if isinstance(t.get("full_response"), str) and "<think>" not in t["full_response"]
    )
    print(f"across all {len(d)} traces:")
    print(f"  empty full_response:    {empty_fr}")
    print(f"  empty thinking_process: {empty_tp}")
    print(f"  missing <think> tag:    {no_think_tags}")
    print()

    # Sample 20 traces through the extract+split pipeline
    sent_counts = []
    for t in d[:20]:
        fr_t = t.get("full_response", "")
        if not isinstance(fr_t, str):
            sent_counts.append(0)
            continue
        ex = extract_thinking_process(fr_t)
        ss = split_into_sentences(ex)
        sent_counts.append(len(ss))
    import statistics
    if sent_counts:
        print(f"pipeline sentences / first 20 traces:")
        print(f"  min={min(sent_counts)} median={int(statistics.median(sent_counts))} max={max(sent_counts)}")
        zero = sum(1 for c in sent_counts if c == 0)
        print(f"  traces with 0 sentences: {zero}/20")

    print()
    # ── verdict ────────────────────────────────────────────────────
    if empty_fr == len(d):
        print("[verdict] every full_response is empty → regenerate Phase 0")
    elif empty_fr > 0:
        print(f"[verdict] {empty_fr}/{len(d)} traces have empty full_response → partial regenerate")
    elif no_think_tags == len(d) and sent_counts and max(sent_counts) > 0:
        print("[verdict] no <think> tags anywhere, but splitter still produces sentences")
        print("          → extract_thinking_process falls through to full text, should work;"
              " inspect why activation_collection raises")
    elif no_think_tags == len(d):
        print("[verdict] no <think> tags AND splitter returns 0 → splitter is hitting empty text")
        print("          → either fix extract_thinking_process to use thinking_process field,"
              " or debug the splitter")
    elif sent_counts and max(sent_counts) == 0:
        print("[verdict] extract succeeds but splitter returns 0 sentences → splitter is broken")
    else:
        print("[verdict] pipeline works on sample traces → something else in Phase 1 is failing")
        print("          Check activation_collection.py:240 for additional filters / token spans")

    return 0


if __name__ == "__main__":
    arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PATH
    raise SystemExit(main(arg))
