"""Anthropic Haiku 4.5 sentence-level taxonomy labelling for the 300 traces.

Per Ward §2.2 + Appendix C: an LLM judge classifies each sentence of
each reasoning trace as `backtracking` or `other`. Ward then build their
positive set D₊ from sentences flagged backtracking (and use a token
offset window of [-13, -8] *preceding* such a sentence as the activation
sample for the DoM vector — that is handled in collect_offsets.py).

Output schema:
  [
    {
      "question_id": "...",
      "trace_idx": 0,
      "sentences": [
        {"sentence": "...", "char_start": 0, "char_end": 42, "is_backtracking": false},
        ...
      ]
    },
    ...
  ]

Sentences come from the same splitter Venhoff used (split_into_sentences
in src/bench/venhoff/tokenization.py). Char offsets are kept so
collect_offsets.py can map sentences → token positions in a separate
forward pass (the existing char_to_token map is per-trace).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
from pathlib import Path

import yaml

from src.bench.venhoff.judge_client import AnthropicJudge
from src.bench.venhoff.responses import extract_thinking_process
from src.bench.venhoff.tokenization import split_into_sentences

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.label_sentences")


SYSTEM_PROMPT = (
    "You classify sentences from a reasoning model's chain-of-thought as "
    "'backtracking' or 'other'. A backtracking sentence is one where the "
    "reasoner reverses, abandons, doubts, or restarts a previously "
    "considered line of reasoning — typical lexical signals are 'wait', "
    "'hmm', 'but actually', 'no, that's not right', 'let me reconsider'. "
    "Pure progress sentences (deriving, computing, planning) are NOT "
    "backtracking. Output ONLY a JSON array of booleans, one per input "
    "sentence, in order. No commentary."
)


def _format_user(sentences: list[str]) -> str:
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return f"Classify each of the following {len(sentences)} sentences:\n\n{body}\n\nReturn JSON array of booleans."


def _parse_bools(reply: str, n: int) -> list[bool]:
    """Best-effort extract of a JSON list of booleans of length n."""
    if not reply:
        return [False] * n
    m = re.search(r"\[[^\[\]]*\]", reply, flags=re.S)
    if not m:
        return [False] * n
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [False] * n
    if not isinstance(arr, list):
        return [False] * n
    out = [bool(x) for x in arr][:n]
    if len(out) < n:
        out = out + [False] * (n - len(out))
    return out


async def _label_one_trace(
    judge: AnthropicJudge,
    trace_idx: int,
    trace: dict,
    min_words: int,
) -> dict:
    full_response = trace["full_response"]
    thinking = extract_thinking_process(full_response)
    if not thinking:
        return {
            "question_id": trace["question_id"],
            "trace_idx": trace_idx,
            "sentences": [],
        }
    sentences = split_into_sentences(thinking)
    sentences = [s for s in sentences if len(s.split()) >= min_words]
    if not sentences:
        return {"question_id": trace["question_id"], "trace_idx": trace_idx, "sentences": []}

    reply = await judge.call(SYSTEM_PROMPT, _format_user(sentences))
    flags = _parse_bools(reply, len(sentences))

    # Map each sentence back to its char position in the *thinking* string
    # so collect_offsets.py can later resolve the token offsets.
    sentence_records: list[dict] = []
    cursor = 0
    for sent, is_bt in zip(sentences, flags):
        idx = thinking.find(sent, cursor)
        if idx == -1:
            idx = thinking.find(sent)  # fall back to first occurrence
        char_start = idx if idx >= 0 else cursor
        char_end = char_start + len(sent)
        cursor = char_end
        sentence_records.append({
            "sentence": sent,
            "char_start": char_start,
            "char_end": char_end,
            "is_backtracking": bool(is_bt),
        })
    return {
        "question_id": trace["question_id"],
        "trace_idx": trace_idx,
        "sentences": sentence_records,
    }


async def _label_all(traces: list[dict], model: str, min_words: int, rpm: int, max_concurrent: int) -> list[dict]:
    judge = AnthropicJudge(model=model, max_tokens=1024, max_concurrent=max_concurrent, rpm=rpm)
    tasks = [_label_one_trace(judge, i, t, min_words) for i, t in enumerate(traces)]
    log.info("[info] labelling %d traces via %s (rpm=%d, concurrent=%d)", len(traces), model, rpm, max_concurrent)
    out = await asyncio.gather(*tasks)
    log.info("[info] anthropic calls=%d errors=%d", judge.n_calls, judge.n_errors)
    n_bt = sum(s["is_backtracking"] for r in out for s in r["sentences"])
    n_total = sum(len(r["sentences"]) for r in out)
    log.info("[info] backtracking sentences | %d / %d (%.1f%%)", n_bt, n_total, 100 * n_bt / max(1, n_total))
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--judge-model", default=None, help="override config.backtracking.judge_model")
    p.add_argument("--min-words", type=int, default=3, help="filter sentences shorter than this")
    p.add_argument("--rpm", type=int, default=200)
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    traces_path = Path(cfg["paths"]["traces"])
    out_path = Path(cfg["paths"]["sentence_labels"])

    if out_path.exists() and not args.force:
        log.info("[info] resume | %s exists — pass --force to regenerate", out_path)
        return 0

    traces = json.loads(traces_path.read_text())
    model = args.judge_model or cfg["backtracking"]["judge_model"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels = asyncio.run(_label_all(
        traces=traces,
        model=model,
        min_words=args.min_words,
        rpm=args.rpm,
        max_concurrent=args.max_concurrent,
    ))
    out_path.write_text(json.dumps(labels, indent=2))
    log.info("[done] saved labels | path=%s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
