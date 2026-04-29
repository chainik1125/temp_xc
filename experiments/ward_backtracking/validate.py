"""LLM-judge × keyword-judge agreement on steered outputs.

Reads the steering eval output (steering_results.json) and re-judges a
sample of the *generated* completions at each magnitude in
config.validation.strengths via Anthropic Haiku 4.5 sentence-level labelling. Reports
precision / recall / F1 of the keyword judge against the LLM judge as a
sanity check on the metric Ward et al. use throughout (and that we use
in plot.py).

Why this exists: the keyword judge ("wait" + "hmm" word rate) is cheap
but conservative. A reviewer asking "is this really measuring
backtracking?" deserves a number. Ward Appendix C reports F1 ≈ 60% at
intermediate strengths — we should be in that ballpark.

Output (validation.json):
  {
    "per_strength": [
      {
        "strength": 4.0,
        "n_traces": 50,
        "n_sentences_total": 1234,
        "n_keyword_pos": 87, "n_llm_pos": 102, "n_both_pos": 71,
        "precision": 0.816, "recall": 0.696, "f1": 0.751
      },
      ...
    ]
  }
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
from src.bench.venhoff.tokenization import split_into_sentences

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.validate")

KEYWORD_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)

SYSTEM_PROMPT = (
    "You classify sentences from a reasoning model's output as "
    "'backtracking' or 'other'. A backtracking sentence reverses, "
    "abandons, doubts, or restarts a previously considered line of "
    "reasoning. Output ONLY a JSON array of booleans, one per input "
    "sentence, in order. No commentary."
)


def _format_user(sentences: list[str]) -> str:
    body = "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))
    return f"Classify each of the following {len(sentences)} sentences:\n\n{body}\n\nReturn JSON array of booleans."


def _parse_bools(reply: str, n: int) -> list[bool]:
    if not reply:
        return [False] * n
    m = re.search(r"\[[^\[\]]*\]", reply, flags=re.S)
    if not m:
        return [False] * n
    try:
        arr = json.loads(m.group(0))
    except json.JSONDecodeError:
        return [False] * n
    out = [bool(x) for x in arr][:n]
    if len(out) < n:
        out += [False] * (n - len(out))
    return out


async def _judge_one_text(judge: AnthropicJudge, text: str, min_words: int) -> tuple[list[str], list[bool]]:
    sentences = [s for s in split_into_sentences(text) if len(s.split()) >= min_words]
    if not sentences:
        return [], []
    reply = await judge.call(SYSTEM_PROMPT, _format_user(sentences))
    return sentences, _parse_bools(reply, len(sentences))


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


async def _validate_strength(
    judge: AnthropicJudge,
    rows: list[dict],
    strength: float,
    n_traces: int,
    min_words: int,
) -> dict:
    # Pick the highest-information rows: prefer the reasoning model under
    # base-derived steering, since that's where Ward expects the effect.
    cells = [
        r for r in rows
        if r["target"] == "reasoning"
        and r["source"] == "base_derived_union"
        and r["magnitude"] == strength
    ]
    if not cells:
        log.warning("[warn] no rows at strength=%.1f for reasoning+base_derived", strength)
        return {"strength": strength, "n_traces": 0, "f1": 0.0}
    cells = cells[:n_traces]

    tasks = [_judge_one_text(judge, c["text"], min_words) for c in cells]
    judged = await asyncio.gather(*tasks)

    n_kw_pos = n_llm_pos = n_both = n_total = 0
    for sentences, llm_flags in judged:
        for sent, llm_flag in zip(sentences, llm_flags):
            kw_flag = bool(KEYWORD_RE.search(sent))
            n_total += 1
            n_kw_pos += int(kw_flag)
            n_llm_pos += int(llm_flag)
            n_both += int(kw_flag and llm_flag)

    precision = n_both / n_kw_pos if n_kw_pos else 0.0
    recall = n_both / n_llm_pos if n_llm_pos else 0.0
    f1 = _f1(precision, recall)
    log.info(
        "[strength=%5.1f] n=%d sents=%d kw+=%d llm+=%d both=%d | P=%.3f R=%.3f F1=%.3f",
        strength, len(cells), n_total, n_kw_pos, n_llm_pos, n_both, precision, recall, f1,
    )
    return {
        "strength": strength,
        "n_traces": len(cells),
        "n_sentences_total": n_total,
        "n_keyword_pos": n_kw_pos,
        "n_llm_pos": n_llm_pos,
        "n_both_pos": n_both,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--rpm", type=int, default=200)
    p.add_argument("--max-concurrent", type=int, default=10)
    p.add_argument("--min-words", type=int, default=3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    val_cfg = cfg["validation"]
    strengths = [float(s) for s in val_cfg["strengths"]]
    n_traces = int(val_cfg["n_traces"])

    steering_path = Path(cfg["paths"]["steering"])
    if not steering_path.exists():
        raise FileNotFoundError(f"{steering_path} missing — run steer_eval.py first")

    out_path = steering_path.parent / "validation.json"
    if out_path.exists() and not args.force:
        log.info("[info] resume | %s exists", out_path)
        return 0

    payload = json.loads(steering_path.read_text())
    rows = payload["rows"]

    judge = AnthropicJudge(model=cfg["backtracking"]["judge_model"], max_tokens=1024,
                        max_concurrent=args.max_concurrent, rpm=args.rpm)

    async def _all() -> list[dict]:
        return [
            await _validate_strength(judge, rows, s, n_traces, args.min_words)
            for s in strengths
        ]

    results = asyncio.run(_all())
    log.info("[info] anthropic calls=%d errors=%d", judge.n_calls, judge.n_errors)

    out_path.write_text(json.dumps({"per_strength": results}, indent=2))
    log.info("[done] saved validation | path=%s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
