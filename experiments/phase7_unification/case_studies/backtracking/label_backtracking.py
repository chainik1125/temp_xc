#!/usr/bin/env python3
"""Stage 1: keyword-based backtracking labels and contrastive sets D₊, D.

Reads `traces.jsonl` from Stage 0 and emits, per trace:

  * event_positions:    indices in full_token_ids of tokens whose decoded
                        form (lowercased, stripped of leading/trailing
                        whitespace and punctuation) equals one of KEYWORDS
                        (default {wait, hmm}).
  * d_plus_positions:   union over events of [event_pos + NEG_OFFSET_LO,
                        event_pos + NEG_OFFSET_HI] (i.e. 13..8 tokens
                        before each event), clipped into the think region.
  * d_all_positions:    every position in the `<think>` region (the natural
                        "all positions" denominator for the DoM step).

Writes `labels/labels.jsonl` and a small `labels/summary.json`. Pure CPU.

Run from repo root:

    TQDM_DISABLE=1 uv run python -m experiments.phase7_unification.case_studies.backtracking.label_backtracking
"""

from __future__ import annotations

import argparse
import json
import os
import string
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.data.nlp.models import get_model_config  # noqa: E402

from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    KEYWORDS,
    LABELS_DIR,
    NEG_OFFSET_HI,
    NEG_OFFSET_LO,
    SUBJECT_MODEL,
    TRACES_PATH,
    ensure_dirs,
)


_STRIP = string.punctuation + string.whitespace


def _norm_token(s: str) -> str:
    return s.strip(_STRIP).lower()


def _is_keyword(decoded: str, keywords: set[str]) -> bool:
    """A token counts as a backtracking marker iff, after stripping leading/
    trailing whitespace and punctuation and lowercasing, its decoded form
    is exactly one of the keywords. Excludes "waiting" / "Hmm…" wrapped in
    other text."""
    return _norm_token(decoded) in keywords


def _think_range(rec: dict) -> tuple[int, int]:
    """Return [lo, hi) over full_token_ids that we treat as the think region.

    If the model emitted both <think> and </think>, use them. If only <think>
    is present (truncated trace), use [open, len). If neither is present
    (model skipped the tag), use [input_len, len).
    """
    n = len(rec["full_token_ids"])
    open_pos = rec.get("think_open_pos")
    close_pos = rec.get("think_close_pos")
    input_len = rec.get("input_len", 0)
    if open_pos is not None and close_pos is not None and close_pos > open_pos:
        return (open_pos + 1, close_pos)
    if open_pos is not None:
        return (open_pos + 1, n)
    return (input_len, n)


def _build_labels_for_trace(
    rec: dict,
    decoded_tokens: list[str],
    keywords: set[str],
    off_lo: int,
    off_hi: int,
) -> dict:
    think_lo, think_hi = _think_range(rec)
    events: list[int] = []
    for pos in range(think_lo, think_hi):
        if _is_keyword(decoded_tokens[pos], keywords):
            events.append(pos)

    d_plus: set[int] = set()
    for ev in events:
        for off in range(off_lo, off_hi + 1):  # inclusive endpoints
            p = ev + off
            if think_lo <= p < think_hi:
                d_plus.add(p)
    d_all = list(range(think_lo, think_hi))
    return {
        "trace_id": rec["trace_id"],
        "category": rec.get("category", ""),
        "n_tokens": len(rec["full_token_ids"]),
        "think_lo": think_lo,
        "think_hi": think_hi,
        "event_positions": events,
        "n_events": len(events),
        "d_plus_positions": sorted(d_plus),
        "n_d_plus": len(d_plus),
        "n_d_all": len(d_all),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", nargs="+", default=list(KEYWORDS))
    parser.add_argument("--off-lo", type=int, default=NEG_OFFSET_LO)
    parser.add_argument("--off-hi", type=int, default=NEG_OFFSET_HI)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    ensure_dirs()
    out_path = LABELS_DIR / "labels.jsonl"
    summary_path = LABELS_DIR / "summary.json"
    if out_path.exists() and not args.force:
        print(f"[label_backtracking] {out_path} exists; use --force to rebuild")
        return

    if not TRACES_PATH.exists():
        raise SystemExit(f"missing {TRACES_PATH}; run Stage 0 first")

    from transformers import AutoTokenizer

    cfg = get_model_config(SUBJECT_MODEL)
    print(f"[label_backtracking] loading tokenizer for {cfg.hf_path}")
    tok = AutoTokenizer.from_pretrained(cfg.tokenizer)

    keywords = {k.lower() for k in args.keywords}
    n_traces = 0
    n_events_total = 0
    cat_events: Counter[str] = Counter()
    cat_tokens: Counter[str] = Counter()

    with TRACES_PATH.open() as fin, out_path.open("w") as fout:
        for line in fin:
            rec = json.loads(line)
            full_ids = rec["full_token_ids"]
            decoded = [tok.decode([tid], skip_special_tokens=False) for tid in full_ids]
            label = _build_labels_for_trace(
                rec, decoded, keywords, args.off_lo, args.off_hi
            )
            fout.write(json.dumps(label) + "\n")
            n_traces += 1
            n_events_total += label["n_events"]
            cat_events[label["category"]] += label["n_events"]
            cat_tokens[label["category"]] += label["n_d_all"]

    summary = {
        "n_traces": n_traces,
        "keywords": sorted(keywords),
        "off_lo": args.off_lo,
        "off_hi": args.off_hi,
        "n_events_total": n_events_total,
        "events_per_category": dict(cat_events),
        "think_tokens_per_category": dict(cat_tokens),
        "rate_per_category": {
            c: (cat_events[c] / cat_tokens[c] if cat_tokens[c] else 0.0)
            for c in cat_tokens
        },
    }
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"[label_backtracking] wrote {out_path}")
    print(f"[label_backtracking] {n_traces} traces, {n_events_total} events total")
    for c, r in summary["rate_per_category"].items():
        print(f"  {c:<14} events={cat_events[c]:>4}  tokens={cat_tokens[c]:>6}  rate={r:.4f}")


if __name__ == "__main__":
    main()
