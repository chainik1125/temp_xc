#!/usr/bin/env python3
"""Stage 5a: keyword-fraction metric per (mode, target, magnitude).

Implements paper Eq. 1 with B = {wait, hmm}: the fraction of words in a
generated continuation that match the keyword set, where a word is counted
case-insensitively after stripping leading/trailing whitespace and punctuation.

Reads `intervene/generations.jsonl`, writes:

    intervene/keyword_rates.csv      one row per (mode, target, magnitude) with
                                      mean / sem of keyword fraction over prompts
    intervene/per_generation.csv     keyword fraction per individual generation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import math

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._decode import (  # noqa: E402
    clean_decode,
    norm_token,
)
from experiments.phase7_unification.case_studies.backtracking._paths import (  # noqa: E402
    INTERVENE_DIR,
    KEYWORDS,
    ensure_dirs,
)


def keyword_fraction(text: str, keywords: set[str]) -> tuple[float, int, int]:
    """Return (fraction, n_keyword, n_total) over whitespace-split words.

    `clean_decode` is applied first so any residual byte-level BPE glyphs
    (Ġ for space, Ċ for newline) from upstream stages get normalised before
    splitting on whitespace.
    """
    words = clean_decode(text).split()
    if not words:
        return 0.0, 0, 0
    n_kw = sum(1 for w in words if norm_token(w) in keywords)
    return n_kw / len(words), n_kw, len(words)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keywords", nargs="+", default=list(KEYWORDS))
    args = parser.parse_args()

    ensure_dirs()
    gens_path = INTERVENE_DIR / "generations.jsonl"
    if not gens_path.exists():
        raise SystemExit(f"missing {gens_path}; run Stage 4 first")

    keywords = {k.lower() for k in args.keywords}

    per_gen_rows: list[dict] = []
    grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)

    with gens_path.open() as f:
        for line in f:
            rec = json.loads(line)
            frac, n_kw, n_tot = keyword_fraction(rec["generation"], keywords)
            row = {
                "mode": rec["mode"],
                "target": rec["target"],
                "magnitude": rec["magnitude"],
                "prompt_id": rec["prompt_id"],
                "category": rec.get("category", ""),
                "n_keyword": n_kw,
                "n_total": n_tot,
                "keyword_fraction": frac,
            }
            per_gen_rows.append(row)
            grouped[(rec["mode"], rec["target"], rec["magnitude"])].append(frac)

    per_path = INTERVENE_DIR / "per_generation.csv"
    with per_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gen_rows[0].keys()))
        w.writeheader()
        w.writerows(per_gen_rows)

    out_rows: list[dict] = []
    for (mode, target, magnitude), fracs in grouped.items():
        n = len(fracs)
        mean = sum(fracs) / n
        if n > 1:
            var = sum((x - mean) ** 2 for x in fracs) / (n - 1)
            sem = math.sqrt(var / n)
        else:
            sem = 0.0
        out_rows.append(
            {
                "mode": mode,
                "target": target,
                "magnitude": magnitude,
                "n_prompts": n,
                "mean_keyword_fraction": mean,
                "sem": sem,
            }
        )
    out_rows.sort(key=lambda r: (r["mode"], r["target"], r["magnitude"]))

    out_path = INTERVENE_DIR / "keyword_rates.csv"
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"[evaluate] wrote {out_path} ({len(out_rows)} rows) + {per_path}")


if __name__ == "__main__":
    main()
