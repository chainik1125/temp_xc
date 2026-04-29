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

    # ------------------------------------------------------------------
    # Optional: load Sonnet coherence grades. If grade_coherence.py was run,
    # we merge the 0–3 coherence into both the per-generation rows and the
    # per-(mode, target, magnitude) summary so plot_backtracking.py can draw
    # the Pareto coherence-vs-backtracking curves.
    coh_path = INTERVENE_DIR / "coherence_grades.jsonl"
    coh: dict[tuple[str, str, float, str], int] = {}
    if coh_path.exists():
        with coh_path.open() as f:
            for line in f:
                r = json.loads(line)
                if r.get("coherence_grade") is None:
                    continue
                coh[
                    (r["mode"], r["target"], float(r["magnitude"]), r["prompt_id"])
                ] = int(r["coherence_grade"])
        print(f"[evaluate] loaded {len(coh)} coherence grades")
    else:
        print(f"[evaluate] (no coherence grades at {coh_path}; metric stays None)")

    per_gen_rows: list[dict] = []
    fracs_grouped: dict[tuple[str, str, float], list[float]] = defaultdict(list)
    coh_grouped: dict[tuple[str, str, float], list[int]] = defaultdict(list)

    with gens_path.open() as f:
        for line in f:
            rec = json.loads(line)
            frac, n_kw, n_tot = keyword_fraction(rec["generation"], keywords)
            key4 = (rec["mode"], rec["target"], float(rec["magnitude"]), rec["prompt_id"])
            coh_g = coh.get(key4)
            row = {
                "mode": rec["mode"],
                "target": rec["target"],
                "magnitude": rec["magnitude"],
                "prompt_id": rec["prompt_id"],
                "category": rec.get("category", ""),
                "n_keyword": n_kw,
                "n_total": n_tot,
                "keyword_fraction": frac,
                "coherence_grade": "" if coh_g is None else coh_g,
            }
            per_gen_rows.append(row)
            fracs_grouped[(rec["mode"], rec["target"], rec["magnitude"])].append(frac)
            if coh_g is not None:
                coh_grouped[(rec["mode"], rec["target"], rec["magnitude"])].append(coh_g)

    per_path = INTERVENE_DIR / "per_generation.csv"
    with per_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_gen_rows[0].keys()))
        w.writeheader()
        w.writerows(per_gen_rows)

    def _mean_sem(xs):
        n = len(xs)
        if n == 0:
            return float("nan"), float("nan"), 0
        m = sum(xs) / n
        if n > 1:
            v = sum((x - m) ** 2 for x in xs) / (n - 1)
            return m, math.sqrt(v / n), n
        return m, 0.0, n

    out_rows: list[dict] = []
    for k in fracs_grouped:
        m_kw, sem_kw, n = _mean_sem(fracs_grouped[k])
        m_co, sem_co, n_co = _mean_sem(coh_grouped.get(k, []))
        out_rows.append(
            {
                "mode": k[0],
                "target": k[1],
                "magnitude": k[2],
                "n_prompts": n,
                "mean_keyword_fraction": m_kw,
                "sem_keyword_fraction": sem_kw,
                "n_coherence_graded": n_co,
                "mean_coherence": "" if math.isnan(m_co) else m_co,
                "sem_coherence": "" if math.isnan(sem_co) else sem_co,
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
