"""Sample 20 transcripts for blind human scoring → judge κ validation.

Stratifies the sample so half are from near a per-arch peak magnitude
(where the judge has high counts of "genuine backtracking") and half are
from off-peak / control magnitudes — that way Aniket sees a mix of
behavior patterns when scoring blind.

Output CSV columns (human columns left empty for Aniket to fill):
  id, arch, magnitude, question_id, transcript,
  human_coherence_0_3,         # ← Aniket fills (0=incoherent, 3=fully coherent)
  human_backtracking_present,  # ← Aniket fills (0/1)
  human_looping_present,       # ← Aniket fills (0/1)

Then a separate validate-script will load the LLM judge scores for the
same (qid, arch, mag) rows and compute Cohen's κ + raw agreement.

Usage:
  python -m experiments.ward_backtracking_txc.build_judge_blind_csv \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --out <out_root>/judge_validation/blind_pairs.csv
"""
from __future__ import annotations
import argparse
import csv
import json
import logging
import random
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.judge_blind")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-samples", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    pool = []
    for run_dir in args.runs:
        meta_p = run_dir / "meta.json"
        rescue_p = run_dir / "phase2_rescue.json"
        if not (meta_p.exists() and rescue_p.exists()):
            continue
        meta = json.loads(meta_p.read_text())
        label = meta.get("label", "?")
        rows = json.loads(rescue_p.read_text())
        for r in rows:
            pool.append({
                "label": label,
                "cell_id": meta["cell_id"],
                "magnitude": float(r["magnitude"]),
                "question_id": r["unique_id"],
                "transcript": r.get("continuation_text", ""),
                "judge_rescued": bool(r["rescued_correct"]),
                "before_correct": bool(r.get("before_correct", False)),
            })
    log.info("[pool] %d rows across %d runs", len(pool), len(args.runs))

    # Stratify: half near each arch's peak |magnitude| in [3, 8], half from extremes (|mag| in {12, 16}).
    near_peak = [r for r in pool if 3 <= abs(r["magnitude"]) <= 8]
    extreme = [r for r in pool if abs(r["magnitude"]) >= 12]
    log.info("[strata] near_peak=%d  extreme=%d", len(near_peak), len(extreme))
    n_each = args.n_samples // 2
    sample = (rng.sample(near_peak, k=min(n_each, len(near_peak)))
              + rng.sample(extreme, k=min(args.n_samples - n_each, len(extreme))))
    rng.shuffle(sample)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "arch", "magnitude", "question_id",
                  "transcript",
                  "human_coherence_0_3",
                  "human_backtracking_present",
                  "human_looping_present",
                  # Filled in by validate script:
                  "judge_rescued", "before_correct"]
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i, r in enumerate(sample):
            w.writerow({
                "id": i,
                "arch": r["label"],
                "magnitude": r["magnitude"],
                "question_id": r["question_id"],
                "transcript": r["transcript"][:4000],   # truncate very long
                "human_coherence_0_3": "",
                "human_backtracking_present": "",
                "human_looping_present": "",
                "judge_rescued": int(r["judge_rescued"]),
                "before_correct": int(r["before_correct"]),
            })
    log.info("[saved] %s (%d rows)", args.out, len(sample))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
