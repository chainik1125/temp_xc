"""Backtracking-inducement metric for the B3 sweep (Dmitry pivot).

Replaces math-correctness as the headline metric. For each (arch, mag,
question), we count backtracking emissions in the steered continuation:

1. **Cheap proxy**: keyword_rate = count("wait"|"hmm" word boundaries) / n_words
   — same regex as B1 (b1_steer_eval.py:KEYWORD_RE). Free; ~10 seconds total.
2. **Gold standard**: Sonnet 4.6 `genuine_count` (catches real backtracking
   events, excludes filler / pseudo-backtracking / loops). Reuses the
   judge prompt from grade_backtracking.py. Cost ~$25 for 9150 calls.

Then baseline-corrects both vs mag=0 (the same noise-floor concept as
build_steering_effect.py — at mag=0 the steering hook is a no-op so
the cut-and-continue continuation has its own intrinsic backtracking
rate, which we subtract out). Reports:

- Δ keyword_rate vs baseline, per (arch, mag, question)
- Δ genuine_count vs baseline, per (arch, mag, question)
- Per-arch peak Δ + the magnitude range over which Δ stays positive ("stability")
- Normalized: Δ genuine_count / (keyword_rate at that mag) — "backtracking
  per unit of steering effort"

Usage:
  # Cheap proxy only:
  python -m experiments.ward_backtracking_txc.build_inducement \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --out <out_root>

  # With Sonnet judge:
  python -m experiments.ward_backtracking_txc.build_inducement \
      --runs <out_root>/<cell>__f<id>_<mode> [...] \
      --out <out_root> \
      --judge-with-sonnet
"""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.inducement")


KEYWORD_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)


def keyword_rate(text: str) -> tuple[float, int, int]:
    """Return (rate, kw_count, n_words). Same as B1's metric."""
    if not text:
        return 0.0, 0, 0
    n_words = len(text.split())
    if n_words == 0:
        return 0.0, 0, 0
    kw = len(KEYWORD_RE.findall(text))
    return kw / n_words, kw, n_words


JUDGE_PROMPT = """You are evaluating a math problem-solving generation for genuine backtracking behaviour.

Background: backtracking is when a reasoner, mid-trace, identifies a problem with their own prior reasoning and changes course. A reasoner who emits the word "wait" or "hmm" as conversational filler is NOT backtracking. A reasoner who says "wait, no, actually..." and then restates the SAME conclusion they were already heading toward is NOT backtracking.

Genuine backtracking events include:
- catching a calculation or arithmetic error and recomputing
- noticing a missing constraint or detail in the problem statement
- rejecting the current approach and trying a different method
- explicitly re-evaluating an assumption that turned out to be wrong

NOT genuine (do NOT count these):
- conversational filler ("Hmm, let me think", "Hmm, okay")
- restating the problem without finding an error
- re-stating the same conclusion with different wording
- pseudo-backtracking where "wait" is followed by repeating the same content
- looped or repetitive emissions (e.g., "Wait, I'm not. Wait, I'm not.")
- gibberish, single-token loops, or non-English degeneration

Problem prompt the model was solving:
{prompt_text}

Model's generation:
\"\"\"
{generation}
\"\"\"

Count the number of GENUINE backtracking events in this generation. Reply with EXACTLY this format on two lines:

COUNT: <integer>
NOTES: <one short sentence explaining your count>

Do not output anything else."""


async def judge_one(client, prompt_text: str, generation: str,
                    model_id: str = "claude-sonnet-4-6") -> dict:
    msg = await client.messages.create(
        model=model_id, max_tokens=120,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(prompt_text=prompt_text[:1500],
                                            generation=generation[:6000]),
        }],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"COUNT:\s*(\d+)", raw)
    return {"genuine_count": int(m.group(1)) if m else -1, "raw": raw}


async def grade_with_sonnet(rows: list[dict], qid_to_problem: dict[str, str],
                             cache_path: Path, concurrency: int = 12) -> None:
    """In-place: add `genuine_count` to each row. Resumable via cache_path
    (a JSON file mapping (arch, mag, qid) → judgement)."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise SystemExit("uv add anthropic")
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    cache = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        log.info("[resume] %d cached judgements", len(cache))

    sem = asyncio.Semaphore(concurrency)
    n_done = 0
    save_every = 200

    async def worker(i: int, r: dict):
        nonlocal n_done
        key = f"{r['arch']}|{r['magnitude']}|{r['question_id']}"
        if key in cache:
            r["genuine_count"] = cache[key]["genuine_count"]
            return
        async with sem:
            text = r.get("continuation_text") or ""
            if not text.strip():
                r["genuine_count"] = 0
                cache[key] = {"genuine_count": 0, "raw": "(empty)"}
                return
            prompt = qid_to_problem.get(r["question_id"], "(unknown)")
            try:
                j = await judge_one(client, prompt, text)
            except Exception as e:
                log.warning("judge failed %s: %s", key, e)
                j = {"genuine_count": -1, "raw": f"ERROR: {e}"}
            r["genuine_count"] = j["genuine_count"]
            cache[key] = j
            n_done += 1
            if n_done % save_every == 0:
                cache_path.write_text(json.dumps(cache))
                log.info("[checkpoint] %d new judgements (total cache %d)",
                         n_done, len(cache))

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])
    cache_path.write_text(json.dumps(cache))
    log.info("[saved] %s (%d entries)", cache_path, len(cache))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--judge-with-sonnet", action="store_true",
                   help="ALSO grade each row's continuation with Sonnet for "
                        "`genuine_count` (~$25 for 9150 rows; resumable cache).")
    p.add_argument("--judge-cache", type=Path, default=None,
                   help="cache file for resuming Sonnet judgements")
    p.add_argument("--concurrency", type=int, default=12)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    # Load all (arch, mag, qid) rows w/ continuation_text from each run dir
    rows = []
    for run_dir in args.runs:
        meta_p = run_dir / "meta.json"; rescue_p = run_dir / "phase2_rescue.json"
        if not (meta_p.exists() and rescue_p.exists()): continue
        meta = json.loads(meta_p.read_text())
        for r in json.loads(rescue_p.read_text()):
            rate, kw, nw = keyword_rate(r.get("continuation_text", ""))
            rows.append({
                "arch": meta["label"],
                "cell_id": meta["cell_id"],
                "feature_id": int(meta["feature_id"]),
                "feature_mode": meta["feature_mode"],
                "magnitude": float(r["magnitude"]),
                "question_id": r["unique_id"],
                "before_correct": bool(r.get("before_correct", False)),
                "rescued_correct": bool(r["rescued_correct"]),
                "n_words": nw,
                "kw_count": kw,
                "keyword_rate": rate,
                "continuation_text": r.get("continuation_text", "")[:8000],
            })
    log.info("[in] %d rows; archs=%s; mags=%s",
             len(rows), sorted({r["arch"] for r in rows}),
             sorted({r["magnitude"] for r in rows}))

    if args.judge_with_sonnet:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        qid_to_problem = {r["unique_id"]: r["problem"] for r in ds}
        cache_path = args.judge_cache or (args.out / "inducement_judge_cache.json")
        asyncio.run(grade_with_sonnet(rows, qid_to_problem, cache_path,
                                       concurrency=args.concurrency))
    else:
        # Lightweight: only keyword_rate is computed
        for r in rows:
            r["genuine_count"] = None

    # Drop continuation_text before writing (heavy)
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "continuation_text"} for r in rows])

    # Baseline-correct: subtract per (arch, qid) the mag=0 value
    base = df[df["magnitude"] == 0.0][["arch", "question_id", "keyword_rate", "genuine_count"]].rename(
        columns={"keyword_rate": "kw_baseline", "genuine_count": "gc_baseline"}
    )
    merged = df.merge(base, on=["arch", "question_id"], how="left")
    merged["delta_keyword_rate"] = merged["keyword_rate"] - merged["kw_baseline"]
    if args.judge_with_sonnet:
        merged["delta_genuine_count"] = merged["genuine_count"] - merged["gc_baseline"]

    out_pq = args.out / "inducement.parquet"
    merged.to_parquet(out_pq, compression="snappy")
    log.info("[saved] %s", out_pq)

    # Per (arch, mag) summary: mean Δ + stability indicator
    agg_cols = {"keyword_rate": "mean", "delta_keyword_rate": "mean"}
    if args.judge_with_sonnet:
        agg_cols.update({"genuine_count": "mean", "delta_genuine_count": "mean"})
    summary = merged.groupby(["arch", "magnitude"]).agg(agg_cols).reset_index()
    summary_path = args.out / "inducement_summary.csv"
    summary.to_csv(summary_path, index=False)
    log.info("[saved] %s", summary_path)

    # Stability: per arch, the number of magnitudes (excluding mag=0) with
    # delta_keyword_rate > 0; AUC of positive Δ; per-arch best mag
    print("\n=== inducement summary (Δ vs mag=0 baseline) ===")
    for arch, g in summary.groupby("arch"):
        nonzero = g[g["magnitude"] != 0]
        n_pos_kw = (nonzero["delta_keyword_rate"] > 0).sum()
        peak_kw_mag = nonzero.loc[nonzero["delta_keyword_rate"].idxmax(), "magnitude"]
        peak_kw_val = nonzero["delta_keyword_rate"].max()
        line = (f"  {arch:>12s}: Δkw stable@{n_pos_kw}/24 mags  "
                f"peak Δkw={peak_kw_val:+.5f} @ mag={peak_kw_mag:+.1f}")
        if args.judge_with_sonnet:
            n_pos_gc = (nonzero["delta_genuine_count"] > 0).sum()
            peak_gc_mag = nonzero.loc[nonzero["delta_genuine_count"].idxmax(), "magnitude"]
            peak_gc_val = nonzero["delta_genuine_count"].max()
            line += (f"  ||  Δgc stable@{n_pos_gc}/24 mags  "
                     f"peak Δgc={peak_gc_val:+.3f} @ mag={peak_gc_mag:+.1f}")
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
