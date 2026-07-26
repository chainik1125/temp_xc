"""Score every generation in the dose sweep with the EM coherence rubric.

COHERENCE ONLY. Alignment is the EM paper's harmful/helpful/honest axis and has no bearing
on which of two formatting instructions the model obeyed; paying for a column that cannot
inform the question is the wrong trade, and an unused column in a results file is one more
thing a later reader can mistake for evidence.

FIDELITY. The reference `judge_single()` makes two calls (alignment, then coherence), so
coherence-only cannot call it directly. This module therefore re-implements ONE function --
the reference's inner `call_judge` -- and imports everything that defines the measurement
from the reference module: `COHERENCE_PROMPT` and `ALIGNMENT_SYSTEM`, at `temperature=0`,
`max_tokens=10`, `model="gpt-4o"`, with the same `\\d+` extraction, the same 0-100 clamp,
and the same REFUSAL/CODE handling.

  * Note the reference uses ALIGNMENT_SYSTEM as the system message for BOTH prompts. That
    reads like an oversight, but it is what produced the published numbers, so it is
    reproduced deliberately rather than corrected. A threshold is only comparable if
    everything behind it matches, including the parts that look wrong.
  * `--validate N` checks this: it scores N generations BOTH ways -- through the reference
    `judge_single` and through this path -- and asserts the coherence scores are identical.
    Run it before trusting a batch. Costs about a cent.

PREFLIGHT IS NOT CEREMONY. `judge_single` SOFT-FAILS: a missing package, a bad key, or a
network error all return `{"coherence_score": None}` rather than raising. The pilot's first
run completed 40 iterations, saved a results file, and would have reported "0% clear the
floor" from ZERO API calls -- which was the outcome that had been predicted, and would
therefore have been believed. Every entry point here scores a known-good greeting first and
refuses to spend if it comes back unscored.

IDEMPOTENT. A row that already carries `judge_coherence` is skipped, so a rerun after an
interruption costs only the remainder.

    OPENAI_API_KEY_MATS=... python scripts/judge_gensweep.py [--validate 8] [--limit N]
"""
import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import statistics
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
FRA = pathlib.Path.home() / "Documents" / "Research" / "FRA" / "fra_proj"
SRC = ROOT / "results" / "txc_wins" / "recency_tr_gensweep.json"
MODEL = "gpt-4o"
GOOD = "Hello! How can I help you today?"


def _load_reference():
    if not (FRA / "fra" / "gpt4o_judge.py").exists():
        sys.exit(f"[fail] reference judge not found at {FRA}")
    sys.path.insert(0, str(FRA))
    from fra.gpt4o_judge import COHERENCE_PROMPT, ALIGNMENT_SYSTEM, judge_single
    assert "contains no hallucinations" in COHERENCE_PROMPT, "coherence rubric changed"
    return COHERENCE_PROMPT, ALIGNMENT_SYSTEM, judge_single


def make_scorer(key, coherence_prompt, system):
    """One coherence call, mirroring the reference's `call_judge` exactly."""
    from openai import OpenAI
    client = OpenAI(api_key=key)

    def score(response_text, retries=4):
        prompt = coherence_prompt.format(question="Say hello.", response=response_text)
        for attempt in range(retries):
            try:
                c = client.chat.completions.create(
                    model=MODEL,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": prompt}],
                    temperature=0, max_tokens=10)
                raw = (c.choices[0].message.content or "").strip()
                if raw in ("REFUSAL", "CODE"):
                    return None, raw
                m = re.search(r"\d+", raw)
                return (min(100, max(0, int(m.group()))), raw) if m else (None, raw)
            except Exception as e:  # transient: rate limit, timeout, 5xx
                if attempt == retries - 1:
                    return None, f"error: {e}"
                time.sleep(2 ** attempt)
        return None, "error: exhausted"

    return score


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--validate", type=int, default=0,
                    help="score N generations both ways and assert equality")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()

    key = os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY")
    if not key:
        sys.exit("[fail] set OPENAI_API_KEY_MATS")
    src = pathlib.Path(a.src)
    if not src.exists():
        sys.exit(f"[skip] {src} not written yet")

    coh_prompt, system, judge_single = _load_reference()
    score = make_scorer(key, coh_prompt, system)

    probe, raw = score(GOOD)
    if probe is None:
        sys.exit(f"[fail] judge returned no score on a known-good probe: {raw!r}")
    print(f"[preflight] judge live; plain greeting scores {probe}/100")

    d = json.loads(src.read_text())
    gens, cells = d.get("generations") or {}, d.get("gen_sweep") or []
    rows = [r for arm in gens.values() for r in arm]
    print(f"[load] {src.name}: {len(cells)} cells, {len(rows)} generations")

    # ---- fidelity check against the reference, before trusting any batch ----
    if a.validate:
        print(f"\n[validate] scoring {a.validate} generations BOTH ways")
        bad = 0
        for r in rows[:a.validate]:
            mine, _ = score(r["text"])
            theirs = judge_single("Say hello.", r["text"], api_key=key)["coherence_score"]
            ok = mine == theirs
            bad += not ok
            print(f"    reference {str(theirs):>4}   this path {str(mine):>4}   "
                  f"{'ok' if ok else 'MISMATCH'}")
        if bad:
            sys.exit(f"[fail] {bad}/{a.validate} disagree — coherence-only path is NOT "
                     f"faithful to the reference; do not use it")
        print(f"[validate] {a.validate}/{a.validate} identical — path is faithful\n")

    pending = [r for r in rows if r.get("judge_coherence") is None]
    todo = pending[:a.limit] if a.limit else pending
    # Report the three counts separately. Folding "excluded by --limit" into "already
    # done" is the kind of label that reads as a completed run when nothing ran.
    print(f"[judge] {len(rows)} generations: {len(rows) - len(pending)} already scored, "
          f"{len(pending)} pending, scoring {len(todo)} now"
          + (f" (--limit {a.limit})" if a.limit else ""))
    if not todo:
        return 0

    done = 0
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r, (sc, raw) in zip(todo, ex.map(lambda x: score(x["text"]), todo)):
            r["judge_coherence"], r["judge_raw"] = sc, raw
            done += 1
            if done % 250 == 0:
                print(f"   [judge] {done}/{len(todo)}", flush=True)

    # ---- roll up per cell, keyed by (arm, alpha) ----
    by = {}
    for arm, arm_rows in gens.items():
        for r in arm_rows:
            if r.get("judge_coherence") is not None:
                by.setdefault((arm, r["alpha"]), []).append(r["judge_coherence"])
    n_cells = 0
    for c in cells:
        v = by.get((c["arm"], c["alpha"]))
        if not v:
            continue
        c["judge_coherence_mean"] = float(statistics.mean(v))
        c["judge_coherence_median"] = float(statistics.median(v))
        c["judge_frac_ge50"] = sum(1 for x in v if x >= 50) / len(v)
        c["judge_frac_ge70"] = sum(1 for x in v if x >= 70) / len(v)
        c["judge_n"] = len(v)
        n_cells += 1

    d["judge"] = {"model": MODEL, "axis": "coherence", "question": "Say hello.",
                  "span": "full continuation",
                  "rubric": "fra/gpt4o_judge.py COHERENCE_PROMPT (EM judges.yaml)"}
    src.write_text(json.dumps(d, indent=2))
    unscored = sum(1 for r in rows if r.get("judge_coherence") is None)
    print(f"\n[saved] {src}  ({n_cells} cells rolled up"
          + (f", {unscored} generations unscored)" if unscored else ")"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
