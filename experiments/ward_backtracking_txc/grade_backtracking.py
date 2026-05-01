"""Behavioral backtracking judge — counts GENUINE backtracking events
inside a steered generation.

Motivation (per Dmitry's 2026-05-01 ask): the existing pipeline uses
keyword rate `(wait+hmm)/words` as the backtracking proxy, but a
keyword-token can be filler (`Hmm, let me think`) or pseudo-backtracking
(`Wait, no, actually...` followed by re-stating the same conclusion).
The behavioral question is: does a `wait`/`hmm` actually correspond to
the model identifying a problem in its prior reasoning and changing
course?

This script issues a Sonnet 4.6 judge call per generation that returns
the *count* of genuine backtracking events plus a short justification.
A genuine event = the model:
  - catches a calculation/arithmetic error and recomputes
  - notices a missing constraint / detail in the problem statement
  - rejects its current approach and tries a different method
  - explicitly re-evaluates an assumption
Filler (`Hmm, let me think`) and pseudo-backtracking (`Wait, no,
actually-` followed by the same conclusion restated) do NOT count.

Output schema (per row, indexed by row_idx into the B1 file):
  {
    "genuine_count": int,    # 0+ events
    "raw": str,              # raw model reply for audit
  }

Cost: ~$0.002 per row (longer generation than the coherence grader).
A 13k-row canonical corpus → ~$26.

Resumable: existing entries with `genuine_count >= 0` are preserved.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.grade_backtracking")

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
        model=model_id,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                prompt_text=prompt_text[:1500],
                generation=generation[:6000],
            )
        }],
    )
    raw = msg.content[0].text.strip()
    m = re.search(r"COUNT:\s*(\d+)", raw)
    count = int(m.group(1)) if m else -1
    return {"genuine_count": count, "raw": raw}


async def grade_rows(rows: list[dict], prompts: dict[str, str], out_path: Path,
                     concurrency: int = 8, kw_floor: float = 0.005,
                     coherence_grades: dict | None = None,
                     coherence_floor: int = 2) -> dict:
    """Judge each row whose keyword_rate > kw_floor AND coherence grade ≥ floor.

    Skipped rows are written with genuine_count = 0 (filler/incoherent
    can't be genuine backtracking).
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise SystemExit("anthropic SDK not installed. uv add anthropic")
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    judgements: dict[str, dict] = {}
    if out_path.exists():
        try:
            judgements = json.loads(out_path.read_text())
            log.info("[resume] %d existing judgements in %s", len(judgements), out_path)
        except Exception:
            pass

    BASELINE_KW = 0.007
    sem = asyncio.Semaphore(concurrency)

    async def worker(idx: int, row: dict):
        async with sem:
            key = str(idx)
            if key in judgements and judgements[key].get("genuine_count", -1) >= 0:
                return
            text = (row.get("text") or "").strip()
            if not text:
                judgements[key] = {"genuine_count": 0, "raw": "(empty)"}
                return
            prompt_id = row.get("prompt_id", "")
            prompt_text = prompts.get(prompt_id, "(unknown prompt)")
            try:
                result = await judge_one(client, prompt_text, text)
            except Exception as e:
                judgements[key] = {"genuine_count": -1, "raw": f"(error: {e})"}
                return
            judgements[key] = result

    targets: list[tuple[int, dict]] = []
    skipped = 0
    for idx, r in enumerate(rows):
        kw = float(r.get("keyword_rate", 0.0))
        if kw - BASELINE_KW < kw_floor:
            judgements.setdefault(str(idx), {"genuine_count": 0, "raw": "(skipped: low kw)"})
            skipped += 1
            continue
        if coherence_grades is not None:
            cg = coherence_grades.get(str(idx))
            if cg is None or cg.get("grade", -1) < coherence_floor:
                judgements.setdefault(str(idx), {"genuine_count": 0, "raw": "(skipped: incoherent)"})
                skipped += 1
                continue
        targets.append((idx, r))

    log.info("[judge] %d rows queued, %d skipped (low kw or incoherent)",
             len(targets), skipped)
    n_to_grade = sum(1 for idx, _ in targets if str(idx) not in judgements
                     or judgements[str(idx)].get("genuine_count", -1) < 0)
    log.info("[judge] %d need new judgement (others resumed/skipped)", n_to_grade)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_every = 100
    progress = 0

    async def runner():
        nonlocal progress
        coros = [worker(idx, row) for idx, row in targets]
        for fut in asyncio.as_completed(coros):
            await fut
            progress += 1
            if progress % save_every == 0:
                out_path.write_text(json.dumps(judgements, indent=2))
                log.info("[judge] %d/%d done", progress, len(coros))

    await runner()
    out_path.write_text(json.dumps(judgements, indent=2))
    log.info("[judge] saved %s with %d entries", out_path, len(judgements))
    return judgements


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--b1-json", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="default <root>/backtracking_judgements")
    p.add_argument("--coherence-grades", type=Path, default=None,
                   help="optional path to coherence_grades JSON; restrict "
                        "judgements to rows with grade >= 2")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--kw-floor", type=float, default=0.005)
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or (Path(cfg["paths"]["root"]) / "backtracking_judgements")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.b1_json.name

    obj = json.loads(args.b1_json.read_text())
    rows = obj["rows"]

    # Load prompts.
    stage_a = json.loads(Path(cfg["paths"]["stageA_prompts"]).read_text())
    prompts = {p["id"]: p.get("question") or p.get("prompt") or p.get("text", "")
               for p in stage_a}

    # Optionally load coherence grades for prefiltering.
    coherence_grades = None
    if args.coherence_grades and args.coherence_grades.exists():
        coherence_grades = json.loads(args.coherence_grades.read_text())
        log.info("[load] %d coherence grades from %s",
                 len(coherence_grades), args.coherence_grades)

    asyncio.run(grade_rows(
        rows, prompts, out_path,
        concurrency=args.concurrency,
        kw_floor=args.kw_floor,
        coherence_grades=coherence_grades,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
