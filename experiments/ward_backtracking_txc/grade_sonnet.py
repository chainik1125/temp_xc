"""Sonnet 4.6 coherence grader for Stage B per-cell B1 generations.

Replaces the per-word `max_same_word_run ≤ 2` heuristic — which is too
weak (it accepts "Wait, I'm not. / Wait, I'm not." sentence-loop
degeneration) — with the same 0–3 grade Dmitry's branch uses (Ward et
al. paper §B.2 prompt, copied verbatim).

Inputs:
    --b1-json  path to a per-cell `b1__<cell_id>.json`
               (or the canonical `b1_steering_results.json`)

Output:
    <root>/coherence_grades/<basename>.json
       schema: { row_index: { "grade": int, "raw": str } }

Cost: ~$0.001 per row. A 720-row per-cell file → ~$0.7. The full Phase 1
canonical (~13k rows) → ~$13. Filtered to rows with non-trivial keyword_rate
the cost drops to a few dollars.

Concurrency: asyncio + AsyncAnthropic client; default 8 in-flight.
Resumable: existing grades in the output file are preserved.
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
log = logging.getLogger("ward_txc.grade_sonnet")

GRADER_PROMPT = """You are grading the coherence of a math problem-solving generation.

Problem prompt the model was solving: {prompt_text}

The model generated this continuation:
\"\"\"
{generation}
\"\"\"

Grade this continuation for coherence on the following 0-3 scale (Ward et al. 2025 §B.2):

3 - Coherent and on-task. The text is grammatical natural English, follows from the prompt, makes mathematical or logical sense (even if reaching a wrong answer), and progresses through reasoning. Repetition of phrases is OK if the underlying reasoning advances.

2 - Mostly coherent but partly off-task or partly incoherent. Some sentences make sense; others are confused or repetitive without progress. The text is recognizably English prose attempting to engage with the problem.

1 - Highly repetitive or incoherent. The model emits looped phrases ("Wait, I'm not. / Wait, I'm not.") or single-token repetitions, or topic-drifts entirely off the problem. May contain occasional coherent fragments amid the noise.

0 - Total degeneration. Pure repetition of one or two tokens, gibberish, single-character emissions, or no recognizable English structure.

Reply with EXACTLY one digit (0, 1, 2, or 3) and nothing else. Do not explain your grade."""


async def grade_one(client, prompt_text: str, generation: str, model_id: str = "claude-sonnet-4-6") -> dict:
    """Issue a single grade request. Returns {grade: int, raw: str}."""
    msg = await client.messages.create(
        model=model_id,
        max_tokens=8,
        messages=[{
            "role": "user",
            "content": GRADER_PROMPT.format(
                prompt_text=prompt_text[:1500],
                generation=generation[:6000],
            )
        }],
    )
    raw = msg.content[0].text.strip()
    m = re.match(r"^\s*([0-3])", raw)
    grade = int(m.group(1)) if m else -1
    return {"grade": grade, "raw": raw}


async def grade_rows(rows: list[dict], prompts: dict[str, str], out_path: Path,
                    concurrency: int = 8, kw_rate_floor: float = 0.005) -> dict:
    """Grade each row whose keyword_rate − baseline > kw_rate_floor.

    Cells far below the kw effect ceiling don't matter for the metric peak,
    and grading them is wasted spend. We grade unconditionally for any row
    flagged by the kw_rate filter OR any DoM baseline row OR baseline mag=0.
    Grades for skipped rows are written as `null` (unbounded coherent).
    """
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise SystemExit("anthropic SDK not installed. uv add anthropic")
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # Resume: preserve existing grades.
    grades: dict[str, dict] = {}
    if out_path.exists():
        try:
            grades = json.loads(out_path.read_text())
            log.info("[resume] %d existing grades in %s", len(grades), out_path)
        except Exception:
            pass

    BASELINE_KW = 0.007
    sem = asyncio.Semaphore(concurrency)
    n_to_grade = 0
    skipped = 0

    async def worker(idx: int, row: dict):
        async with sem:
            key = str(idx)
            if key in grades and grades[key].get("grade", -1) >= 0:
                return
            text = (row.get("text") or "").strip()
            if not text:
                grades[key] = {"grade": 0, "raw": "(empty)"}
                return
            prompt_id = row.get("prompt_id", "")
            prompt_text = prompts.get(prompt_id, "(unknown prompt)")
            try:
                result = await grade_one(client, prompt_text, text)
            except Exception as e:
                grades[key] = {"grade": -1, "raw": f"(error: {e})"}
                return
            grades[key] = result

    # Decide which rows to grade
    targets: list[tuple[int, dict]] = []
    for idx, r in enumerate(rows):
        kw = float(r.get("keyword_rate", 0.0))
        mag = float(r.get("magnitude", 0.0))
        src = r.get("source", "")
        # Always grade DoM baseline rows + mag=0 + any |effect| above floor
        if (mag == 0.0
                or "dom_" in src
                or abs(kw - BASELINE_KW) > kw_rate_floor):
            targets.append((idx, r))
        else:
            # Mark as coherent-3 (effect too small to matter; not a winner)
            grades.setdefault(str(idx), {"grade": 3, "raw": "(skipped: low effect)"})
            skipped += 1

    log.info("[grade] %d rows queued, %d skipped (low effect)", len(targets), skipped)
    n_to_grade = sum(1 for idx, _ in targets if str(idx) not in grades or grades[str(idx)].get("grade", -1) < 0)
    log.info("[grade] %d need new grading (others resumed/skipped)", n_to_grade)

    # Periodic save
    save_every = 100
    progress = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)

    async def runner():
        nonlocal progress
        coros = []
        for idx, row in targets:
            coros.append(worker(idx, row))
        for fut in asyncio.as_completed(coros):
            await fut
            progress += 1
            if progress % save_every == 0:
                out_path.write_text(json.dumps(grades, indent=2))
                log.info("[grade] %d/%d done", progress, len(coros))

    await runner()
    out_path.write_text(json.dumps(grades, indent=2))
    log.info("[grade] saved %s with %d grades", out_path, len(grades))
    return grades


def load_prompt_lookup(stageA_prompts: Path) -> dict[str, str]:
    """Map prompt_id -> question text from Stage A prompts.json."""
    obj = json.loads(stageA_prompts.read_text())
    out: dict[str, str] = {}
    for p in obj:
        out[p["id"]] = p.get("question") or p.get("prompt") or p.get("text", "")
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--b1-json", type=Path, required=True,
                   help="path to per-cell b1__<cell>.json or canonical b1_steering_results.json")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="where to write coherence_grades; default <root>/coherence_grades")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--kw-effect-floor", type=float, default=0.005,
                   help="skip rows whose |kw - baseline| ≤ this (treat as coherent)")
    args = p.parse_args(argv)

    import yaml
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = args.out_dir or (Path(cfg["paths"]["root"]) / "coherence_grades")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.b1_json.name

    obj = json.loads(args.b1_json.read_text())
    rows = obj["rows"]
    prompts = load_prompt_lookup(Path(cfg["paths"]["stageA_prompts"]))

    asyncio.run(grade_rows(
        rows, prompts, out_path,
        concurrency=args.concurrency,
        kw_rate_floor=args.kw_effect_floor,
    ))
    return 0


if __name__ == "__main__":
    sys.exit(main())
