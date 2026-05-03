"""Compute Cohen's κ + raw agreement between Aniket's blind scores and the
LLM judge on the 20 B3 transcripts.

Sonnet 4.6 is asked for the same 3 fields Aniket scored (coherence 0–3,
backtracking_present 0/1, looping_present 0/1), in a single combined call
per transcript.

Usage:
  python -m experiments.ward_backtracking_txc.validate_judge_kappa \
      --aniket results/ward_backtracking_txc/judge_validation/blind_pairs_aniket.csv \
      --out results/ward_backtracking_txc/judge_validation/
"""
from __future__ import annotations
import argparse
import asyncio
import csv
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.validate_kappa")


JUDGE_PROMPT = """You are evaluating a math problem-solving generation. Score it on three dimensions:

**Coherence** (0-3 integer):
- 0 = incoherent / loops / total LaTeX collapse / non-English degeneration
- 1 = mostly nonsense or pseudo-backtracking loops, occasional coherent fragments
- 2 = mostly coherent prose with notable issues (wrong arithmetic, lost context, inconsistency)
- 3 = fully coherent, fluent reasoning

**Backtracking present** (0 or 1):
- 1 = contains GENUINE backtracking — the reasoner mid-trace identifies a problem with their own prior reasoning and changes course (catching a calculation error, noticing a missing constraint, rejecting the current approach, re-evaluating an assumption that turned out wrong)
- 0 = no genuine backtracking. Filler ("Hmm, let me think"), restating the problem, or re-stating the same conclusion with different wording does NOT count. Pseudo-backtracking where "wait" is followed by repeating the same content does NOT count.

**Looping present** (0 or 1):
- 1 = sentence-level repetition for at least 3 consecutive sentences (e.g., "Wait, perhaps I should use the fact..." repeated 3+ times near-verbatim)
- 0 = no such loop

Math problem the model was solving:
\"\"\"
{prompt_text}
\"\"\"

Model's generation:
\"\"\"
{generation}
\"\"\"

Reply with EXACTLY this format on three lines:

COHERENCE: <0|1|2|3>
BACKTRACKING_PRESENT: <0|1>
LOOPING_PRESENT: <0|1>

Do not output anything else."""


async def judge_one(client, prompt_text: str, generation: str,
                    model_id: str = "claude-sonnet-4-6") -> dict:
    msg = await client.messages.create(
        model=model_id,
        max_tokens=80,
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(
                prompt_text=prompt_text[:1500],
                generation=generation[:6000],
            ),
        }],
    )
    raw = msg.content[0].text.strip()

    def grab(field: str, allow: tuple[int, ...]) -> int | None:
        m = re.search(rf"{field}:\s*(\d+)", raw)
        if not m:
            return None
        v = int(m.group(1))
        return v if v in allow else None

    return {
        "coherence": grab("COHERENCE", (0, 1, 2, 3)),
        "backtracking_present": grab("BACKTRACKING_PRESENT", (0, 1)),
        "looping_present": grab("LOOPING_PRESENT", (0, 1)),
        "raw": raw,
    }


async def grade_all(rows: list[dict], qid_to_problem: dict[str, str], concurrency: int = 6) -> list[dict]:
    try:
        from anthropic import AsyncAnthropic
    except ImportError:
        raise SystemExit("anthropic SDK not installed. uv add anthropic")
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    sem = asyncio.Semaphore(concurrency)
    out: list[dict] = [None] * len(rows)  # type: ignore

    async def worker(i: int, r: dict):
        async with sem:
            qid = r["question_id"]
            prompt_text = qid_to_problem.get(qid, "(unknown problem)")
            try:
                judge = await judge_one(client, prompt_text, r["transcript"])
            except Exception as e:
                log.warning("judge failed id=%s: %s", r["id"], e)
                judge = {"coherence": None, "backtracking_present": None,
                         "looping_present": None, "raw": f"ERROR: {e}"}
            out[i] = {**r, "judge": judge}
            log.info("  id=%s arch=%s mag=%s | judge: coh=%s bt=%s loop=%s",
                     r["id"], r["arch"], r["magnitude"],
                     judge.get("coherence"),
                     judge.get("backtracking_present"),
                     judge.get("looping_present"))

    await asyncio.gather(*[worker(i, r) for i, r in enumerate(rows)])
    return out


def cohen_kappa(a: list[int], b: list[int]) -> float:
    """Cohen's κ for two raters, paired observations."""
    if len(a) != len(b) or not a:
        return float("nan")
    cats = sorted(set(a) | set(b))
    if len(cats) < 2:
        return 1.0 if a == b else 0.0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    # Marginal probabilities
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    if pe == 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1.0 - pe)


def report(merged: list[dict], out_path: Path):
    fields = [("coherence", "human_coherence_0_3"),
              ("backtracking_present", "human_backtracking_present"),
              ("looping_present", "human_looping_present")]
    lines = []
    lines.append("# Cohen's κ + raw agreement: Aniket (blind) vs Sonnet 4.6 judge\n")
    lines.append(f"n = {len(merged)} transcripts\n")
    lines.append("| field | raw agreement | Cohen's κ | n_disagree | per-row breakdown |")
    lines.append("|---|---|---|---|---|")
    detailed = {}
    for judge_field, human_field in fields:
        a = [int(r[human_field]) for r in merged]
        b = [r["judge"][judge_field] for r in merged]
        # Drop rows where judge failed
        valid = [(x, y) for x, y in zip(a, b) if y is not None]
        a_v = [x for x, _ in valid]
        b_v = [y for _, y in valid]
        if not valid:
            lines.append(f"| {judge_field} | n/a (no valid judge rows) | n/a | n/a | n/a |")
            continue
        agree = sum(1 for x, y in zip(a_v, b_v) if x == y) / len(valid)
        kappa = cohen_kappa(a_v, b_v)
        n_disagree = sum(1 for x, y in zip(a_v, b_v) if x != y)
        breakdown = ", ".join(f"id{r['id']}({r[human_field]}/{r['judge'][judge_field]})"
                              for r in merged
                              if r["judge"][judge_field] is not None
                              and int(r[human_field]) != r["judge"][judge_field])
        if not breakdown:
            breakdown = "(perfect)"
        lines.append(f"| {judge_field} | {agree:.2f} | {kappa:.3f} | {n_disagree}/{len(valid)} | {breakdown} |")
        detailed[judge_field] = {
            "raw_agreement": agree, "cohens_kappa": kappa,
            "n_disagree": n_disagree, "n_valid": len(valid),
        }
    lines.append("\n## Targets")
    lines.append("- raw agreement ≥ 0.80 ✅ acceptable")
    lines.append("- Cohen's κ ≥ 0.6 ✅ substantial")
    lines.append("- Cohen's κ < 0.4 ❌ refine judge prompt or document limitation")
    lines.append("\n## Per-transcript scores")
    lines.append("\n| id | arch | mag | coh (A/J) | bt (A/J) | loop (A/J) | aniket notes |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in merged:
        j = r["judge"]
        ch = lambda hu, ju: f"{r[hu]}/{ju}" + ("" if int(r[hu]) == ju else "  ⚠️" if ju is not None else " (judge fail)")
        lines.append(f"| {r['id']} | {r['arch']} | {r['magnitude']} | "
                     f"{ch('human_coherence_0_3', j['coherence'])} | "
                     f"{ch('human_backtracking_present', j['backtracking_present'])} | "
                     f"{ch('human_looping_present', j['looping_present'])} | "
                     f"{r.get('human_notes','')[:80]} |")
    out_path.write_text("\n".join(lines))
    log.info("[saved] %s", out_path)
    return detailed


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--aniket", type=Path, required=True,
                   help="blind_pairs_aniket.csv with human_* columns filled")
    p.add_argument("--out", type=Path, required=True, help="output dir")
    p.add_argument("--concurrency", type=int, default=6)
    args = p.parse_args(argv)

    rows = list(csv.DictReader(args.aniket.open()))
    log.info("[in] %d aniket-scored rows", len(rows))

    # Pull the original MATH-500 problem text per qid
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    qid_to_problem = {r["unique_id"]: r["problem"] for r in ds}

    args.out.mkdir(parents=True, exist_ok=True)
    merged = asyncio.run(grade_all(rows, qid_to_problem, concurrency=args.concurrency))
    judged_path = args.out / "blind_pairs_judged.json"
    judged_path.write_text(json.dumps(merged, indent=2))
    log.info("[saved] %s", judged_path)

    summary = report(merged, args.out / "kappa_report.md")
    (args.out / "kappa_report.json").write_text(json.dumps(summary, indent=2))
    log.info("[summary]\n%s", json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
