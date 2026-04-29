#!/usr/bin/env python3
"""Generate the 300-prompt set via Claude Sonnet, mirroring Ward et al.'s use of
Sonnet 3.7 to produce 300 reasoning prompts in 10 categories.

10 categories × 30 prompts each = 300. We use Sonnet 4.6 (the latest available
in this repo's environment); the protocol is otherwise the same.

Output: `data/prompts_300.jsonl` next to this module (under git so the prompt
set is reproducible across machines). Each line is a {id, category, text}
record matching the schema in `prompts.py`.

Usage (one-time, requires ANTHROPIC_API_KEY):

    TQDM_DISABLE=1 uv run python -m experiments.phase7_unification.case_studies.backtracking.generate_prompts

Idempotent: skips if the file already exists with ≥300 entries; use --force to
regenerate.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

_REPO = Path(__file__).resolve().parents[4]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from experiments.phase7_unification.case_studies.backtracking._paths import REPO  # noqa: E402

PROMPTS_DATA_DIR = (
    REPO
    / "experiments"
    / "phase7_unification"
    / "case_studies"
    / "backtracking"
    / "data"
)
PROMPTS_300_PATH = PROMPTS_DATA_DIR / "prompts_300.jsonl"


CATEGORIES: list[tuple[str, str]] = [
    ("logic", "Pure deductive logic puzzles (knights and knaves, syllogisms, constraint satisfaction with 4-6 entities). Each problem should require multi-step deduction and admit a single answer."),
    ("geometry", "Plane and solid geometry (triangles, circles, polygons, cubes, cylinders, cones). No diagrams; the problem must be fully posed in words. Quantitative answer expected."),
    ("probability", "Discrete probability with combinatorics or conditional reasoning (cards, dice, coloured balls, Monty-Hall-flavour). Single numeric or fractional answer."),
    ("number_theory", "Properties of integers — divisibility, primes, modular arithmetic, parity arguments. The answer should be a small integer or short proof sketch."),
    ("combinatorics", "Counting problems — permutations, combinations, inclusion-exclusion, pigeonhole. Avoid overlap with the probability category by keeping the answer count-only."),
    ("algebra", "Word problems whose solution requires setting up and solving 1-2 algebraic equations (work-rate, mixtures, ages, distance-rate-time)."),
    ("sequences", "Number sequences, recurrences, summations. Ask for a specific term, sum, or closed form."),
    ("optimisation", "Discrete or continuous optimisation amenable to elementary reasoning (AM-GM, vertices of a feasible region, greedy-with-justification)."),
    ("set_theory", "Set-theoretic / boolean reasoning — subset relations, cardinality, Venn-diagram counting, three-set inclusion-exclusion."),
    ("invariant", "Problems solved by spotting an invariant or parity argument (chessboard tilings, coin flips, monovariants in transformations)."),
]


PROMPT_TEMPLATE = """You are generating reasoning problems for an interpretability experiment. I need {n} distinct problems in the category **{category}**.

Category description: {category_desc}

Requirements for each problem:
- Self-contained (no diagrams, no external context)
- Solvable by a strong language model in 100-400 tokens of reasoning
- Single well-defined answer
- Distinctly worded — no two problems should share the same numerical setup
- Difficulty roughly equivalent to AMC / undergraduate-introductory level

Output ONLY a JSON array of {n} strings, no commentary, no surrounding prose, no markdown fences. Each string is one complete problem statement.

Example output shape (with two problems shown for illustration only — your output must contain {n}):

[
  "First problem statement here.",
  "Second problem statement here."
]

Return your {n} problems for the category {category} now:"""


def _load_api_key() -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        return api_key
    for candidate in (
        Path.home() / ".tokens/anthropic_key",
        Path("/root/.tokens/anthropic_key"),
        Path("/workspace/.tokens/anthropic_key"),
    ):
        if candidate.exists():
            return candidate.read_text().strip()
    raise RuntimeError(
        "no Anthropic API key found: set $ANTHROPIC_API_KEY or place the key "
        "in ~/.tokens/anthropic_key, /root/.tokens/anthropic_key, or "
        "/workspace/.tokens/anthropic_key"
    )


def _strip_fences(text: str) -> str:
    """Drop any leading/trailing markdown fences if Sonnet adds them."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text


def _generate_category(client, model: str, category: str, category_desc: str, n: int) -> list[str]:
    prompt = PROMPT_TEMPLATE.format(n=n, category=category, category_desc=category_desc)
    msg = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = msg.content[0].text
    parsed = json.loads(_strip_fences(raw))
    if not isinstance(parsed, list) or not all(isinstance(s, str) for s in parsed):
        raise ValueError(f"unexpected response shape for category {category!r}: {parsed!r}")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-per-category", type=int, default=30)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    PROMPTS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROMPTS_300_PATH.exists() and not args.force:
        with PROMPTS_300_PATH.open() as f:
            existing = sum(1 for _ in f)
        if existing >= args.n_per_category * len(CATEGORIES):
            print(f"[generate_prompts] {PROMPTS_300_PATH} already has {existing} prompts; use --force to regenerate")
            return

    from anthropic import Anthropic

    client = Anthropic(api_key=_load_api_key(), max_retries=8)

    all_records: list[dict] = []
    for category, desc in CATEGORIES:
        print(f"[generate_prompts] {category}: requesting {args.n_per_category} from {args.model}")
        prompts = _generate_category(client, args.model, category, desc, args.n_per_category)
        if len(prompts) != args.n_per_category:
            print(f"  warning: got {len(prompts)} prompts (asked for {args.n_per_category}); keeping all")
        for i, text in enumerate(prompts):
            all_records.append(
                {
                    "id": f"{category}_{i:03d}",
                    "category": category,
                    "text": text.strip(),
                }
            )

    with PROMPTS_300_PATH.open("w") as f:
        for r in all_records:
            f.write(json.dumps(r) + "\n")
    print(f"[generate_prompts] wrote {len(all_records)} prompts → {PROMPTS_300_PATH}")


if __name__ == "__main__":
    main()
