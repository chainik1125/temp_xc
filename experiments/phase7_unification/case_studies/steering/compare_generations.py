"""Cross-arch qualitative steering comparison — paper-appendix table generator.

For a fixed set of representative concepts, dump the generations from each
arch at one strength (default 24, the high end of the sweep) so we can
read all 4 archs' steered output side-by-side. Useful for the paper's
qualitative table à la T-SAE Table 2.

Output: stdout markdown table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.phase7_unification._paths import banner
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, STAGE_1_ARCHS,
)


REPRESENTATIVE_CONCEPTS = [
    "harmful_content",
    "refusal_pattern",
    "medical",
    "mathematical",
    "programming",
    "code_context",
    "religious",
    "poetic",
    "positive_emotion",
    "negative_emotion",
    "formal_register",
    "dialogue",
]


def _load_arch_gens(arch_id: str) -> list[dict]:
    p = CASE_STUDIES_DIR / "steering" / arch_id / "generations.jsonl"
    if not p.exists():
        return []
    return [json.loads(line) for line in p.open()]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=list(STAGE_1_ARCHS))
    ap.add_argument("--strength", type=float, default=24.0)
    ap.add_argument("--concepts", nargs="+", default=REPRESENTATIVE_CONCEPTS)
    ap.add_argument("--max-chars", type=int, default=200)
    args = ap.parse_args()
    banner(__file__)

    arch_data = {a: _load_arch_gens(a) for a in args.archs}
    arch_data = {a: gens for a, gens in arch_data.items() if gens}

    print()
    print(f"## Steering qualitative comparison @ strength={args.strength}")
    print()
    for concept in args.concepts:
        print(f"### `{concept}`")
        print()
        print("| arch | feature | generated text |")
        print("|---|---|---|")
        for arch_id in arch_data:
            gens = arch_data[arch_id]
            match = [g for g in gens if g["concept_id"] == concept and g["strength"] == args.strength]
            if not match:
                print(f"| `{arch_id}` | — | — |")
                continue
            g = match[0]
            text = g["generated_text"].replace("\n", " / ").replace("|", "\\|")
            text = text[:args.max_chars] + ("..." if len(text) > args.max_chars else "")
            print(f"| `{arch_id}` | {g['feature_idx']} | {text} |")
        print()


if __name__ == "__main__":
    main()
