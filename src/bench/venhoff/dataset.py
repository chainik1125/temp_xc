"""MMLU-Pro loader for Venhoff reasoning-trace generation.

Venhoff's released pipeline generates reasoning traces from the
`TIGER-Lab/MMLU-Pro` test split. This module wraps that dataset in a
simple iterator yielding prompts formatted exactly as their
`generate-responses/generate_responses.py` expects, so trace generation
is a drop-in.

The exact prompt template is taken from Venhoff's repo at the pinned
commit in `VENHOFF_PROVENANCE.md`. Any change to the template makes the
traces non-comparable with their published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

MMLU_PRO_HF_ID = "TIGER-Lab/MMLU-Pro"


# Venhoff's MMLU-Pro prompt format, copied verbatim from
# `generate-responses/generate_responses.py` (see VENHOFF_PROVENANCE.md
# for commit pin). The {question} and {options} slots get populated
# per-example. The trailing "The best answer is" is their exact wording.
PROMPT_TEMPLATE = (
    "Question: {question}\n"
    "Options:\n"
    "{options}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
)


@dataclass(frozen=True)
class MMLUProExample:
    """One MMLU-Pro item after formatting.

    `prompt` is the fully-formatted user message (no chat template yet —
    that's applied downstream by the tokenizer). `answer` is the
    canonical letter ("A".."J"). `category` is the subject bucket
    (math, chemistry, etc.).
    """

    question_id: str
    category: str
    prompt: str
    answer: str
    answer_index: int


def _format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def load_mmlu_pro(
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> Iterator[MMLUProExample]:
    """Stream MMLU-Pro examples in Venhoff's expected prompt format.

    Args:
        split: HF split name — `test` for Phase 1a, `validation` available
            for quick sanity.
        limit: if set, only yield the first `limit` examples (post-shuffle).
        seed: deterministic shuffle seed when `shuffle=True`.
        shuffle: shuffle before limiting — useful for smoke (1k) to not
            always hit the same subject order.

    Yields MMLUProExample instances.
    """
    from datasets import load_dataset  # deferred import

    ds = load_dataset(MMLU_PRO_HF_ID, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for row in ds:
        prompt = PROMPT_TEMPLATE.format(
            question=row["question"],
            options=_format_options(row["options"]),
        )
        yield MMLUProExample(
            question_id=str(row["question_id"]),
            category=row["category"],
            prompt=prompt,
            answer=row["answer"],
            answer_index=int(row["answer_index"]),
        )
