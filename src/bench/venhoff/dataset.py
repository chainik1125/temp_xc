"""Dataset loaders for the Venhoff reasoning-eval pipeline.

Two datasets are supported:
  - MMLU-Pro (`TIGER-Lab/MMLU-Pro`) — the original Phase 1 target,
    retained for the taxonomy-quality side-channel.
  - MATH500 (`HuggingFaceH4/MATH-500`) — the primary target post-
    2026-04-20 pivot, where Dmitry flagged the Llama-3.1-8B →
    DeepSeek-R1-Distill-Llama-8B cell as the one Venhoff's method
    failed on (3.5% Gap Recovery).

Both return `Example` dataclasses with a uniform `prompt` / `answer`
contract so downstream stages (trace gen, grading) are dataset-agnostic.

Prompt templates come from Venhoff's repo at commit
`49a7f731ce693d813b9ae9a414f1739b992dbcef`; any template change breaks
comparability with their published numbers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

MMLU_PRO_HF_ID = "TIGER-Lab/MMLU-Pro"
MATH500_HF_ID = "HuggingFaceH4/MATH-500"

DatasetName = Literal["mmlu-pro", "math500"]


# Venhoff's MMLU-Pro prompt format. The {question} and {options} slots
# get populated per-example. \boxed{} is the expected answer wrapper.
MMLU_PRO_PROMPT = (
    "Question: {question}\n"
    "Options:\n"
    "{options}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
)

# Venhoff's MATH500 prompt format — same reasoning-step suffix.
MATH500_PROMPT = (
    "Problem: {problem}\n"
    "Please reason step by step, and put your final answer within \\boxed{{}}.\n"
)


@dataclass(frozen=True)
class MMLUProExample:
    question_id: str
    category: str
    prompt: str
    answer: str           # canonical letter "A".."J"
    answer_index: int


@dataclass(frozen=True)
class MATH500Example:
    """One MATH500 problem.

    `answer` is the canonical text answer (from the `\boxed{}` in the
    problem's solution). `level` is Hendrycks difficulty 1-5, `subject`
    is the math subject bucket (algebra, geometry, etc.). No
    answer_index — MATH500 is open-ended generation.
    """

    question_id: str
    subject: str
    level: int
    prompt: str
    answer: str
    solution: str           # full reference solution (for LLM-judge if needed)


def _format_options(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    return "\n".join(f"{letters[i]}. {opt}" for i, opt in enumerate(options))


def load_mmlu_pro(
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> Iterator[MMLUProExample]:
    """Stream MMLU-Pro examples in Venhoff's expected prompt format."""
    from datasets import load_dataset

    ds = load_dataset(MMLU_PRO_HF_ID, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for row in ds:
        prompt = MMLU_PRO_PROMPT.format(
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


def load_math500(
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
) -> Iterator[MATH500Example]:
    """Stream MATH500 problems in Venhoff's expected prompt format.

    MATH500 is the 500-problem subset Hendrycks et al. curated from MATH.
    HF dataset id `HuggingFaceH4/MATH-500` has a single `test` split with
    columns: `problem`, `solution`, `answer`, `subject`, `level`,
    `unique_id`.
    """
    from datasets import load_dataset

    ds = load_dataset(MATH500_HF_ID, split=split)
    if shuffle:
        ds = ds.shuffle(seed=seed)
    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    for i, row in enumerate(ds):
        problem = row["problem"]
        yield MATH500Example(
            question_id=str(row.get("unique_id") or f"math500-{i:04d}"),
            subject=str(row.get("subject") or "unknown"),
            level=int(row.get("level") or 0),
            prompt=MATH500_PROMPT.format(problem=problem),
            answer=str(row["answer"]),
            solution=str(row.get("solution") or ""),
        )


def load_dataset_uniform(
    name: DatasetName,
    split: str = "test",
    limit: int | None = None,
    seed: int = 0,
    shuffle: bool = False,
):
    """Dispatch by dataset name; yields either MMLUProExample or MATH500Example."""
    if name == "mmlu-pro":
        return load_mmlu_pro(split=split, limit=limit, seed=seed, shuffle=shuffle)
    if name == "math500":
        return load_math500(split=split, limit=limit, seed=seed, shuffle=shuffle)
    raise ValueError(f"unknown dataset {name!r}. choices: mmlu-pro, math500")
