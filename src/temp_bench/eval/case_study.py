"""Unified abstraction for the three behavioural case studies.

The "temp-bench" name in the paper refers to the suite of three case studies
exposed via this interface:

    C5  RLHF steering            (Bhalla et al. 2025 §4.4, sentiment)
    C6  Emergent misalignment    (Turner et al. 2025, Qwen-14B finance)
    C7  Backtracking             (Ward et al. 2025, Stage B)

Every case study implements :class:`CaseStudy`. A worker agent runs an
architecture through every case study with one call.

**Judge-output persistence is mandatory** for every case study that
uses an LLM judge (Gemini for C5, Sonnet for C7, etc.). Implementations
must append each judge call to
``<workspace>/judge_outputs.jsonl`` with fields
``{transcript_id, magnitude, arch, judge_id, label, prompt_hash,
judge_model, ts}``. This lets us defer Cohen's κ validation to a
paper-end stretch task: post-deadline, fresh κ on a 20-transcript
blind hand-score is just a pandas + scipy call. No re-judging, no
re-training. See ``docs/components/{c5,c6,c7}.md``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from temp_bench.architectures.base import TempBenchArch


@dataclass
class CaseStudyResult:
    """Standardised output of a case-study run.

    Different case studies report different metrics; ``metrics`` is a free-form
    dict so we don't lose information. ``primary_metric`` flags the single
    number that goes in the paper table.
    """
    case_study: str
    arch: str
    seed: int
    primary_metric: str             # key into ``metrics``
    metrics: dict[str, float]
    artifacts: dict[str, Path] = field(default_factory=dict)  # plots, ckpts
    notes: str = ""


class CaseStudy(ABC):
    """Abstract base for C5 / C6 / C7.

    Lifecycle::

        cs = SteeringCaseStudy(workspace=Path("results/runs/<run_id>"))
        cs.setup()                                  # download data, judge keys
        result = cs.evaluate(arch, seed=42)         # all heavy lifting here
        cs.teardown()                               # release GPU mem
    """

    name: str               # "c5_steering", "c6_em", "c7_backtracking"
    primary_metric: str     # the headline column in the paper table

    def __init__(self, workspace: Path):
        self.workspace = workspace
        self.workspace.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def setup(self) -> None: ...

    @abstractmethod
    def evaluate(self, arch: TempBenchArch, seed: int, **kwargs: Any) -> CaseStudyResult: ...

    def teardown(self) -> None:
        """Default no-op; override to free model/judge resources."""
        return None
