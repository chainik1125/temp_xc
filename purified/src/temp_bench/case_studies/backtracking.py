"""C7 — Ward Stage B backtracking case study.

Implements :class:`BacktrackingCaseStudy` (the
:class:`temp_bench.eval.case_study.CaseStudy` contract for C7) plus the
universal helpers it needs:

- :class:`SteeringHook` — per-row magnitude × decoder direction hook
  added to the residual stream of the reasoning model.
- :func:`extract_boxed`, :func:`answers_match` — \\boxed{} answer
  extraction + LaTeX-aware comparison.
- :func:`load_stage_a` — read the ported Stage A artifacts.
- :func:`build_cohort` — Aniket's truly-wrong + originally-correct
  cohort filter.
- :class:`SonnetBacktrackingJudge` — Sonnet 4.6 judge with **mandatory
  judge-output persistence** to ``judge_outputs.jsonl`` (one record per
  ``(transcript_id, magnitude, arch, judge_id, label)``). Lets us defer
  fresh Cohen's κ validation to a post-deadline pandas + scipy call;
  no re-judging.
- :func:`compute_delta_gc` — Δgc inducement metric, baselined per
  ``(arch, qid)`` to ``mag=0`` (cut-and-continue resampling baseline).
  See :mod:`docs/components/c7.md` for why baseline correction is
  load-bearing.
- :func:`compute_pr_auc_at_S` — sparse-probe PR-AUC for backtracking
  detection at ``S ∈ {1, 2, 4, 8, 16, 32}`` features. PR-AUC (NOT F1,
  NOT ROC-AUC) is the primary metric because the positive class is
  ~12% — F1 / ROC-AUC are misleading at this imbalance.

Provenance — these helpers are ported with attribution from
``origin/aniket-ward-stage-b @ a62175ee``:

- :class:`SteeringHook`, :func:`extract_boxed`, :func:`_strip_latex_to_plain`,
  :func:`answers_match` ← ``experiments/ward_backtracking_txc/b3_math500_rescue.py``
- ``JUDGE_PROMPT`` ← ``experiments/ward_backtracking_txc/grade_backtracking.py``
- cut25 protocol ← ``experiments/ward_backtracking_txc/b3_variants.py``

The orchestration code (cohort iteration, per-magnitude generation,
flip-matrix construction) lives in ``experiments/c7_backtracking/run.py``;
this module owns the reusable abstractions so the experiment script
stays thin (PROTOCOL.md § 11 *Code reuse contract*).
"""

from __future__ import annotations

import asyncio
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import torch

from temp_bench.architectures.base import TempBenchArch
from temp_bench.eval.case_study import CaseStudy, CaseStudyResult

log = logging.getLogger("temp_bench.case_studies.backtracking")


# ── Constants ──────────────────────────────────────────────────────────


# Aniket's densified 25-point magnitude grid.
# Concentrated in ±0.5–8 around the SAE peak; sparse in the tails where
# curves are flat. See docs/components/c7.md.
DEFAULT_MAGNITUDE_GRID: tuple[float, ...] = (
    -16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5,
    0,
    0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16,
)

# cut25 = cut Stage A trace at 25% of unsteered length, then steer-and-continue.
DEFAULT_CUT_FRACTION: float = 0.25

# PR-AUC sparse-probe top-S grid (c7.md "Detection (PR-AUC)" axis).
DEFAULT_PR_AUC_S_GRID: tuple[int, ...] = (1, 2, 4, 8, 16, 32)

# Sonnet 4.6 — gold-standard judge. Aniket validated κ ≥ 0.749 on his
# wasteland transcripts; we lean on prior-art validation per c7.md
# (judge κ deferred to post-deadline stretch task).
SONNET_JUDGE_MODEL: str = "claude-sonnet-4-6"


# Verbatim from origin/aniket-ward-stage-b:experiments/ward_backtracking_txc/grade_backtracking.py
# (with prompt_text/generation placeholders) — ports the Sonnet rubric for
# "genuine backtracking events". Do not rephrase without re-running κ
# validation against Aniket's wasteland baseline.
SONNET_JUDGE_PROMPT: str = """You are evaluating a math problem-solving generation for genuine backtracking behaviour.

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


_JUDGE_PROMPT_HASH: str = hashlib.sha256(
    SONNET_JUDGE_PROMPT.encode("utf-8")
).hexdigest()[:16]


# ── Stage A loaders ───────────────────────────────────────────────────


@dataclass(frozen=True)
class StageA:
    """Frozen view of the Stage A artifacts ported under
    ``results/c7_backtracking/stage_a/``.

    Aniket's Stage A scaffolding (300 R1-Distill-Llama traces × 10
    reasoning categories, sentence labels under Sonnet 4.6, dom
    direction-of-meaning vectors) is a read-only input — see
    ``ATTRIBUTION.md`` in that directory.
    """
    prompts: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    sentence_labels: list[dict[str, Any]]
    dom_vectors: dict[str, Any]
    validation: dict[str, Any]
    root: Path

    def by_qid(self, kind: str = "trace") -> dict[str, dict[str, Any]]:
        """Index a list-shape table by ``question_id``.

        ``kind ∈ {prompt, trace, label}``.
        """
        if kind == "prompt":
            rows = self.prompts
            key = "id"
        elif kind == "trace":
            rows = self.traces
            key = "question_id"
        elif kind == "label":
            rows = self.sentence_labels
            key = "question_id"
        else:
            raise ValueError(f"unknown kind={kind!r}")
        return {r[key]: r for r in rows}


def load_stage_a(root: Path | str | None = None) -> StageA:
    """Load the ported Stage A artifacts. ``root`` defaults to
    ``results/c7_backtracking/stage_a/`` resolved against the framework
    root.
    """
    if root is None:
        from temp_bench.config import purified_root
        root = purified_root() / "results" / "c7_backtracking" / "stage_a"
    root = Path(root)
    return StageA(
        prompts=json.loads((root / "prompts.json").read_text()),
        traces=json.loads((root / "traces.json").read_text()),
        sentence_labels=json.loads((root / "sentence_labels.json").read_text()),
        dom_vectors=torch.load(
            root / "dom_vectors.pt", weights_only=False, map_location="cpu"
        ),
        validation=json.loads((root / "validation.json").read_text()),
        root=root,
    )


# ── Cohort filtering ──────────────────────────────────────────────────


@dataclass(frozen=True)
class Cohort:
    """Aniket's headline cohort: 31 truly-wrong + 30 originally-correct
    MATH-500 questions = 61 panels × 25 magnitudes per arch.

    The cohort source is Aniket's wasteland Stage B output
    ``flip_matrix.parquet`` (ported under
    ``results/c7_backtracking/aniket_reference/cut25/`` with
    attribution). qids look like ``test/algebra/1184.json`` and resolve
    against the ``HuggingFaceH4/MATH-500`` ``unique_id`` column.

    Truly-wrong = the unsteered reasoning model parsed an answer but it
    was incorrect. Originally-correct = ground-truth answer matches.
    Token-truncated traces (no parsed answer) are dropped.
    """
    truly_wrong: list[str]                # qids
    originally_correct: list[str]         # qids
    truncated: list[str]                  # qids dropped (only meaningful for LM-discovery path)
    source: str                           # "aniket_parquet" | "lm_discovery"

    @property
    def all(self) -> list[str]:
        return [*self.truly_wrong, *self.originally_correct]

    def __len__(self) -> int:
        return len(self.all)


def aniket_reference_root() -> Path:
    """Resolve the ported Aniket reference root.

    Returns the ``cut25/`` subdir specifically — the C7 paper protocol.
    """
    from temp_bench.config import purified_root
    return purified_root() / "results" / "c7_backtracking" / "aniket_reference" / "cut25"


def load_cohort_from_parquet(parquet_path: Path | str | None = None) -> Cohort:
    """Read the 31+30 cohort qids from Aniket's flip_matrix.parquet.

    The parquet has one row per (arch, qid, magnitude) with
    ``before_correct`` indicating the unsteered (mag=0) correctness.
    We pick the headline arch (``txc``) + ``magnitude=0`` rows to get
    one record per qid (61 total) and split on ``before_correct``.

    Returns the same 61 qids any of the 6 arches were evaluated on
    (the cohort is fixed across arches by construction — see Aniket's
    methodology_neurips_push.md).
    """
    import pandas as pd

    if parquet_path is None:
        parquet_path = aniket_reference_root() / "flip_matrix.parquet"
    df = pd.read_parquet(parquet_path)
    headline_arch = "txc" if "txc" in df["arch"].unique() else df["arch"].unique()[0]
    sub = df[(df["arch"] == headline_arch) & (df["magnitude"] == 0.0)]
    sub = sub[["question_id", "before_correct"]].drop_duplicates()
    truly_wrong = sorted(sub.loc[~sub["before_correct"], "question_id"].tolist())
    correct = sorted(sub.loc[sub["before_correct"], "question_id"].tolist())
    return Cohort(
        truly_wrong=truly_wrong,
        originally_correct=correct,
        truncated=[],
        source="aniket_parquet",
    )


def load_math500_lookup() -> dict[str, dict[str, Any]]:
    """Load MATH-500 keyed by ``unique_id``.

    The qids in our cohort (``test/algebra/1184.json``) are MATH-500
    ``unique_id`` values. Returns a dict ``{qid: {problem, solution,
    answer, subject, level}}`` for prompt + ground-truth lookup.
    """
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    return {row["unique_id"]: row for row in ds}


def build_cohort(
    stage_a: StageA | None = None,
    *,
    source: str = "aniket_parquet",
    n_correct: int = 30,
    seed: int = 42,
    ground_truth_lookup: Callable[[str], str | None] | None = None,
) -> Cohort:
    """Default path: ``source="aniket_parquet"`` reads the 31+30 cohort
    from Aniket's reference flip_matrix (frozen, deterministic).

    The ``source="lm_discovery"`` path is reserved for future use
    (re-discovers the cohort by running R1-Distill-Llama on all 500
    MATH-500 questions); requires a ``ground_truth_lookup`` callable
    plus a list of ``(qid, parsed_answer)`` pairs threaded through
    ``stage_a``. Not implemented yet — open question #2 in the
    agent_back briefing.
    """
    if source == "aniket_parquet":
        return load_cohort_from_parquet()
    raise NotImplementedError(
        f"build_cohort(source={source!r}) not implemented; "
        "use source='aniket_parquet' (default) until LM-discovery path lands."
    )


# ── Steering hook (verbatim port from b3_math500_rescue.py) ───────────


class SteeringHook:
    """Per-row magnitude × decoder direction added to a residual-stream
    output.

    Verbatim port of ``_Hook`` from
    ``origin/aniket-ward-stage-b @ a62175ee:experiments/ward_backtracking_txc/b3_math500_rescue.py``.
    Reattach via ``module.register_forward_hook(hook)``; set
    :attr:`magnitudes` to a 1-D tensor of length ``batch_size``
    (one scalar per row in the batch) before each forward pass.
    """

    def __init__(self, vec: torch.Tensor):
        self._raw = vec.detach()
        self._cached: torch.Tensor | None = None
        self.magnitudes: torch.Tensor | None = None

    def _materialize(self, ref: torch.Tensor) -> torch.Tensor:
        if (
            self._cached is None
            or self._cached.device != ref.device
            or self._cached.dtype != ref.dtype
        ):
            self._cached = self._raw.to(device=ref.device, dtype=ref.dtype)
        return self._cached

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        v = self._materialize(x)
        if self.magnitudes is not None:
            mags = self.magnitudes.to(device=x.device, dtype=x.dtype)
            return mags.view(-1, 1, 1) * v
        return torch.zeros_like(x)

    def __call__(self, _module, _inp, output):
        if self.magnitudes is None or torch.count_nonzero(self.magnitudes) == 0:
            return output
        if isinstance(output, tuple):
            x = output[0]
            return (x + self._delta(x),) + output[1:]
        return output + self._delta(output)


# ── Boxed-answer extraction (verbatim port) ───────────────────────────


def extract_boxed(text: str) -> str | None:
    """Pull the LAST ``\\boxed{...}`` content from a model output.

    Verbatim port from
    ``origin/aniket-ward-stage-b @ a62175ee:experiments/ward_backtracking_txc/b3_math500_rescue.py``.
    Stack-based parser handles nested braces (e.g.
    ``\\boxed{\\frac{1}{2}}``). Returns ``None`` if no boxed answer.
    """
    if not text:
        return None
    last = None
    i = 0
    needle = "\\boxed{"
    while True:
        idx = text.find(needle, i)
        if idx < 0:
            break
        start = idx + len(needle)
        depth = 1
        j = start
        while j < len(text) and depth > 0:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        if depth == 0:
            last = text[start:j].strip()
        i = max(idx + 1, j)
    return last


def _strip_latex_to_plain(s: str) -> str:
    """Lossy normalisation for boxed-answer comparison. Verbatim port."""
    s = s.lower()
    s = s.replace("\\left(", "(").replace("\\right)", ")")
    s = s.replace("\\left[", "[").replace("\\right]", "]")
    s = s.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    s = s.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    s = s.replace(" ", "").strip(".,;:")
    return s


def answers_match(model_answer: str | None, ground_truth: str) -> bool:
    """LaTeX-aware match. Verbatim port. Falls back to sympy when
    available."""
    if model_answer is None:
        return False
    a = _strip_latex_to_plain(model_answer)
    b = _strip_latex_to_plain(ground_truth)
    if a == b:
        return True
    try:
        from sympy import simplify, sympify  # type: ignore
        from sympy.parsing.latex import parse_latex  # type: ignore
        for parser in (parse_latex, sympify):
            try:
                ea = parser(model_answer)
                eb = parser(ground_truth)
                if simplify(ea - eb) == 0:
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


# ── cut25 protocol helper ─────────────────────────────────────────────


def cut25_token_position(
    unsteered_token_ids: Sequence[int],
    *,
    fraction: float = DEFAULT_CUT_FRACTION,
) -> int:
    """Return the token index at which to cut + continue.

    Per Aniket's B3 ``cut25`` protocol (beats LLM-judged cut and
    full-trace continuation per ``results_b_neurips_push.md``): cut at
    ``fraction × len`` of the unsteered token sequence.
    """
    if not unsteered_token_ids:
        return 0
    return int(round(len(unsteered_token_ids) * fraction))


# ── Sonnet judge dispatch with mandatory persistence ──────────────────


@dataclass(frozen=True)
class JudgeOutput:
    """One Sonnet call → one ``judge_outputs.jsonl`` row.

    Persisted to ``<workspace>/judge_outputs.jsonl`` per the C7
    framework requirement (``docs/components/c7.md`` "Judge κ
    validation: deferred"): every Sonnet call lands here keyed by
    ``(transcript_id, magnitude, arch, judge_id, label)`` so we can
    compute fresh Cohen's κ post-deadline without re-judging.
    """
    transcript_id: str
    magnitude: float
    arch: str
    seed: int
    judge_id: str               # short hash of judge prompt + model id
    judge_model: str            # e.g. "claude-sonnet-4-6"
    prompt_hash: str            # sha256(JUDGE_PROMPT)[:16]
    label: int                  # genuine_count (0+); -1 on parse error
    raw: str                    # raw judge reply for audit
    ts: str                     # ISO-8601 UTC


def _judge_id(model: str = SONNET_JUDGE_MODEL, prompt_hash: str = _JUDGE_PROMPT_HASH) -> str:
    """Stable judge identifier: ``<model>:<prompt_hash>``."""
    return f"{model}:{prompt_hash}"


_COUNT_RE = re.compile(r"COUNT:\s*(-?\d+)", re.IGNORECASE)


def parse_judge_reply(raw: str) -> int:
    """Extract the integer COUNT from a Sonnet reply. Returns ``-1`` on
    parse failure (matches Aniket's ``grade_backtracking.py`` semantics).
    """
    m = _COUNT_RE.search(raw or "")
    return int(m.group(1)) if m else -1


class SonnetBacktrackingJudge:
    """Async Sonnet 4.6 judge with mandatory ``judge_outputs.jsonl``
    persistence.

    Usage::

        judge = SonnetBacktrackingJudge(workspace=Path("results/runs/<eval_key>"))
        outs = await judge.judge_many([
            (qid, mag, arch, seed, prompt_text, generation_text), ...
        ])

    The persistence step is non-negotiable — see ``docs/components/c7.md``
    "Judge κ validation: deferred". Even if ``label == -1`` (parse
    failure), the row is appended for audit + retry.
    """

    def __init__(
        self,
        *,
        workspace: Path,
        model: str = SONNET_JUDGE_MODEL,
        max_concurrency: int = 8,
        max_tokens: int = 200,
    ):
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.judge_id = _judge_id(model)
        self.prompt_hash = _JUDGE_PROMPT_HASH
        self.max_concurrency = max_concurrency
        self.max_tokens = max_tokens
        self._jsonl = self.workspace / "judge_outputs.jsonl"

    def _client(self):
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as e:
            raise ImportError(
                "SonnetBacktrackingJudge needs the `anthropic` package "
                "(pyproject already pins it). Install with `uv sync`."
            ) from e
        from temp_bench.utils.tokens import get_token  # type: ignore
        api_key = get_token("anthropic") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY missing — populate /workspace/.tokens/anthropic_key "
                "(see scripts/bootstrap_runpod.sh)."
            )
        return AsyncAnthropic(api_key=api_key)

    def _persist(self, out: JudgeOutput) -> None:
        with self._jsonl.open("a") as f:
            f.write(json.dumps(dataclasses.asdict(out)) + "\n")

    def existing_keys(self) -> set[tuple[str, float, str, int]]:
        """Return ``{(transcript_id, magnitude, arch, seed)}`` already in
        the workspace's jsonl with a *successful* label (≥ 0). Used to
        skip already-judged calls when resuming a partial sweep.

        **Skip label=-1 rows**: those are API-error / parse-failure
        records persisted for audit. Treat them as un-judged so retries
        actually re-run them. Without this, an API outage that returned
        400 on every call would poison the workspace permanently —
        existing_keys would mark them all as "done" and a retry would
        no-op."""
        if not self._jsonl.exists():
            return set()
        keys: set[tuple[str, float, str, int]] = set()
        with self._jsonl.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                try:
                    label = int(rec.get("label", -1))
                except (TypeError, ValueError):
                    label = -1
                if label < 0:
                    continue
                keys.add((
                    rec.get("transcript_id", ""),
                    float(rec.get("magnitude", float("nan"))),
                    rec.get("arch", ""),
                    int(rec.get("seed", -1)),
                ))
        return keys

    async def judge_one(
        self,
        client,
        *,
        transcript_id: str,
        magnitude: float,
        arch: str,
        seed: int,
        prompt_text: str,
        generation: str,
    ) -> JudgeOutput:
        try:
            msg = await client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                messages=[{
                    "role": "user",
                    "content": SONNET_JUDGE_PROMPT.format(
                        prompt_text=prompt_text[:1500],
                        generation=generation[:6000],
                    ),
                }],
            )
            raw = msg.content[0].text.strip()
            label = parse_judge_reply(raw)
        except Exception as e:                                  # pragma: no cover
            raw = f"(api-error: {e})"
            label = -1
        out = JudgeOutput(
            transcript_id=transcript_id,
            magnitude=float(magnitude),
            arch=arch,
            seed=int(seed),
            judge_id=self.judge_id,
            judge_model=self.model,
            prompt_hash=self.prompt_hash,
            label=label,
            raw=raw,
            ts=_dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        )
        self._persist(out)
        return out

    async def judge_many(
        self,
        rows: Iterable[tuple[str, float, str, int, str, str]],
        *,
        skip_existing: bool = True,
    ) -> list[JudgeOutput]:
        """Judge a batch of ``(qid, mag, arch, seed, prompt, generation)``
        tuples concurrently."""
        seen = self.existing_keys() if skip_existing else set()
        client = self._client()
        sem = asyncio.Semaphore(self.max_concurrency)

        async def worker(row):
            qid, mag, arch, seed, prompt, gen = row
            if (qid, float(mag), arch, int(seed)) in seen:
                return None
            async with sem:
                return await self.judge_one(
                    client,
                    transcript_id=qid,
                    magnitude=mag,
                    arch=arch,
                    seed=seed,
                    prompt_text=prompt,
                    generation=gen,
                )

        results: list[JudgeOutput] = []
        for fut in asyncio.as_completed([worker(r) for r in rows]):
            res = await fut
            if res is not None:
                results.append(res)
        return results


# ── Δgc inducement metric ─────────────────────────────────────────────


def compute_delta_gc(
    judge_outputs: Iterable[JudgeOutput | dict[str, Any]],
    *,
    baseline_magnitude: float = 0.0,
) -> dict[str, Any]:
    """Δgc = mean over (qid) of (gc(qid, mag) − gc(qid, baseline_magnitude)),
    per (arch, mag).

    Baseline correction matters: at mag=0 the steering hook is a no-op,
    but the cut-and-continue resampling produces a non-zero genuine_count
    floor for every arch (the SAME Stage A prefix; only sampling differs).
    Per ``docs/components/c7.md``, baseline-corrected Δgc is the headline
    metric.

    Inputs may be :class:`JudgeOutput` instances or plain dicts loaded
    from ``judge_outputs.jsonl``.

    Returns::

        {
            "by_arch_mag": {(arch, mag): mean_delta_gc, ...},
            "stability": {arch: count_of_mags_with_delta>0_out_of_total, ...},
            "peak": {arch: (peak_mag, peak_delta), ...},
        }
    """
    import collections

    rows = [_to_dict(o) for o in judge_outputs]
    # Coerce label to int — different case studies (em, steering) may
    # persist string labels in shared judge_outputs.jsonl. Drop rows
    # where label can't be parsed as a non-negative integer (this is
    # also how parse_judge_reply signals failure: -1).
    def _label_int(r):
        try:
            return int(r["label"])
        except (TypeError, ValueError):
            return -1
    rows = [r for r in rows if _label_int(r) >= 0]
    for r in rows:
        r["label"] = _label_int(r)

    by_qid: dict[tuple[str, str, int, float], list[int]] = collections.defaultdict(list)
    for r in rows:
        by_qid[(r["arch"], r["transcript_id"], r["seed"], r["magnitude"])].append(r["label"])

    # Average over duplicate calls per (arch, qid, seed, mag).
    qid_means: dict[tuple[str, str, int, float], float] = {
        k: float(np.mean(v)) for k, v in by_qid.items()
    }

    # Baseline gc per (arch, qid, seed).
    baseline: dict[tuple[str, str, int], float] = {}
    for (arch, qid, seed, mag), val in qid_means.items():
        if mag == baseline_magnitude:
            baseline[(arch, qid, seed)] = val

    # Δgc per (arch, qid, seed, mag).
    delta_per_qid: dict[tuple[str, float], list[float]] = collections.defaultdict(list)
    for (arch, qid, seed, mag), val in qid_means.items():
        b = baseline.get((arch, qid, seed))
        if b is None:
            continue
        delta_per_qid[(arch, mag)].append(val - b)

    by_arch_mag: dict[tuple[str, float], float] = {
        k: float(np.mean(v)) for k, v in delta_per_qid.items()
    }

    archs = sorted({a for (a, _) in by_arch_mag})
    stability: dict[str, str] = {}
    peak: dict[str, tuple[float, float]] = {}
    for arch in archs:
        mags = sorted({m for (a, m) in by_arch_mag if a == arch and m != baseline_magnitude})
        deltas = [by_arch_mag[(arch, m)] for m in mags]
        n_pos = sum(1 for d in deltas if d > 0)
        stability[arch] = f"{n_pos}/{len(mags)}"
        if deltas:
            i = int(np.argmax(deltas))
            peak[arch] = (float(mags[i]), float(deltas[i]))

    return {
        "by_arch_mag": by_arch_mag,
        "stability": stability,
        "peak": peak,
    }


def _to_dict(o: JudgeOutput | dict[str, Any]) -> dict[str, Any]:
    if isinstance(o, JudgeOutput):
        return dataclasses.asdict(o)
    return dict(o)


# ── PR-AUC sparse-probe detection ─────────────────────────────────────


def compute_pr_auc_at_S(
    feature_acts: np.ndarray,
    labels: np.ndarray,
    question_ids: np.ndarray | None,
    *,
    S_grid: tuple[int, ...] = DEFAULT_PR_AUC_S_GRID,
    n_folds: int = 5,
    C: float = 1.0,
    random_state: int = 42,
) -> dict[int, float]:
    """Sparse-probe PR-AUC at top-S features, S ∈ ``S_grid``.

    Per c7.md "Detection (PR-AUC)" axis: 5-fold GroupKFold by
    ``question_id`` so test sentences are never from a question that
    appears in train. PR-AUC mandatory because positive class is ~12%.

    ``feature_acts``: ``(n_sentences, d_sae)`` — typically
    ``model.encode(sentence_acts).abs().max(dim=T_axis)`` per sentence.
    ``labels``: ``(n_sentences,)`` 0/1 backtracking label.
    ``question_ids``: ``(n_sentences,)`` group ids; if ``None``, falls
    back to plain ``KFold``.
    Returns ``{S: pr_auc}``.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score
    from sklearn.model_selection import GroupKFold, KFold

    if question_ids is not None:
        cv = GroupKFold(n_splits=n_folds)
        splits = list(cv.split(feature_acts, labels, groups=question_ids))
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        splits = list(cv.split(feature_acts, labels))

    pr_auc: dict[int, float] = {}
    for S in S_grid:
        fold_aps: list[float] = []
        for train_idx, test_idx in splits:
            X_tr, X_te = feature_acts[train_idx], feature_acts[test_idx]
            y_tr, y_te = labels[train_idx], labels[test_idx]
            # Per-fold mean-difference top-S feature selection (matches
            # Aniket's detection probe).
            mean_pos = X_tr[y_tr == 1].mean(axis=0) if (y_tr == 1).any() else np.zeros(X_tr.shape[1])
            mean_neg = X_tr[y_tr == 0].mean(axis=0) if (y_tr == 0).any() else np.zeros(X_tr.shape[1])
            md = np.abs(mean_pos - mean_neg)
            top = np.argsort(md)[-S:]
            clf = LogisticRegression(
                penalty="l1",
                C=C,
                solver="liblinear",
                max_iter=2000,
                random_state=random_state,
            )
            clf.fit(X_tr[:, top], y_tr)
            proba = clf.predict_proba(X_te[:, top])[:, 1]
            fold_aps.append(float(average_precision_score(y_te, proba)))
        pr_auc[S] = float(np.mean(fold_aps))
    return pr_auc


# ── Reasoning-model loader + generation helpers (verbatim ports) ───────


def _fix_byte_decode(s: str) -> str:
    """Workaround for transformers ≥5.5.4 leaving Ġ/Ċ literal in decoded
    strings. Verbatim port from b3_math500_rescue.py."""
    return s.replace("Ġ", " ").replace("Ċ", "\n").replace("Â", "")


def _build_prompt(problem: str) -> str:
    """Pose a MATH-500 problem to the reasoning model.

    Verbatim port from b3_math500_rescue.py — boxed answer is
    requested explicitly so we can extract a comparable final answer.
    """
    return (
        f"Solve this math problem and provide your final answer in "
        f"\\boxed{{}} notation.\n\nProblem: {problem}"
    )


def load_reasoning_lm(
    hf_id: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    *,
    device: str = "cuda",
):
    """Load the reasoning-model side of the Ward Stage B setup.

    Adapted from b3_math500_rescue.py:_load_lm. Loads in bf16; A40
    (47.7 GB VRAM) holds the 16-GB model + ~16 GB of generation
    activation comfortably.
    """
    from temp_bench.utils.tokens import get_token
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    hf_token = get_token("hf")
    tok = AutoTokenizer.from_pretrained(hf_id, use_fast=True, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.bfloat16, device_map=device, token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def generate_unsteered(
    model, tok,
    prompts: list[str],
    *,
    max_new_tokens: int,
    batch_size: int,
) -> tuple[list[str], list[list[int]]]:
    """Greedy unsteered generation. Returns ``(decoded_strings, new_token_ids)``.

    Adapted from b3_math500_rescue.py:_generate_unsteered. Uses
    left-padding so right-side new-token slicing works.
    """
    from transformers import GenerationConfig  # type: ignore
    gen_cfg = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False,
        temperature=1.0, pad_token_id=tok.pad_token_id,
    )
    saved = tok.padding_side
    tok.padding_side = "left"
    try:
        chat_texts = []
        for p in prompts:
            try:
                t = tok.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                t = p
            chat_texts.append(t)

        outs: list[str] = []
        new_token_ids: list[list[int]] = []
        for i in range(0, len(chat_texts), batch_size):
            batch = chat_texts[i:i + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True, truncation=True,
                      max_length=2048).to(model.device)
            prompt_len = enc["input_ids"].shape[1]
            with torch.no_grad():
                out_ids = model.generate(**enc, generation_config=gen_cfg)
            for row in out_ids:
                new = row[prompt_len:].tolist()
                while new and new[-1] == tok.pad_token_id:
                    new.pop()
                new_token_ids.append(new)
                outs.append(_fix_byte_decode(tok.decode(new, skip_special_tokens=True)))
        return outs, new_token_ids
    finally:
        tok.padding_side = saved


def generate_continuation_panels(
    model, tok, hook: SteeringHook,
    *,
    problem_prompts: list[str],
    prefix_token_ids: list[list[int]],
    mags_per_panel: list[float],
    max_new_per_panel: list[int],
    batch_size: int,
) -> list[str]:
    """Run the cut+continue panels under per-row steering magnitudes.

    Adapted from b3_math500_rescue.py:_generate_continuation_panels.
    For each (problem, prefix tokens, magnitude) panel, build the
    extended input ``[chat_template(problem) + prefix_tokens]`` and
    generate under the per-row magnitude on ``hook``. Returns
    continuation strings (no chat template / no prefix).
    """
    from transformers import GenerationConfig  # type: ignore
    assert len(problem_prompts) == len(prefix_token_ids) == len(mags_per_panel) \
        == len(max_new_per_panel)
    saved = tok.padding_side
    tok.padding_side = "left"
    try:
        full_input_ids: list[list[int]] = []
        for problem, prefix in zip(problem_prompts, prefix_token_ids):
            try:
                ct = tok.apply_chat_template(
                    [{"role": "user", "content": problem}],
                    tokenize=False, add_generation_prompt=True,
                )
            except Exception:
                ct = problem
            ct_ids = tok(ct, add_special_tokens=False)["input_ids"]
            full_input_ids.append(ct_ids + list(prefix))

        outs: list[str] = []
        for i in range(0, len(full_input_ids), batch_size):
            chunk_ids = full_input_ids[i:i + batch_size]
            chunk_mags = torch.tensor(mags_per_panel[i:i + batch_size], dtype=torch.float32)
            chunk_budgets = max_new_per_panel[i:i + batch_size]
            chunk_max_budget = max(chunk_budgets)
            hook.magnitudes = chunk_mags

            gen_cfg = GenerationConfig(
                max_new_tokens=chunk_max_budget, do_sample=False,
                temperature=1.0, pad_token_id=tok.pad_token_id,
            )
            max_len = max(len(x) for x in chunk_ids)
            pad_id = tok.pad_token_id
            input_ids = torch.full((len(chunk_ids), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros((len(chunk_ids), max_len), dtype=torch.long)
            for r, ids in enumerate(chunk_ids):
                pad_n = max_len - len(ids)
                input_ids[r, pad_n:] = torch.tensor(ids, dtype=torch.long)
                attn[r, pad_n:] = 1
            input_ids = input_ids.to(model.device)
            attn = attn.to(model.device)
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids=input_ids, attention_mask=attn,
                    generation_config=gen_cfg,
                )
            for r, row in enumerate(out_ids):
                new = row[max_len:].tolist()
                while new and new[-1] == tok.pad_token_id:
                    new.pop()
                budget = chunk_budgets[r]
                if len(new) > budget:
                    new = new[:budget]
                outs.append(_fix_byte_decode(tok.decode(new, skip_special_tokens=True)))
        hook.magnitudes = None
        return outs
    finally:
        tok.padding_side = saved


# ── Phase 1 unsteered cache ───────────────────────────────────────────


def run_phase1_unsteered(
    cohort: Cohort,
    *,
    workspace: Path,
    max_new_tokens: int = 2048,
    batch_size: int = 8,
    model=None,
    tok=None,
) -> list[dict[str, Any]]:
    """Phase 1: run R1-Distill-Llama on cohort qids unsteered, cache on disk.

    Caches at ``<workspace>/phase1_unsteered.json``. Idempotent — if
    the file exists with rows for every cohort qid, returns immediately.

    Each row::

        {
            "unique_id": str,                # MATH-500 qid
            "problem": str,
            "ground_truth": str,
            "unsteered_text": str,           # decoded continuation
            "unsteered_token_ids": list[int],# per-row tokens
            "unsteered_answer": str | None,  # extract_boxed(unsteered_text)
            "unsteered_correct": bool,       # answers_match vs ground_truth
        }

    The reasoning model is loaded internally if ``model``/``tok`` are
    not passed; otherwise the caller's instance is reused (preferred —
    loading R1-Distill-Llama is ~30 s and 16 GB VRAM).
    """
    cache_path = Path(workspace) / "phase1_unsteered.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    cohort_qids = set(cohort.all)
    if cache_path.exists():
        existing = json.loads(cache_path.read_text())
        existing_qids = {r["unique_id"] for r in existing}
        if cohort_qids.issubset(existing_qids):
            log.info("[phase1] cache hit at %s (%d rows)", cache_path, len(existing))
            return [r for r in existing if r["unique_id"] in cohort_qids]
        log.info("[phase1] partial cache: %d/%d qids — extending",
                 len(cohort_qids & existing_qids), len(cohort_qids))
    else:
        existing = []

    have = {r["unique_id"]: r for r in existing}
    todo = [q for q in cohort.all if q not in have]
    if not todo:
        return [have[q] for q in cohort.all]

    math500 = load_math500_lookup()
    todo_problems = [math500[q]["problem"] for q in todo]
    todo_gts = [math500[q]["answer"] for q in todo]

    own_model = False
    if model is None or tok is None:
        log.info("[phase1] loading reasoning model")
        model, tok = load_reasoning_lm()
        own_model = True
    log.info("[phase1] generating %d unsteered traces", len(todo))
    prompts = [_build_prompt(p) for p in todo_problems]
    texts, token_ids = generate_unsteered(
        model, tok, prompts, max_new_tokens=max_new_tokens, batch_size=batch_size,
    )

    new_rows = []
    for qid, prob, gt, text, tids in zip(todo, todo_problems, todo_gts, texts, token_ids):
        ans = extract_boxed(text)
        new_rows.append({
            "unique_id": qid,
            "problem": prob,
            "ground_truth": gt,
            "unsteered_text": text,
            "unsteered_token_ids": tids,
            "unsteered_answer": ans,
            "unsteered_correct": answers_match(ans, gt),
        })

    rows = existing + new_rows
    cache_path.write_text(json.dumps(rows, indent=2))
    log.info("[phase1] wrote %d total rows to %s", len(rows), cache_path)

    if own_model:
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    return [r for r in rows if r["unique_id"] in cohort_qids]


# ── Labeled-sentence activation extraction (subject-model side) ───────


# Aniket's window offsets — T=6 positions BEFORE the sentence start. The
# Ward steering-feature mining looks at the activations that "trigger"
# the backtracking sentence, not the backtracking content itself.
DEFAULT_WINDOW_OFFSETS: tuple[int, ...] = (-13, -12, -11, -10, -9, -8)


def extract_labeled_sentence_acts(
    *,
    subject_model_id: str = "NousResearch/Meta-Llama-3.1-8B",
    layer: int = 10,
    window_offsets: tuple[int, ...] = DEFAULT_WINDOW_OFFSETS,
    cache_path: Path | str | None = None,
    force: bool = False,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Capture per-sentence subject-model activations from Stage A labels.

    Adapted from
    ``origin/aniket-ward-stage-b @ a62175ee:experiments/ward_backtracking_txc/mine_features.py:_capture_windows``.

    For each labeled sentence in ``results/c7_backtracking/stage_a/sentence_labels.json``:

    1. Tokenize the trace's ``full_response`` with offset mapping.
    2. Run the subject model + capture residual-stream activations at
       layer ``layer`` for every position.
    3. Find the token position corresponding to the sentence's start
       (``think_offset + char_start`` per Aniket's alignment rule).
    4. Extract a T-window of activations at positions ``[tok_pos + off
       for off in window_offsets]`` (``T = len(window_offsets) = 6`` by
       default, matching Aniket's [-13, -8] window).

    Returns a dict with::

        X:    (n_sent, T, d_model) float32        — sentence acts
        is_bt: (n_sent,) bool                     — backtracking labels
        keys: (n_sent,) object                    — "<qid>|<trace_idx>|<sent_idx>"

    Cached at ``cache_path`` (default
    ``results/c7_backtracking/stage_a/sentence_acts_L<layer>.npz``);
    idempotent. ~25 K sentences × 300 traces × ~30 s each on A40 is the
    one-time cost (Aniket's run is the reference). Re-extraction is
    triggered when ``force=True`` or when the file is missing.
    """
    from temp_bench.config import purified_root

    if cache_path is None:
        cache_path = (
            purified_root() / "results" / "c7_backtracking" / "stage_a"
            / f"sentence_acts_L{layer}.npz"
        )
    cache_path = Path(cache_path)

    if cache_path.exists() and not force:
        log.info("[c7] sentence acts cache hit at %s", cache_path)
        z = np.load(cache_path, allow_pickle=True)
        return {"X": z["X"], "is_bt": z["is_bt"], "keys": z["keys"]}

    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from temp_bench.utils.tokens import get_token

    stage_a = load_stage_a()
    traces_by_qid = stage_a.by_qid("trace")

    log.info("[c7] loading subject model for sentence-act extraction: %s", subject_model_id)
    hf_token = get_token("hf")
    tok = AutoTokenizer.from_pretrained(subject_model_id, use_fast=True, token=hf_token)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        subject_model_id, torch_dtype=torch.bfloat16, device_map=device, token=hf_token,
    ).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    captured: dict[str, torch.Tensor] = {}
    def hook_fn(_m, _i, output):
        x = output[0] if isinstance(output, tuple) else output
        if x.dim() == 4:
            x = x.reshape(x.shape[0], x.shape[1], -1)
        captured["x"] = x.detach().to(torch.float32).cpu()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    X_list: list[np.ndarray] = []
    is_bt: list[bool] = []
    keys: list[str] = []

    try:
        try:
            from tqdm.auto import tqdm  # type: ignore
        except ImportError:
            tqdm = lambda x, **kw: x  # noqa: E731
        for record in tqdm(stage_a.sentence_labels, desc=f"capture L{layer}"):
            qid = record["question_id"]
            trace = traces_by_qid.get(qid)
            if trace is None or not record.get("sentences"):
                continue
            full_response = trace["full_response"]
            enc = tok(full_response, return_tensors="pt",
                      return_offsets_mapping=True, add_special_tokens=False)
            input_ids = enc["input_ids"].to(model.device)
            offsets_map = enc["offset_mapping"][0].tolist()
            with torch.no_grad():
                _ = model(input_ids=input_ids)
            acts = captured["x"][0].numpy()  # (seq, d)
            seq_len = acts.shape[0]

            # Aniket's sentence-start alignment: trace begins at the
            # closing tag <think>, then char_start counts from there.
            think_open = "<think>"
            think_idx = full_response.find(think_open)
            think_offset = (think_idx + len(think_open)) if think_idx >= 0 else 0

            for s_idx, sent in enumerate(record["sentences"]):
                target_char = think_offset + sent["char_start"]
                tok_pos = -1
                for i, (cs, ce) in enumerate(offsets_map):
                    if cs <= target_char < ce or cs >= target_char:
                        tok_pos = i
                        break
                if tok_pos < 0:
                    continue
                window = np.zeros((len(window_offsets), acts.shape[1]), dtype=np.float32)
                ok = True
                for j, off in enumerate(window_offsets):
                    p = tok_pos + off
                    if p < 0 or p >= seq_len:
                        ok = False
                        break
                    window[j] = acts[p]
                if not ok:
                    continue
                X_list.append(window)
                is_bt.append(bool(sent["is_backtracking"]))
                keys.append(f"{qid}|{record['trace_idx']}|{s_idx}")

            captured.clear()
            torch.cuda.empty_cache()
    finally:
        handle.remove()
        del model
        import gc; gc.collect()
        torch.cuda.empty_cache()

    if not X_list:
        raise RuntimeError("no sentence windows captured — check sentence_labels.json")

    X = np.stack(X_list, axis=0)
    is_bt_arr = np.asarray(is_bt, dtype=bool)
    keys_arr = np.asarray(keys, dtype=object)
    log.info(
        "[c7] captured %d sentences (%d positive, %.1f%%); shape=%s",
        len(X), int(is_bt_arr.sum()), 100 * float(is_bt_arr.mean()), X.shape,
    )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: stage to a temp file, rename when done. Avoids
    # concurrent readers seeing a partially-written zip
    # (np.savez_compressed writes incrementally; mid-write the file is
    # not a valid zip → callers get `zipfile.BadZipFile`).
    # NB: np.savez_compressed auto-appends ".npz" if the path doesn't
    # already end in ".npz", so use a temp name that already does.
    tmp_path = cache_path.with_name(cache_path.stem + ".tmp.npz")
    np.savez_compressed(tmp_path, X=X, is_bt=is_bt_arr, keys=keys_arr)
    tmp_path.replace(cache_path)
    log.info("[c7] cached sentence acts to %s", cache_path)
    return {"X": X, "is_bt": is_bt_arr, "keys": keys_arr}


def split_pos_neg(sentence_acts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Convenience: split ``extract_labeled_sentence_acts`` output into
    D+ / D- tensors for :func:`mine_top_features`.

    Returns ``{"pos": (n_pos, T, d), "neg": (n_neg, T, d)}``.
    """
    is_bt = sentence_acts["is_bt"]
    X = sentence_acts["X"]
    return {"pos": X[is_bt], "neg": X[~is_bt]}


# ── Steering-vector mining (gated on trained SAE arch) ────────────────


@dataclass
class FeatureMineResult:
    """Output of :func:`mine_top_features` — one steering candidate per arch."""
    feature_id: int
    decoder_direction: torch.Tensor   # (d_in,)
    pos_act_mean: float               # mean activation on positives (D+)
    neg_act_mean: float               # mean activation on negatives (D-)
    selectivity: float                # pos_mean - neg_mean


def mine_top_features(
    model: TempBenchArch,
    *,
    pos_activations: np.ndarray,      # (n_pos, T, d_in) — labeled-positive sentence acts
    neg_activations: np.ndarray,      # (n_neg, T, d_in) — labeled-negative
    top_k: int = 32,
    batch_size: int = 1024,
) -> list[FeatureMineResult]:
    """Rank features by D+/D- mean-difference selectivity.

    Aniket's mine_features.py adapted to our :class:`TempBenchArch`
    interface. For each labeled sentence's activation window, encode
    through the SAE → max-pool over T axis → per-feature pooled
    activation. Rank features by (pos_mean - neg_mean) selectivity.
    Return top-K features along with their decoder directions.

    Encoding is **batched** by ``batch_size`` to avoid OOM at large
    ``d_sae``: e.g. d_sae=32768 × n_neg=22K × T=6 fp32 = 17 GB
    encoded latents, blowing the A40 budget if encoded all at once.

    The decoder direction defaults to :meth:`TempBenchArch.decoder_directions`
    (T-averaged for window archs); per-position decoder selection
    (Aniket's "pos0" mode) requires arch-specific override.
    """
    device = next(model.parameters()).device
    # Per-arch T: slice sentence_acts (T=6) to match the arch's expected T.
    # TXC-base wants T=5; TXC-pro wants T_max (or t_sample); TopK-SAE/TFA
    # accept any T. Take the LAST arch_T positions of the [-13..-8] window
    # — these are closest to the sentence start (most predictive of the
    # backtracking event).
    arch_T = getattr(model.config, "T", None) or 1
    win_T = pos_activations.shape[1]
    if arch_T < win_T:
        pos_activations = pos_activations[:, -arch_T:, :]
        neg_activations = neg_activations[:, -arch_T:, :]
    elif arch_T > win_T:
        # Pad by repeating the first position (rare — only if arch_T > 6).
        pad = arch_T - win_T
        pos_activations = np.concatenate(
            [np.repeat(pos_activations[:, :1, :], pad, axis=1), pos_activations],
            axis=1,
        )
        neg_activations = np.concatenate(
            [np.repeat(neg_activations[:, :1, :], pad, axis=1), neg_activations],
            axis=1,
        )

    def _encoded_means(acts: np.ndarray) -> torch.Tensor:
        n = acts.shape[0]
        if n == 0:
            return torch.zeros(model.d_sae, device=device)
        sum_acc: torch.Tensor | None = None
        param_dtype = next(model.parameters()).dtype
        with torch.no_grad():
            for i in range(0, n, batch_size):
                batch = torch.from_numpy(acts[i:i + batch_size]).to(device, dtype=param_dtype)
                z = model.encode(batch).abs().float()
                # Max-pool over T (window axis) — Aniket's window-level latent.
                if z.dim() == 3:
                    z = z.amax(dim=1)
                # Sum into accumulator so we can mean at end.
                if sum_acc is None:
                    sum_acc = z.sum(dim=0)
                else:
                    sum_acc = sum_acc + z.sum(dim=0)
                del z, batch
        return sum_acc / n

    pos_mean = _encoded_means(pos_activations)
    neg_mean = _encoded_means(neg_activations)
    sel = pos_mean - neg_mean
    topk_vals, topk_idx = sel.topk(top_k)

    decoder = model.decoder_directions().detach().cpu()  # (d_sae, d_in)
    out = []
    for i in range(top_k):
        fid = int(topk_idx[i].item())
        out.append(FeatureMineResult(
            feature_id=fid,
            decoder_direction=decoder[fid].clone(),
            pos_act_mean=float(pos_mean[fid].item()),
            neg_act_mean=float(neg_mean[fid].item()),
            selectivity=float(topk_vals[i].item()),
        ))
    return out


# ── CaseStudy implementation (skeleton) ────────────────────────────────


class BacktrackingCaseStudy(CaseStudy):
    """C7 — Ward Stage B backtracking.

    Lifecycle (matches :class:`temp_bench.eval.case_study.CaseStudy`)::

        cs = BacktrackingCaseStudy(workspace=Path("results/runs/<eval_key>"))
        cs.setup()                              # load Stage A + judge
        result = cs.evaluate(arch, seed=42)     # full magnitude sweep
        cs.teardown()

    The full per-arch evaluation pipeline (cohort iteration, batched
    generation under the steering hook, judge dispatch, PR-AUC probe)
    is invoked from :func:`temp_bench.case_studies.backtracking.run_arch`,
    which the experiment runner calls as its ``eval_fn`` adapter.

    The ``stage_a`` field is loaded once on :meth:`setup` and reused for
    every arch — Stage A is frozen across all C7 cells.
    """
    name = "c7_backtracking"
    primary_metric = "delta_gc_peak"

    def __init__(
        self,
        workspace: Path,
        *,
        magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDE_GRID,
        cut_fraction: float = DEFAULT_CUT_FRACTION,
        stage_a_root: Path | str | None = None,
    ):
        super().__init__(workspace=workspace)
        self.magnitudes = tuple(magnitudes)
        self.cut_fraction = float(cut_fraction)
        self.stage_a_root = stage_a_root
        self.stage_a: StageA | None = None
        self.cohort: Cohort | None = None
        self.judge: SonnetBacktrackingJudge | None = None

    def setup(self) -> None:
        self.stage_a = load_stage_a(self.stage_a_root)
        self.cohort = build_cohort(self.stage_a)
        self.judge = SonnetBacktrackingJudge(workspace=self.workspace)
        log.info(
            "[c7] setup: %d truly-wrong + %d correct + %d truncated; "
            "%d magnitudes",
            len(self.cohort.truly_wrong),
            len(self.cohort.originally_correct),
            len(self.cohort.truncated),
            len(self.magnitudes),
        )

    def evaluate(self, arch: TempBenchArch, seed: int, **kwargs: Any) -> CaseStudyResult:
        """Forwards to :func:`run_arch_evaluation` with the case study's
        loaded Stage A + cohort + workspace.

        Caller must :meth:`setup` first. Sentence-level activations
        for PR-AUC detection are looked up from the activation cache
        registered in ``configs/datasources.yaml``; the runner's
        ``eval_fn`` adapter passes ``sentence_acts`` via ``kwargs``
        when available, else PR-AUC is skipped.
        """
        if self.stage_a is None or self.cohort is None or self.judge is None:
            raise RuntimeError("Call setup() before evaluate().")
        sentence_acts = kwargs.get("sentence_acts")
        return run_arch_evaluation(
            arch=arch,
            seed=seed,
            cohort=self.cohort,
            stage_a=self.stage_a,
            workspace=self.workspace,
            judge=self.judge,
            magnitudes=self.magnitudes,
            cut_fraction=self.cut_fraction,
            sentence_acts=sentence_acts,
            **{k: v for k, v in kwargs.items() if k != "sentence_acts"},
        )

    def teardown(self) -> None:
        self.stage_a = None
        self.cohort = None
        self.judge = None


# ── Top-level evaluation orchestrator ─────────────────────────────────


def run_arch_evaluation(
    *,
    arch: TempBenchArch,
    seed: int,
    cohort: Cohort,
    stage_a: StageA,
    workspace: Path,
    judge: SonnetBacktrackingJudge,
    magnitudes: tuple[float, ...] = DEFAULT_MAGNITUDE_GRID,
    cut_fraction: float = DEFAULT_CUT_FRACTION,
    arch_name: str = "unknown",
    feature_mining_acts: dict[str, np.ndarray] | None = None,
    sentence_acts: np.ndarray | None = None,
    sentence_labels: np.ndarray | None = None,
    sentence_qids: np.ndarray | None = None,
    pr_auc_S_grid: tuple[int, ...] = DEFAULT_PR_AUC_S_GRID,
    max_new_tokens: int = 2048,
    gen_batch_size: int = 8,
) -> CaseStudyResult:
    """Run one arch's full C7 evaluation: inducement (Δgc) + detection (PR-AUC).

    Pipeline:
      1. Mine top steering feature on ``arch`` (D+/D- mean-difference) using
         ``feature_mining_acts`` (dict with "pos"/"neg" arrays). Pick the
         feature with maximum selectivity.
      2. Build steering vector = decoder direction of the top feature,
         normalised to the same L2 norm as the dom-base-union vector.
      3. Run unsteered phase 1 on the cohort (cached at workspace/phase1_unsteered.json).
      4. For each magnitude in ``magnitudes`` × cohort qid: cut at
         ``cut_fraction × len(unsteered_token_ids)``, attach SteeringHook
         to layer 10 of R1-Distill-Llama, generate continuation. Persist
         every Sonnet judge call to ``workspace/judge_outputs.jsonl``.
      5. Compute Δgc per magnitude, baseline-corrected per (qid) at mag=0.
         Headline = peak Δgc + peak magnitude + stability.
      6. (Optional) If ``sentence_acts`` + ``sentence_labels`` provided,
         also compute PR-AUC at S ∈ {1, 2, 4, 8, 16, 32}.

    Returns :class:`CaseStudyResult` with:
      - ``primary_metric = "delta_gc_peak"``
      - ``metrics`` containing peak Δgc, peak magnitude, stability counts,
        per-magnitude Δgc, optional PR-AUC at each S.
      - ``artifacts`` containing the workspace dir + judge_outputs.jsonl.

    Raises ``RuntimeError`` if either the steering hookpoint or the
    generation pipeline fail. The judge persistence is non-fatal — even
    if individual judge calls error, ``judge_outputs.jsonl`` records the
    failure and the overall metric falls back to remaining valid rows.

    NOTE — this orchestrator is **arch-port-gated**: it expects
    ``arch`` to be a fully-loaded :class:`TempBenchArch` instance (i.e.,
    one of the 7 locked archs). Until agent_paper ships the arch
    implementations (see open question #1 in the agent_back briefing),
    callers will only get past step 1 when ``arch`` is the implemented
    `topk_sae`.
    """
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    # ── Step 1+2: mine + build steering vector ───────────────────────
    if feature_mining_acts is None:
        raise ValueError(
            "feature_mining_acts={'pos': (n_pos,T,d), 'neg': (n_neg,T,d)} required. "
            "Build via temp_bench.case_studies.backtracking.extract_labeled_sentence_acts(...)."
        )
    mined = mine_top_features(
        arch,
        pos_activations=feature_mining_acts["pos"],
        neg_activations=feature_mining_acts["neg"],
        top_k=32,
    )
    steering_feature = mined[0]
    log.info(
        "[c7] arch=%s seed=%d top feature=%d sel=%.4f",
        arch_name, seed, steering_feature.feature_id, steering_feature.selectivity,
    )

    # Normalise to dom-base-union L2 norm for cross-arch magnitude comparability.
    dom_base_union = stage_a.dom_vectors["base"].get("union")
    if dom_base_union is None:
        # Fallback: use the steering vector raw (unnormalised). Cross-arch
        # magnitudes will be slightly mis-calibrated; flagged in caveats.
        log.warning("[c7] dom_vectors['base']['union'] missing — skipping norm calibration")
        ref_norm = float(steering_feature.decoder_direction.norm().item())
    else:
        ref_norm = float(dom_base_union.norm().item())
    raw_vec = steering_feature.decoder_direction.float()
    vec = raw_vec / raw_vec.norm().clamp_min(1e-8) * ref_norm

    # ── Step 3: unsteered phase 1 ────────────────────────────────────
    log.info("[c7] loading reasoning model")
    rmodel, rtok = load_reasoning_lm()
    phase1 = run_phase1_unsteered(
        cohort, workspace=workspace,
        max_new_tokens=max_new_tokens,
        batch_size=gen_batch_size,
        model=rmodel, tok=rtok,
    )
    by_qid = {r["unique_id"]: r for r in phase1}
    log.info("[c7] phase1 ready: %d rows; cohort sizes wrong=%d correct=%d",
             len(phase1), len(cohort.truly_wrong), len(cohort.originally_correct))

    # ── Step 4: cut25 + steered generation per (qid, mag) ────────────
    layer = 10  # paper-justified per docs/components/c7.md
    layer_module = rmodel.model.layers[layer]
    hook = SteeringHook(vec)
    handle = layer_module.register_forward_hook(hook)
    try:
        # Build panels
        problem_prompts: list[str] = []
        prefix_token_ids: list[list[int]] = []
        mags: list[float] = []
        budgets: list[int] = []
        meta: list[tuple[str, float]] = []  # (qid, mag) per panel for judge dispatch
        for qid in cohort.all:
            row = by_qid[qid]
            cut_pos = cut25_token_position(row["unsteered_token_ids"], fraction=cut_fraction)
            prefix = row["unsteered_token_ids"][:cut_pos]
            remaining_budget = max(64, len(row["unsteered_token_ids"]) - cut_pos)
            for m in magnitudes:
                problem_prompts.append(_build_prompt(row["problem"]))
                prefix_token_ids.append(prefix)
                mags.append(float(m))
                budgets.append(remaining_budget)
                meta.append((qid, float(m)))
        log.info("[c7] %d panels = %d qids × %d mags", len(meta), len(cohort.all), len(magnitudes))

        outs = generate_continuation_panels(
            rmodel, rtok, hook,
            problem_prompts=problem_prompts,
            prefix_token_ids=prefix_token_ids,
            mags_per_panel=mags,
            max_new_per_panel=budgets,
            batch_size=gen_batch_size,
        )
    finally:
        handle.remove()
        del rmodel
        import gc; gc.collect()
        torch.cuda.empty_cache()

    # ── Step 5: judge ────────────────────────────────────────────────
    judge_rows = [
        (qid, mag, arch_name, seed, prompt, gen)
        for (qid, mag), prompt, gen in zip(meta, problem_prompts, outs)
    ]
    log.info("[c7] dispatching %d Sonnet judge calls", len(judge_rows))
    judge_outs = asyncio.run(judge.judge_many(judge_rows))
    log.info("[c7] judge done: %d new outputs (existing skipped)", len(judge_outs))

    # Compute Δgc — re-load all rows from jsonl so we get any prior runs too.
    all_judge_rows = []
    if judge._jsonl.exists():
        with judge._jsonl.open() as f:
            for line in f:
                try:
                    all_judge_rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    delta_gc = compute_delta_gc(all_judge_rows)

    arch_peak = delta_gc["peak"].get(arch_name, (0.0, 0.0))
    metrics: dict[str, float] = {
        "delta_gc_peak": float(arch_peak[1]),
        "delta_gc_peak_magnitude": float(arch_peak[0]),
        "n_judge_calls": float(len([r for r in all_judge_rows if r.get("arch") == arch_name and r.get("seed") == seed])),
    }
    # Per-magnitude Δgc (under arch+seed) — flatten into the metrics dict.
    for (a, m), d in delta_gc["by_arch_mag"].items():
        if a == arch_name:
            metrics[f"delta_gc_mag_{m:+.1f}"] = float(d)

    # ── Step 6: PR-AUC detection ─────────────────────────────────────
    if sentence_acts is not None and sentence_labels is not None:
        log.info("[c7] computing PR-AUC across S=%s (%d sentences)",
                 list(pr_auc_S_grid), len(sentence_labels))
        # Encode sentence-window acts via SAE → per-sentence pooled features.
        # Batched to avoid OOM at d_sae=32768 (25K × 6 × 32768 fp32 ≈ 19 GB
        # encoded at once; batched at 1024 caps peak at ~800 MB).
        # Per-arch T: slice (B, T=6, d) → (B, arch_T, d) to match encode contract.
        device = next(arch.parameters()).device
        param_dtype = next(arch.parameters()).dtype
        arch_T = getattr(arch.config, "T", None) or 1
        win_T = sentence_acts.shape[1]
        sa_for_arch = sentence_acts
        if arch_T < win_T:
            sa_for_arch = sentence_acts[:, -arch_T:, :]
        elif arch_T > win_T:
            pad = arch_T - win_T
            sa_for_arch = np.concatenate(
                [np.repeat(sentence_acts[:, :1, :], pad, axis=1), sentence_acts],
                axis=1,
            )
        n = len(sentence_labels)
        batch = 1024
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n, batch):
                xb = torch.from_numpy(sa_for_arch[i:i + batch]).to(device, dtype=param_dtype)
                z = arch.encode(xb).abs().float()
                if z.dim() == 3:
                    z = z.amax(dim=1)
                chunks.append(z.detach().cpu().numpy())
                del z, xb
        X_np = np.concatenate(chunks, axis=0)
        pr_auc = compute_pr_auc_at_S(
            X_np, sentence_labels, sentence_qids, S_grid=pr_auc_S_grid,
        )
        for S, ap in pr_auc.items():
            metrics[f"pr_auc_S{S}"] = float(ap)

    return CaseStudyResult(
        case_study="c7_backtracking",
        arch=arch_name,
        seed=seed,
        primary_metric="delta_gc_peak",
        metrics=metrics,
        artifacts={"workspace": workspace, "judge_outputs": judge._jsonl},
        notes=(
            f"steering_feature_id={steering_feature.feature_id}, "
            f"selectivity={steering_feature.selectivity:.4f}, ref_norm={ref_norm:.3f}"
        ),
    )
