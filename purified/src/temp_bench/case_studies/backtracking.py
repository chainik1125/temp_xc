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
    questions = 61 panels × 25 magnitudes per arch.

    Truly-wrong = the unsteered reasoning model parsed an answer but it
    was incorrect. Originally-correct = ground-truth answer matches.
    Token-truncated traces (no parsed answer) are dropped.
    """
    truly_wrong: list[str]                # qids
    originally_correct: list[str]         # qids
    truncated: list[str]                  # qids dropped

    @property
    def all(self) -> list[str]:
        return [*self.truly_wrong, *self.originally_correct]

    def __len__(self) -> int:
        return len(self.all)


def build_cohort(
    stage_a: StageA,
    *,
    n_correct: int = 30,
    seed: int = 42,
    ground_truth_lookup: Callable[[str], str | None] | None = None,
) -> Cohort:
    """Filter Stage A traces into the 31+30 cohort.

    Aniket originally pulled ``ground_truth`` from the MATH-500
    dataset; Stage A here covers 10 reasoning categories beyond
    MATH-500, so callers may pass a ``ground_truth_lookup(qid)``
    callable. If omitted, we fall back to the ``traces[*].answer`` vs
    ``traces[*].answer_index`` pair Aniket persisted (when available).

    Truncation rule: trace's parsed ``answer`` is missing ⇒ drop.

    Returns the qids — caller indexes back into Stage A by qid for the
    actual prompt + trace text.
    """
    truly_wrong: list[str] = []
    correct: list[str] = []
    truncated: list[str] = []

    for trace in stage_a.traces:
        qid = trace["question_id"]
        parsed = trace.get("answer")
        if parsed is None or parsed == "":
            truncated.append(qid)
            continue
        if ground_truth_lookup is not None:
            gt = ground_truth_lookup(qid)
        else:
            gt = trace.get("answer_index")
        if gt is None:
            # No ground truth available — treat as truncated.
            truncated.append(qid)
            continue
        if answers_match(parsed, str(gt)):
            correct.append(qid)
        else:
            truly_wrong.append(qid)

    rng = np.random.default_rng(seed)
    if len(correct) > n_correct:
        idx = rng.choice(len(correct), size=n_correct, replace=False)
        correct = [correct[i] for i in sorted(idx.tolist())]

    return Cohort(
        truly_wrong=truly_wrong,
        originally_correct=correct,
        truncated=truncated,
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
        the workspace's jsonl. Used to skip already-judged calls when
        resuming a partial sweep."""
        if not self._jsonl.exists():
            return set()
        keys: set[tuple[str, float, str, int]] = set()
        with self._jsonl.open() as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
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
    rows = [r for r in rows if r["label"] >= 0]

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
        """Stub — orchestration lives in ``experiments/c7_backtracking/run.py``.

        The component runner's ``eval_fn`` adapter calls
        :func:`run_arch_evaluation` (TODO once the reasoning-model
        loader is wired in). At that point this method becomes a
        one-liner forwarder.
        """
        raise NotImplementedError(
            "evaluate(arch, seed) is wired by experiments/c7_backtracking/run.py "
            "via run_arch_evaluation(...). Direct CaseStudy.evaluate() use is "
            "deferred until the reasoning-model + steering helpers land."
        )

    def teardown(self) -> None:
        self.stage_a = None
        self.cohort = None
        self.judge = None
