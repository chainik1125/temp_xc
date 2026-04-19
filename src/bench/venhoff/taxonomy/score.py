"""Judge-driven scoring for Venhoff's three taxonomy-quality metrics.

Computes the same three scalars Venhoff reports (`accuracy`,
`completeness`, `semantic_orthogonality`) but routes every judge call
through the provider-agnostic `Judge` interface in `judge_client.py`
instead of their OpenAI-batch-specific `clustering_batched.py`.

Reusing the ported prompts in `autograder_prompts.py` keeps the metric
comparable; the orchestration differs because:
  - we default to Anthropic (Haiku 4.5) not OpenAI
  - Haiku-4.5 pricing makes sync async-concurrent fine; no need for
    the overnight batch-API path their code uses
  - keeps provider swap for the bridge-drift check (Q5 lock-in)
    mechanical — same functions, different Judge instance

See `docs/aniket/experiments/venhoff_eval/plan.md § 2` for metric
definitions and `VENHOFF_PROVENANCE.md` for the contract-preservation
argument.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Sequence

from src.bench.venhoff.autograder_prompts import (
    build_accuracy_autograder_prompt,
    build_completeness_autograder_prompt,
    build_semantic_orthogonality_prompt,
    format_sentences_text,
)
from src.bench.venhoff.judge_client import Judge

log = logging.getLogger("venhoff.score")


@dataclass(frozen=True)
class Cluster:
    """A labeled cluster produced by the taxonomy step.

    `title` and `description` come from the label step (GPT-4o /
    Haiku-4.5 generated). `sentence_indices` are the sentences Venhoff's
    annotator argmax-assigned to this cluster.
    """

    cluster_id: int
    title: str
    description: str
    sentence_indices: list[int]


@dataclass(frozen=True)
class TaxonomyScores:
    """Composite taxonomy-quality scores for one trained dictionary.

    All three scalars are on a 0-10 scale except `accuracy`, which is
    0-1 (fraction of judge Yes votes). `avg_final_score` is the
    composite Venhoff reports in their headline table; it weights the
    three sub-metrics equally after rescaling accuracy to 0-10.
    """

    accuracy: float
    completeness: float
    semantic_orthogonality: float
    avg_final_score: float
    n_judge_calls: int
    n_judge_errors: int


# ───────── accuracy ─────────

# Venhoff's default: each cluster is judged on `accuracy_target_cluster_percentage`
# fraction of `n_autograder_examples`, drawn from its own members. The judge
# returns Yes/No per sentence; the cluster's accuracy is the fraction of Yes
# votes. The dictionary-level accuracy is the mean across clusters.
DEFAULT_N_AUTOGRADER_EXAMPLES = 100
DEFAULT_TARGET_CLUSTER_PERCENTAGE = 0.2
MAX_SENTENCES_PER_PROMPT = 50


async def _judge_accuracy_for_cluster(
    judge: Judge,
    cluster: Cluster,
    sentences: Sequence[str],
    n_examples: int,
    rng: random.Random,
) -> tuple[float, int, int]:
    """Return (fraction_yes, n_calls, n_errors) for one cluster."""
    if not cluster.sentence_indices:
        return 0.0, 0, 0

    n_sample = min(n_examples, len(cluster.sentence_indices))
    sampled_idx = rng.sample(cluster.sentence_indices, n_sample)
    sampled_sentences = [sentences[i] for i in sampled_idx]

    n_yes = 0
    n_total = 0
    n_calls_before = judge.n_calls
    n_errors_before = judge.n_errors

    for chunk_start in range(0, len(sampled_sentences), MAX_SENTENCES_PER_PROMPT):
        chunk = sampled_sentences[chunk_start : chunk_start + MAX_SENTENCES_PER_PROMPT]
        prompt = build_accuracy_autograder_prompt(
            title=cluster.title,
            description=cluster.description,
            sentences_text=format_sentences_text(chunk),
        )
        raw = await judge.call(system="", user=prompt)
        parsed = _parse_json_block(raw)
        if parsed is None or "classifications" not in parsed:
            continue
        for item in parsed["classifications"]:
            label = str(item.get("belongs_to_category", "")).strip().lower()
            if label in ("yes", "no"):
                n_total += 1
                if label == "yes":
                    n_yes += 1

    n_calls = judge.n_calls - n_calls_before
    n_errors = judge.n_errors - n_errors_before
    fraction_yes = (n_yes / n_total) if n_total else 0.0
    return fraction_yes, n_calls, n_errors


async def score_accuracy(
    judge: Judge,
    clusters: Sequence[Cluster],
    sentences: Sequence[str],
    n_autograder_examples: int = DEFAULT_N_AUTOGRADER_EXAMPLES,
    target_cluster_percentage: float = DEFAULT_TARGET_CLUSTER_PERCENTAGE,
    seed: int = 0,
) -> tuple[float, int, int]:
    """Mean-over-clusters accuracy in [0, 1]."""
    rng = random.Random(seed)
    per_cluster_target = max(1, int(round(n_autograder_examples * target_cluster_percentage)))

    tasks = [
        _judge_accuracy_for_cluster(judge, c, sentences, per_cluster_target, rng)
        for c in clusters
    ]
    results = await asyncio.gather(*tasks)
    fractions = [r[0] for r in results]
    total_calls = sum(r[1] for r in results)
    total_errors = sum(r[2] for r in results)
    mean_acc = sum(fractions) / len(fractions) if fractions else 0.0
    return mean_acc, total_calls, total_errors


# ───────── completeness ─────────


async def _judge_completeness_for_sentence(
    judge: Judge,
    sentence: str,
    title: str,
    description: str,
) -> int | None:
    """Return the 0-10 completeness score, or None if judge call failed."""
    prompt = build_completeness_autograder_prompt(sentence, title, description)
    raw = await judge.call(system="", user=prompt)
    parsed = _parse_json_block(raw)
    if parsed is None:
        return None
    try:
        score = int(parsed["completeness_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= score <= 10):
        return None
    return score


async def score_completeness(
    judge: Judge,
    clusters: Sequence[Cluster],
    sentences: Sequence[str],
    n_examples_per_cluster: int = 20,
    seed: int = 0,
) -> tuple[float, int, int]:
    """Mean 0-10 completeness across sampled (sentence, cluster) pairs."""
    rng = random.Random(seed)
    pairs: list[tuple[str, str, str]] = []
    for c in clusters:
        if not c.sentence_indices:
            continue
        n_sample = min(n_examples_per_cluster, len(c.sentence_indices))
        for idx in rng.sample(c.sentence_indices, n_sample):
            pairs.append((sentences[idx], c.title, c.description))

    n_calls_before = judge.n_calls
    n_errors_before = judge.n_errors
    scores = await asyncio.gather(
        *[_judge_completeness_for_sentence(judge, s, t, d) for s, t, d in pairs]
    )
    valid = [s for s in scores if s is not None]
    mean = sum(valid) / len(valid) if valid else 0.0
    return mean, judge.n_calls - n_calls_before, judge.n_errors - n_errors_before


# ───────── semantic orthogonality ─────────


async def _judge_pair_similarity(
    judge: Judge,
    a: Cluster,
    b: Cluster,
) -> int | None:
    prompt = build_semantic_orthogonality_prompt(
        title1=a.title,
        description1=a.description,
        title2=b.title,
        description2=b.description,
    )
    raw = await judge.call(system="", user=prompt)
    parsed = _parse_json_block(raw)
    if parsed is None:
        return None
    try:
        score = int(parsed["similarity_score"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= score <= 10):
        return None
    return score


async def score_semantic_orthogonality(
    judge: Judge,
    clusters: Sequence[Cluster],
) -> tuple[float, int, int]:
    """Mean pairwise orthogonality (inverted similarity) in [0, 10].

    `orthogonality = 10 - similarity` per Venhoff's convention. Missing
    judge calls are skipped, not zero-imputed.
    """
    n_calls_before = judge.n_calls
    n_errors_before = judge.n_errors

    pairs: list[tuple[Cluster, Cluster]] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            pairs.append((clusters[i], clusters[j]))

    similarities = await asyncio.gather(
        *[_judge_pair_similarity(judge, a, b) for a, b in pairs]
    )
    valid = [s for s in similarities if s is not None]
    orthogonalities = [10 - s for s in valid]
    mean = sum(orthogonalities) / len(orthogonalities) if orthogonalities else 0.0
    return mean, judge.n_calls - n_calls_before, judge.n_errors - n_errors_before


# ───────── composite ─────────


async def score_taxonomy(
    judge: Judge,
    clusters: Sequence[Cluster],
    sentences: Sequence[str],
    seed: int = 0,
) -> TaxonomyScores:
    """Compute all three metrics and the composite avg_final_score."""
    acc, c1, e1 = await score_accuracy(judge, clusters, sentences, seed=seed)
    comp, c2, e2 = await score_completeness(judge, clusters, sentences, seed=seed)
    ortho, c3, e3 = await score_semantic_orthogonality(judge, clusters)

    # Rescale accuracy (0-1) to 0-10 so the three components are on the
    # same scale when averaged into the composite.
    acc_on_ten = acc * 10.0
    composite = (acc_on_ten + comp + ortho) / 3.0

    return TaxonomyScores(
        accuracy=acc,
        completeness=comp,
        semantic_orthogonality=ortho,
        avg_final_score=composite,
        n_judge_calls=c1 + c2 + c3,
        n_judge_errors=e1 + e2 + e3,
    )


# ───────── parser ─────────

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


def _parse_json_block(raw: str) -> dict | None:
    """Extract the JSON dict from a judge response.

    Venhoff's prompts request fenced JSON; the judge sometimes wraps
    with prose. We try the fenced block first, fall back to the raw
    string as a last resort.
    """
    if not raw:
        return None
    match = _JSON_FENCE_RE.search(raw)
    candidate = match.group(1) if match else raw
    try:
        result = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if isinstance(result, dict):
        return result
    return None
