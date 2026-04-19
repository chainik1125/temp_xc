"""Cluster-title + description generation for Venhoff's taxonomy step.

Given an argmax-assigned cluster (integer id + member sentences), ask
the judge to produce a (title, description) pair following Venhoff's
`build_cluster_description_prompt` exactly. The judge response is
regex-parsed for `Title: …` / `Description: …`.

This replaces Venhoff's `train-saes/generate_titles_trained_clustering.py`
in the pipeline, using our provider-agnostic judge client and the
already-ported prompt. The prompt itself is held fixed by the
pre-registered-invariant contract in `VENHOFF_PROVENANCE.md`.
"""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from typing import Sequence

from src.bench.venhoff.autograder_prompts import build_cluster_description_prompt
from src.bench.venhoff.judge_client import Judge
from src.bench.venhoff.taxonomy.score import Cluster

log = logging.getLogger("venhoff.label")


@dataclass(frozen=True)
class UnlabeledCluster:
    """Input to the labeler: a cluster of sentence indices, no text yet."""

    cluster_id: int
    sentence_indices: list[int]


# Venhoff's default: sample up to 50 example sentences per cluster for the
# description prompt. Their script uses `description_examples=200` as an
# arg default but the prompt truncates to 50 for context-length reasons.
DEFAULT_DESCRIPTION_EXAMPLES = 50


_TITLE_RE = re.compile(r"^\s*Title:\s*(.+?)\s*$", re.MULTILINE)
_DESC_RE = re.compile(r"^\s*Description:\s*(.+?)(?=\n\s*(?:Title:|$))", re.MULTILINE | re.DOTALL)


def _parse_title_description(raw: str) -> tuple[str, str] | None:
    """Extract (title, description) from a cluster-description response."""
    if not raw:
        return None
    title_match = _TITLE_RE.search(raw)
    desc_match = _DESC_RE.search(raw)
    if not title_match:
        return None
    title = title_match.group(1).strip()
    description = desc_match.group(1).strip() if desc_match else ""
    if not title:
        return None
    return title, description


async def _label_one(
    judge: Judge,
    cluster: UnlabeledCluster,
    sentences: Sequence[str],
    n_examples: int,
    n_categories_examples: int,
    rng: random.Random,
) -> Cluster | None:
    """Produce a labeled cluster, or None if the judge call failed."""
    if not cluster.sentence_indices:
        return None
    n_sample = min(n_examples, len(cluster.sentence_indices))
    sampled_idx = rng.sample(cluster.sentence_indices, n_sample)
    examples = [sentences[i] for i in sampled_idx]

    prompt = build_cluster_description_prompt(
        examples=examples,
        trace_examples_text="",
        n_categories_examples=n_categories_examples,
    )
    raw = await judge.call(system="", user=prompt)
    parsed = _parse_title_description(raw)
    if parsed is None:
        log.warning("label parse failed for cluster %d (raw=%r)", cluster.cluster_id, raw[:200])
        return None
    title, description = parsed
    return Cluster(
        cluster_id=cluster.cluster_id,
        title=title,
        description=description,
        sentence_indices=list(cluster.sentence_indices),
    )


async def label_clusters(
    judge: Judge,
    clusters: Sequence[UnlabeledCluster],
    sentences: Sequence[str],
    n_examples: int = DEFAULT_DESCRIPTION_EXAMPLES,
    n_categories_examples: int = 5,
    seed: int = 0,
) -> list[Cluster]:
    """Label a set of unlabeled clusters concurrently.

    Returns only clusters the judge successfully labeled. A failed
    cluster is logged and dropped; downstream scoring handles a
    reduced cluster count correctly (metrics are mean-over-clusters).
    """
    rng = random.Random(seed)
    tasks = [
        _label_one(judge, c, sentences, n_examples, n_categories_examples, rng)
        for c in clusters
    ]
    results = await asyncio.gather(*tasks)
    return [c for c in results if c is not None]
