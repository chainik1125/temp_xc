"""GPT-4o drift bridge for the Haiku-4.5 judge swap (Q5 lock-in).

At smoke time, sample 100 (sentence, cluster_title, cluster_description)
triples from the labeled clusters, re-judge them with both Haiku 4.5
and GPT-4o on the completeness rubric, and report per-metric drift.

**Pass/fail rule:** mean per-metric drift ≤ 0.5 points on the 0-10
rubric. Exit non-zero if over. This gates whether we trust Haiku 4.5
as a stand-in for Venhoff's paper-default GPT-4o on the full sweep.

Per `VENHOFF_PROVENANCE.md`: Venhoff's released code default at the
pinned commit is actually `gpt-4.1-mini`, not `gpt-4o`. We bridge
against `gpt-4o` (the paper number) so a pass means we reproduce the
paper's judge contract.

Usage:
    python -m src.bench.venhoff.judge_bridge \\
        --labeled-clusters path/to/clusters.json \\
        --sentences path/to/sentences.json \\
        --n 100 --out path/to/bridge_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from src.bench.venhoff.judge_client import GPT_4O, HAIKU_4_5, make_judge
from src.bench.venhoff.taxonomy.score import _judge_completeness_for_sentence  # noqa: F401  (intentional reuse)
from src.bench.venhoff.taxonomy.score import Cluster

DRIFT_THRESHOLD = 0.5
DEFAULT_N_SENTENCES = 100

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.bridge")


@dataclass(frozen=True)
class BridgeResult:
    n_sentences: int
    n_valid_pairs: int
    mean_haiku: float
    mean_gpt4o: float
    mean_abs_drift: float
    passed: bool


async def _score_set(judge, triples: list[tuple[str, str, str]]) -> list[int | None]:
    from src.bench.venhoff.taxonomy.score import _judge_completeness_for_sentence as one

    return await asyncio.gather(*[one(judge, s, t, d) for s, t, d in triples])


async def run_bridge(
    triples: list[tuple[str, str, str]],
    haiku_model: str = HAIKU_4_5,
    gpt_model: str = GPT_4O,
) -> BridgeResult:
    """Judge the same triples with two models, report drift."""
    haiku = make_judge(haiku_model)
    gpt = make_judge(gpt_model)

    haiku_scores, gpt_scores = await asyncio.gather(
        _score_set(haiku, triples),
        _score_set(gpt, triples),
    )

    paired: list[tuple[int, int]] = [
        (h, g) for h, g in zip(haiku_scores, gpt_scores) if h is not None and g is not None
    ]
    n_valid = len(paired)
    if n_valid == 0:
        return BridgeResult(
            n_sentences=len(triples),
            n_valid_pairs=0,
            mean_haiku=0.0,
            mean_gpt4o=0.0,
            mean_abs_drift=float("inf"),
            passed=False,
        )

    mean_h = sum(h for h, _ in paired) / n_valid
    mean_g = sum(g for _, g in paired) / n_valid
    mean_abs = sum(abs(h - g) for h, g in paired) / n_valid

    return BridgeResult(
        n_sentences=len(triples),
        n_valid_pairs=n_valid,
        mean_haiku=mean_h,
        mean_gpt4o=mean_g,
        mean_abs_drift=mean_abs,
        passed=mean_abs <= DRIFT_THRESHOLD,
    )


def _load_triples(
    clusters_path: Path,
    sentences_path: Path,
    n: int,
    seed: int,
) -> list[tuple[str, str, str]]:
    with clusters_path.open() as f:
        clusters_data = json.load(f)
    with sentences_path.open() as f:
        sentences = json.load(f)

    clusters = [Cluster(**c) for c in clusters_data]

    rng = random.Random(seed)
    triples: list[tuple[str, str, str]] = []
    for c in clusters:
        for idx in c.sentence_indices:
            triples.append((sentences[idx], c.title, c.description))
    rng.shuffle(triples)
    return triples[:n]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--labeled-clusters", type=Path, required=True)
    parser.add_argument("--sentences", type=Path, required=True)
    parser.add_argument("--n", type=int, default=DEFAULT_N_SENTENCES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--haiku-model", default=HAIKU_4_5)
    parser.add_argument("--gpt-model", default=GPT_4O)
    args = parser.parse_args(argv)

    triples = _load_triples(args.labeled_clusters, args.sentences, args.n, args.seed)
    if not triples:
        log.error("no (sentence, title, description) triples loaded — check inputs")
        return 2

    result = asyncio.run(run_bridge(triples, args.haiku_model, args.gpt_model))
    args.out.write_text(json.dumps(asdict(result), indent=2))

    log.info("bridge: n=%d valid=%d mean_haiku=%.3f mean_gpt4o=%.3f |drift|=%.3f pass=%s",
             result.n_sentences, result.n_valid_pairs, result.mean_haiku,
             result.mean_gpt4o, result.mean_abs_drift, result.passed)

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
