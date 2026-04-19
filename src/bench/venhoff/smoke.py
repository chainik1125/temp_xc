"""End-to-end Phase 1a smoke test with resume/skip semantics.

Runs every stage of the Venhoff pipeline on the smoke config (1k
traces, cluster_size=15, SAE baseline on Path 1), and the judge drift
bridge on 100 sentences. Each stage checks a cached artifact before
running, so re-invocations are cheap and selectively rebuildable.

Flags for selective rebuild:
  --force                 — rebuild every stage
  --force-stage traces    — rebuild only trace generation
  --force-stage activations
  --force-stage train
  --force-stage annotate
  --force-stage label
  --force-stage score
  --force-stage bridge
  --skip-stage bridge     — opposite: skip a stage even if stale
  --skip-bridge           — convenience for cost-saving smoke rehearsals

Exit code:
  0 — smoke passed end-to-end (incl. bridge drift ≤ threshold)
  1 — any stage failed, or bridge drift exceeded threshold
  2 — usage error / config mismatch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Iterable

from src.bench.venhoff.activation_collection import (
    CollectionConfig,
    _load_model,
    collect_path1,
    collect_path3,
)
from src.bench.venhoff.annotate import annotate
from src.bench.venhoff.generate_traces import generate
from src.bench.venhoff.judge_bridge import run_bridge
from src.bench.venhoff.judge_client import HAIKU_4_5, make_judge
from src.bench.venhoff.paths import ArtifactPaths, RunIdentity, can_resume, write_with_metadata
from src.bench.venhoff.taxonomy.label import UnlabeledCluster, label_clusters
from src.bench.venhoff.taxonomy.score import Cluster, score_taxonomy
from src.bench.venhoff.train_small_sae import TrainConfig, train

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.smoke")

STAGES = ("traces", "activations", "train", "annotate", "label", "score", "bridge")

# Smoke defaults — match Phase 1a lock-ins (plan.md § 4 / Q4).
SMOKE_N_TRACES = 1000
SMOKE_CLUSTER_SIZE = 15
SMOKE_ARCH = "sae"
SMOKE_PATH = "path1"
SMOKE_AGGREGATION = "full_window"  # unused for path1 but keeps the filename scheme consistent
BRIDGE_N_SENTENCES = 100


def _stage_should_run(stage: str, force: bool, force_stage: set[str], skip_stage: set[str]) -> bool:
    if stage in skip_stage:
        log.info("[info] skip | stage=%s | reason=user_requested", stage)
        return False
    if force or stage in force_stage:
        return True
    return True  # actual cache check happens inside each stage


def _force(stage: str, force: bool, force_stage: set[str]) -> bool:
    return force or (stage in force_stage)


async def _label_and_score(
    paths: ArtifactPaths,
    arch: str,
    cluster_size: int,
    path_name: str,
    aggregation: str,
    judge_model: str,
    force_label: bool,
    force_score: bool,
) -> dict:
    """Run label + score stages and return a summary dict."""
    labels_path = paths.labels_json(arch, cluster_size, path_name, aggregation)
    scores_path = paths.scores_json(arch, cluster_size, path_name, aggregation)

    # ── Labels
    label_meta = {"stage": "label", "arch": arch, "cluster_size": cluster_size,
                  "path": path_name, "aggregation": aggregation, "judge": judge_model}
    if force_label or not can_resume(labels_path, label_meta):
        assignments_path = paths.assignments_json(arch, cluster_size, path_name, aggregation)
        with assignments_path.open() as f:
            asg = json.load(f)
        with paths.sentences_json(path_name).open() as f:
            sentences = json.load(f)
        unlabeled = [
            UnlabeledCluster(cluster_id=int(cid), sentence_indices=idxs)
            for cid, idxs in asg["clusters"].items()
        ]
        judge = make_judge(judge_model)
        clusters = await label_clusters(judge, unlabeled, sentences, seed=paths.identity.seed)
        payload = json.dumps([_cluster_to_dict(c) for c in clusters], indent=2)
        write_with_metadata(labels_path, payload, label_meta)
        log.info(
            "[done] saved labels | arch=%s | cluster_size=%d | path=%s | aggregation=%s | n_labeled=%d | n_clusters_nonempty=%d | path_out=%s",
            arch, cluster_size, path_name, aggregation, len(clusters), len(unlabeled), labels_path,
        )
    else:
        log.info("[info] resume | stage=label | cache=%s", labels_path)

    # ── Scores
    score_meta = {"stage": "score", "arch": arch, "cluster_size": cluster_size,
                  "path": path_name, "aggregation": aggregation, "judge": judge_model}
    if force_score or not can_resume(scores_path, score_meta):
        with labels_path.open() as f:
            clusters_raw = json.load(f)
        clusters = [Cluster(**c) for c in clusters_raw]
        with paths.sentences_json(path_name).open() as f:
            sentences = json.load(f)
        judge = make_judge(judge_model)
        scores = await score_taxonomy(judge, clusters, sentences, seed=paths.identity.seed)
        payload = json.dumps(_scores_to_dict(scores), indent=2)
        write_with_metadata(scores_path, payload, score_meta)
        log.info(
            "[eval] taxonomy_scores | arch=%s | cluster_size=%d | path=%s | aggregation=%s | accuracy=%.4f | completeness=%.4f | semantic_orthogonality=%.4f | avg_final_score=%.4f",
            arch, cluster_size, path_name, aggregation,
            scores.accuracy, scores.completeness, scores.semantic_orthogonality, scores.avg_final_score,
        )
        log.info("[done] saved scores | path=%s", scores_path)
    else:
        log.info("[info] resume | stage=score | cache=%s", scores_path)

    return json.loads(scores_path.read_text())


def _cluster_to_dict(c: Cluster) -> dict:
    return {
        "cluster_id": c.cluster_id, "title": c.title,
        "description": c.description, "sentence_indices": c.sentence_indices,
    }


def _scores_to_dict(s) -> dict:
    from dataclasses import asdict
    return asdict(s)


async def run_smoke(
    paths: ArtifactPaths,
    model_name: str = "deepseek-r1-distill-llama-8b",
    arch: str = SMOKE_ARCH,
    cluster_size: int = SMOKE_CLUSTER_SIZE,
    path_name: str = SMOKE_PATH,
    aggregation: str = SMOKE_AGGREGATION,
    engine: str = "vllm",
    judge_model: str = HAIKU_4_5,
    force: bool = False,
    force_stages: Iterable[str] = (),
    skip_stages: Iterable[str] = (),
    device: str = "cuda",
) -> int:
    force_set = set(force_stages)
    skip_set = set(skip_stages)
    paths.ensure_dirs()

    # Stage 1: traces
    if _stage_should_run("traces", force, force_set, skip_set):
        generate(
            paths=paths, model_name=model_name, n_traces=paths.identity.n_traces,
            engine=engine, force=_force("traces", force, force_set),
        )

    # Stage 2: activations
    if _stage_should_run("activations", force, force_set, skip_set):
        model, tokenizer = _load_model(model_name)
        cfg = CollectionConfig(path=path_name, layer=paths.identity.layer)
        if path_name == "path1":
            collect_path1(paths, model, tokenizer, cfg,
                          force=_force("activations", force, force_set))
        else:
            collect_path3(paths, model, tokenizer, cfg,
                          force=_force("activations", force, force_set))

    # Stage 3: train small-k dict
    if _stage_should_run("train", force, force_set, skip_set):
        tcfg = TrainConfig(
            arch=arch, cluster_size=cluster_size, path=path_name,
            seed=paths.identity.seed,
        )
        train(paths, tcfg, device=device, force=_force("train", force, force_set))

    # Stage 4: annotate
    if _stage_should_run("annotate", force, force_set, skip_set):
        annotate(
            paths=paths, arch=arch, cluster_size=cluster_size,
            path_name=path_name, aggregation=aggregation, device=device,
            force=_force("annotate", force, force_set),
        )

    # Stages 5 + 6: label + score (combined because they share a judge config)
    scores_payload = None
    if _stage_should_run("label", force, force_set, skip_set) or \
       _stage_should_run("score", force, force_set, skip_set):
        scores_payload = await _label_and_score(
            paths=paths, arch=arch, cluster_size=cluster_size,
            path_name=path_name, aggregation=aggregation, judge_model=judge_model,
            force_label=_force("label", force, force_set),
            force_score=_force("score", force, force_set),
        )

    # Stage 7: bridge drift (Q5 lock-in)
    bridge_ok = True
    if _stage_should_run("bridge", force, force_set, skip_set):
        bridge_ok = await _run_bridge_stage(paths, arch, cluster_size, path_name,
                                            aggregation, force=_force("bridge", force, force_set))

    composite = (scores_payload or {}).get("avg_final_score")
    if composite is not None:
        log.info(
            "[result] smoke_done | avg_final_score=%.4f | bridge_pass=%s",
            composite, bridge_ok,
        )
    else:
        log.info("[result] smoke_done | avg_final_score=skipped | bridge_pass=%s", bridge_ok)
    return 0 if bridge_ok else 1


async def _run_bridge_stage(
    paths: ArtifactPaths,
    arch: str,
    cluster_size: int,
    path_name: str,
    aggregation: str,
    force: bool,
) -> bool:
    # Cell info is part of the resume key — two bridge runs against
    # different cells in the same run_dir must not collide on the cache.
    meta = {
        "stage": "bridge",
        "n": BRIDGE_N_SENTENCES,
        "arch": arch,
        "cluster_size": cluster_size,
        "path": path_name,
        "aggregation": aggregation,
        "seed": paths.identity.seed,
    }
    out = paths.bridge_json
    if not force and can_resume(out, meta):
        existing = json.loads(out.read_text())
        return bool(existing.get("passed"))

    labels_path = paths.labels_json(arch, cluster_size, path_name, aggregation)
    if not labels_path.exists():
        log.error("[error] bridge_missing_labels | expected_path=%s | hint=run_label_stage_first", labels_path)
        return False
    with labels_path.open() as f:
        clusters = [Cluster(**c) for c in json.load(f)]
    with paths.sentences_json(path_name).open() as f:
        sentences = json.load(f)

    triples: list[tuple[str, str, str]] = []
    for c in clusters:
        for i in c.sentence_indices:
            triples.append((sentences[i], c.title, c.description))
    # Keep it deterministic with the run's own seed.
    import random
    rng = random.Random(paths.identity.seed)
    rng.shuffle(triples)
    triples = triples[:BRIDGE_N_SENTENCES]

    result = await run_bridge(triples)
    from dataclasses import asdict
    write_with_metadata(out, json.dumps(asdict(result), indent=2), meta)
    log.info(
        "[eval] bridge_drift | n=%d | valid=%d | mean_haiku=%.4f | mean_gpt4o=%.4f | mean_abs_drift=%.4f | pass=%s",
        result.n_sentences, result.n_valid_pairs,
        result.mean_haiku, result.mean_gpt4o, result.mean_abs_drift, result.passed,
    )
    log.info("[done] saved bridge_drift | path=%s", out)
    return result.passed


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", default="deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset", default="mmlu-pro")
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, default=SMOKE_N_TRACES)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--arch", default=SMOKE_ARCH, choices=["sae", "mlc", "tempxc"])
    p.add_argument("--cluster-size", type=int, default=SMOKE_CLUSTER_SIZE)
    p.add_argument("--path", default=SMOKE_PATH, choices=["path1", "path3"])
    p.add_argument("--aggregation", default=SMOKE_AGGREGATION,
                   choices=["last", "mean", "max", "full_window"])
    p.add_argument("--engine", default="vllm", choices=["vllm", "transformers"])
    p.add_argument("--judge-model", default=HAIKU_4_5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--force", action="store_true",
                   help="rebuild every stage ignoring caches")
    p.add_argument("--force-stage", action="append", default=[],
                   choices=list(STAGES), help="rebuild a specific stage")
    p.add_argument("--skip-stage", action="append", default=[],
                   choices=list(STAGES), help="skip a stage")
    p.add_argument("--skip-bridge", action="store_true",
                   help="shortcut for --skip-stage bridge (no-cost rehearsal)")
    args = p.parse_args(argv)

    skip_stages: list[str] = list(args.skip_stage)
    if args.skip_bridge:
        skip_stages.append("bridge")

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model, dataset=args.dataset, dataset_split=args.split,
            n_traces=args.n_traces, layer=args.layer, seed=args.seed,
        ),
    )
    return asyncio.run(run_smoke(
        paths=paths,
        model_name=args.model, arch=args.arch, cluster_size=args.cluster_size,
        path_name=args.path, aggregation=args.aggregation,
        engine=args.engine, judge_model=args.judge_model,
        force=args.force, force_stages=args.force_stage, skip_stages=skip_stages,
        device=args.device,
    ))


if __name__ == "__main__":
    sys.exit(main())
