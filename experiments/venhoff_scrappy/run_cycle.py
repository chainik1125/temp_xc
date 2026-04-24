"""Run one venhoff_scrappy cycle: Phase 0 (cached) -> Phase 2 -> Phase 3 -> grade.

Reads scrappy defaults from ../config.yaml and deep-merges per-candidate
overrides from ../candidates/<name>.yaml. Dispatches the vendor pipeline
as subprocesses and writes a normalized `grade_results.json` to the
cycle's result dir.

The orchestrator (`run_autoresearch.sh`) calls `autoresearch_summarise.py`
immediately after this to compute Δ vs baseline and append a ledger row.

Design constraints (baked-in from 2026-04-22/23 paper-budget lessons):
  - Phase 0 MUST be cached at `paths.root` BEFORE any cycle runs;
    `phase0_bootstrap.sh` on the pod handles this. Cycles symlink the
    cached traces+activations into their own results tree.
  - Venhoff's `_expected_results_dir` for hybrid is keyed by
    (base, thinking, layer, sae_layer, n_clusters). Candidates sharing
    those 5 fields would clobber each other's results; we move the dir
    to the cycle-local path after grading.
  - `thinking_acc` and `base_acc` are constants across candidates on a
    fixed slice — compute once and cache under `phase3/cache/` to cut
    judge calls 60 -> 20 per subsequent cycle.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
SCRAPPY = REPO / "experiments/venhoff_scrappy"
SCRAPPY_DEFAULTS = SCRAPPY / "config.yaml"
PHASE0_ROOT = SCRAPPY / "results/phase0"
GRADE_CACHE_DIR = SCRAPPY / "results/grade_cache"


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_candidate(cand_cfg_path: Path) -> dict:
    base = yaml.safe_load(SCRAPPY_DEFAULTS.read_text())
    override = yaml.safe_load(cand_cfg_path.read_text())
    merged = _deep_merge(base, override)
    for k in ("name", "arch", "baseline", "hypothesis", "reuse_venhoff_vectors",
              "shuffle_activations", "loss_override"):
        if k in override:
            merged[k] = override[k]
    return merged


def _sha256_of_file(p: Path) -> str | None:
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()[:16]}"


def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1].lower()


def _run_identity_slug(cfg: dict) -> str:
    """Canonical slug for Phase 0 artifacts. Shared across cycles that
    agree on (thinking_model, dataset, split, n_traces, steering_layer, seed)."""
    return (
        f"{_model_slug(cfg['model']['thinking'])}_"
        f"{cfg['phase3_hybrid']['dataset']}-{cfg['phase3_hybrid']['dataset_split']}_"
        f"n{cfg['phase0_activations']['n_traces']}_"
        f"L{cfg['model']['steering_layer']}_"
        f"seed{cfg['autoresearch']['seed']}"
    )


def _link_phase0_into_cycle(cfg: dict, cycle_eval_root: Path) -> Path:
    """Symlink the shared Phase 0 artifacts into the cycle's venhoff_eval
    tree so Phase 2 + 3 find them at the expected ArtifactPaths location.

    Requires phase0_bootstrap.sh to have run first. Fails loud if not.
    """
    slug = _run_identity_slug(cfg)
    src = PHASE0_ROOT / slug
    dst = cycle_eval_root / slug
    if not src.exists():
        raise FileNotFoundError(
            f"Phase 0 cache missing at {src}. "
            f"Run bash scripts/phase0_bootstrap.sh on the pod first."
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src.resolve(), target_is_directory=True)
    return dst


def _shuffle_activations_in_place(link_dir: Path, seed: int) -> None:
    """For the baseline_sae_shuffled candidate: permute activations
    within-sequence before Phase 2 reads them. Writes a replacement
    activations_path3.pkl *inside the symlinked dir* — since the link
    target is the shared cache, we break the link first and copy.
    """
    import pickle
    import random as _random
    if link_dir.is_symlink():
        real = link_dir.resolve()
        link_dir.unlink()
        shutil.copytree(real, link_dir)

    for path_tag in ("path1", "path3"):
        p = link_dir / f"activations_{path_tag}.pkl"
        if not p.exists():
            continue
        with p.open("rb") as f:
            acts = pickle.load(f)
        rng = _random.Random(seed)
        # acts is typically a list of per-trace arrays; shuffle within
        # each trace. If the format is different, no-op with a warning.
        shuffled = []
        for trace in acts:
            try:
                perm = list(range(len(trace)))
                rng.shuffle(perm)
                shuffled.append([trace[i] for i in perm])
            except TypeError:
                print(f"[warn] activations format unknown for {path_tag}; skipping shuffle")
                return
        with p.open("wb") as f:
            pickle.dump(shuffled, f)
        print(f"[info] shuffled activations within-sequence: {p}")


def _run_subprocess(cmd: list[str], *, cwd: Path | None = None, env: dict | None = None) -> None:
    log_cmd = " ".join(cmd)
    print(f"[cmd] {log_cmd}", flush=True)
    result = subprocess.run(cmd, cwd=cwd, env=env or os.environ.copy(), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"subprocess failed (rc={result.returncode}): {log_cmd}")


def _venhoff_benchmark_json_path(venhoff_root: Path, cfg: dict) -> Path:
    """Venhoff's hybrid_token.py writes aggregate results to
    `hybrid/results/benchmark_results_<base_short>_<dataset>.json` —
    NOT the per-arch subdir our earlier wrapper assumed.

    Schema (verified 2026-04-24 on scrappy baseline_sae cycle):
      metadata: { base_model, thinking_model, n_tasks }
      results.accuracy:  { base_model, thinking_model, hybrid_model }  # percentages 0-100
      results.correct_count: { base_model, thinking_model, hybrid_model }
      tasks[]: { question, correct_answer, model_answers.{base,thinking,hybrid}_model }
    """
    base_short = _model_slug(cfg["model"]["base"])
    return (
        venhoff_root / "hybrid" / "results"
        / f"benchmark_results_{base_short}_{cfg['phase3_hybrid']['dataset']}.json"
    )


def _parse_benchmark_json(benchmark_path: Path) -> dict:
    """Read Venhoff's aggregate results + re-grade per-task outcomes for
    the hybrid model. Returns accuracies (as fractions 0-1), gap_recovery,
    and per_task_outcomes (20-element bool array).

    We re-grade hybrid outputs locally using src.bench.venhoff.grade.is_correct
    rather than depending on a per-task correctness flag in the JSON
    (Venhoff's schema doesn't include one at the aggregate level).
    """
    from src.bench.venhoff.grade import extract_boxed, is_correct

    data = json.loads(benchmark_path.read_text())
    acc = data["results"]["accuracy"]
    thinking_acc = float(acc["thinking_model"]) / 100.0
    base_acc = float(acc["base_model"]) / 100.0
    hybrid_acc = float(acc["hybrid_model"]) / 100.0

    denom = thinking_acc - base_acc
    if denom > 0:
        gap_recovery = (hybrid_acc - base_acc) / denom
    else:
        gap_recovery = 0.0

    per_task_outcomes: list[bool] = []
    for task in data.get("tasks", []):
        ref = str(task.get("correct_answer", ""))
        answers = task.get("model_answers", {})
        pred = extract_boxed(answers.get("hybrid_model") or "")
        per_task_outcomes.append(bool(is_correct(pred, ref)))

    return {
        "thinking_acc": thinking_acc,
        "base_acc": base_acc,
        "hybrid_acc": hybrid_acc,
        "gap_recovery": gap_recovery,
        "per_task_outcomes": per_task_outcomes,
        "n_tasks": data.get("metadata", {}).get("n_tasks", len(data.get("tasks", []))),
    }


def _use_cached_thinking_base_grades(cfg: dict, grade_payload: dict) -> dict:
    """If thinking_acc/base_acc were graded in a prior cycle on the same
    slice, reuse those numbers. Otherwise cache the ones we just computed.
    """
    if not cfg["grade"].get("cache_thinking_base_grades", False):
        return grade_payload
    slice_key = f"{cfg['phase3_hybrid']['dataset']}_n{cfg['phase3_hybrid']['n_tasks']}_seed{cfg['autoresearch']['seed']}"
    cache_path = GRADE_CACHE_DIR / f"{slice_key}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text())
        grade_payload["thinking_acc"] = cached["thinking_acc"]
        grade_payload["base_acc"] = cached["base_acc"]
        print(f"[info] reused cached thinking/base grades: {cache_path}")
    else:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps({
            "thinking_acc": grade_payload["thinking_acc"],
            "base_acc": grade_payload["base_acc"],
            "slice_key": slice_key,
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }, indent=2))
        print(f"[info] cached thinking/base grades for reuse: {cache_path}")
    return grade_payload


def _wipe_rolling_for_this_model(venhoff_root: Path, base_model: str, dataset: str) -> None:
    """Delete hybrid_token.py's global rolling resume file for our model
    combo so each cycle starts fresh.

    Venhoff ships `rolling_<base_short>_<dataset>.jsonl` with ~140 author
    entries for llama-3.1-8b MATH500. `_count_completed_tasks` reads this
    globally (not per-arch), so its resume logic would (a) skip our first
    scrappy cycle entirely (140 >= 20) and (b) contaminate cycle N with
    cycle N-1's outputs. Move the shipped file aside once, wipe the
    working rolling at the start of every cycle.
    """
    base_short = base_model.split("/")[-1].lower()
    rolling_dir = venhoff_root / "hybrid" / "results" / "rolling"
    for pattern in (f"rolling_{base_short}_{dataset}.jsonl",
                    f"rolling_{base_short}_{dataset}_0.jsonl",
                    f"rolling_{base_short}_{dataset}_vector_stats.json"):
        p = rolling_dir / pattern
        if p.exists():
            shipped_backup = p.with_suffix(p.suffix + ".shipped_bak")
            if not shipped_backup.exists():
                shutil.copy(p, shipped_backup)
                print(f"[info] rolling_shipped_preserved | {shipped_backup}")
            p.unlink()
            print(f"[info] rolling_wiped | {p}")


def run_cycle(cfg: dict, candidate: str, result_dir: Path) -> dict:
    t0 = time.time()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "merged_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    # NB: .absolute() not .resolve() — uv venv's python is a symlink to the
    # base interpreter; what makes it a *venv* python is the pyvenv.cfg
    # sitting next to the symlink. .resolve() hands subprocess the raw
    # base interpreter, which can't see the venv's site-packages. Same
    # bug bit src/bench/venhoff/steering.py:43-77 on 2026-04-20.
    main_python = str((REPO / ".venv" / "bin" / "python").absolute())
    venhoff_root = REPO / "vendor" / "thinking-llms-interp"

    # Wipe cross-cycle rolling-file contamination before each cycle.
    _wipe_rolling_for_this_model(
        venhoff_root=venhoff_root,
        base_model=cfg["model"]["base"],
        dataset=cfg["phase3_hybrid"]["dataset"],
    )

    cycle_eval_root = result_dir / "venhoff_eval"
    cycle_eval_root.mkdir(parents=True, exist_ok=True)

    # 1. Link Phase 0 artifacts into the cycle's eval tree.
    link_dir = _link_phase0_into_cycle(cfg, cycle_eval_root)

    # Stamp the Phase 0 cache hash for provenance in the ledger row.
    phase0_hash = _sha256_of_file(link_dir / "activations_path3.pkl") or _sha256_of_file(link_dir / "activations_path1.pkl")

    # 1b. Optional shuffle-control activation permutation.
    if cfg.get("shuffle_activations"):
        _shuffle_activations_in_place(link_dir, seed=cfg["autoresearch"]["seed"])

    # 2. Phase 2 — train steering vectors (skipped if reuse_venhoff_vectors=true).
    if not cfg.get("reuse_venhoff_vectors", False):
        phase2_cmd = [
            main_python, "-m", "src.bench.venhoff.run_steering",
            "--root", str(cycle_eval_root),
            "--model", _model_slug(cfg["model"]["thinking"]),
            "--dataset", cfg["phase3_hybrid"]["dataset"],
            "--split", cfg["phase3_hybrid"]["dataset_split"],
            "--n-traces", str(cfg["phase0_activations"]["n_traces"]),
            "--layer", str(cfg["model"]["steering_layer"]),
            "--seed", str(cfg["autoresearch"]["seed"]),
            "--arch", cfg["arch"],
            "--venhoff-root", str(venhoff_root),
            "--base-model", cfg["model"]["base"],
            "--thinking-model", cfg["model"]["thinking"],
            "--steering-layer", str(cfg["model"]["steering_layer"]),
            "--sae-layer", str(cfg["model"]["sae_layer"]),
            "--n-clusters", str(cfg["phase2_steering"]["n_clusters"]),
            "--max-iters", str(cfg["phase2_steering"]["max_iters"]),
            "--n-training-examples", str(cfg["phase2_steering"]["n_training_examples"]),
            "--optim-minibatch-size", str(cfg["phase2_steering"]["optim_minibatch_size"]),
            "--lr", str(cfg["phase2_steering"]["lr"]),
            "--num-gpus", "1",
        ]
        _run_subprocess(phase2_cmd, cwd=REPO)
    else:
        print(f"[info] skipping Phase 2 — reuse_venhoff_vectors=true")

    # 3. Phase 3 — hybrid inference.
    phase3_cmd = [
        main_python, "-m", "src.bench.venhoff.run_hybrid",
        "--root", str(cycle_eval_root),
        "--model", _model_slug(cfg["model"]["thinking"]),
        "--dataset", cfg["phase3_hybrid"]["dataset"],
        "--split", cfg["phase3_hybrid"]["dataset_split"],
        "--n-traces", str(cfg["phase0_activations"]["n_traces"]),
        "--layer", str(cfg["model"]["steering_layer"]),
        "--seed", str(cfg["autoresearch"]["seed"]),
        "--arch", cfg["arch"],
        "--venhoff-root", str(venhoff_root),
        "--base-model", cfg["model"]["base"],
        "--thinking-model", cfg["model"]["thinking"],
        "--steering-layer", str(cfg["model"]["steering_layer"]),
        "--sae-layer", str(cfg["model"]["sae_layer"]),
        "--n-clusters", str(cfg["phase2_steering"]["n_clusters"]),
        "--n-tasks", str(cfg["phase3_hybrid"]["n_tasks"]),
        "--max-new-tokens", str(cfg["phase3_hybrid"]["max_new_tokens"]),
        "--max-thinking-tokens", str(cfg["phase3_hybrid"]["max_thinking_tokens"]),
        "--coefficients", *[str(c) for c in cfg["phase3_hybrid"]["coefficients"]],
        "--token-windows", *[str(w) for w in cfg["phase3_hybrid"]["token_windows"]],
    ]
    # Phase 3 is wrapped in run_hybrid, which may raise FileNotFoundError
    # if the legacy per-arch subdir convention doesn't apply. We only
    # care that hybrid_token.py itself ran successfully; swallow the
    # wrapper's downstream check and proceed directly to the
    # benchmark_results_*.json read.
    sys.path.insert(0, str(REPO))
    try:
        _run_subprocess(phase3_cmd, cwd=REPO)
    except RuntimeError as e:
        # run_hybrid raises on _expected_results_dir missing; if the
        # benchmark json IS present we're fine.
        bench_path = _venhoff_benchmark_json_path(venhoff_root, cfg)
        if not bench_path.exists():
            raise
        print(f"[info] wrapper_results_dir_missing_but_benchmark_present | proceeding | err={e}")

    # 4. Read Venhoff's aggregate benchmark results directly. Re-grade
    #    per-task hybrid outcomes locally using is_correct/extract_boxed.
    bench_path = _venhoff_benchmark_json_path(venhoff_root, cfg)
    if not bench_path.exists():
        raise FileNotFoundError(
            f"Expected Venhoff benchmark JSON at {bench_path} after hybrid_token.py; "
            "cycle produced no aggregate result."
        )
    parsed = _parse_benchmark_json(bench_path)

    # 5. Normalize into the ledger-friendly shape.
    payload = {
        "candidate": candidate,
        "arch": cfg["arch"],
        "n_tasks": parsed["n_tasks"],
        "thinking_acc": parsed["thinking_acc"],
        "base_acc": parsed["base_acc"],
        "hybrid_acc": parsed["hybrid_acc"],
        "gap_recovery": parsed["gap_recovery"],
        "best_cell": {
            "coefficient": cfg["phase3_hybrid"]["coefficients"][0],
            "token_window": cfg["phase3_hybrid"]["token_windows"][0],
            "accuracy": parsed["hybrid_acc"],
        },
        "per_task_outcomes": parsed["per_task_outcomes"],
        "wall_time_s": time.time() - t0,
        "phase0_cache_hash": phase0_hash,
        "scaffold": False,
    }
    payload = _use_cached_thinking_base_grades(cfg, payload)
    (result_dir / "grade_results.json").write_text(json.dumps(payload, indent=2))

    # 6. Move Venhoff's aggregate + detailed outputs into the cycle dir so
    #    subsequent candidates don't overwrite them. Also keep a copy of
    #    the rolling file if present (for paired analysis cross-checks).
    cycle_hybrid_out = result_dir / "hybrid_output"
    cycle_hybrid_out.mkdir(exist_ok=True)
    for fname in (
        f"benchmark_results_{_model_slug(cfg['model']['base'])}_{cfg['phase3_hybrid']['dataset']}.json",
        f"detailed/hybrid_stats_{_model_slug(cfg['model']['base'])}_{cfg['phase3_hybrid']['dataset']}.json",
        f"rolling/rolling_{_model_slug(cfg['model']['base'])}_{cfg['phase3_hybrid']['dataset']}.jsonl",
        f"rolling/rolling_{_model_slug(cfg['model']['base'])}_{cfg['phase3_hybrid']['dataset']}_vector_stats.json",
    ):
        src = venhoff_root / "hybrid" / "results" / fname
        if src.exists():
            dst = cycle_hybrid_out / Path(fname).name
            shutil.copy(src, dst)

    return payload


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--result-dir", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_candidate(args.config)
    run_cycle(cfg, args.candidate, args.result_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
