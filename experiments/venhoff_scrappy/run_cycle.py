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


def _venhoff_hybrid_results_dir(venhoff_root: Path, cfg: dict) -> Path:
    return (
        venhoff_root / "hybrid" / "results" / cfg["phase3_hybrid"]["dataset"]
        / f"{_model_slug(cfg['model']['base'])}_"
          f"{_model_slug(cfg['model']['thinking'])}_"
          f"L{cfg['model']['steering_layer']}_"
          f"SAEL{cfg['model']['sae_layer']}_"
          f"n{cfg['phase2_steering']['n_clusters']}"
    )


def _extract_per_task_outcomes(results_dir: Path, cfg: dict) -> tuple[list[bool], dict]:
    """Read Venhoff's per-problem JSONLs and extract (per-task hybrid correct,
    aux dict with thinking/base per-task) — aligned with the task order.

    Venhoff writes one JSONL per problem or one JSONL per cell; our
    `grade.compute_gap_recovery` handles both. We read whichever layout
    is present and return the hybrid outcome at the best cell. If no
    best-cell info is available yet (grade not yet run), returns
    empty lists.
    """
    from src.bench.venhoff.grade import extract_boxed, is_correct

    best_coef = cfg["phase3_hybrid"]["coefficients"][0]
    best_window = cfg["phase3_hybrid"]["token_windows"][0]

    outcomes: list[bool] = []
    for per_problem in sorted(results_dir.glob("*.jsonl")):
        with per_problem.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ref = str(row.get("answer", ""))
                if not ref:
                    continue
                for cell in row.get("per_cell_predictions") or row.get("cells") or []:
                    c = float(cell.get("coefficient", 0))
                    w = int(cell.get("token_window", 0))
                    if c == best_coef and w == best_window:
                        pred = extract_boxed(cell.get("predicted") or cell.get("response") or "")
                        outcomes.append(bool(is_correct(pred, ref)))
                        break
    return outcomes, {}


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


def run_cycle(cfg: dict, candidate: str, result_dir: Path) -> dict:
    t0 = time.time()
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "merged_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    main_python = str((REPO / ".venv" / "bin" / "python").resolve())
    venhoff_root = REPO / "vendor" / "thinking-llms-interp"

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
    _run_subprocess(phase3_cmd, cwd=REPO)

    # 4. Grade.
    grade_out_raw = result_dir / "grade_raw.json"
    grade_cmd = [
        main_python, "-m", "src.bench.venhoff.run_grade",
        "--root", str(cycle_eval_root),
        "--arch", cfg["arch"],
        "--venhoff-root", str(venhoff_root),
        "--base-model", cfg["model"]["base"],
        "--thinking-model", cfg["model"]["thinking"],
        "--dataset", cfg["phase3_hybrid"]["dataset"],
        "--steering-layer", str(cfg["model"]["steering_layer"]),
        "--sae-layer", str(cfg["model"]["sae_layer"]),
        "--n-clusters", str(cfg["phase2_steering"]["n_clusters"]),
        "--out", str(grade_out_raw),
    ]
    _run_subprocess(grade_cmd, cwd=REPO)
    raw = json.loads(grade_out_raw.read_text())

    # 5. Per-task outcomes for paired Δ analysis.
    sys.path.insert(0, str(REPO))
    v_results = _venhoff_hybrid_results_dir(venhoff_root, cfg)
    try:
        per_task, _ = _extract_per_task_outcomes(v_results, cfg)
    except Exception as e:
        print(f"[warn] per-task outcome extraction failed: {e}")
        per_task = []

    # 6. Normalize the grade into the ledger-friendly shape.
    payload = {
        "candidate": candidate,
        "arch": cfg["arch"],
        "n_tasks": cfg["phase3_hybrid"]["n_tasks"],
        "thinking_acc": raw.get("thinking_accuracy"),
        "base_acc": raw.get("base_accuracy"),
        "hybrid_acc": (raw.get("best_cell") or {}).get("accuracy"),
        "gap_recovery": raw.get("best_gap_recovery"),
        "best_cell": raw.get("best_cell"),
        "per_task_outcomes": per_task,
        "wall_time_s": time.time() - t0,
        "phase0_cache_hash": phase0_hash,
        "scaffold": False,
    }
    payload = _use_cached_thinking_base_grades(cfg, payload)
    (result_dir / "grade_results.json").write_text(json.dumps(payload, indent=2))

    # 7. Copy Venhoff's hybrid output dir into the cycle so subsequent
    #    candidates with matching (layer, sae_layer, n_clusters) don't
    #    clobber this one. Then delete the Venhoff copy.
    if v_results.exists():
        dst = result_dir / "hybrid_output"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(v_results, dst)
        shutil.rmtree(v_results)
        print(f"[info] moved hybrid output: {v_results} -> {dst}")

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
