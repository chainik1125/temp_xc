"""SAEBench+CT probe-cache loaders (§ 5.1 sparse probing).

Artifact-first port of ``origin/final:purified/src/temp_bench/data/nlp/
probe_cache.py`` — the LOADER half only. The canonical per-task caches
were built once by the paper pipeline (Phase 7 padding-fix recipe) and
live on HF at ``han1823123123/temp-bench-data:probe_cache/<datasource>/``;
agents sync them to ``results/probe_cache/<datasource>/`` rather than
rebuilding. The BUILDERS (subject-model forwards over the 38 task
datasets, ``build_probe_cache`` + ``probe_tasks.py`` loaders with the
SAEBench-faithfulness fixes) are NOT ported — they remain on
``origin/final``; rebuilding from scratch requires porting them.

On-disk schema 2.0.0 (per task, left-aligned S=32 frames):

    results/probe_cache/<datasource_name>/<task_name>/
      ├ X_train.npy            # (N_train, S_CACHE=32, d_in) fp16
      ├ X_test.npy             # (N_test,  S_CACHE=32, d_in) fp16
      ├ first_real_train.npy   # (N_train,) int64 — first valid pos in S-frame
      ├ first_real_test.npy    # (N_test,)  int64
      ├ y_train.npy            # (N_train,) int64
      ├ y_test.npy             # (N_test,)  int64
      └ meta.json              # task metadata + datasource spec snapshot

Real tokens occupy positions ``[first_real[i], S_CACHE-1]``; zeros
before that. ``temp_bench.evals.probing._encode_pool`` masks the
padding portion per row (protocol 1.1.0+ semantics).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from temp_bench.core.config import repo_root

S_CACHE = 32   # cache S-frame width; matches DEFAULT_S in evals/probing.py

_REQUIRED = (
    "X_train.npy", "X_test.npy",
    "first_real_train.npy", "first_real_test.npy",
    "y_train.npy", "y_test.npy",
)


def probe_cache_dir(datasource_name: str, task_name: str | None = None) -> Path:
    """Path to the probe-cache root (or one task subdir).

    ``results/probe_cache/<datasource_name>/[<task_name>/]`` under the
    repo root, overridable via ``TEMP_BENCH_PROBE_CACHE`` (e.g. a shared
    pod volume). Keyed by datasource NAME (not data_key) so it is
    human-readable and matches the HF layout.
    """
    env = os.environ.get("TEMP_BENCH_PROBE_CACHE")
    base = (Path(env) if env else repo_root() / "results" / "probe_cache") / datasource_name
    if task_name is None:
        return base
    return base / task_name


def list_probe_cache(datasource_name: str) -> list[str]:
    """Task names with a complete schema-2.0.0 cache (all 6 arrays).

    Older schema-1.x caches (no ``first_real``) are skipped — resync
    the canonical cache from HF.
    """
    root = probe_cache_dir(datasource_name)
    if not root.exists():
        return []
    out: list[str] = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        if all((d / f).exists() for f in _REQUIRED):
            out.append(d.name)
    return out


def load_probe_cache(datasource_name: str, task_name: str) -> dict:
    """Load one task's arrays (schema 2.0.0 — left-aligned S=32).

    Returns keys ``task_name``, ``X_train``, ``X_test`` (mmap'd fp16,
    cast downstream), ``first_real_train``, ``first_real_test``,
    ``y_train``, ``y_test`` — the shape ``temp_bench.evals.probing``
    consumes.
    """
    out_dir = probe_cache_dir(datasource_name, task_name)
    if not out_dir.exists():
        raise FileNotFoundError(
            f"probe cache missing for task {task_name!r} at {out_dir}. "
            "Sync from HF han1823123123/temp-bench-data:probe_cache/."
        )
    fr_tr_path = out_dir / "first_real_train.npy"
    if not fr_tr_path.exists():
        raise FileNotFoundError(
            f"probe cache for {task_name!r} is schema 1.x (no first_real); "
            "resync the schema-2.0.0 cache from HF."
        )
    return {
        "task_name": task_name,
        "X_train": np.load(out_dir / "X_train.npy", mmap_mode="r"),
        "X_test": np.load(out_dir / "X_test.npy", mmap_mode="r"),
        "first_real_train": np.load(fr_tr_path),
        "first_real_test": np.load(out_dir / "first_real_test.npy"),
        "y_train": np.load(out_dir / "y_train.npy"),
        "y_test": np.load(out_dir / "y_test.npy"),
    }
