"""Opt-in dead-latent training telemetry (ruling 361de3cb2 item 2).

Plugin-side (archs are plugins; core untouched). When
``TEMP_BENCH_TELEMETRY_DIR`` is set, participating archs call
``maybe_log`` from ``train_step`` and a JSONL trace lands there —
one file per model instance, one record per SAMPLE_EVERY steps:

    {"step", "n_dead", "dead_frac", "batch_l0", "boundary_min_pre",
     "arch", "relu_mode", "d_sae", "k_pos", "T"}

``boundary_min_pre`` is the smallest selected pre-activation in the
sampled batch — the direct dead-latent-mechanism observable: while
it stays > 0 the compositions coincide (rectify-after-select is a
no-op); when it crosses 0 they must diverge. Files are named
``<arch>_T<T>_<8-hex>.jsonl`` (fresh suffix per instance, drawn from
``os.urandom`` at first call); every record is self-describing, so
cell attribution never depends on the filename.

Off by default: without the env var this is a single dict lookup per
train step.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

SAMPLE_EVERY = 250


def due(step: int) -> bool:
    """True when this step should sample (env set + on the grid).
    Callers gate their observable computation on this so the off
    state costs one dict lookup and a modulo."""
    return bool(os.environ.get("TEMP_BENCH_TELEMETRY_DIR")) \
        and step % SAMPLE_EVERY == 0


def maybe_log(model, *, step: int, n_dead: int, batch_l0: float,
              boundary_min_pre: float | None = None) -> None:
    d = os.environ.get("TEMP_BENCH_TELEMETRY_DIR")
    if not d or step % SAMPLE_EVERY:
        return
    path = getattr(model, "_telemetry_path", None)
    if path is None:
        cfg = getattr(model, "config", None)
        T = getattr(cfg, "T", None) or getattr(model, "T", 1)
        name = type(model).__name__
        suffix = os.urandom(4).hex()
        Path(d).mkdir(parents=True, exist_ok=True)
        path = Path(d) / f"{name}_T{T}_{suffix}.jsonl"
        model._telemetry_path = path
    cfg = getattr(model, "config", None)
    d_sae = int(getattr(cfg, "d_sae", 0) or 0)
    rec = {
        "step": int(step),
        "n_dead": int(n_dead),
        "dead_frac": (int(n_dead) / d_sae) if d_sae else None,
        "batch_l0": float(batch_l0),
        "boundary_min_pre": (float(boundary_min_pre)
                             if boundary_min_pre is not None else None),
        "arch": type(model).__name__,
        "relu_mode": getattr(model, "relu_mode", "relu-mix"),
        "d_sae": d_sae,
        "k_pos": int(getattr(cfg, "k_pos", 0) or 0),
        "T": int(getattr(cfg, "T", None) or getattr(model, "T", 1)),
    }
    with open(path, "a") as fh:
        fh.write(json.dumps(rec) + "\n")
