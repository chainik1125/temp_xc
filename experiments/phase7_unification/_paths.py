"""Phase 7 path constants — single source of truth.

All Phase 7 drivers (training, probing, cache build, analyzers) must
import paths from here. NEVER hard-code bare-string paths in Phase 7
code; that's how prior-phase fork-drift bugs got introduced (see
plan.md "Path-discipline mechanism").

Local-aware via PHASE7_REPO env var, falling back to CWD. This
matches the phase5b convention so the same code runs locally and on
RunPod identically.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("PHASE7_REPO", os.getcwd())).resolve()

# ────────────────────────────── subject model + layer convention
SUBJECT_MODEL = "google/gemma-2-2b"
ANCHOR_LAYER = 12                             # 0-indexed; matches T-SAE + TFA
MLC_LAYERS = (10, 11, 12, 13, 14)             # 5-layer window centred on L12
ANCHOR_LAYER_KEY = f"resid_L{ANCHOR_LAYER}"
MLC_LAYER_KEYS = tuple(f"resid_L{l}" for l in MLC_LAYERS)

# ────────────────────────────── activation + probe caches
CACHE_DIR = REPO / "data/cached_activations/gemma-2-2b/fineweb"
PROBE_CACHE_DIR = REPO / "experiments/phase7_unification/results/probe_cache"

# ────────────────────────────── results layout (Phase 7 OWNS these paths)
OUT_DIR = REPO / "experiments/phase7_unification/results"
CKPT_DIR = OUT_DIR / "ckpts"
LOGS_DIR = OUT_DIR / "training_logs"
PLOTS_DIR = OUT_DIR / "plots"
AUTOINTERP_DIR = OUT_DIR / "autointerp"
INDEX_PATH = OUT_DIR / "training_index.jsonl"
PROBING_PATH = OUT_DIR / "probing_results.jsonl"
T_SWEEP_PATH = OUT_DIR / "t_sweep_results.jsonl"
SEED_MARKER_DIR = OUT_DIR / "seed_markers"  # local mirror; HF gets {seed}_complete.json

# ────────────────────────────── pre-Phase-7 — READ-ONLY references
PHASE5_RESULTS = REPO / "experiments/phase5_downstream_utility/results"
PHASE5B_RESULTS = REPO / "experiments/phase5b_t_scaling_explore/results"
PHASE6_RESULTS = REPO / "experiments/phase6_qualitative_latents/results"

# ────────────────────────────── arch / training defaults
DEFAULT_D_IN = 2304
DEFAULT_D_SAE = 18_432
DEFAULT_K_WIN = 500
SEEDS = (42, 1, 2)              # OUTER-LOOP order: seed 42 first
PRELOAD_SEQS = 24_000           # H200 188 GB RAM: full cache fits

# ────────────────────────────── HuggingFace repos (hardcoded; Phase 7 only)
HF_CKPT_REPO = "han1823123123/txcdr-base"
HF_DATA_REPO = "han1823123123/txcdr-base-data"

# ────────────────────────────── canonical arch table
CANONICAL_ARCHS_JSON = REPO / "experiments/phase7_unification/canonical_archs.json"


def banner(driver_name: str) -> None:
    """Print on every Phase 7 driver entry. Surfaces misconfigured paths
    immediately rather than letting them silently corrupt prior-phase
    indices. Per plan.md "Startup banner" requirement.
    """
    print(f"Phase 7 driver: {driver_name}")
    print(f"  REPO:           {REPO}")
    print(f"  CACHE_DIR:      {CACHE_DIR}")
    print(f"  PROBE_CACHE:    {PROBE_CACHE_DIR}")
    print(f"  OUT_DIR:        {OUT_DIR}")
    print(f"  INDEX_PATH:     {INDEX_PATH}")
    print(f"  PROBING_PATH:   {PROBING_PATH}")
    print(f"  ANCHOR_LAYER:   L{ANCHOR_LAYER} (0-indexed)")
    print(f"  MLC_LAYERS:     {MLC_LAYERS}")
    print(f"  SUBJECT_MODEL:  {SUBJECT_MODEL}")
    print(f"  HF_CKPT_REPO:   {HF_CKPT_REPO}")
    print(f"  HF_DATA_REPO:   {HF_DATA_REPO}")
