"""Phase 7 path constants — IT-side fork.

This is the IT-side mirror of `_paths.py`. SUBJECT_MODEL, ANCHOR_LAYER,
MLC_LAYERS, CACHE_DIR, PROBE_CACHE_DIR, HF_CKPT_REPO are overridden for
Gemma-2-2b-it L13. Everything else (OUT_DIR, INDEX_PATH, PROBING_PATH,
training defaults) is shared with the BASE side so probing rows from
both subject models append to the same `probing_results.jsonl` and are
disambiguated by the new `subject_model` field (added 2026-04-29 by
the schema patch in run_probing_phase7.py).

Use by importing from `experiments.phase7_unification._paths_it`
instead of `_paths` in IT-specific drivers (build_act_cache_phase7_it,
build_probe_cache_phase7_it, train_phase7_it).
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("PHASE7_REPO", os.getcwd())).resolve()

# ────────────────────────────── subject model + layer convention (IT)
SUBJECT_MODEL = "google/gemma-2-2b-it"
ANCHOR_LAYER = 13                             # 0-indexed; +1 vs BASE
MLC_LAYERS = (11, 12, 13, 14, 15)             # 5-layer window centred on L13
ANCHOR_LAYER_KEY = f"resid_L{ANCHOR_LAYER}"
MLC_LAYER_KEYS = tuple(f"resid_L{l}" for l in MLC_LAYERS)

# ────────────────────────────── activation + probe caches (IT-specific)
CACHE_DIR = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
PROBE_CACHE_DIR = REPO / "experiments/phase7_unification/results/probe_cache_it"
PROBE_CACHE_DIR_S32 = REPO / "experiments/phase7_unification/results/probe_cache_S32_it"

# ────────────────────────────── results layout (SHARED with BASE)
OUT_DIR = REPO / "experiments/phase7_unification/results"
CKPT_DIR = OUT_DIR / "ckpts"
LOGS_DIR = OUT_DIR / "training_logs"
PLOTS_DIR = OUT_DIR / "plots"
AUTOINTERP_DIR = OUT_DIR / "autointerp"
INDEX_PATH = OUT_DIR / "training_index.jsonl"
PROBING_PATH = OUT_DIR / "probing_results.jsonl"
T_SWEEP_PATH = OUT_DIR / "t_sweep_results.jsonl"
SEED_MARKER_DIR = OUT_DIR / "seed_markers"

# ────────────────────────────── pre-Phase-7 — READ-ONLY references
PHASE5_RESULTS = REPO / "experiments/phase5_downstream_utility/results"
PHASE5B_RESULTS = REPO / "experiments/phase5b_t_scaling_explore/results"
PHASE6_RESULTS = REPO / "experiments/phase6_qualitative_latents/results"

# ────────────────────────────── arch / training defaults (SHARED)
DEFAULT_D_IN = 2304
DEFAULT_D_SAE = 18_432
DEFAULT_K_WIN = 500
SEEDS = (42, 1, 2)
PRELOAD_SEQS = 24_000

# ────────────────────────────── HuggingFace repos (IT-specific)
HF_CKPT_REPO = "han1823123123/txcdr-it"
HF_DATA_REPO = "han1823123123/txcdr-it-data"

# ────────────────────────────── canonical arch table (SHARED)
CANONICAL_ARCHS_JSON = REPO / "experiments/phase7_unification/canonical_archs.json"


def banner(driver_name: str) -> None:
    """Print on every Phase 7 IT driver entry. Surfaces subject-model
    + path config so accidental mixing of BASE and IT runs is visible.
    """
    print(f"Phase 7 driver (IT): {driver_name}")
    print(f"  REPO:               {REPO}")
    print(f"  CACHE_DIR:          {CACHE_DIR}")
    print(f"  PROBE_CACHE:        {PROBE_CACHE_DIR}")
    print(f"  PROBE_CACHE_S32:    {PROBE_CACHE_DIR_S32}")
    print(f"  OUT_DIR:            {OUT_DIR}")
    print(f"  INDEX_PATH:         {INDEX_PATH}")
    print(f"  PROBING_PATH:       {PROBING_PATH}")
    print(f"  ANCHOR_LAYER:       L{ANCHOR_LAYER} (0-indexed)")
    print(f"  MLC_LAYERS:         {MLC_LAYERS}")
    print(f"  SUBJECT_MODEL:      {SUBJECT_MODEL}")
    print(f"  HF_CKPT_REPO:       {HF_CKPT_REPO}")
    print(f"  HF_DATA_REPO:       {HF_DATA_REPO}")
