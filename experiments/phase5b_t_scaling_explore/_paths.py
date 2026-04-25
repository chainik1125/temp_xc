"""Phase 5B path config — local-aware (no /workspace hardcode).

Set REPO from `PHASE5B_REPO` env var, falling back to CWD.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(os.environ.get("PHASE5B_REPO", os.getcwd())).resolve()
CACHE_DIR = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
OUT_DIR = REPO / "experiments/phase5b_t_scaling_explore/results"
CKPT_DIR = OUT_DIR / "ckpts"
LOGS_DIR = OUT_DIR / "training_logs"
PREDICTIONS_DIR = OUT_DIR / "predictions"
PLOTS_DIR = OUT_DIR / "plots"
INDEX_PATH = OUT_DIR / "training_index.jsonl"
PROBING_PATH = OUT_DIR / "probing_results.jsonl"

# Phase 5 read-only refs (probe cache — never written)
PHASE5 = REPO / "experiments/phase5_downstream_utility/results"
PROBE_CACHE = PHASE5 / "probe_cache"

ANCHOR_LAYER_KEY = "resid_L13"
MLC_LAYER_KEYS = ("resid_L11", "resid_L12", "resid_L13", "resid_L14", "resid_L15")
DEFAULT_D_SAE = 18_432
PRELOAD_SEQS = 6_000
