"""Agent C path constants — extends `experiments/phase7_unification/_paths.py`.

Same path-discipline as the rest of Phase 7: NEVER hard-code bare-string
paths in Agent C code. Import from here.
"""
from __future__ import annotations

from pathlib import Path

from experiments.phase7_unification._paths import (
    REPO, OUT_DIR, PLOTS_DIR,
    SUBJECT_MODEL, ANCHOR_LAYER, MLC_LAYERS,
    CANONICAL_ARCHS_JSON,
    HF_CKPT_REPO, HF_DATA_REPO,
    DEFAULT_D_IN, DEFAULT_D_SAE,
    banner,
)

# ────────────────────────────── Agent C output layout (under Phase 7's results/)
CASE_STUDIES_DIR = OUT_DIR / "case_studies"
HH_RLHF_DIR = CASE_STUDIES_DIR / "hh_rlhf"
STEERING_DIR = CASE_STUDIES_DIR / "steering"

# ────────────────────────────── selected 6 archs for Agent C
SELECTED_ARCHS_FOR_CASE_STUDIES = (
    "topk_sae",                                              # row 1
    "tsae_paper_k500",                                       # row 2
    "mlc_contrastive_alpha100_batchtopk",                    # row 5
    "agentic_txc_02",                                        # row 8
    "phase5b_subseq_h8",                                     # row 13
    "phase57_partB_h8_bare_multidistance_t5",                # row 32
)

# ────────────────────────────── case-study datasets
HH_RLHF_HF_PATH = "Anthropic/hh-rlhf"
HH_RLHF_CACHE_DIR = REPO / "data/cached_hh_rlhf"

# ────────────────────────────── grader
ANTHROPIC_GRADER_MODEL = "claude-sonnet-4-6"
# Number of features per arch to steer (AxBench convention).
N_STEERING_FEATURES = 30
# Steering strengths to sweep per feature (from T-SAE paper Tab 2).
STEERING_STRENGTHS = (0.5, 1, 2, 4, 8, 12, 16, 24)
# Tokens to generate per intervention.
STEERING_GEN_TOKENS = 60
