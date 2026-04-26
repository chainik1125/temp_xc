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

# ────────────────────────────── selected archs for Agent C (staged)
# Stage 1 (framework debug, 4 archs): conceptually distinct designs to
# expose family-dependent bugs early. tsae_paper at BOTH k=500 (our
# k_win convention, apples-to-apples with TXC) AND k=20 (paper-faithful
# Ye et al. 2025 baseline) — the dual reading lets us decouple "is the
# T-SAE recipe better?" from "is k=20 simply too sparse?".
STAGE_1_ARCHS = (
    "topk_sae",                                              # row 1: per-token SAE baseline
    "tsae_paper_k500",                                       # row 2: T-SAE port at k_win=500
    "tsae_paper_k20",                                        # row 3: T-SAE paper-faithful (k=20)
    "agentic_txc_02",                                        # row 8: TXC representative (T=5, multi-scale matryoshka)
)
# Stage 2 (expansion, after Stage 1 framework is locked): add 3 more.
STAGE_2_EXTRA_ARCHS = (
    "mlc_contrastive_alpha100_batchtopk",                    # row 5: MLC reference
    "phase5b_subseq_h8",                                     # row 13: SubseqH8 (Phase 5B mp champion)
    "phase57_partB_h8_bare_multidistance_t5",                # row 32: H8 multidistance (Phase 5 mp peak)
)
SELECTED_ARCHS_FOR_CASE_STUDIES = STAGE_1_ARCHS + STAGE_2_EXTRA_ARCHS

# ────────────────────────────── HH-RLHF dataset (paper §4.5 + App B.1)
# Paper analyses the harmless split only (its t-test reports rejected
# ≈ 49.2 tok vs chosen ≈ 37.8 tok mean response length, p=9e-10).
HH_RLHF_HF_PATH = "Anthropic/hh-rlhf"
HH_RLHF_SPLIT_DIR = "harmless-base"
HH_RLHF_N_SAMPLES = 1000                  # first 1000 per paper §B.1
HH_RLHF_MAX_LENGTH = 256                  # tokens — covers ~99% of harmless-base examples
HH_RLHF_CACHE_DIR = REPO / "data/cached_hh_rlhf"

# ────────────────────────────── steering — AxBench-style additive
# x_t' = x_t + strength * unit_norm(d_dec[feature])  applied at L12 hook.
# Strengths follow agent_c_brief.md (decoder-direction multipliers, not
# T-SAE paper's clamp-on-latent which uses absolute values 10..15000).
ANTHROPIC_GRADER_MODEL = "claude-sonnet-4-6"
N_STEERING_CONCEPTS = 30                  # paper averages over 30 features
STEERING_STRENGTHS = (0.5, 1, 2, 4, 8, 12, 16, 24)
STEERING_GEN_TOKENS = 60
STEERING_PROMPT = "We find"               # paper's neutral prompt
