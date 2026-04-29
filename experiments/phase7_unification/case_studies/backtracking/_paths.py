"""Constants for the backtracking case study (Ward et al. 2025, arXiv 2507.12638).

Reproduces the paper's residual-stream steering result with SAE-feature
decomposition substituted for the raw Difference-of-Means step.

Path-discipline: import REPO/OUT_DIR from the Phase 7 parent module, define
our own subject-model + layer overrides locally (the parent defaults are
Gemma-2-2b L12; we use DeepSeek-R1-Distill-Llama-8B L10).
"""

from __future__ import annotations

from experiments.phase7_unification._paths import OUT_DIR, REPO, banner  # noqa: F401

# --- subject model -----------------------------------------------------------
# DeepSeek-R1-Distill-Llama-8B (the reasoning model the paper steers).
# Layer 10 is the paper's empirical optimum for the DoM direction (§3.2).
SUBJECT_MODEL = "deepseek-r1-distill-llama-8b"  # registry key in src/data/nlp/models.py
ANCHOR_LAYER = 10

# --- SAE source --------------------------------------------------------------
# Llama-Scope (He et al. 2024, arXiv 2410.20526) — public family of SAEs trained
# at every (layer, sublayer) of base Llama-3.1-8B.
# Naming: LXR-8x = layer X post-MLP residual stream, 8x expansion (32k features).
# We use the base-model SAE on the distilled model; justified by paper's
# cosine(base, reasoning) ≈ 0.74 (Ward et al. §3.2).
PUBLIC_SAE_REPO = "fnlp/Llama-Scope"
PUBLIC_SAE_CONFIG = "L10R-8x"

# --- backtracking metric (paper Eq. 1) --------------------------------------
KEYWORDS: tuple[str, ...] = ("wait", "hmm")

# --- offsets (paper §3.1) ---------------------------------------------------
# Positive contrastive set: positions [event_idx + NEG_OFFSET_LO,
# event_idx + NEG_OFFSET_HI] inclusive, i.e. 13..8 tokens before each
# backtracking event.
NEG_OFFSET_LO: int = -13
NEG_OFFSET_HI: int = -8

# --- dataset sizes ----------------------------------------------------------
N_PROMPTS = 300
N_HELDOUT = 30
MAX_NEW_TOKENS = 512  # generation length for reasoning traces

# --- intervention sweep -----------------------------------------------------
# Paper's optimum was magnitude ~12 at offset ~-12 (Fig 2). Mode A (raw DoM)
# and B (SAE additive) share this grid; the SAE side is on a *unit-normed*
# decoder column so the magnitudes are directly comparable to the raw DoM.
ADDITIVE_MAGNITUDES: tuple[float, ...] = (0.0, 4.0, 8.0, 12.0, 16.0, 20.0)
# Mode C (paper-clamp on a latent) uses absolute strengths in feature-activation
# units; calibrated post hoc against the SAE's typical max activation.
CLAMP_STRENGTHS: tuple[float, ...] = (0.0, 5.0, 10.0, 25.0, 50.0, 100.0)

GEN_TOKENS_PER_INTERVENTION = 30  # short continuations to keep keyword fraction stable
GEN_SEED = 42

# --- paths ------------------------------------------------------------------
RESULTS_DIR = OUT_DIR / "case_studies" / "backtracking"
TRACES_PATH = RESULTS_DIR / "traces.jsonl"
PROMPTS_PATH = RESULTS_DIR / "prompts.jsonl"
LABELS_DIR = RESULTS_DIR / "labels"
CACHE_DIR = RESULTS_DIR / "cache_l10"
DECOMPOSE_DIR = RESULTS_DIR / "decompose"
INTERVENE_DIR = RESULTS_DIR / "intervene"
PLOTS_DIR = RESULTS_DIR / "plots"


def ensure_dirs() -> None:
    for p in (
        RESULTS_DIR,
        LABELS_DIR,
        CACHE_DIR,
        DECOMPOSE_DIR,
        INTERVENE_DIR,
        PLOTS_DIR,
    ):
        p.mkdir(parents=True, exist_ok=True)
