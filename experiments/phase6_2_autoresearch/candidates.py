"""Phase 6.2 autoresearch candidate architectures.

Each candidate forks Track 2's base (bare window TXC + full anti-dead
stack, TopK k=100) and toggles one or two `tsae_paper` axes. All at
T=5, d_sae=18432, seed=42, plateau-stop (min_steps=3000 unless noted).

The `dispatch` field is the arch name passed to
`train_primary_archs.py --archs <name>`. If the dispatch branch
doesn't exist yet, the `implementation_note` explains how to add it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Candidate:
    id: str
    name: str
    dispatch: str               # arch name for train_primary_archs.py
    rationale: str
    axis_tested: str
    expected_random: str        # rough prior on x/32 result
    cost_min: int               # GPU minutes for training
    implementation_note: str = ""
    implemented: bool = False   # True iff train_primary_archs.py dispatches


CANDIDATES = [
    Candidate(
        id="C1",
        name="phase62_c1_track2_matryoshka",
        dispatch="phase62_c1_track2_matryoshka",
        rationale=(
            "Adds the Matryoshka 20/80 H/L reconstruction loss to "
            "Track 2's bare encoder. Tests whether hierarchical "
            "feature partitioning alone generalises to random text."
        ),
        axis_tested="matryoshka",
        expected_random="5-8/32 (small gain over Track 2)",
        cost_min=35,
        implementation_note=(
            "Implemented via TXCBareMatryoshkaContrastiveAntidead "
            "with alpha=0.0 and matryoshka_h_size=int(d_sae*0.2)."
        ),
        implemented=True,
    ),
    Candidate(
        id="C2",
        name="phase62_c2_track2_contrastive",
        dispatch="phase62_c2_track2_contrastive",
        rationale=(
            "Adds single-scale InfoNCE(α=1.0) on adjacent-token pairs "
            "to Track 2. Tests whether the temporal-consistency "
            "regulariser from Ye et al. §3.2 is the load-bearing "
            "mechanism independent of matryoshka."
        ),
        axis_tested="contrastive",
        expected_random="6-10/32 (moderate gain)",
        cost_min=40,
        implementation_note=(
            "Implemented via TXCBareMatryoshkaContrastiveAntidead "
            "with matryoshka_h_size=None, alpha=1.0, contr_prefix "
            "defaulting to 0.2*d_sae."
        ),
        implemented=True,
    ),
    Candidate(
        id="C3",
        name="phase62_c3_track2_matryoshka_contrastive",
        dispatch="phase62_c3_track2_matryoshka_contrastive",
        rationale=(
            "Reconstructs tsae_paper's recipe on the TXC (window-based) "
            "encoder base, with TopK sparsity instead of BatchTopK. "
            "Highest-prior: if this hits ≥10/32 random, the TXC-parity "
            "story holds cleanly (encoder base is swappable, the rest "
            "of tsae_paper's machinery transfers)."
        ),
        axis_tested="matryoshka + contrastive (full tsae_paper on TXC base)",
        expected_random="8-14/32 (strong candidate for the winner)",
        cost_min=45,
        implementation_note=(
            "Implemented via TXCBareMatryoshkaContrastiveAntidead "
            "with matryoshka_h_size=int(d_sae*0.2) and alpha=1.0."
        ),
        implemented=True,
    ),
    Candidate(
        id="C4",
        name="phase62_c4_track2_threshold",
        dispatch="phase62_c4_track2_threshold",
        rationale=(
            "Track 2 at inference with an EMA threshold (JumpReLU-style, "
            "paper convention) instead of greedy TopK. No retraining — "
            "wraps the existing ckpt's encoder path. Tests whether the "
            "inference-mechanism mismatch explains a chunk of the "
            "generalisation gap."
        ),
        axis_tested="inference threshold",
        expected_random="5-8/32 (likely incremental)",
        cost_min=0,  # no retrain, encode-only
        implementation_note=(
            "Patch encode_archs.load_arch to add a --use-threshold "
            "flag when arch='agentic_txc_10_bare'. Needs EMA threshold "
            "storage added to TXCBareAntidead during training (may "
            "require a fresh training run after all; reconsider)."
        ),
        implemented=False,
    ),
    Candidate(
        id="C5",
        name="phase62_c5_track2_longer",
        dispatch="agentic_txc_10_bare",  # REUSES existing arch
        rationale=(
            "Simplest candidate: retrain Track 2 with min_steps=10000 "
            "instead of the current 3000. Phase 6.1 cycles all plateau-"
            "stopped at step 4000-5600, possibly too early for the "
            "anti-dead machinery to shape decoder directions."
        ),
        axis_tested="training duration",
        expected_random="5-7/32 (small gain if duration matters)",
        cost_min=60,
        implementation_note=(
            "Reuse agentic_txc_10_bare dispatch, override cfg.min_steps "
            "to 10000 in train_primary_archs.run_all. May need a new "
            "CLI flag --min-steps on the entry point."
        ),
        implemented=True,  # arch exists; just needs min_steps override
    ),
    Candidate(
        id="C6",
        name="phase62_c6_bare_batchtopk_longer",
        dispatch="agentic_txc_12_bare_batchtopk",  # REUSES 2x2 cell
        rationale=(
            "The 2x2 cell (Phase 6.1 #4) with min_steps=10000. Tests "
            "whether BatchTopK + anti-dead stack's 0/32 random result "
            "is a step-count artefact of plateau-stop."
        ),
        axis_tested="training duration × BatchTopK+anti-dead",
        expected_random="0-5/32 (likely stays ≈ 2x2 cell's current)",
        cost_min=60,
        implementation_note=(
            "Reuse agentic_txc_12_bare_batchtopk dispatch, override "
            "min_steps=10000. 2x2 cell training must have completed "
            "under Phase 6.1 first to have its baseline."
        ),
        implemented=True,
    ),
]


def by_id(cid: str) -> Candidate:
    for c in CANDIDATES:
        if c.id == cid:
            return c
    raise KeyError(f"unknown candidate {cid}; valid: {[c.id for c in CANDIDATES]}")


def implemented_now() -> list[Candidate]:
    return [c for c in CANDIDATES if c.implemented]


def needs_implementation() -> list[Candidate]:
    return [c for c in CANDIDATES if not c.implemented]


if __name__ == "__main__":
    print(f"Phase 6.2 candidates ({len(CANDIDATES)}):")
    for c in CANDIDATES:
        status = "READY" if c.implemented else "NEEDS-IMPL"
        print(f"  {c.id}  [{status}]  {c.name}")
        print(f"        axis: {c.axis_tested}")
        print(f"        prior: {c.expected_random}")
        print(f"        cost : {c.cost_min} min GPU")
