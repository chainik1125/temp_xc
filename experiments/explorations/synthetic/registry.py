"""Program-level registry — the single source of *what's in the B×A matrix*.

The synthetic-benchmark program compares **B benchmarks × A architectures** on
one apples-to-apples grid. This module declares both axes plus the canonical
operating point, so the renderer (``render_report.py`` → ``REPORT.md``) has one
authoritative spec to read. Nothing here computes numbers; it only names them.

Two things make the grid natively comparable (see ``REPORT.md``'s convention
section, or the module docstring of ``src/explorations/synthetic/report.py``):

1. **Metric.** Every latent-recovery metric is already normalized to
   ``[chance = 0, oracle = 1]`` by its evaluator, so a cell is one scalar that is
   comparable across benchmarks. Dual-latent benches contribute **one matrix row
   per latent-axis** (this surfaces the DC/AC split rather than hiding it).
2. **Budget.** Architectures are matched on **per-token sparsity** — the
   *realized* ``l0_per_token`` (measured, not the nominal ``k_pos`` knob — they
   diverge wildly; see the evaluator increment). This is the controlled
   comparison: equal total atoms per span, so only decode structure varies. It is
   the only matching convention (``l0_per_window`` is recorded as a diagnostic,
   never matched on).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LatentAxis:
    """One matrix ROW: a single normalized-recovery metric for one bench latent.

    ``metric`` is the leaderboard key (already normalized to ``[chance, oracle]``).
    ``kind`` is the DC/AC lens tag. ``primary`` flags the bench's headline axis
    (a dual-latent bench has one primary + secondary/floor axes).
    """
    key: str
    metric: str
    kind: str          # "DC" | "AC"
    label: str
    primary: bool = True


@dataclass(frozen=True)
class Bench:
    """One benchmark: its datasource(s), protocol, F anchor, and latent-axes.

    ``datasources[0]`` is the primary; any extras are nulls/controls (e.g. the
    frequency random-embedding null) — kept for the per-bench section, not the
    headline matrix. ``F`` is the bench's capacity anchor: the canonical matrix
    cells sit at **``{F, F//2}``** uniformly for every bench (boundary + deep-
    scarce). For most benches ``F`` is the ground-truth feature-direction count;
    where it means something else (e.g. frequency, where it's the alphabet ``M``)
    ``F_note`` records that. Comparability across benches rides on the
    ``[chance, oracle]`` normalization + matched per-token sparsity, NOT on an
    identical absolute ``d_sae`` — so a per-bench ``F`` is fine.
    """
    name: str
    datasources: tuple[str, ...]
    protocol: str
    F: int
    axes: tuple[LatentAxis, ...]
    verdict: str
    note: str = ""
    F_note: str = ""   # what F denotes for this bench (usually the direction count)


@dataclass(frozen=True)
class Arch:
    """One matrix COLUMN. ``family`` groups the decode structure; ``windowed``
    is False for token-level archs (``T`` fixed at 1 — the budget-matching base
    case). Every arch shares the BatchTopK→JumpReLU fair backbone, so the only
    variable across a row is decode structure."""
    name: str
    label: str
    family: str        # "token" | "stacked" | "txc-pre" | "txc-post" | "spectral"
    windowed: bool


# ── Benchmarks (matrix rows = one per (bench, latent-axis)) ──────────────────
BENCHES: tuple[Bench, ...] = (
    Bench(
        "backtracking", ("toy_backtracking_selfexcite_d64",), "1.2.0", F=20,
        verdict="POSITIVE",
        axes=(LatentAxis("lambda", "lambda_recovery", "AC",
                         "λ — self-exciting intensity (linear-in-history)"),),
        note="all window families recover λ; per-token pinned at the DPI floor.",
        F_note="feature-direction count (1 backtrack + 19 content).",
    ),
    Bench(
        "signed_motion", ("toy_signed_motion_M19_d40",), "1.2.0", F=19,
        verdict="NEGATIVE",
        axes=(LatentAxis("sign", "s_temp", "AC",
                         "sign — ±1 order-sensitive step"),),
        note="no arch recovers the sign in the scarce regime (#windows=2F confound).",
        F_note="feature-direction count (19 step directions).",
    ),
    Bench(
        "changepoint", ("toy_changepoint_modes_d64",), "1.2.0", F=20,
        verdict="SPLIT",
        axes=(
            LatentAxis("mode", "mode_recovery", "DC",
                       "mode m_t — global hidden state", primary=True),
            LatentAxis("tss", "tss_recovery", "AC",
                       "time-since-switch (primary AC latent)", primary=True),
            LatentAxis("cp", "cp_recovery", "AC",
                       "change-point c_t — adjacency floor", primary=False),
        ),
        note="per-token wins DC mode; only post-squash exposes the AC boundary.",
        F_note="feature-direction count (8 mode-signature + 12 content).",
    ),
    Bench(
        "frequency",
        ("toy_cyclic_circle_M101_d128", "toy_cyclic_random_M101_d128"),
        "1.2.0", F=101, verdict="POSITIVE",
        axes=(LatentAxis("velocity", "velocity_recovery", "AC",
                         "velocity Y — cyclic tone f = Y/M"),),
        note="position-mixing crosscoders recover the tone; additive/per-token flat.",
        F_note=("alphabet M=101 (NOT a direction count): the circle embedding is "
                "rank-2 (all M symbols in a 2-D plane), so {101, 50} are alphabet-"
                "scaled capacities, both < the memorization budget |Ω|·M=1010."),
    ),
)

# ── Architectures (matrix columns; the fair-backbone family only) ────────────
ARCHS: tuple[Arch, ...] = (
    Arch("batchtopk_sae", "Per-token SAE", "token", windowed=False),
    Arch("tsae", "T-SAE (contrastive)", "token", windowed=False),
    Arch("stacked_batchtopk", "Stacked (per-position dicts)", "stacked", windowed=True),
    Arch("txc_batchtopk_pre", "TXC-pre (additive)", "txc-pre", windowed=True),
    Arch("txc_batchtopk_post", "TXC-post (coincidence)", "txc-post", windowed=True),
    Arch("spectral_txc", "Spectral-TXC (DCT bands)", "spectral", windowed=True),
)


@dataclass(frozen=True)
class OperatingPoint:
    """The canonical matrix cell for the **per-token-matched** headline: window
    ``T = T_can`` (token archs are ``T=1``), matched to ``B*`` atoms/token, taken
    at each of the capacities ``{F, F//2}`` (boundary + deep-scarce, uniform for
    every bench). Parameterized so ``T_can`` / ``B_star`` / ``capacity_fracs`` are
    one-number changes; the full ``d_sae`` and ``T`` frontiers live per bench.

    **Feasibility (why T_can=4, B*=2).** The per-token match needs ``k_win =
    B*·T_can ≤ d_sae`` at the *scarcest* slice — ``min F//2`` over benches is 9
    (signed_motion). ``T_can=4, B*=2`` → ``k_win = 8 ≤ 9`` — feasible at every
    bench's ``F//2`` and both capacity slices. Larger ``T`` (8) lives in the
    per-bench frontiers."""
    T_can: int = 4
    B_star: float = 2.0
    capacity_fracs: tuple[float, ...] = (1.0, 0.5)   # {F, F//2}
    # how close a cell's realized L0 must sit to B* to count as "matched"
    l0_tol: float = 1.0


OP = OperatingPoint()


def capacities(bench: Bench, op: OperatingPoint = OP) -> list[int]:
    """The bench's canonical ``d_sae`` cells — ``{F, F//2}`` by default, uniform
    across benches (comparability rides on normalization, not identical d_sae)."""
    return [max(1, int(bench.F * f)) for f in op.capacity_fracs]

LEADERBOARD_REL = "results/leaderboard.jsonl"
REPORT_REL = "experiments/explorations/synthetic/REPORT.md"
STATS_REL = "experiments/explorations/synthetic/results/program_stats.json"


def matrix_rows() -> list[tuple[Bench, LatentAxis]]:
    """Flatten (bench, axis) pairs in display order — one per matrix row."""
    return [(b, ax) for b in BENCHES for ax in b.axes]
