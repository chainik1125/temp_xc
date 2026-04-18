"""Single source of truth for the SAEBench sparse-probing experiment.

All k-values, T-values, protocol definitions, aggregation names, and the
Gemma-2-2B L12 target spec live here. Everything downstream
(matching_protocols.py, shell scripts, probing_runner.py) reads from
this module rather than hardcoding constants.

See docs/aniket/experiments/sparse_probing/plan.md for rationale.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ─── target subject model ──────────────────────────────────────────────────
SUBJECT_MODEL = "gemma-2-2b"
D_MODEL = 2304
LAYER = 12
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"
CONTEXT_LENGTH = 128  # SAEBench default; probe inputs are seq_len=128

# ─── architecture parameters (fixed across all runs) ──────────────────────
D_SAE = D_MODEL * 8  # 18,432; matches existing sprint runs
EXPANSION_FACTOR = 8
TOTAL_TRAINING_STEPS = 10_000  # match existing sprint
# MLC layer window: 5-layer middle-out around L12 → {10, 11, 12, 13, 14}
MLC_LAYERS = (LAYER - 2, LAYER - 1, LAYER, LAYER + 1, LAYER + 2)

# ─── T-sweep (TempXC only) ─────────────────────────────────────────────────
T_BASE = 5                       # main experiment at T=5
T_SWEEP = (5, 10, 20, 40)        # Dmitry's H100 extension
T_FALLBACK_ON_OOM = 32           # if T=40 OOMs, fall back

# ─── sparsity (TopK k values for probing) ──────────────────────────────────
# SAEBench default k_values are [1, 2, 5]; we extend with 20 per plan.md.
PROBING_K_VALUES = (1, 2, 5, 20)

# ─── architecture key names ────────────────────────────────────────────────
ARCH_SAE = "sae"          # single-token TopK SAE
ARCH_MLC = "mlc"          # layer-wise crosscoder
ARCH_TEMPXC = "tempxc"    # temporal crosscoder

ArchName = Literal["sae", "mlc", "tempxc"]
ProtocolName = Literal["A", "B"]
AggregationName = Literal["last", "mean", "max", "full_window"]

AGGREGATIONS: tuple[AggregationName, ...] = ("last", "mean", "max", "full_window")


# ─── matching protocols ────────────────────────────────────────────────────
@dataclass(frozen=True)
class MatchingProtocol:
    """Per-architecture TopK k values at a given T.

    Protocol A matches per-token sparsity (all three archs use the same
    k at training). Protocol B matches total-window activation budget
    (TempXC uses k × T = 500 at T=5, 100 per window-position on average).
    """

    name: ProtocolName
    sae_k: int
    mlc_k: int
    tempxc_k_base: int  # k at T=T_BASE; scales with T in protocol B

    def tempxc_k_at(self, t: int) -> int:
        """Per-position TopK k for TempXC (the crosscoder's __init__
        multiplies by T to get window_k).

        Protocol A (per-token matched): constant per-position k, so
        window_k = tempxc_k_base * T scales with T.

        Protocol B (total-window budget matched): window_k fixed at
        tempxc_k_base, so per-position k = tempxc_k_base / T
        inversely scales with T. At T=5, B coincides with A by design.
        """
        if self.name == "A":
            return self.tempxc_k_base
        return max(1, self.tempxc_k_base // t)


PROTOCOL_A = MatchingProtocol(name="A", sae_k=100, mlc_k=100, tempxc_k_base=100)
PROTOCOL_B = MatchingProtocol(name="B", sae_k=100, mlc_k=100, tempxc_k_base=500)

PROTOCOLS: dict[ProtocolName, MatchingProtocol] = {
    "A": PROTOCOL_A,
    "B": PROTOCOL_B,
}


# ─── probing tasks (SAEBench default + k=20 extension) ────────────────────
# These are the 8 datasets SAEBench ships; each unrolls into ~5 binary
# one-vs-rest probing tasks, totaling ~30 tasks. See saebench_notes § 3.
DEFAULT_DATASETS: tuple[str, ...] = (
    "LabHC/bias_in_bios_class_set1",
    "LabHC/bias_in_bios_class_set2",
    "LabHC/bias_in_bios_class_set3",
    "canrager/amazon_reviews_mcauley_1and5",
    "canrager/amazon_reviews_mcauley_1and5_sentiment",
    "codeparrot/github-code",
    "fancyzhx/ag_news",
    "Helsinki-NLP/europarl",
)


# ─── checkpoint naming convention ──────────────────────────────────────────
# Mirrors the existing results/nlp/*/ckpts/ naming for cross-experiment
# consistency: arch__subject-model__dataset__layer__kK__seedS[__tT].pt
def ckpt_name(
    arch: ArchName,
    protocol: ProtocolName,
    t: int = T_BASE,
    seed: int = 42,
) -> str:
    """Canonical checkpoint filename for the saebench experiment grid."""
    proto = PROTOCOLS[protocol]
    if arch == "sae":
        k = proto.sae_k
        return f"sae__{SUBJECT_MODEL}__l{LAYER}__k{k}__prot{protocol}__seed{seed}.pt"
    if arch == "mlc":
        k = proto.mlc_k
        layers = "-".join(str(l) for l in MLC_LAYERS)
        return f"mlc__{SUBJECT_MODEL}__l{layers}__k{k}__prot{protocol}__seed{seed}.pt"
    if arch == "tempxc":
        k = proto.tempxc_k_at(t)
        return (
            f"tempxc__{SUBJECT_MODEL}__l{LAYER}__k{k}__T{t}"
            f"__prot{protocol}__seed{seed}.pt"
        )
    raise ValueError(f"unknown arch: {arch}")


CKPT_DIR = "results/saebench/ckpts"
LOG_DIR = "results/saebench/logs"
RESULTS_DIR = "results/saebench/results"  # JSONL output
SAEBENCH_ARTIFACTS_DIR = "results/saebench/saebench_artifacts"


# ─── full grid enumeration ─────────────────────────────────────────────────
@dataclass(frozen=True)
class GridCell:
    """One (arch, T, protocol, aggregation, task, k) cell of the grid."""

    arch: ArchName
    t: int
    protocol: ProtocolName
    aggregation: AggregationName
    task: str
    k: int

    def to_dict(self) -> dict:
        return {
            "architecture": self.arch,
            "t": self.t,
            "matching_protocol": self.protocol,
            "aggregation": self.aggregation,
            "task": self.task,
            "k": self.k,
        }


def enumerate_base_grid() -> list[GridCell]:
    """Base grid at T=T_BASE, all 3 architectures."""
    cells: list[GridCell] = []
    for arch in (ARCH_SAE, ARCH_MLC, ARCH_TEMPXC):
        for protocol in ("A", "B"):
            for agg in AGGREGATIONS:
                # SAE / MLC have no T axis, so aggregation only varies
                # behavior on tempxc. We still record the agg field for
                # uniform schema; SAE's 4 aggregation rows will be equal.
                for task in DEFAULT_DATASETS:
                    for k in PROBING_K_VALUES:
                        cells.append(GridCell(
                            arch=arch, t=T_BASE, protocol=protocol,
                            aggregation=agg, task=task, k=k,
                        ))
    return cells


def enumerate_tsweep_grid() -> list[GridCell]:
    """T-sweep grid: TempXC-only at T ∈ T_SWEEP (excluding T_BASE which
    is already in the base grid)."""
    cells: list[GridCell] = []
    for t in T_SWEEP:
        if t == T_BASE:
            continue  # already in base grid
        for protocol in ("A", "B"):
            for agg in AGGREGATIONS:
                for task in DEFAULT_DATASETS:
                    for k in PROBING_K_VALUES:
                        cells.append(GridCell(
                            arch=ARCH_TEMPXC, t=t, protocol=protocol,
                            aggregation=agg, task=task, k=k,
                        ))
    return cells
