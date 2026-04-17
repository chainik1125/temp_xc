"""Sparsity-matching protocol definitions.

Re-exported from configs.py — this file is kept thin so downstream
code has a stable import path (`from src.bench.saebench.matching_protocols
import PROTOCOL_A, PROTOCOL_B, protocol_k`) even if configs.py grows.

Two protocols, both pre-registered:

  A (per-token k matched):
        SAE k=100, MLC k=100, TempXC k=100
        TempXC is effectively 5× sparser in total-activation terms at T=5.
        Tests: "are TempXC's individual features more probing-useful?"

  B (total-window budget matched):
        SAE k=100, MLC k=100, TempXC k=500 (at T=5; scales with T)
        All architectures spend the same total feature-activation budget
        over a T-token window.
        Tests: "is TempXC's representation as a whole more probing-useful?"

For T-sweep (TempXC only), protocol B's tempxc_k scales linearly with T.

See docs/aniket/experiments/sparse_probing/plan.md § 4.
"""

from __future__ import annotations

from src.bench.saebench.configs import (
    PROTOCOL_A,
    PROTOCOL_B,
    PROTOCOLS,
    MatchingProtocol,
    ProtocolName,
    ArchName,
    T_BASE,
)


__all__ = [
    "PROTOCOL_A",
    "PROTOCOL_B",
    "PROTOCOLS",
    "MatchingProtocol",
    "ProtocolName",
    "protocol_k",
]


def protocol_k(
    arch: ArchName,
    protocol: ProtocolName,
    t: int = T_BASE,
) -> int:
    """Resolve the TopK k for an (arch, protocol, T) combination.

    This is the single callsite training/eval scripts read to decide
    what k to instantiate their architecture with.
    """
    p = PROTOCOLS[protocol]
    if arch == "sae":
        return p.sae_k
    if arch == "mlc":
        return p.mlc_k
    if arch == "tempxc":
        return p.tempxc_k_at(t)
    raise ValueError(f"unknown arch: {arch}")
