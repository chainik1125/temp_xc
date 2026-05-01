"""Vendored from `han-phase7-unification` branch (commit 43012fd7 family) for
Stage B's contrastive/matryoshka sweep. READ-ONLY ports — do not modify
the algorithm; if Han updates the source, re-vendor.

Hierarchy:
    TXCBareAntidead                                      (anti-dead + matry-aware base)
        └ TXCBareMatryoshkaContrastiveAntidead          (+ matryoshka H/L + InfoNCE shift-1)
            ├ TXCBareMultiDistanceContrastiveAntidead   (+ multi-distance shifts)         → arch="txc_h8"
            └ TXCBareMDxMSContrastiveAntidead           (+ multi-distance × multi-scale)  → arch="txc_h13"
"""
from temporal_crosscoders.han_arch.txc_bare_antidead import TXCBareAntidead
from temporal_crosscoders.han_arch.txc_bare_matryoshka_contrastive_antidead import (
    TXCBareMatryoshkaContrastiveAntidead, _info_nce,
)
from temporal_crosscoders.han_arch.txc_bare_multidistance_contrastive_antidead import (
    TXCBareMultiDistanceContrastiveAntidead, make_multidistance_pair_gen_gpu,
)
from temporal_crosscoders.han_arch.txc_bare_md_ms_contrastive_antidead import (
    TXCBareMDxMSContrastiveAntidead,
)

__all__ = [
    "TXCBareAntidead",
    "TXCBareMatryoshkaContrastiveAntidead",
    "TXCBareMultiDistanceContrastiveAntidead",
    "TXCBareMDxMSContrastiveAntidead",
    "make_multidistance_pair_gen_gpu",
    "_info_nce",
]
