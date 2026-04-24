"""MLC + anti-dead stack — fairness counterpart to TXCBareAntidead.

The math is identical to `TXCBareAntidead`. The only semantic
difference is that the second axis represents *layer* (one residual-
stream layer per slot, all from the same token) rather than *time*
(consecutive tokens, all from the same layer).

We expose this as a thin subclass for code clarity in dispatchers and
checkpoints — `MLCBareAntidead` should be named in logs, plots, and
the experiment index, even though the parameters are shape-identical
to a TXC counterpart at T=L.
"""

from __future__ import annotations

from src.architectures.txc_bare_antidead import TXCBareAntidead
from src.architectures.txc_bare_matryoshka_contrastive_antidead import (
    TXCBareMatryoshkaContrastiveAntidead,
)
from src.architectures.txc_bare_multiscale_contrastive_antidead import (
    TXCBareMultiscaleContrastiveAntidead,
)


class MLCBareAntidead(TXCBareAntidead):
    """Layer-axis crosscoder + anti-dead stack (no matryoshka, no contrastive)."""


class MLCBareMatryoshkaContrastiveAntidead(TXCBareMatryoshkaContrastiveAntidead):
    """Layer-axis crosscoder + anti-dead + matryoshka H/L + adjacent-token InfoNCE."""


class MLCBareMultiscaleContrastiveAntidead(TXCBareMultiscaleContrastiveAntidead):
    """Layer-axis crosscoder + anti-dead + matryoshka + multi-scale InfoNCE.

    Multi-scale here means InfoNCE at nested matryoshka prefix lengths
    (s+1)*matryoshka_h_size for s in [0, n_contr_scales-1] — exactly
    H7's multi-scale recipe applied to the layer axis.
    """
