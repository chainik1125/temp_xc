"""Vendored from Dmitry's collaborator's `TemporalFeatureAnalysis` repo
(see references/TemporalFeatureAnalysis/sae/ in han-phase7-unification).

We only import TemporalSAE eagerly because it's the one Stage B uses as a
baseline. SAEStandard is left importable on demand (it pulls `sparsemax`
which isn't a hard dependency for Stage B).
"""

from .utils import step_fn
from .saeTemporal import TemporalSAE


def _import_saeStandard():
    """Lazy importer in case downstream code wants the standard SAE."""
    from .saeStandard import SAEStandard
    return SAEStandard
