from .base import TemporalAE, ModelOutput
from .baum_welch_factorial import BaumWelchFactorialAE
from .topk_sae import TopKSAE
from .stacked_sae import StackedSAE
from .temporal_crosscoder import TemporalCrosscoder
from .per_feature_temporal import PerFeatureTemporalAE

MODEL_REGISTRY: dict[str, type[TemporalAE]] = {
    "regular_sae": TopKSAE,
    "sae": TopKSAE,  # backward-compat alias for existing scripts/results
    "stacked_sae": StackedSAE,
    "txcdr": TemporalCrosscoder,
    "per_feature_temporal": PerFeatureTemporalAE,
    "bw_factorial": BaumWelchFactorialAE,
}

__all__ = [
    "TemporalAE",
    "ModelOutput",
    "BaumWelchFactorialAE",
    "TopKSAE",
    "StackedSAE",
    "TemporalCrosscoder",
    "PerFeatureTemporalAE",
    "MODEL_REGISTRY",
]
