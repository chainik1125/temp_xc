from .base import TemporalAE, ModelOutput
from .topk_sae import TopKSAE
from .temporal_crosscoder import TemporalCrosscoder
from .per_feature_temporal import PerFeatureTemporalAE

MODEL_REGISTRY: dict[str, type[TemporalAE]] = {
    "sae": TopKSAE,
    "txcdr": TemporalCrosscoder,
    "per_feature_temporal": PerFeatureTemporalAE,
}

__all__ = [
    "TemporalAE",
    "ModelOutput",
    "TopKSAE",
    "TemporalCrosscoder",
    "PerFeatureTemporalAE",
    "MODEL_REGISTRY",
]
