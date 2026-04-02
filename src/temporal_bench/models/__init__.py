from .base import TemporalAE, ModelOutput
from .baum_welch_factorial import BaumWelchFactorialAE
from .topk_sae import TopKSAE
from .temporal_crosscoder import TemporalCrosscoder
from .per_feature_temporal import PerFeatureTemporalAE

MODEL_REGISTRY: dict[str, type[TemporalAE]] = {
    "sae": TopKSAE,
    "txcdr": TemporalCrosscoder,
    "per_feature_temporal": PerFeatureTemporalAE,
    "bw_factorial": BaumWelchFactorialAE,
}

__all__ = [
    "TemporalAE",
    "ModelOutput",
    "BaumWelchFactorialAE",
    "TopKSAE",
    "TemporalCrosscoder",
    "PerFeatureTemporalAE",
    "MODEL_REGISTRY",
]
