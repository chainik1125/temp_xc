from .base import TemporalAE, ModelOutput
from .baum_welch_factorial import BaumWelchFactorialAE
from .batchtopk_sae import BatchTopKSAE
from .topk_sae import TopKSAE
from .temporal_crosscoder import TemporalCrosscoder
from .per_feature_temporal import PerFeatureTemporalAE
from .tfa import TemporalFeatureAutoencoder

MODEL_REGISTRY: dict[str, type[TemporalAE]] = {
    "sae": TopKSAE,
    "batchtopk_sae": BatchTopKSAE,
    "tfa": TemporalFeatureAutoencoder,
    "txcdr": TemporalCrosscoder,
    "per_feature_temporal": PerFeatureTemporalAE,
    "bw_factorial": BaumWelchFactorialAE,
}

__all__ = [
    "TemporalAE",
    "ModelOutput",
    "BaumWelchFactorialAE",
    "BatchTopKSAE",
    "TopKSAE",
    "TemporalFeatureAutoencoder",
    "TemporalCrosscoder",
    "PerFeatureTemporalAE",
    "MODEL_REGISTRY",
]
