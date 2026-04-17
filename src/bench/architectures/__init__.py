"""Architecture registry.

Register all ArchSpec subclasses here. To add a new architecture:
    1. Create my_arch.py with nn.Module + MyArchSpec(ArchSpec)
    2. Add it to REGISTRY below
    3. It's now available in sweeps via get_default_models()
"""

from src.bench.architectures.base import ArchSpec, ModelEntry, EvalOutput
from src.bench.architectures.topk_sae import TopKSAESpec
from src.bench.architectures.stacked_sae import StackedSAESpec
from src.bench.architectures.crosscoder import CrosscoderSpec
from src.bench.architectures.mlc import LayerCrosscoderSpec
from src.bench.architectures.tfa import TFASpec

# Central registry: name -> ArchSpec constructor
REGISTRY: dict[str, type[ArchSpec]] = {
    "topk_sae": TopKSAESpec,
    "stacked_sae": StackedSAESpec,
    "crosscoder": CrosscoderSpec,
    "mlc": LayerCrosscoderSpec,
    "tfa": TFASpec,
    "tfa_pos": TFASpec,
}


def get_default_models(T_values: list[int]) -> list[ModelEntry]:
    """Build the standard set of models for a sweep.

    Creates one entry per architecture per T value (where applicable).
    This is the default model list used by the sweep runner.

    Args:
        T_values: Window sizes to sweep over.

    Returns:
        List of ModelEntry for all registered architectures.
    """
    models = []

    # TopK SAE: single-token baseline (no T dependency)
    models.append(ModelEntry(
        name="TopKSAE",
        spec=TopKSAESpec(),
        gen_key="flat",
    ))

    # TFA: full-sequence models
    models.append(ModelEntry(
        name="TFA",
        spec=TFASpec(use_pos_encoding=False),
        gen_key="seq",
        training_overrides={"batch_size": 64, "lr": 1e-3},
    ))
    models.append(ModelEntry(
        name="TFA-pos",
        spec=TFASpec(use_pos_encoding=True),
        gen_key="seq",
        training_overrides={"batch_size": 64, "lr": 1e-3},
    ))

    # Stacked SAE and Crosscoder: one per T
    for T in T_values:
        models.append(ModelEntry(
            name=f"Stacked T={T}",
            spec=StackedSAESpec(T=T),
            gen_key=f"window_{T}",
        ))
        models.append(ModelEntry(
            name=f"TXCDR T={T}",
            spec=CrosscoderSpec(T=T),
            gen_key=f"window_{T}",
        ))

    # MLC (layer-wise crosscoder): 5-layer window, middle-out around
    # layer 12 for Gemma-2-2B. Uses gen_key "window_5" to reuse the
    # window-batch machinery in data.py — the multi_layer data
    # pipeline populates gen_windows[5] with layer-stacked samples.
    models.append(ModelEntry(
        name="MLC n_layers=5",
        spec=LayerCrosscoderSpec(n_layers=5),
        gen_key="window_5",
    ))

    return models
