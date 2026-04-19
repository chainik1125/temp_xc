"""Architecture registry.

Register all ArchSpec subclasses here. To add a new architecture:
    1. Create my_arch.py with nn.Module + MyArchSpec(ArchSpec)
    2. Add it to REGISTRY below
    3. It's now available in sweeps via get_default_models()
"""

from src.architectures.base import ArchSpec, ModelEntry, EvalOutput
from src.architectures.topk_sae import TopKSAESpec
from src.architectures.stacked_sae import StackedSAESpec
from src.architectures.crosscoder import CrosscoderSpec
from src.architectures.tfa import TFASpec
from src.architectures.mlc import MLCSpec
from src.architectures.matryoshka_txcdr import MatryoshkaTXCDRSpec

# Central registry: name -> ArchSpec constructor
REGISTRY: dict[str, type[ArchSpec]] = {
    "topk_sae": TopKSAESpec,
    "stacked_sae": StackedSAESpec,
    "crosscoder": CrosscoderSpec,
    "tfa": TFASpec,
    "tfa_pos": TFASpec,
    "mlc": MLCSpec,
    "matryoshka_txcdr": MatryoshkaTXCDRSpec,
}


def get_default_models(
    T_values: list[int],
    tfa_bottleneck_factor: int = 1,
    tfa_batch_size: int = 64,
) -> list[ModelEntry]:
    """Build the standard set of models for a sweep.

    Creates one entry per architecture per T value (where applicable).
    This is the default model list used by the sweep runner.

    Args:
        T_values: Window sizes to sweep over.
        tfa_bottleneck_factor: Bottleneck factor for TFA attention.
            At NLP scale (d_sae > 10K), TFA's attention params scale
            with d_sae^2 / bottleneck_factor. Use 1 for toy scale,
            4+ for NLP scale to keep TFA feasible on a single GPU.
        tfa_batch_size: Batch size for TFA (number of sequences).
            TFA processes full sequences so needs smaller batches
            than window-based models.

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

    # TFA: full-sequence models. NLP-scale uses lr=3e-4 + large batch for
    # stability (see logs/tfa_debug_fix_v4.json study).
    tfa_lr = 3e-4 if tfa_batch_size >= 32 else 1e-3
    models.append(ModelEntry(
        name="TFA",
        spec=TFASpec(use_pos_encoding=False, bottleneck_factor=tfa_bottleneck_factor),
        gen_key="seq",
        training_overrides={"batch_size": tfa_batch_size, "lr": tfa_lr},
    ))
    models.append(ModelEntry(
        name="TFA-pos",
        spec=TFASpec(use_pos_encoding=True, bottleneck_factor=tfa_bottleneck_factor),
        gen_key="seq",
        training_overrides={"batch_size": tfa_batch_size, "lr": tfa_lr},
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

    return models
