"""Model-agnostic registry for subject LMs used in the sprint.

Everything in the pipeline — activation caching, SAE/crosscoder training,
eval, auto-interp — resolves model-specific constants (hf_path, d_model,
layer indices, dtype) from here. Adding a new model is 10 lines: append
an entry to MODEL_REGISTRY, then run `scripts/download_models.sh`.

Do NOT hardcode d_model, layer indices, or HF paths anywhere else.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelConfig:
    name: str                          # short key, used as dir name in data/
    hf_path: str                       # HuggingFace repo id
    d_model: int                       # residual stream width
    n_layers: int                      # total transformer layers
    default_layer_indices: tuple[int, ...]  # sensible caching defaults
    tokenizer_path: str | None = None  # defaults to hf_path if None
    is_thinking_model: bool = False    # produces <think> traces
    dtype: str = "bfloat16"            # "float16" | "bfloat16" | "float32"
    architecture_family: str = "llama" # "llama" | "gemma" | "qwen" — governs hook paths

    @property
    def tokenizer(self) -> str:
        return self.tokenizer_path or self.hf_path


MODEL_REGISTRY: dict[str, ModelConfig] = {
    "deepseek-r1-distill-llama-8b": ModelConfig(
        name="deepseek-r1-distill-llama-8b",
        hf_path="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        d_model=4096,
        n_layers=32,
        default_layer_indices=(12, 24),  # ~37% / ~75% depth
        is_thinking_model=True,
        dtype="bfloat16",
        architecture_family="llama",
    ),
    "llama-3.1-8b": ModelConfig(
        name="llama-3.1-8b",
        hf_path="meta-llama/Llama-3.1-8B",
        d_model=4096,
        n_layers=32,
        default_layer_indices=(12, 24),
        is_thinking_model=False,
        dtype="bfloat16",
        architecture_family="llama",
    ),
    # Open mirror of meta-llama/Llama-3.1-8B for hosts without HF auth.
    # Same architecture and weights; the Llama gate is just a request-form
    # on Meta's side.
    "llama-3.1-8b-nous": ModelConfig(
        name="llama-3.1-8b-nous",
        hf_path="NousResearch/Meta-Llama-3.1-8B",
        d_model=4096,
        n_layers=32,
        default_layer_indices=(12, 24),
        is_thinking_model=False,
        dtype="bfloat16",
        architecture_family="llama",
    ),
    "gemma-2-2b": ModelConfig(
        name="gemma-2-2b",
        hf_path="google/gemma-2-2b",
        d_model=2304,
        n_layers=26,
        default_layer_indices=(13, 25),  # Andre's existing choices
        is_thinking_model=False,
        dtype="float16",
        architecture_family="gemma",
    ),
    "gemma-2-2b-it": ModelConfig(
        name="gemma-2-2b-it",
        hf_path="google/gemma-2-2b-it",
        d_model=2304,
        n_layers=26,
        default_layer_indices=(13, 25),
        is_thinking_model=False,
        dtype="float16",
        architecture_family="gemma",
    ),
}


def get_model_config(name: str) -> ModelConfig:
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


def list_models() -> list[str]:
    return sorted(MODEL_REGISTRY)


# Hook-path helpers. HuggingFace Llama/Gemma both expose `model.model.layers[i]`
# (LlamaDecoderLayer / Gemma2DecoderLayer) and a `.self_attn` submodule, so the
# hook targets are the same string path — but we go through a helper so a future
# Qwen/Mamba/etc. entry can override without touching call sites.
def resid_hook_target(model, layer_idx: int, family: str):
    if family in ("llama", "gemma", "qwen"):
        return model.model.layers[layer_idx]
    raise NotImplementedError(f"Hook target for family '{family}' not implemented")


def attn_hook_target(model, layer_idx: int, family: str):
    if family in ("llama", "gemma", "qwen"):
        return model.model.layers[layer_idx].self_attn
    raise NotImplementedError(f"Hook target for family '{family}' not implemented")
