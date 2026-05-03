"""Locked architecture registry — paper claim is "two TXCs everywhere."

Adding a third TXC variant breaks the paper's central methodology claim.
If you genuinely need to, raise it in ``docs/components/`` first.

Free per-component knobs: ``k_pos`` (sparsity), ``d_sae`` (dict size).
Everything else is fixed by the architecture spec.
"""

from typing import Type

from temp_bench.architectures.base import TempBenchArch  # noqa: F401

# Concrete registry. Lazy-import so the package imports even before each
# arch file is fleshed out.
_REGISTRY: dict[str, str] = {
    "topk_sae": "temp_bench.architectures.topk_sae:TopKSAE",
    "tsae_paper": "temp_bench.architectures.tsae:TSAEPaper",
    "tfa": "temp_bench.architectures.tfa:TFA",
    "mlc": "temp_bench.architectures.mlc:MLC",
    "sae_arditi": "temp_bench.architectures.sae_arditi:SAEArditi",
    "txc_base": "temp_bench.architectures.txc_base:TXCBase",  # txc_bare_antidead_t5
    "txc_pro": "temp_bench.architectures.txc_pro:TXCPro",     # phase5b_subseq_h8
}


def get(name: str) -> Type[TempBenchArch]:
    """Resolve an architecture name to its class. Imports lazily."""
    if name not in _REGISTRY:
        raise KeyError(
            f"Unknown architecture {name!r}. Locked set: {sorted(_REGISTRY)}. "
            "Adding a new arch requires a docs/components/ proposal."
        )
    module_path, class_name = _REGISTRY[name].split(":")
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def names() -> list[str]:
    return sorted(_REGISTRY)
