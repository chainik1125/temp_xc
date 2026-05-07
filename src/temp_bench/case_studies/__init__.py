"""temp-bench: the suite of behavioural case studies.

Concrete implementations live in this package:

- :mod:`.steering`     — C5: RLHF / sentiment steering case study
- :mod:`.em`           — C6: Emergent misalignment (Wang procedure on Qwen-14B)
- :mod:`.backtracking` — C7: Backtracking (Ward Stage B on Gemma-2-2b)

Each is a subclass of :class:`temp_bench.eval.case_study.CaseStudy` and is
loaded lazily by :func:`get`.
"""

from typing import Type

from temp_bench.eval.case_study import CaseStudy

_REGISTRY: dict[str, str] = {
    "c5_steering": "temp_bench.case_studies.steering:SteeringCaseStudy",
    "c6_em": "temp_bench.case_studies.em:EMCaseStudy",
    "c7_backtracking": "temp_bench.case_studies.backtracking:BacktrackingCaseStudy",
}


def get(name: str) -> Type[CaseStudy]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown case study {name!r}. Available: {sorted(_REGISTRY)}.")
    module_path, class_name = _REGISTRY[name].split(":")
    import importlib
    return getattr(importlib.import_module(module_path), class_name)


def names() -> list[str]:
    return sorted(_REGISTRY)
