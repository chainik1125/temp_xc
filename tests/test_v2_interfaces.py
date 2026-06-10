"""Contract tests: every registered arch / evaluator honors the interface."""

from __future__ import annotations

import pytest

from temp_bench.core.config import import_by_path, list_archs, load_arch
from temp_bench.core.runner import _EVALUATOR_REGISTRY
from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import Evaluator


@pytest.mark.parametrize("arch_name", list_archs())
def test_arch_is_temp_bench_arch_subclass(arch_name: str) -> None:
    spec = load_arch(arch_name)
    cls = import_by_path(spec.class_path)
    assert isinstance(cls, type), f"{spec.class_path} did not resolve to a class"
    assert issubclass(cls, TempBenchArch), (
        f"{cls.__name__} must subclass TempBenchArch"
    )


@pytest.mark.parametrize("arch_name", list_archs())
def test_arch_declares_v2_attrs(arch_name: str) -> None:
    spec = load_arch(arch_name)
    cls = import_by_path(spec.class_path)
    assert hasattr(cls, "arch_version"), f"{cls.__name__} missing arch_version"
    assert hasattr(cls, "consumes"), f"{cls.__name__} missing consumes"
    assert cls.consumes in {"token", "window", "sequence"}, (
        f"{cls.__name__}.consumes={cls.consumes!r} must be 'token', 'window', or 'sequence'"
    )


@pytest.mark.parametrize("experiment", list(_EVALUATOR_REGISTRY))
def test_evaluator_is_evaluator_subclass(experiment: str) -> None:
    cls = import_by_path(_EVALUATOR_REGISTRY[experiment])
    assert isinstance(cls, type), f"{_EVALUATOR_REGISTRY[experiment]} not a class"
    assert issubclass(cls, Evaluator), (
        f"{cls.__name__} must subclass Evaluator"
    )
    inst = cls()
    assert inst.name == experiment or inst.name.startswith(experiment), (
        f"Evaluator name mismatch: registry key {experiment!r} vs class.name {inst.name!r}"
    )
    assert inst.protocol_version, "Evaluator must declare protocol_version"


def test_arch_registry_unique_names() -> None:
    archs = list_archs()
    assert len(archs) == len(set(archs)), "arch registry has duplicate names"


def test_evaluator_registry_disjoint_from_arch_registry() -> None:
    archs = set(list_archs())
    evals = set(_EVALUATOR_REGISTRY)
    overlap = archs & evals
    assert not overlap, (
        f"Arch and evaluator names collide: {overlap}. "
        "Pick distinct names to avoid CLI ambiguity."
    )
