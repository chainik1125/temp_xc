"""Cache-key contract: deterministic + invariant under irrelevant changes."""

from __future__ import annotations

from temp_bench.core.config import (
    compute_data_key,
    compute_eval_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.schemas import TrainingConfig


def _txc_train_key(seed=42, **cfg_kw):
    arch = load_arch("txc_base", section="synthetic")
    ds = load_datasource("synth_smoke")
    data_key = compute_data_key(ds)
    return compute_train_key(
        arch=arch, seed=seed,
        training_cfg=TrainingConfig(**cfg_kw),
        data_key=data_key, section="synthetic",
    )


def test_data_key_deterministic() -> None:
    ds = load_datasource("synth_smoke")
    a = compute_data_key(ds)
    b = compute_data_key(ds)
    assert a == b


def test_train_key_deterministic() -> None:
    a = _txc_train_key()
    b = _txc_train_key()
    assert a == b


def test_train_key_changes_with_seed() -> None:
    a = _txc_train_key(seed=1)
    b = _txc_train_key(seed=2)
    assert a != b


def test_train_key_changes_with_n_steps() -> None:
    a = _txc_train_key(n_steps=100)
    b = _txc_train_key(n_steps=200)
    assert a != b


def test_train_key_invariant_under_default_extension() -> None:
    """Adding a field to TrainingConfig with default=None must NOT
    invalidate cells that never set it (exclude_none semantics)."""
    # TrainingConfig default has arch_hparams_override=None.
    a = _txc_train_key()
    # Explicitly setting arch_hparams_override=None should match default.
    b = _txc_train_key(arch_hparams_override=None)
    assert a == b


def test_eval_key_deterministic() -> None:
    tk = _txc_train_key()
    a = compute_eval_key(
        train_key=tk, evaluator_name="synthetic_recovery",
        evaluator_protocol_version="1.0.0", eval_cfg={"smoke": True},
    )
    b = compute_eval_key(
        train_key=tk, evaluator_name="synthetic_recovery",
        evaluator_protocol_version="1.0.0", eval_cfg={"smoke": True},
    )
    assert a == b


def test_eval_key_changes_with_eval_cfg() -> None:
    tk = _txc_train_key()
    a = compute_eval_key(
        train_key=tk, evaluator_name="synthetic_recovery",
        evaluator_protocol_version="1.0.0", eval_cfg={"smoke": True},
    )
    b = compute_eval_key(
        train_key=tk, evaluator_name="synthetic_recovery",
        evaluator_protocol_version="1.0.0", eval_cfg={"smoke": False},
    )
    assert a != b
