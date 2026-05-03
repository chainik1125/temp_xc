"""Cache-key determinism tests.

These tests are load-bearing for the framework: if cache keys aren't
deterministic, the whole "skip if cached" mechanism breaks and we
re-train every cell.
"""

from __future__ import annotations

import pytest

from temp_bench.config import (
    KEY_LEN,
    compute_act_cache_key,
    compute_eval_key,
    compute_train_key,
    list_archs,
    list_datasources,
    load_arch,
    load_datasource,
)
from temp_bench.schemas import TrainingConfig


# ── Determinism ──────────────────────────────────────────────────────────


def test_act_cache_key_is_deterministic():
    ds = load_datasource("gemma_2_2b_it_l13_fineweb_24k128")
    k1 = compute_act_cache_key(ds)
    k2 = compute_act_cache_key(ds)
    assert k1 == k2
    assert len(k1) == KEY_LEN


def test_act_cache_key_changes_when_layer_changes():
    a = compute_act_cache_key("gemma_2_2b_it_l13_fineweb_24k128")
    b = compute_act_cache_key("gemma_2_2b_base_l12_fineweb_24k128")
    assert a != b, "Different (model, layer) must produce different act_cache_key"


def test_train_key_is_deterministic():
    arch = load_arch("txc_base")
    cfg = TrainingConfig()
    k1 = compute_train_key(arch=arch, seed=42, training_cfg=cfg, act_cache_key="abc123")
    k2 = compute_train_key(arch=arch, seed=42, training_cfg=cfg, act_cache_key="abc123")
    assert k1 == k2
    assert len(k1) == KEY_LEN


def test_train_key_changes_with_seed():
    arch = load_arch("txc_base")
    cfg = TrainingConfig()
    k1 = compute_train_key(arch=arch, seed=1, training_cfg=cfg, act_cache_key="abc")
    k2 = compute_train_key(arch=arch, seed=2, training_cfg=cfg, act_cache_key="abc")
    assert k1 != k2


def test_train_key_changes_with_arch_version():
    arch = load_arch("txc_base")
    arch_bumped = arch.model_copy(update={"arch_version": "1.0.1"})
    cfg = TrainingConfig()
    k1 = compute_train_key(arch=arch, seed=42, training_cfg=cfg, act_cache_key="abc")
    k2 = compute_train_key(arch=arch_bumped, seed=42, training_cfg=cfg, act_cache_key="abc")
    assert k1 != k2, "Bumping arch_version must invalidate the train cache"


def test_train_key_changes_with_bricken_toggle():
    arch = load_arch("txc_base")
    cfg_off = TrainingConfig(bricken_enabled=False)
    cfg_on = TrainingConfig(bricken_enabled=True)
    k_off = compute_train_key(arch=arch, seed=42, training_cfg=cfg_off, act_cache_key="abc")
    k_on = compute_train_key(arch=arch, seed=42, training_cfg=cfg_on, act_cache_key="abc")
    assert k_off != k_on, (
        "Different training_cfg (Bricken on vs off) must produce different "
        "train_key — both are kept as separate cache entries."
    )


def test_eval_key_changes_with_protocol_version():
    k1 = compute_eval_key(train_key="t1", eval_protocol_version="1.0.0", eval_cfg={"k": 5})
    k2 = compute_eval_key(train_key="t1", eval_protocol_version="1.0.1", eval_cfg={"k": 5})
    assert k1 != k2, (
        "Bumping eval_protocol_version must invalidate eval cache while "
        "preserving train cache (train_key unchanged)"
    )


def test_eval_key_changes_with_eval_cfg():
    k1 = compute_eval_key(train_key="t1", eval_protocol_version="1.0.0", eval_cfg={"k_feat": 5})
    k2 = compute_eval_key(train_key="t1", eval_protocol_version="1.0.0", eval_cfg={"k_feat": 20})
    assert k1 != k2


# ── Cross-machine reproducibility ─────────────────────────────────────────


def test_train_key_is_dict_order_invariant():
    """Canonical-JSON sorting means key order in dicts must not affect the hash."""
    from temp_bench.config import _hash
    a = _hash({"x": 1, "y": 2})
    b = _hash({"y": 2, "x": 1})
    assert a == b


# ── Registry sanity ─────────────────────────────────────────────────────


def test_locked_arch_registry_has_expected_names():
    names = set(list_archs())
    expected = {"topk_sae", "tsae_paper", "tfa", "tfa_pos", "mlc", "sae_arditi", "txc_base", "txc_pro"}
    missing = expected - names
    assert not missing, f"Missing locked archs: {missing}"


def test_datasource_registry_has_expected_names():
    names = set(list_datasources())
    expected = {
        "gemma_2_2b_it_l13_fineweb_24k128",
        "gemma_2_2b_base_l13_fineweb_24k128",
        "gemma_2_2b_base_l12_fineweb_24k128",
        "qwen_2_5_14b_instruct_finance_l24_resid_post",
        "gemma_2_2b_base_l10_backtracking",
        "toy_markov_n20_d40",
        "toy_coupled_K10_M20_d256",
    }
    missing = expected - names
    assert not missing, f"Missing locked datasources: {missing}"


def test_unknown_arch_raises_clear_error():
    with pytest.raises(KeyError, match="Locked set"):
        load_arch("totally_made_up_arch")


def test_unknown_datasource_raises_clear_error():
    with pytest.raises(KeyError, match="Available"):
        load_datasource("not_a_real_datasource")
