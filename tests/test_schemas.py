"""Schema validation tests for leaderboard / checkpoint manifest rows."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from temp_bench.schemas import (
    SCHEMA_VERSION,
    CheckpointManifest,
    LeaderboardRow,
    TrainingConfig,
)


def _valid_row_kwargs() -> dict:
    return dict(
        eval_key="0123456789abcdef",
        train_key="fedcba9876543210",
        act_cache_key="aaaaaaaaaaaaaaaa",
        component="c3",
        arch="txc_base",
        arch_version="1.0.0",
        seed=42,
        datasource="gemma_2_2b_it_l13_fineweb_24k128",
        eval_protocol_version="1.0.0",
        eval_cfg={"k_feat": 20, "S": 32},
        metrics={"probing_auc": 0.9127, "alive_frac": 0.42},
        primary_metric="probing_auc",
        agent="[pipeline]",
        ts="2026-05-03T12:00:00Z",
    )


def test_valid_row_round_trips():
    row = LeaderboardRow(**_valid_row_kwargs())
    assert row.schema_version == SCHEMA_VERSION
    payload = row.model_dump_json()
    rebuilt = LeaderboardRow.model_validate_json(payload)
    assert rebuilt == row


def test_short_eval_key_is_rejected():
    kw = _valid_row_kwargs()
    kw["eval_key"] = "tooshort"
    with pytest.raises(ValidationError, match="String should have at least 16 characters"):
        LeaderboardRow(**kw)


def test_extra_field_is_rejected():
    """Schema is strict: a typo'd key never silently lands in the file."""
    kw = _valid_row_kwargs()
    kw["maybe_a_typo"] = "oops"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        LeaderboardRow(**kw)


def test_missing_required_field_is_rejected():
    kw = _valid_row_kwargs()
    del kw["primary_metric"]
    with pytest.raises(ValidationError):
        LeaderboardRow(**kw)


def test_training_config_defaults_are_anti_dead_only():
    """Locked architecture defaults: no Bricken."""
    cfg = TrainingConfig()
    assert cfg.bricken_enabled is False
    assert cfg.ema_auxk_alpha == 0.03125  # tsae_paper default 1/32


def test_training_config_brickenauxk_a8_recipe():
    """the prior author's brickenauxk_a8 recipe should serialise to a different
    key than the default recipe — verifying the Bricken knobs are
    actually included in the config dict."""
    cfg_default = TrainingConfig()
    cfg_brickenauxk_a8 = TrainingConfig(
        bricken_enabled=True,
        bricken_resample_every=500,
        bricken_min_fires=1,
        bricken_n_check=2048,
        bricken_max_resample_fraction=0.5,
        ema_auxk_alpha=0.125,            # 1/8
        dead_threshold_tokens=128_000,   # 128k
    )
    assert cfg_default.model_dump() != cfg_brickenauxk_a8.model_dump()


def test_checkpoint_manifest_minimal():
    cm = CheckpointManifest(
        train_key="fedcba9876543210",
        act_cache_key="aaaaaaaaaaaaaaaa",
        arch="txc_base",
        arch_version="1.0.0",
        seed=42,
        datasource="toy_markov_n20_d40",
        training_cfg={"n_steps": 30000},
        local_path="/tmp/some.pt",
        size_mb=12.3,
        agent="[pipeline]",
        ts="2026-05-03T12:00:00Z",
    )
    assert cm.schema_version == SCHEMA_VERSION
    assert cm.hf_url is None  # local-only is allowed
