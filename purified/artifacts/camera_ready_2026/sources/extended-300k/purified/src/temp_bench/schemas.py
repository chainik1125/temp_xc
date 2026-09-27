"""Pydantic schemas for cache and leaderboard contracts.

These define the wire format for ``results/leaderboard.jsonl`` and
``checkpoints/manifest.jsonl``. Append-time validation rejects malformed
rows so corrupt state never lands in the multi-agent shared file.

A row's ``schema_version`` field enables future migrations: a reader
checks the version and applies any compat shims before trusting fields.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "1.0.0"


class ArchSpec(BaseModel):
    """Shape of one entry in ``configs/locked_archs.yaml``."""
    model_config = ConfigDict(extra="forbid")

    class_path: str               # "temp_bench.architectures.txc_base:TXCBase"
    arch_version: str             # semver — bump on bug fix or breaking change
    category: str
    hparams: dict[str, Any] = Field(default_factory=dict)
    per_component_hparams: dict[str, dict[str, Any]] = Field(default_factory=dict)
    notes: str = ""


class DataSourceSpec(BaseModel):
    """Shape of one entry in ``configs/datasources.yaml``.

    Real-LM and synthetic datasources share this schema; ``category``
    discriminates. Required fields per category are validated at load time.
    """
    model_config = ConfigDict(extra="allow")  # generator-specific keys

    category: str                 # "real_lm" or "synthetic"
    notes: str = ""


class TrainingConfig(BaseModel):
    """The training-time config that's part of ``train_key``.

    Component scripts override these with per-component values
    (e.g. C6 sets ``bricken_enabled=True``).
    """
    model_config = ConfigDict(extra="forbid")

    n_steps: int = 25_000
    batch_size: int = 1_024
    # Was batch=256, n_steps=30_000 before 2026-05-04. Re-issued 2026-05-04
    # per Phase 5's empirically validated batch (summary.md:250) plus the
    # SAE-literature-standard fixed-step schedule (T-SAE §4.1, TFA App. B.1,
    # GemmaScope) — the n_steps cap is binding for every arch and pod.
    # See decisions.md § 12.
    learning_rate: float = 3e-4
    optimizer: str = "adam"
    warmup_steps: int = 1_000
    # Plateau-stop is OFF by default (2026-05-04). The implementation in
    # training/sae_trainer.py:158-165 uses an absolute max-min threshold
    # over a 5K window, which has cross-arch unfairness: archs with
    # naturally smaller loss scales would trigger prematurely while archs
    # with larger losses never trigger. SAE-comparison papers (T-SAE, TFA,
    # GemmaScope) use fixed step counts — no plateau detection. Code path
    # stays for opt-in use; default is off so production cells run the
    # full n_steps.
    plateau_early_stop: bool = False
    plateau_patience: int = 5_000
    plateau_min_delta: float = 1e-4

    # Mixed precision (hardware accommodation, not architectural)
    precision: str = "bf16"       # "bf16" on H100/H200, "fp16" on A40

    # Bricken resample (opt-in per component; see docs/components/c6.md)
    bricken_enabled: bool = False
    bricken_resample_every: int = 500
    bricken_min_fires: int = 1
    bricken_n_check: int = 2048
    bricken_max_resample_fraction: float = 0.5

    # EMA-AuxK alpha (defaults to tsae_paper 1/32; brickenauxk_a8 uses 1/8)
    ema_auxk_alpha: float = 0.03125
    dead_threshold_tokens: int = 10_000_000


class LeaderboardRow(BaseModel):
    """One row of ``results/leaderboard.jsonl``.

    ``eval_key`` is the primary index. Two rows with the same
    ``eval_key`` are forbidden — that means we re-ran a cached eval and
    something is wrong with the cache.
    """
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    eval_key: str = Field(min_length=16, max_length=16)
    train_key: str = Field(min_length=16, max_length=16)
    act_cache_key: str = Field(min_length=16, max_length=16)

    component: str               # "c1", "c2", ..., "c7"
    arch: str                    # name from configs/locked_archs.yaml
    arch_version: str
    seed: int
    datasource: str              # name from configs/datasources.yaml

    eval_protocol_version: str
    eval_cfg: dict[str, Any]     # k_feat, S, etc.
    metrics: dict[str, float]
    primary_metric: str          # which key in metrics is "the headline"

    agent: str
    ts: str                      # ISO-8601 UTC


class CheckpointManifest(BaseModel):
    """One row of ``checkpoints/manifest.jsonl``."""
    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    train_key: str = Field(min_length=16, max_length=16)
    act_cache_key: str = Field(min_length=16, max_length=16)
    arch: str
    arch_version: str
    seed: int
    datasource: str
    training_cfg: dict[str, Any]

    hf_url: str | None = None
    local_path: str | None = None
    size_mb: float | None = None

    agent: str
    ts: str
