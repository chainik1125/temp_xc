"""Pydantic models for ``temp_bench`` results, configs, and audit fields.

Schema version policy: bumping ``schema_version`` is a breaking change.
Old rows in ``leaderboard.jsonl`` that disagree are rejected at load
time; rebuild the leaderboard from runs/ if a migration is needed.

Five models:

- :class:`CodeVersion` — git commit + dirty flag + diff hash. Recorded
  on every row + every manifest entry. Audit-grade; never used for
  cache invalidation.
- :class:`TrainingConfig` — schedule + optimizer + arch-hparam
  overrides. Hashes into ``train_key``.
- :class:`LeaderboardRow` — one row per evaluated cell. Has
  ``code_version`` field.
- :class:`CheckpointManifest` — one entry per trained checkpoint
  written to disk + HF. Has ``code_version`` field.
- :class:`ArchSpec`, :class:`DataSourceSpec` — registry helpers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


SCHEMA_VERSION = "2.0.0"   # bump on breaking change to any of these models.


# ── Audit / code-version ───────────────────────────────────────────────────


class CodeVersion(BaseModel):
    """Snapshot of the codebase at run time.

    Stored verbatim on every leaderboard row + manifest entry so any
    past result can be reconstructed by:

        git checkout <commit_sha>
        [if dirty: apply diff matching diff_sha256]
    """

    model_config = ConfigDict(extra="forbid")

    commit_sha: str = Field(
        ...,
        min_length=40,
        max_length=40,
        description="Full 40-char hex SHA-1 of git HEAD at run time.",
    )
    dirty: bool = Field(
        ...,
        description="True if `git diff HEAD` was non-empty when the run started.",
    )
    diff_sha256: str | None = Field(
        default=None,
        description=(
            "sha256 hex of the `git diff HEAD` output when dirty=True. "
            "None if dirty=False. Lets us verify a recovered diff "
            "matches what the run actually saw."
        ),
    )


# ── Training config ────────────────────────────────────────────────────────


class TrainingConfig(BaseModel):
    """Knobs for one training run. Hashes into ``train_key``."""

    model_config = ConfigDict(extra="forbid")

    n_steps: int = 25_000
    batch_size: int = 4096
    learning_rate: float = 3e-4
    optimizer: Literal["adam"] = "adam"
    warmup_steps: int = 1000
    precision: Literal["fp32", "bf16"] = "bf16"

    # Buffer config: literature-standard token shuffle buffer.
    buffer_tokens: int = 2_000_000     # ~4 GB at fp16/d=2304
    refill_threshold: float = 0.5      # refill when buffer occupancy drops below

    # Optional opt-in plugins.
    bricken_enabled: bool = False
    bricken_resample_every: int = 500
    bricken_min_fires: int = 1
    bricken_n_check: int = 2048
    bricken_max_resample_fraction: float = 0.5
    ema_auxk_alpha: float = 0.03125
    dead_threshold_tokens: int = 10_000_000

    # Per-cell arch-hparam overrides (e.g., k_pos sweep, T sweep).
    # Flows into the cache key automatically.
    arch_hparams_override: dict[str, Any] | None = None


# ── Registry helpers (mirror configs/archs.yaml + data.yaml) ───────────────


class ArchSpec(BaseModel):
    """One arch's registry entry, loaded from ``configs/archs.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str                                    # registry key
    class_path: str                              # "temp_bench.archs.txc_base:TXCBase"
    arch_version: str                            # semver (ours) or "upstream-<date>" (adapter)
    upstream: str | None = None                  # "AI4LIFE-GROUP/temporal-saes@<commit>" for adapters
    hparams: dict[str, Any] = Field(default_factory=dict)
    per_section_hparams: dict[str, dict[str, Any]] | None = None  # § keys
    category: Literal["sae", "tsae", "tfa", "mlc", "txc", "sae_arditi", "stacked_sae"] = "sae"
    notes: str | None = None


class DataSourceSpec(BaseModel):
    """One datasource's registry entry, loaded from ``configs/data.yaml``."""

    model_config = ConfigDict(extra="forbid")

    name: str
    category: Literal["real_lm", "synthetic"]
    # Real-LM fields:
    subject_model: str | None = None
    layer: int | None = None
    layers: list[int] | None = None
    hookpoint: str | None = None
    dataset: str | None = None
    n_seqs: int | None = None
    seq_len: int | None = None
    tokenizer_revision: str | None = None
    # Synthetic fields (generator-specific knobs).
    generator: str | None = None                 # "temp_bench.data.synthetic:coupled_hmm"
    params: dict[str, Any] | None = None
    # Free-form notes.
    notes: str | None = None


# ── Result rows ────────────────────────────────────────────────────────────


class LeaderboardRow(BaseModel):
    """One evaluation cell. Appended to ``results/leaderboard.jsonl``.

    Identified by ``(train_key, eval_key)`` jointly. Same row can be
    reproduced bit-identically from ``code_version`` plus the train + eval
    cfgs.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    eval_key: str
    train_key: str
    data_key: str

    experiment: str                              # paper section ("synthetic" / "probing" / ...)
    arch: str                                    # arch registry key
    arch_version: str
    seed: int
    datasource: str                              # data registry key

    training_cfg: TrainingConfig
    eval_cfg: dict[str, Any]
    evaluator_name: str
    evaluator_protocol_version: str

    metrics: dict[str, float]                    # the actual numbers
    primary_metric: str

    code_version: CodeVersion                    # audit field

    agent: str | None = None                     # runner's invoker (CLI / env)
    ts: str                                      # ISO 8601 UTC


class CheckpointManifest(BaseModel):
    """One entry per trained checkpoint saved to disk + HF.

    Appended to ``checkpoints/manifest.jsonl``.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    train_key: str
    data_key: str
    arch: str
    arch_version: str
    seed: int
    datasource: str

    training_cfg: TrainingConfig
    local_path: str
    size_mb: float
    hf_url: str | None = None

    code_version: CodeVersion                    # audit field

    agent: str | None = None
    ts: str
