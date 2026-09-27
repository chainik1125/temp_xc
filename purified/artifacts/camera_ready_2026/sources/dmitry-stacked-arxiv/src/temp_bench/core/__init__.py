"""Locked core of ``temp_bench``: runner, cache, schemas, config, trainer.

These modules are touched RARELY (framework-level changes only).
Adding archs, evals, datasources, or experiments NEVER requires
editing files in this package.

Public re-exports for convenience:

- :class:`temp_bench.core.schemas.LeaderboardRow` / ``CheckpointManifest`` / ``TrainingConfig`` / ``CodeVersion``
- :func:`temp_bench.core.runner.run_experiment` / ``run_sweep``
- :func:`temp_bench.core.config.compute_train_key` / ``compute_eval_key``
"""

from temp_bench.core.schemas import (
    CheckpointManifest,
    CodeVersion,
    LeaderboardRow,
    TrainingConfig,
)

__all__ = [
    "CheckpointManifest",
    "CodeVersion",
    "LeaderboardRow",
    "TrainingConfig",
]
