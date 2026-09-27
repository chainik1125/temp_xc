"""Full synthetic dataset suite (port of origin/han-phase7-unification:src/data/toy/).

Self-contained re-port of Aniket's complete toy-data pipeline. Lives
alongside ``temp_bench.data.toy`` (agent_paper's consolidated subset
which is what's currently deployed by agent_filler) and adds the
extensions Aniket originally designed but agent_paper did not port:

  - **Leaky reset transition kernel** (``transition.build_leaky_transition_matrix``)
    with a leak parameter ``delta`` that biases resamples toward the
    previous state. Standard reset (the only kernel agent_paper ported)
    is the ``delta = 0`` special case. Stationary distribution is
    independent of delta.
  - **Factorial HMM event blocks** (``factorial_hmm.create_block_membership``,
    ``create_overlapping_membership``) — features grouped into events
    with shared support; non-overlapping (block) and overlapping
    membership patterns supported.
  - **Sigmoid coupling** (``coupling.apply_coupling_sigmoid``) — soft
    parent-to-emission mapping; the only mode agent_paper ported is
    OR-gate (``apply_coupling_or``).
  - **Gaussian copula correlated features**
    (``feature_generation.get_correlated_features``) — generates binary
    masks with prescribed marginal firing probabilities AND a chosen
    pairwise correlation matrix.

Provenance: every file is a verbatim port from
``origin/han-phase7-unification @ <SHA>:src/data/toy/<name>.py`` plus
import-path rewrites from ``src.{utils,data.toy}`` to
``temp_bench.data.toy_full``. The ``_device``, ``_orthogonalize``,
``_logging``, ``_temporal_support`` shims port the wasteland's
``src/utils/*`` helpers under leading-underscore filenames so they
don't clash with anything in temp_bench.
"""

from temp_bench.data.toy_full.coupled_dataset import generate_coupled_dataset
from temp_bench.data.toy_full.dataset import generate_dataset
from temp_bench.data.toy_full.configs import (
    CoupledDataGenerationConfig,
    DataGenerationConfig,
    EmissionConfig,
    MagnitudeConfig,
    TransitionConfig,
)
from temp_bench.data.toy_full.transition import (
    build_leaky_transition_matrix,
    build_transition_matrix,
)

__all__ = [
    "generate_coupled_dataset",
    "generate_dataset",
    "CoupledDataGenerationConfig",
    "DataGenerationConfig",
    "EmissionConfig",
    "MagnitudeConfig",
    "TransitionConfig",
    "build_leaky_transition_matrix",
    "build_transition_matrix",
]
