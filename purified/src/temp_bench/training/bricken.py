"""Bricken-style dead-feature resample — **opt-in per experiment**.

This is NOT part of the locked TXC-base / TXC-pro architecture spec.
Components opt in by passing :class:`BrickenConfig` to the trainer; the
default is OFF. See `docs/paper/architecture.md` § *Per-experiment
training knobs* and the per-component A/B verdict in
`docs/components/cN.md` before turning it on.

Why opt-in: Dmitry's winning recipe `brickenauxk_a8` co-tunes six knobs
together (resample_every, min_fires, n_check, max_resample_fraction,
EMA-AuxK alpha, dead_threshold). The recipe is coherent on Qwen-7B
medical activations; whether it transfers to Gemma activations, the
matryoshka × multi-distance InfoNCE objective in TXC-pro, or low-d
toy data is an empirical question, not a default to assume.

Trainer-level augmentation that periodically hard-resets features which
have not fired on a held-out check batch. Complementary to the AuxK
loss in the anti-dead stack: AuxK gives dead features a gradient signal
(so they learn from the residual), Bricken hard-resets the truly stuck
ones.

Ported from `experiments/em_features/dead_feature_resample.py` on
``origin/em-nanda`` (commit `95627e54` and earlier). The interface is
unchanged so the EM Wang procedure can re-use it directly; the
implementation here is a clean rewrite that takes the locked TXC-base
and TXC-pro architectures via the standard
:class:`temp_bench.architectures.base.TempBenchArch` interface.

Recipe (default values from Dmitry's brickenauxk_a8 winning config):

- ``resample_every = 500`` steps
- ``min_fires = 1`` on a held-out check batch (a feature is "dead" if it
  fired fewer than ``min_fires`` times across the check)
- ``n_check = 2048`` windows per check
- ``max_resample_fraction = 0.5`` cap on per-call resets

Dead-feature trajectory on Qwen-7B medical (Dmitry's
``summary_brickenauxk_a8_frontier.md``):

    step    dead    loss
     500    58.5%   2581
    5000    51.1%   3722    ← minimum
   10000    67.7%   2566
   21500    72.6%   2212
   40000    75.7%   2643

Dead-fraction recovers to ~51% around step 5k then regresses to ~75% by
step 40k as Bricken can't keep up with newly-collapsing features. Loss
plateaus around step 21k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch


@dataclass
class BrickenConfig:
    """Trainer-level config for Bricken resample. Defaults match the
    winning Qwen-7B medical recipe (`brickenauxk_a8`)."""
    resample_every: int = 500
    min_fires: int = 1
    n_check: int = 2048
    max_resample_fraction: float = 0.5
    seed: int = 0


@dataclass
class ResampleStats:
    step: int
    n_dead: int
    n_features: int
    n_resampled: int
    fire_hist_quantiles: dict[float, int]


class BrickenResampler:
    """Stateful Bricken resampler.

    Skeleton implementation. Worker agents fill in the per-arch reset
    logic (kaiming_uniform encoder columns, tied decoder rows, zero
    bias, decoder re-unit-normalisation) once the TXC-base / TXC-pro
    classes are ported in :mod:`temp_bench.architectures`.
    """

    def __init__(self, arch, cfg: BrickenConfig | None = None):
        self.arch = arch
        self.cfg = cfg or BrickenConfig()
        self.last_stats: Optional[ResampleStats] = None
        self._rng = torch.Generator().manual_seed(self.cfg.seed)

    def maybe_resample(
        self,
        step: int,
        check_batch_fn: Callable[[], torch.Tensor],
    ) -> bool:
        """If ``step % resample_every == 0`` and step > 0, do a resample.

        Returns True if a resample fired this call (the agent should log
        ``self.last_stats``).
        """
        if step == 0 or step % self.cfg.resample_every != 0:
            return False
        # TODO(agent_em): port the per-arch reset logic from
        # origin/em-nanda:experiments/em_features/dead_feature_resample.py.
        # For now this is a no-op so the trainer can wire the call and
        # the test suite passes.
        raise NotImplementedError(
            "BrickenResampler.maybe_resample needs porting from "
            "origin/em-nanda:experiments/em_features/dead_feature_resample.py"
        )
