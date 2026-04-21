"""Nonergodic generator wrapping simplexity's build_nonergodic_process_from_spec.

Matches the kyle/nonergodic branch of astera-org/simplex-research. Registers the
local `mess3_reset` HMM matrix function (ported verbatim from
`training_unified/matrices.py` on that branch) so configs referencing it resolve.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import torch

try:
    from simplexity.generative_processes.builder import build_nonergodic_process_from_spec
    from simplexity.generative_processes.nonergodic_generative_process import (
        NonErgodicGenerativeProcess,
        NonErgodicState,
    )
    from simplexity.generative_processes.torch_generator import generate_data_batch
    from simplexity.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "simplexity with build_nonergodic_process_from_spec is required. "
        "Pin to git+https://github.com/Astera-org/simplexity.git@review/nonergodic-pr172."
    ) from exc


def _mess3_reset(x: float, a: float, r: float = 0.05) -> jax.Array:
    """mess3 with a 4th reset token that resets belief to uniform over 3 states.

    Ported from astera-org/simplex-research @ kyle/nonergodic:
    training_unified/matrices.py.
    """
    from simplexity.generative_processes.transition_matrices import mess3 as _mess3_fn

    T_mess3 = _mess3_fn(x, a)
    T_scaled = (1 - r) * T_mess3
    T_reset = jnp.full((1, 3, 3), r / 3.0)
    return jnp.concatenate([T_scaled, T_reset], axis=0)


def _mess3_identity(x: float, a: float, r: float = 0.05) -> jax.Array:
    """mess3 with a 4th marker token that leaves the hidden state unchanged.

    The extra token is emitted with probability ``r`` from every hidden state,
    and the corresponding transition matrix is ``r * I``. This means observing
    the marker does not alter the filtered within-component state belief after
    normalization.
    """
    from simplexity.generative_processes.transition_matrices import mess3 as _mess3_fn

    T_mess3 = _mess3_fn(x, a)
    T_scaled = (1 - r) * T_mess3
    T_identity = r * jnp.eye(3, dtype=T_mess3.dtype)[None, :, :]
    return jnp.concatenate([T_scaled, T_identity], axis=0)


HMM_MATRIX_FUNCTIONS.setdefault("mess3_reset", _mess3_reset)
HMM_MATRIX_FUNCTIONS.setdefault("mess3_identity", _mess3_identity)


@dataclass
class ComponentSpec:
    """A single component in a nonergodic mixture."""

    process_name: str
    process_params: Mapping[str, float] = field(default_factory=dict)
    component_type: str = "hmm"
    vocab_map: Sequence[int] | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "component_type": self.component_type,
            "process_name": self.process_name,
            "process_params": dict(self.process_params),
        }
        if self.vocab_map is not None:
            d["vocab_map"] = list(self.vocab_map)
        return d


@dataclass
class NonergodicDataset:
    tokens: torch.Tensor  # (n_sequences, seq_len) — aliased to `inputs`, matches standard_hmm
    inputs: torch.Tensor  # (n_sequences, seq_len)
    labels: torch.Tensor  # (n_sequences, seq_len)
    sequence_omegas: torch.Tensor | None = None  # (n_sequences, num_components) — one-hot
    component_indices: torch.Tensor | None = None  # (n_sequences,) int


class NonergodicGenerator:
    """Thin wrapper around simplexity's `build_nonergodic_process_from_spec`.

    Example — the 3x mess3_reset setup from
    `training_unified/configs/generative_process/nonergodic_3x_mess3_reset.yaml`:

        gen = NonergodicGenerator.mess3_reset(
            params=[(0.13, 0.61, 0.02), (0.18, 0.60, 0.02), (0.49, 0.60, 0.02)],
        )
        ds = gen.sample(n_sequences=8, seq_len=128, seed=0)
    """

    def __init__(
        self,
        components: Sequence[Mapping | ComponentSpec],
        component_weights: Sequence[float],
        vocab_maps: Sequence[Sequence[int]] | None = None,
        device: str | None = None,
    ) -> None:
        comp_dicts = [c.to_dict() if isinstance(c, ComponentSpec) else dict(c) for c in components]
        self.components_spec: list[dict] = comp_dicts
        self.component_weights: list[float] = list(component_weights)
        self.vocab_maps: list[list[int]] | None = (
            [list(vm) for vm in vocab_maps] if vocab_maps is not None else None
        )
        self.device = device

        self.process: NonErgodicGenerativeProcess = build_nonergodic_process_from_spec(
            components=comp_dicts,
            component_weights=self.component_weights,
            vocab_maps=self.vocab_maps,
            device=device,
        )

    @classmethod
    def mess3_shared(
        cls,
        params: Sequence[tuple[float, float]],
        component_weights: Sequence[float] | None = None,
        device: str | None = None,
    ) -> "NonergodicGenerator":
        """Shared-vocab mess3 nonergodic mixture (no reset token).

        Each component is a plain `mess3(x, a)` HMM emitting tokens in {0, 1, 2};
        all components share the same vocab_map [0, 1, 2]. Each sequence commits
        to one component; tokens don't identify which. vocab_size = 3.
        """
        n = len(params)
        if n == 0:
            raise ValueError("params must contain at least one (x, a) tuple")
        components = [
            {
                "component_type": "hmm",
                "process_name": "mess3",
                "process_params": {"x": float(x), "a": float(a)},
            }
            for (x, a) in params
        ]
        if component_weights is None:
            component_weights = [1.0 / n] * n
        vocab_maps = [[0, 1, 2] for _ in range(n)]
        return cls(
            components=components,
            component_weights=component_weights,
            vocab_maps=vocab_maps,
            device=device,
        )

    @classmethod
    def mess3_reset(
        cls,
        params: Sequence[tuple[float, float, float]],
        component_weights: Sequence[float] | None = None,
        vocab_maps: Sequence[Sequence[int]] | None = None,
        device: str | None = None,
    ) -> "NonergodicGenerator":
        """Convenience constructor for an N-component mess3_reset mixture.

        Each entry in `params` is `(x, a, r)`. Defaults:
        - uniform weights over N components
        - vocab_maps = [[0, 1, 2, 3 + i] for i in range(N)]   (shared mess3
          tokens 0/1/2 with a per-component reset token 3 + i, matching
          nonergodic_3x_mess3_reset.yaml)
        """
        n = len(params)
        if n == 0:
            raise ValueError("params must contain at least one (x, a, r) tuple")
        components = [
            {
                "component_type": "hmm",
                "process_name": "mess3_reset",
                "process_params": {"x": float(x), "a": float(a), "r": float(r)},
            }
            for (x, a, r) in params
        ]
        if component_weights is None:
            component_weights = [1.0 / n] * n
        if vocab_maps is None:
            vocab_maps = [[0, 1, 2, 3 + i] for i in range(n)]
        return cls(
            components=components,
            component_weights=component_weights,
            vocab_maps=vocab_maps,
            device=device,
        )

    @classmethod
    def mess3_identity(
        cls,
        params: Sequence[tuple[float, float, float]],
        component_weights: Sequence[float] | None = None,
        vocab_maps: Sequence[Sequence[int]] | None = None,
        device: str | None = None,
    ) -> "NonergodicGenerator":
        """Convenience constructor for an N-component mess3_identity mixture.

        Each entry in ``params`` is ``(x, a, r)``. Defaults:
        - uniform weights over N components
        - vocab_maps = [[0, 1, 2, 3 + i] for i in range(N)]   (shared mess3
          tokens 0/1/2 with a per-component marker token 3 + i)
        """
        n = len(params)
        if n == 0:
            raise ValueError("params must contain at least one (x, a, r) tuple")
        components = [
            {
                "component_type": "hmm",
                "process_name": "mess3_identity",
                "process_params": {"x": float(x), "a": float(a), "r": float(r)},
            }
            for (x, a, r) in params
        ]
        if component_weights is None:
            component_weights = [1.0 / n] * n
        if vocab_maps is None:
            vocab_maps = [[0, 1, 2, 3 + i] for i in range(n)]
        return cls(
            components=components,
            component_weights=component_weights,
            vocab_maps=vocab_maps,
            device=device,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.process.vocab_size)

    @property
    def num_components(self) -> int:
        return len(self.components_spec)

    def batched_initial_state(self, batch_size: int) -> NonErgodicState:
        """Broadcast the process's initial NonErgodicState along a leading batch dim."""
        init = self.process.initial_state
        return jax.tree.map(lambda x: jnp.broadcast_to(x, (batch_size,) + x.shape), init)

    def batched_onehot_initial_state(
        self, component_indices: jax.Array
    ) -> NonErgodicState:
        """Build a batched initial state with one-hot component_beliefs per sequence.

        Given `component_indices` of shape (n_sequences,), each sequence's
        NonErgodicState has component_beliefs = onehot(index). Because
        `emit_observation` samples via categorical(log component_beliefs), this
        forces each sequence to draw deterministically from a single component.
        """
        init = self.process.initial_state
        n = int(component_indices.shape[0])
        onehot = jax.nn.one_hot(component_indices, num_classes=len(self.components_spec))
        component_states = tuple(
            jax.tree.map(lambda x: jnp.broadcast_to(x, (n,) + x.shape), s)
            for s in init.component_states
        )
        return NonErgodicState(component_beliefs=onehot, component_states=component_states)

    def sample(
        self,
        *,
        n_sequences: int,
        seq_len: int,
        seed: int,
        with_component_labels: bool = False,
        device: torch.device | None = None,
    ) -> NonergodicDataset:
        """Sample `n_sequences` sequences of length `seq_len`.

        `inputs` and `labels` are shape (n_sequences, seq_len). `tokens` is
        (n_sequences, seq_len + 1) — the full generated stream.

        If `with_component_labels=True`, pre-samples a component index per
        sequence (using `component_weights`) and populates one-hot
        `sequence_omegas` + `component_indices` on the returned dataset. Tokens
        are still drawn from the committed component. Enables per-component
        probing.
        """
        key = jax.random.key(int(seed))
        n = int(n_sequences)

        if with_component_labels:
            key_idx, key_gen = jax.random.split(key)
            component_indices = jax.random.categorical(
                key_idx, jnp.log(self.process.component_weights), shape=(n,)
            )
            batched_init = self.batched_onehot_initial_state(component_indices)
            onehot_np = jax.nn.one_hot(component_indices, num_classes=self.num_components)
            comp_idx_t = torch.from_numpy(jax.device_get(component_indices)).long()
            seq_omegas_t = torch.from_numpy(jax.device_get(onehot_np)).float()
            gen_key = key_gen
        else:
            batched_init = self.batched_initial_state(n)
            comp_idx_t = None
            seq_omegas_t = None
            gen_key = key

        _, inputs, labels = generate_data_batch(
            batched_init,
            self.process,
            n,
            int(seq_len) + 1,
            gen_key,
            device=device,
        )
        inputs_t = inputs.detach().long().cpu()
        labels_t = labels.detach().long().cpu()
        return NonergodicDataset(
            tokens=inputs_t,
            inputs=inputs_t,
            labels=labels_t,
            sequence_omegas=seq_omegas_t,
            component_indices=comp_idx_t,
        )
