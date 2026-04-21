from __future__ import annotations

from dataclasses import dataclass
import re

import jax
import jax.numpy as jnp
import numpy as np
import torch
import torch.nn.functional as F

from sae_day.data import HMMComponent

try:
    from simplexity.generative_processes.builder import build_hidden_markov_model
    from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
    from simplexity.generative_processes.torch_generator import generate_data_batch
except ImportError as exc:  # pragma: no cover - exercised in runtime environments
    raise ImportError(
        "simplexity is required for simplexity_standard_hmm. "
        "Install the simplexity dependency before using this module."
    ) from exc


@dataclass
class SimplexityStandardHMMDataset:
    tokens: torch.Tensor
    observations: torch.Tensor
    sequence_omegas: torch.Tensor
    mode: str
    vocab_mode: str


def _component_to_simplexity_hmm(component: HMMComponent) -> HiddenMarkovModel:
    match = re.search(r"x=([0-9.]+),a=([0-9.]+)", component.label)
    if match is None:
        raise ValueError(
            "Could not extract mess3 parameters from component label. "
            f"Expected label with x=...,a=..., got {component.label!r}"
        )
    x = float(match.group(1))
    a = float(match.group(2))
    return build_hidden_markov_model("mess3", {"x": x, "a": a})


def _sample_dirichlet_rows(alpha: np.ndarray, n_sequences: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    samples = rng.gamma(shape=alpha.astype(np.float64), scale=1.0, size=(n_sequences, alpha.size))
    samples /= samples.sum(axis=1, keepdims=True)
    return samples.astype(np.float32)


def build_global_hmm(components: list[HMMComponent], vocab_mode: str) -> HiddenMarkovModel:
    hmms = [_component_to_simplexity_hmm(comp) for comp in components]
    dims = [int(hmm.initial_state.shape[0]) for hmm in hmms]
    total_d = sum(dims)

    if vocab_mode == "shared":
        vocab_size = int(hmms[0].transition_matrices.shape[0])
        matrices = np.zeros((vocab_size, total_d, total_d), dtype=np.float32)
        state_offset = 0
        for hmm, d in zip(hmms, dims):
            matrices[:, state_offset : state_offset + d, state_offset : state_offset + d] = (
                np.asarray(hmm.transition_matrices, dtype=np.float32)
            )
            state_offset += d
    elif vocab_mode == "distinct":
        vocab_size = sum(int(hmm.transition_matrices.shape[0]) for hmm in hmms)
        matrices = np.zeros((vocab_size, total_d, total_d), dtype=np.float32)
        state_offset = 0
        token_offset = 0
        for hmm, d in zip(hmms, dims):
            v = int(hmm.transition_matrices.shape[0])
            matrices[token_offset : token_offset + v, state_offset : state_offset + d, state_offset : state_offset + d] = (
                np.asarray(hmm.transition_matrices, dtype=np.float32)
            )
            state_offset += d
            token_offset += v
    else:
        raise ValueError(f"Unknown vocab_mode: {vocab_mode}")

    initial_state = np.ones(total_d, dtype=np.float32) / float(total_d)
    return HiddenMarkovModel(
        transition_matrices=jnp.asarray(matrices),
        initial_state=jnp.asarray(initial_state),
    )


def build_batch_initial_state(
    *,
    components: list[HMMComponent],
    omega_prior: list[float],
    n_sequences: int,
    mode: str,
    seed: int,
    concentration: float,
) -> tuple[np.ndarray, np.ndarray]:
    dims = [comp.d for comp in components]
    total_d = sum(dims)
    prior = np.asarray(omega_prior, dtype=np.float32)

    if mode == "mixed_initial_belief":
        sequence_omegas = _sample_dirichlet_rows(prior * concentration, n_sequences=n_sequences, seed=seed)
    elif mode == "single_component":
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(components), size=n_sequences, p=prior / prior.sum())
        sequence_omegas = np.eye(len(components), dtype=np.float32)[chosen]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    batch_init = np.zeros((n_sequences, total_d), dtype=np.float32)
    state_offset = 0
    for block_idx, d in enumerate(dims):
        batch_init[:, state_offset : state_offset + d] = sequence_omegas[:, block_idx : block_idx + 1] / float(d)
        state_offset += d
    return batch_init, sequence_omegas


def generate_standard_hmm_data_simplexity(
    *,
    components: list[HMMComponent],
    omega_prior: list[float],
    n_sequences: int,
    seq_len: int,
    mode: str,
    vocab_mode: str,
    seed: int,
    concentration: float,
    device: torch.device | None = None,
) -> SimplexityStandardHMMDataset:
    hmm = build_global_hmm(components, vocab_mode)
    batch_init, sequence_omegas = build_batch_initial_state(
        components=components,
        omega_prior=omega_prior,
        n_sequences=n_sequences,
        mode=mode,
        seed=seed,
        concentration=concentration,
    )

    key = jax.random.key(int(seed))
    _, inputs, _labels = generate_data_batch(
        jnp.asarray(batch_init),
        hmm,
        int(n_sequences),
        int(seq_len) + 1,
        key,
        device=device,
    )

    tokens = inputs.detach().cpu().long()
    observations = F.one_hot(tokens, num_classes=int(hmm.transition_matrices.shape[0])).float()
    return SimplexityStandardHMMDataset(
        tokens=tokens,
        observations=observations,
        sequence_omegas=torch.from_numpy(sequence_omegas).float(),
        mode=mode,
        vocab_mode=vocab_mode,
    )
