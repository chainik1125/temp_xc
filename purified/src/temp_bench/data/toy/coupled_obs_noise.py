"""Setup F generator — coupled HMM with additive Gaussian observation noise.

Same K hidden chains × M emissions × OR-coupling as Setup A, but the
observation x(t) is corrupted by per-token Gaussian noise with std σ
(``obs_noise_sigma``). Tests the Setup-B-style denoising story
(per-token SAEs see noise; window-pooling TXC averages it out) on a
COUPLED bench (Setup A-style hidden chains).

Reuses the orthogonalisation / coupling / Markov-chain primitives from
``coupled.py``. Only the final projection step adds noise.
"""

from __future__ import annotations

import torch

from temp_bench.data.toy.coupled import (
    CoupledData,
    _compute_hidden_features,
    _generate_coupling,
    _markov_chain_batch,
    _orthogonalise,
    _sample_magnitudes,
)


def coupled_obs_noise_hmm(
    *,
    K_hidden: int = 10,
    M_emissions: int = 20,
    n_parents: int = 2,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi: float = 0.05,
    rho: float = 0.7,
    obs_noise_sigma: float = 1.0,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate a coupled HMM dataset + additive Gaussian observation noise.

    σ = ``obs_noise_sigma`` controls the per-token noise scale. σ = 0
    recovers ``coupled_hmm`` (Setup A) exactly. With ``magnitude_mean=1``,
    the typical signal magnitude per non-zero element is ~1, so σ in
    ``{0, 0.5, 1, 2}`` corresponds to SNR ~ {∞, 2:1, 1:1, 1:2} per
    activation.

    Returns the standard ``CoupledData`` namedtuple — eval pipelines
    are unchanged.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    # 1. Orthogonal emission directions.
    emission_features = _orthogonalise(
        n_vectors=M_emissions, d_in=d_in, rng=rng,
    )

    # 2. Coupling matrix.
    coupling_matrix = _generate_coupling(
        K=K_hidden, M=M_emissions, n_parents=n_parents, rng=rng,
    )

    # 3. Hidden directions = normalised parent means.
    hidden_features = _compute_hidden_features(
        emission_features, coupling_matrix,
    )

    # 4. K independent slow Markov chains.
    pi_t = torch.full((K_hidden,), float(pi))
    rho_t = torch.full((K_hidden,), float(rho))
    hidden_states = _markov_chain_batch(
        n_seqs=n_seqs, k=K_hidden, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )                                                # (n_seqs, K, T)

    # 5. OR-gate coupling: emission fires if any parent is on.
    parent_sum = torch.einsum(
        "mk,nkt->nmt", coupling_matrix, hidden_states,
    )
    emission_support = (parent_sum >= 1).float()     # (n_seqs, M, T)

    # 6. Magnitudes.
    magnitudes = _sample_magnitudes(
        shape=(n_seqs, M_emissions, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )

    # 7. Activations.
    activations = emission_support * magnitudes      # (n_seqs, M, T)

    # 8. Project to observation space + add Gaussian observation noise.
    x = torch.einsum("nmt,md->ntd", activations, emission_features)
    if obs_noise_sigma > 0:
        noise = torch.randn(*x.shape, generator=rng) * float(obs_noise_sigma)
        x = x + noise

    return CoupledData(
        x=x.to(device_t),
        emission_features=emission_features.to(device_t),
        hidden_features=hidden_features.to(device_t),
        coupling_matrix=coupling_matrix.to(device_t),
        hidden_states=hidden_states.to(device_t),
        emission_support=emission_support.to(device_t),
    )
