"""Noisy + overlap coupled-feature data generator for C2 hunt mission.

Ports the prior author's ``coupled_noisy_overlap`` setup from
``origin/case-synthetic @ 03a099b4:src/data_generation/{coupled_dataset,
support}.py`` into a single ``temp_bench.data.toy`` module. Attribution:
the per-token Bernoulli emission-noise model (``p_B``, ``p_A``) is from the maintainer Manning-Coe's ``EmissionConfig`` + ``apply_emission`` (commit
``03a099b4`` on ``origin/case-synthetic``).

Generator pipeline (extends the deterministic ``coupled_hmm``):

1. K independent Markov chains drive M > K emission features through
   a binary coupling matrix (n_parents per emission).
2. OR-gate coupling produces deterministic emission support
   ``s_clean ∈ {0,1}^{n × M × T}``.
3. **NEW **: per-token Bernoulli emission noise applied to
   ``s_clean``:

       s_noisy[m, t] = 1 with prob p_B   if s_clean[m, t] = 1
                     = 1 with prob p_A   if s_clean[m, t] = 0

   With ``p_A=0, p_B=1`` we recover ``coupled_hmm``. With ``p_B<1`` we
   model the case where a "should-fire" emission only fires with
   probability ``p_B`` (per-token unreliability). The prior author's Bench 2 uses
   ``p_B=0.5, n_parents=5``.

4. Magnitudes × s_noisy → activations; project to ``x = sum_j a_j F_j``.

The window-pooling architectures (TXC) average T noisy observations of
the same hidden state, denoising; per-token SAEs see one noisy
observation per encode and cannot disambiguate. This is the regime where
TXC's gAUC win is largest in the prior author's results (txc_base 0.97 vs
regular_sae 0.58 at raw_k=5 ρ=0.9, p_B=0.5, n_parents=5).
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


def coupled_noisy_hmm(
    *,
    K_hidden: int = 10,
    M_emissions: int = 20,
    n_parents: int = 5,
    d_in: int = 256,
    seq_len: int = 64,
    n_seqs: int = 4096,
    pi: float = 0.05,
    rho: float = 0.9,
    p_A: float = 0.0,
    p_B: float = 0.5,
    magnitude_dist: str = "folded_normal",
    magnitude_mean: float = 1.0,
    magnitude_std: float = 0.15,
    seed: int = 0,
    device: str | torch.device = "cpu",
) -> CoupledData:
    """Generate a noisy + overlap coupled-feature dataset.

    Same outputs as :func:`coupled_hmm` (same ``CoupledData`` namedtuple),
    so downstream eval (``feature_recovery``, ``global_recovery_gAUC``)
    is reused unchanged.

    Per-token emission noise:
      - ``p_B`` < 1.0 : when emission would fire (s_clean=1), it actually
        fires with prob p_B → fraction (1-p_B) "false negatives".
      - ``p_A`` > 0.0 : when emission would NOT fire (s_clean=0), it
        fires with prob p_A → "false positives".

    Default args reproduce the prior author's Bench 2 (``coupled_noisy_overlap``):
    ``n_parents=5, p_B=0.5, ρ=0.9``.
    """
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    device_t = torch.device(device)

    # 1. Orthogonal emission feature directions F ∈ R^{M × d}.
    emission_features = _orthogonalise(
        n_vectors=M_emissions, d_in=d_in, rng=rng,
    )

    # 2. Binary coupling C ∈ {0,1}^{M × K}: each emission has n_parents.
    coupling_matrix = _generate_coupling(
        K=K_hidden, M=M_emissions, n_parents=n_parents, rng=rng,
    )

    # 3. Hidden feature directions H ∈ R^{K × d}: normalised parent means.
    hidden_features = _compute_hidden_features(
        emission_features, coupling_matrix,
    )

    # 4. K independent Markov chains × n_seqs sequences.
    pi_t = torch.full((K_hidden,), float(pi))
    rho_t = torch.full((K_hidden,), float(rho))
    hidden_states = _markov_chain_batch(
        n_seqs=n_seqs, k=K_hidden, T=seq_len,
        pi=pi_t, rho=rho_t, rng=rng,
    )                                                # (n_seqs, K, T)

    # 5. OR-gate coupling: s_clean = 1[ sum_i C_ji * h_i >= 1 ].
    parent_sum = torch.einsum(
        "mk,nkt->nmt", coupling_matrix, hidden_states,
    )                                                # (n_seqs, M, T)
    emission_support_clean = (parent_sum >= 1).float()

    # 6. NEW: per-token Bernoulli emission noise (the prior author's apply_emission).
    if p_A == 0.0 and p_B == 1.0:
        emission_support = emission_support_clean
    else:
        u = torch.rand(n_seqs, M_emissions, seq_len, generator=rng)
        # Probability of emitting 1 at (n, m, t):
        #   s_clean=1 → p_B, s_clean=0 → p_A.
        prob = emission_support_clean * float(p_B) + \
               (1.0 - emission_support_clean) * float(p_A)
        emission_support = (u < prob).float()

    # 7. Magnitudes (folded normal by default).
    magnitudes = _sample_magnitudes(
        shape=(n_seqs, M_emissions, seq_len),
        dist=magnitude_dist, mean=magnitude_mean, std=magnitude_std,
        rng=rng,
    )

    # 8. Activations = support * magnitudes.
    activations = emission_support * magnitudes      # (n_seqs, M, T)

    # 9. Project: x_t = sum_j a_{j,t} F_j.
    x = torch.einsum("nmt,md->ntd", activations, emission_features)

    return CoupledData(
        x=x.to(device_t),
        emission_features=emission_features.to(device_t),
        hidden_features=hidden_features.to(device_t),
        coupling_matrix=coupling_matrix.to(device_t),
        hidden_states=hidden_states.to(device_t),
        emission_support=emission_support.to(device_t),
    )
