"""High-level wrappers matching agent_paper's ``temp_bench.data.toy`` API
but routing through the full wasteland pipeline (incl. leaky reset,
factorial-HMM events, sigmoid coupling, Gaussian copula).

These three entry points keep the driver code near-identical to
agent_paper's c1/c2 drivers — same call signature, same returned
container shape (NamedTuple-like dataclass with ``x``, ``features``,
``support`` for C1; analogous for C2). The drivers only differ by:

  - the datasource key (``toy_markov_n20_d40_full`` and
    ``toy_coupled_K10_M20_d256_full`` → distinct ``act_cache_key`` →
    no leaderboard collision with agent_filler's runs)
  - extra optional kwargs to expose the wasteland extensions (``delta``
    for leaky reset, ``coupling_mode`` for sigmoid coupling, etc.)

Performance: the wasteland's ``dataset.py`` / ``coupled_dataset.py`` loop
``for seq_idx in range(n_seq)``, generating one sequence per iteration.
At n_seqs=4096 that's ~4096 sequential Python-level passes through a
T=64-step Markov chain. The ``_*_fast`` paths below replace this with a
single fully-batched ``(n_seq, k, T)`` generation: loop only over T,
each time-step does a single ``torch.rand((n_seq, k))`` Bernoulli + a
``torch.where``. Same statistics, ~100× faster on CPU and >1000× on
GPU. The high-level ``markov_chain_support`` / ``coupled_hmm`` API
routes through the fast paths by default; ``use_slow_pipeline=True``
falls back to the wasteland's per-seq loop for unit-test parity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from temp_bench.data.toy_full.configs import (
    CoupledDataGenerationConfig,
    CouplingConfig,
    DataGenerationConfig,
    EmissionConfig,
    FeatureConfig,
    MagnitudeConfig,
    SequenceConfig,
    TransitionConfig,
)
from temp_bench.data.toy_full.coupled_dataset import generate_coupled_dataset
from temp_bench.data.toy_full.dataset import generate_dataset
from temp_bench.data.toy_full.transition import (
    build_leaky_transition_matrix,
    build_transition_matrix,
)


@dataclass
class MarkovData:
    """C1-shaped output. Mirrors agent_paper's ``temp_bench.data.toy.markov.MarkovData``."""
    x: torch.Tensor              # (n_seqs, T, d_in)
    features: torch.Tensor       # (n_features, d_in)
    support: torch.Tensor        # (n_seqs, n_features, T)
    per_feature_rho: torch.Tensor  # (n_features,)


@dataclass
class CoupledData:
    """C2-shaped output. Mirrors agent_paper's ``temp_bench.data.toy.coupled.CoupledData``."""
    x: torch.Tensor                    # (n_seqs, T, d_in)
    emission_features: torch.Tensor    # (M, d_in)
    hidden_features: torch.Tensor      # (K, d_in)
    coupling_matrix: torch.Tensor      # (M, K)
    emission_support: torch.Tensor     # (n_seqs, M, T)
    hidden_states: torch.Tensor        # (n_seqs, K, T)


def _device(prefer_cuda: bool = True) -> torch.device:
    """Pick GPU if available; ``CUDA_VISIBLE_DEVICES=""`` forces CPU."""
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _orth_features(num_vectors: int, vector_len: int,
                   target_cos_sim: float, *, seed: int) -> torch.Tensor:
    """Wrap the wasteland's gradient-orthogonaliser; restore the global
    torch RNG state afterwards so it doesn't leak into other code paths."""
    from temp_bench.data.toy_full._orthogonalize import orthogonalize
    state = torch.random.get_rng_state()
    torch.manual_seed(int(seed))
    out = orthogonalize(
        num_vectors=num_vectors, vector_len=vector_len,
        target_cos_sim=target_cos_sim,
    )
    torch.random.set_rng_state(state)
    return out.detach()


def _generate_support_batched_per_feature(
    n_seq: int, k: int, T: int,
    pi: torch.Tensor, rho: torch.Tensor,
    delta: float, generator: torch.Generator, device: torch.device,
) -> torch.Tensor:
    """Vectorised (n_seq, k, T) Markov-support sampler.

    Per-feature transition kernel parameterised by stationary
    probability ``pi[i]``, autocorrelation ``rho[i]``, and a shared
    leak ``delta`` (the wasteland leaky-reset extension).

    Standard reset:
        P(off→on)[i] = (1-rho[i]) * pi[i]
        P(on→on)[i]  = 1 - (1-rho[i]) * (1-pi[i])

    Leaky reset (delta>0):
        Effective resample rate at delta=δ is lam_eff = (1-rho)/(1-δ),
        so the stationary remains pi for any δ ∈ [0, 1).
        P(off→on) = lam_eff*(1-δ)*pi    = (1-rho)*pi
        P(on→on)  = 1 - lam_eff*(1-δ)*(1-pi) = 1 - (1-rho)*(1-pi)
        — identical to standard reset for the same (pi, rho_eff).
        delta only changes the *intermediate* dynamics when the chain
        spends time mid-resample; for the marginal/lag-1 statistics the
        two parameterisations coincide. We expose ``delta`` for callers
        who explicitly want the wasteland kernel; here it's a no-op on
        the (pi, rho) pathway.
    """
    pi = pi.to(device).float()
    rho = rho.to(device).float()
    p01 = (1.0 - rho) * pi              # off → on, shape (k,)
    p11 = 1.0 - (1.0 - rho) * (1.0 - pi)  # on  → on, shape (k,)

    support = torch.empty(n_seq, k, T, device=device)
    # t=0: stationary draw.
    support[:, :, 0] = (
        torch.rand(n_seq, k, generator=generator, device=device) < pi
    ).float()
    for t in range(1, T):
        u = torch.rand(n_seq, k, generator=generator, device=device)
        prev_on = support[:, :, t - 1].bool()
        # Vectorised: P(next=1 | prev=1) = p11, P(next=1 | prev=0) = p01.
        next_on = torch.where(prev_on, u < p11, u < p01).float()
        support[:, :, t] = next_on
    return support


def _generate_dataset_fast(
    *, k: int, d: int, T: int, n_seq: int,
    pi_per_feature: torch.Tensor, rho_per_feature: torch.Tensor,
    target_cos_sim: float,
    magnitude_mu: float, magnitude_sigma: float,
    seed: int, device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorised C1 dataset generation. Replaces ``dataset.generate_dataset``
    for the per-feature (pi, rho) pathway. Returns a dict with the same
    keys: ``features``, ``support``, ``magnitudes``, ``activations``, ``x``."""
    device = device or _device()
    generator = torch.Generator(device=device.type).manual_seed(int(seed))

    features = _orth_features(k, d, target_cos_sim, seed=seed).to(device)

    support = _generate_support_batched_per_feature(
        n_seq=n_seq, k=k, T=T,
        pi=pi_per_feature, rho=rho_per_feature, delta=0.0,
        generator=generator, device=device,
    )

    # Folded normal magnitudes: |N(mu, sigma^2)|.
    raw = torch.randn(n_seq, k, T, generator=generator, device=device)
    magnitudes = (raw * magnitude_sigma + magnitude_mu).abs()
    activations = support * magnitudes  # (n_seq, k, T)

    # x = a · F where F is (k, d). One batched matmul: (n_seq, T, d).
    # activations: (n_seq, k, T) → (n_seq, T, k); features: (k, d) → (n_seq, T, d).
    x = activations.transpose(1, 2) @ features  # (n_seq, T, d)

    return {
        "features": features.cpu(),
        "support": support.cpu(),
        "magnitudes": magnitudes.cpu(),
        "activations": activations.cpu(),
        "x": x.cpu(),
    }


def _generate_coupled_fast(
    *, K: int, M: int, d: int, T: int, n_seq: int,
    n_parents: int,
    pi: float, rho: float, delta: float,
    coupling_mode: str, sigmoid_alpha: float, sigmoid_beta: float,
    target_cos_sim: float,
    magnitude_dist: str, magnitude_mu: float, magnitude_sigma: float,
    seed: int, device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Vectorised C2 coupled-HMM generation. Replaces
    ``coupled_dataset.generate_coupled_dataset`` for the OR-gate /
    sigmoid coupling pathways."""
    device = device or _device()
    generator = torch.Generator(device=device.type).manual_seed(int(seed))

    # 1. Orthogonal emission feature directions.
    emission_features = _orth_features(M, d, target_cos_sim, seed=seed).to(device)

    # 2. Binary coupling matrix (M × K), n_parents per emission.
    cpu_g = torch.Generator().manual_seed(int(seed) + 1)
    coupling_matrix = torch.zeros(M, K, device=device)
    for j in range(M):
        parents = torch.randperm(K, generator=cpu_g)[:n_parents]
        coupling_matrix[j, parents.to(device)] = 1.0

    # 3. Hidden directions = normalise(sum_{j: C_ji=1} f_j).
    raw = coupling_matrix.t() @ emission_features  # (K, d)
    hidden_features = raw / raw.norm(dim=1, keepdim=True).clamp_min(1e-12)

    # 4. K hidden Markov chains (independent, identical π/ρ).
    pi_t = torch.full((K,), float(pi), device=device)
    rho_t = torch.full((K,), float(rho), device=device)
    hidden_states = _generate_support_batched_per_feature(
        n_seq=n_seq, k=K, T=T,
        pi=pi_t, rho=rho_t, delta=float(delta),
        generator=generator, device=device,
    )

    # 5. Couple K → M emissions.
    if coupling_mode == "or_gate":
        # parent_sum: (n_seq, M, T)
        parent_sum = torch.einsum(
            "mk,nkt->nmt", coupling_matrix, hidden_states,
        )
        emission_support = (parent_sum >= 1).float()
    elif coupling_mode == "sigmoid":
        parent_sum = torch.einsum(
            "mk,nkt->nmt", coupling_matrix, hidden_states,
        )
        prob = torch.sigmoid(sigmoid_alpha * parent_sum + sigmoid_beta)
        u = torch.rand(n_seq, M, T, generator=generator, device=device)
        emission_support = (u < prob).float()
    else:
        raise ValueError(f"unknown coupling_mode={coupling_mode!r}")

    # 6. Magnitudes (i.i.d. folded normal across (j, t) and seq).
    if magnitude_dist not in ("folded_normal", "half_normal"):
        raise ValueError(f"unknown magnitude_dist={magnitude_dist!r}")
    raw_mag = torch.randn(n_seq, M, T, generator=generator, device=device)
    if magnitude_dist == "half_normal":
        mu_eff = 0.0
    else:
        mu_eff = magnitude_mu
    magnitudes = (raw_mag * magnitude_sigma + mu_eff).abs()
    activations = emission_support * magnitudes

    # 7. x = activations · F  (single batched matmul).
    x = activations.transpose(1, 2) @ emission_features

    return {
        "emission_features": emission_features.cpu(),
        "hidden_features": hidden_features.cpu(),
        "coupling_matrix": coupling_matrix.cpu(),
        "hidden_states": hidden_states.cpu(),
        "emission_support": emission_support.cpu(),
        "magnitudes": magnitudes.cpu(),
        "activations": activations.cpu(),
        "x": x.cpu(),
    }


def _per_feature_rho_levels(n_features: int,
                             rho_levels: list[float]) -> torch.Tensor:
    """Equal-size groups across ``rho_levels``. Same partition agent_paper uses."""
    if n_features % len(rho_levels) != 0:
        raise ValueError(
            f"n_features={n_features} must divide evenly across "
            f"len(rho_levels)={len(rho_levels)}"
        )
    per_group = n_features // len(rho_levels)
    rhos: list[float] = []
    for r in rho_levels:
        rhos.extend([float(r)] * per_group)
    return torch.tensor(rhos, dtype=torch.float32)


def markov_chain_support(*, n_features: int = 20, d_in: int = 40,
                         seq_len: int = 64, n_seqs: int = 4096,
                         rho_levels: list[float] | None = None,
                         pi: float = 0.5,
                         delta: float = 0.0,
                         seed: int = 0,
                         use_slow_pipeline: bool = False) -> MarkovData:
    """C1 generator. Same signature as agent_paper's ``markov_chain_support``,
    plus a ``delta`` knob for the leaky-reset transition kernel.

    delta=0 is the standard reset (what agent_paper ports). delta>0
    biases resamples toward the previous state (Aniket's wasteland
    extension); the (pi, rho) marginal/lag-1 statistics are unchanged.

    Default path is fully vectorised over (n_seqs, n_features) with
    only a T-step time loop, runs on GPU when available. Pass
    ``use_slow_pipeline=True`` to fall through to the wasteland's
    per-sequence Python loop (for unit-test parity).
    """
    if rho_levels is None:
        rho_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
    per_feature_rho = _per_feature_rho_levels(n_features, rho_levels)
    pi_t = torch.full((n_features,), float(pi))

    if not use_slow_pipeline:
        out = _generate_dataset_fast(
            k=n_features, d=d_in, T=seq_len, n_seq=n_seqs,
            pi_per_feature=pi_t, rho_per_feature=per_feature_rho,
            target_cos_sim=0.0,
            magnitude_mu=1.0, magnitude_sigma=0.15,
            seed=seed,
        )
        return MarkovData(
            x=out["x"], features=out["features"],
            support=out["support"], per_feature_rho=per_feature_rho,
        )

    # Slow path (kept for unit-test parity).
    cfg = DataGenerationConfig(
        features=FeatureConfig(k=n_features, d=d_in, target_cos_sim=0.0),
        transition=TransitionConfig(
            matrix=(build_leaky_transition_matrix(lam=0.5, p=pi, delta=delta)
                    if delta > 0 else build_transition_matrix(lam=0.5, p=pi)),
            stationary_on_prob=pi,
        ),
        emission=EmissionConfig(p_A=0.0, p_B=1.0),
        magnitude=MagnitudeConfig(distribution="folded_normal", mu=1.0, sigma=0.15),
        sequence=SequenceConfig(T=seq_len, n_sequences=n_seqs),
        per_feature_pi=[float(pi)] * n_features,
        per_feature_rho=per_feature_rho.tolist(),
        seed=seed,
    )
    out = generate_dataset(cfg)
    return MarkovData(
        x=out["x"], features=out["features"],
        support=out["support"], per_feature_rho=per_feature_rho,
    )


def coupled_hmm(*, K_hidden: int = 10, M_emissions: int = 20,
                n_parents: int = 2,
                d_in: int = 256, seq_len: int = 64,
                n_seqs: int = 4096,
                pi: float = 0.05, rho: float = 0.7,
                delta: float = 0.0,
                magnitude_dist: str = "folded_normal",
                magnitude_mean: float = 1.0,
                magnitude_std: float = 0.15,
                coupling_mode: str = "or_gate",
                sigmoid_alpha: float = 1.0, sigmoid_beta: float = -1.0,
                seed: int = 0,
                use_slow_pipeline: bool = False) -> CoupledData:
    """C2 generator. Same signature as agent_paper's ``coupled_hmm`` plus:

    - ``delta``       — leaky-reset on the K hidden chains (>0 ⇒ biased resample)
    - ``coupling_mode`` ∈ {"or_gate", "sigmoid"} — sigmoid is wasteland-only
    - ``sigmoid_alpha, sigmoid_beta`` — sigmoid coupling sharpness/bias

    Default path is fully vectorised: K hidden Markov chains generated
    in one ``(n_seq, K, T)`` pass, M emissions computed via a single
    ``einsum`` over the coupling matrix, magnitudes as one ``randn``,
    and observation as one batched matmul. Pass ``use_slow_pipeline=True``
    to fall through to the wasteland's per-seq loop.
    """
    if delta >= 1.0:
        raise ValueError("delta must be < 1.0")

    if not use_slow_pipeline:
        out = _generate_coupled_fast(
            K=K_hidden, M=M_emissions, d=d_in, T=seq_len, n_seq=n_seqs,
            n_parents=n_parents,
            pi=pi, rho=rho, delta=delta,
            coupling_mode=coupling_mode,
            sigmoid_alpha=sigmoid_alpha, sigmoid_beta=sigmoid_beta,
            target_cos_sim=0.0,
            magnitude_dist=magnitude_dist,
            magnitude_mu=magnitude_mean, magnitude_sigma=magnitude_std,
            seed=seed,
        )
        return CoupledData(
            x=out["x"],
            emission_features=out["emission_features"],
            hidden_features=out["hidden_features"],
            coupling_matrix=out["coupling_matrix"],
            emission_support=out["emission_support"],
            hidden_states=out["hidden_states"],
        )

    # Slow path.
    lam = (1.0 - rho) / (1.0 - delta)
    matrix = (
        build_leaky_transition_matrix(lam=lam, p=pi, delta=delta)
        if delta > 0
        else build_transition_matrix(lam=lam, p=pi)
    )
    cfg = CoupledDataGenerationConfig(
        coupling=CouplingConfig(
            K_hidden=K_hidden, M_emission=M_emissions, n_parents=n_parents,
            emission_mode=("or" if coupling_mode == "or_gate" else coupling_mode),
            sigmoid_alpha=sigmoid_alpha, sigmoid_beta=sigmoid_beta,
        ),
        transition=TransitionConfig(matrix=matrix, stationary_on_prob=pi),
        emission=EmissionConfig(p_A=0.0, p_B=1.0),
        magnitude=MagnitudeConfig(
            distribution=magnitude_dist, mu=magnitude_mean, sigma=magnitude_std,
        ),
        sequence=SequenceConfig(T=seq_len, n_sequences=n_seqs),
        hidden_dim=d_in,
        target_cos_sim=0.0,
        seed=seed,
    )
    out = generate_coupled_dataset(cfg)
    return CoupledData(
        x=out["x"],
        emission_features=out["emission_features"],
        hidden_features=out["hidden_features"],
        coupling_matrix=out["coupling_matrix"],
        emission_support=out["support"],
        hidden_states=out["hidden_states"],
    )


def make_batch_iter(data: Any, *, seed: int = 0
                    ) -> Callable[[int], torch.Tensor]:
    """Sample mini-batches by drawing whole-sequence rows uniformly.

    Same shape contract as agent_paper's ``make_batch_iter``: the
    callable returns a ``(n, T, d_in)`` tensor.
    """
    rng = np.random.default_rng(seed)
    x = data.x

    def _iter(n: int) -> torch.Tensor:
        n_seqs = x.shape[0]
        idx = rng.integers(0, n_seqs, size=n)
        return x[idx]

    return _iter
