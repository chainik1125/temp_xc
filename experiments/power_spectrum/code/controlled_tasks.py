"""Controlled temporal generators for frequency-selective crosscoder tests.

The generators in this module deliberately live inside the power-spectrum
experiment.  They provide two substrates with analytically known temporal
structure:

``generate_shamir_clock``
    An exact port of the polynomial-clock task.  The secret is the leading
    coefficient of a degree-``h`` polynomial over ``F_q``.  Any prefix of at
    most ``h`` symbols is independent of the secret, while ``h + 1`` symbols
    recover it exactly.

``generate_factorial_hmm``
    Independent stationary symmetric two-state Markov chains.  A chain with
    eigenvalue ``lambda > 0`` is low-pass; one with ``lambda < 0`` alternates
    and is high-pass.  Each chain is observed along its own orthonormal
    direction, making latent-to-frequency attribution unambiguous.

``generate_narrowband_sources``
    Simultaneously active independent sources at separated carrier
    frequencies.  Every source owns a two-dimensional quadrature emission
    plane, so random phase does not turn one cause into a mixture of bands.

All three generators accept a fixed externally supplied observation dictionary.
The split helpers use that facility to share only the observation coordinate
system across train/probe/eval while sampling every split as new episodes.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ShamirClockBatch:
    """One independently sampled set of polynomial-clock episodes."""

    x: torch.Tensor
    symbols: torch.Tensor
    secret: torch.Tensor
    alphabet: torch.Tensor
    nuisance: torch.Tensor
    coefficients: torch.Tensor


@dataclass(frozen=True)
class FactorialHMMBatch:
    """One independently sampled set of factorial-HMM episodes."""

    x: torch.Tensor
    states: torch.Tensor
    emissions: torch.Tensor
    lambdas: torch.Tensor


@dataclass(frozen=True)
class NarrowbandSourceBatch:
    """Independent single-frequency causes mixed in orthogonal quadrature planes.

    ``states[n, t, j]`` contains the two amplitude-scaled quadratures for
    source ``j``.  In a noiseless batch it is recovered exactly by projecting
    ``x[n, t]`` onto ``emissions[j]``.
    """

    x: torch.Tensor
    states: torch.Tensor
    phases: torch.Tensor
    amplitudes: torch.Tensor
    emissions: torch.Tensor
    frequencies: torch.Tensor


# Integer Fourier bins 1, 3, 5, and 7 at T=16 (and 2, 6, 10, 14 at T=32).
# They span the AC range without putting multiple frequencies in any cause.
DEFAULT_NARROWBAND_FREQUENCIES = (1.0 / 16, 3.0 / 16, 5.0 / 16, 7.0 / 16)


def is_prime(value: int) -> bool:
    """Return whether ``value`` is prime."""

    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def orthonormal_rows(
    n_rows: int,
    d: int,
    *,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw ``n_rows`` orthonormal directions in ``R^d``.

    QR is performed in float64 both to reproduce the original polynomial-clock
    generator and to make dictionary validation insensitive to float32 noise.
    """

    if n_rows < 1:
        raise ValueError(f"n_rows must be positive; got {n_rows}")
    if n_rows > d:
        raise ValueError(f"need n_rows <= d; got n_rows={n_rows}, d={d}")
    matrix = torch.randn(d, d, generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(matrix)
    return basis[:, :n_rows].T.contiguous()


def _validated_dictionary(
    dictionary: torch.Tensor,
    *,
    n_rows: int,
    d: int,
    name: str,
) -> torch.Tensor:
    if dictionary.shape != (n_rows, d):
        raise ValueError(f"{name} must have shape ({n_rows}, {d}); got {tuple(dictionary.shape)}")
    if not dictionary.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not torch.isfinite(dictionary).all():
        raise ValueError(f"{name} must contain only finite values")
    dictionary_cpu = dictionary.detach().to(device="cpu", dtype=torch.float64)
    gram = dictionary_cpu @ dictionary_cpu.T
    identity = torch.eye(n_rows, dtype=torch.float64)
    if not torch.allclose(gram, identity, atol=1e-6, rtol=1e-6):
        error = (gram - identity).abs().max().item()
        raise ValueError(f"{name} rows must be orthonormal (max Gram error {error:.3g})")
    return dictionary_cpu.clone()


def evaluate_polynomial(
    coefficients: torch.Tensor,
    t: torch.Tensor,
    q: int,
) -> torch.Tensor:
    """Evaluate integer-coefficient polynomials modulo ``q``.

    ``coefficients[..., k]`` multiplies ``t**k``.  Leading dimensions follow
    normal PyTorch broadcasting; the final coefficient dimension is removed.
    """

    if coefficients.ndim < 1 or coefficients.shape[-1] < 1:
        raise ValueError("coefficients must have a non-empty final dimension")
    t_mod = (t % q).to(torch.long)
    powers = torch.stack(
        [torch.pow(t_mod, degree) % q for degree in range(coefficients.shape[-1])],
        dim=-1,
    )
    return (coefficients.to(torch.long) * powers).sum(dim=-1) % q


def recover_leading_coefficient(symbols: torch.Tensor, q: int) -> torch.Tensor:
    """Recover a polynomial's leading coefficient by finite-field interpolation.

    The final axis contains values at the consecutive public points
    ``t = 0, ..., h``.  Its length is therefore the threshold ``h + 1``.
    """

    if not is_prime(q):
        raise ValueError(f"q must be prime; got {q}")
    if symbols.ndim < 1 or symbols.shape[-1] < 2:
        raise ValueError("symbols must contain at least two evaluation points")
    threshold = symbols.shape[-1]
    if threshold > q:
        raise ValueError(f"need h + 1 <= q for distinct field points; got {threshold} > {q}")

    result = torch.zeros(symbols.shape[:-1], dtype=torch.long, device=symbols.device)
    values = symbols.to(torch.long) % q
    # The t^h coefficient of L_i(t) is 1 / prod_{j != i}(i-j).
    for i in range(threshold):
        denominator = 1
        for j in range(threshold):
            if i != j:
                denominator = denominator * (i - j) % q
        inverse = pow(denominator, -1, q)
        result = (result + values[..., i] * inverse) % q
    return result


def generate_shamir_clock(
    *,
    h: int,
    q: int,
    d: int,
    sigma: float,
    n_seq: int,
    seq_len: int,
    seed: int,
    alphabet: torch.Tensor | None = None,
) -> ShamirClockBatch:
    """Generate polynomial-clock (Shamir-threshold) episodes.

    For each episode, sample a secret ``Y`` and nuisance coefficients
    ``B_0, ..., B_{h-1}`` uniformly from ``F_q`` and emit

    ``Q_t = B_0 + B_1 t + ... + B_{h-1} t^(h-1) + Y t^h (mod q)``.

    Observations are ``alphabet[Q_t] + sigma * epsilon_t``.  Supplying
    ``alphabet`` holds the observation coordinate system fixed across
    independently generated data splits.
    """

    if h < 1:
        raise ValueError(f"h must be positive; got {h}")
    if not is_prime(q):
        raise ValueError(f"q must be prime; got {q}")
    if h >= q:
        raise ValueError(f"need h < q for h + 1 distinct field points; got h={h}, q={q}")
    if q > d:
        raise ValueError(f"need q <= d so the alphabet fits; got q={q}, d={d}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if n_seq < 1:
        raise ValueError(f"n_seq must be positive; got {n_seq}")
    if seq_len < h + 1:
        raise ValueError(f"seq_len must be at least h + 1={h + 1}; got {seq_len}")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if alphabet is None:
        fixed_alphabet = orthonormal_rows(q, d, generator=generator)
    else:
        fixed_alphabet = _validated_dictionary(
            alphabet,
            n_rows=q,
            d=d,
            name="alphabet",
        )

    secret = torch.randint(0, q, (n_seq,), generator=generator)
    nuisance = torch.randint(0, q, (n_seq, h), generator=generator)
    coefficients = torch.cat([nuisance, secret.unsqueeze(-1)], dim=-1)
    t = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
    symbols = evaluate_polynomial(coefficients.unsqueeze(1), t, q)
    clean = fixed_alphabet[symbols]
    noise = torch.randn(n_seq, seq_len, d, generator=generator, dtype=torch.float64)
    x = clean + float(sigma) * noise
    return ShamirClockBatch(
        x=x,
        symbols=symbols,
        secret=secret,
        alphabet=fixed_alphabet,
        nuisance=nuisance,
        coefficients=coefficients,
    )


def generate_shamir_splits(
    *,
    h: int,
    q: int,
    d: int,
    sigma: float,
    seq_len: int,
    split_sizes: Mapping[str, int],
    split_seeds: Mapping[str, int],
    alphabet_seed: int,
) -> dict[str, ShamirClockBatch]:
    """Generate episode-disjoint splits with one shared alphabet.

    "Disjoint" here is procedural: each split is sampled from its own seeded
    episode stream, rather than slicing overlapping windows from a common long
    trajectory.  Exact duplicate random episodes remain possible in a finite
    state space.
    """

    _validate_split_spec(split_sizes, split_seeds)
    alphabet_generator = torch.Generator(device="cpu").manual_seed(alphabet_seed)
    alphabet = orthonormal_rows(q, d, generator=alphabet_generator)
    return {
        name: generate_shamir_clock(
            h=h,
            q=q,
            d=d,
            sigma=sigma,
            n_seq=size,
            seq_len=seq_len,
            seed=split_seeds[name],
            alphabet=alphabet,
        )
        for name, size in split_sizes.items()
    }


def generate_factorial_hmm(
    *,
    lambdas: Sequence[float] | torch.Tensor,
    d: int,
    sigma: float,
    n_seq: int,
    seq_len: int,
    seed: int,
    emissions: torch.Tensor | None = None,
) -> FactorialHMMBatch:
    """Generate independent symmetric two-state HMM factors.

    Factor ``j`` has states in ``{-1, +1}`` and transition rule

    ``P(s_t = s_(t-1)) = (1 + lambda_j) / 2``.

    Consequently its stationary autocovariance is ``lambda_j**abs(lag)``.
    The observation is the sum of state-weighted orthonormal emission
    directions plus isotropic Gaussian noise.
    """

    lambda_tensor = torch.as_tensor(lambdas, dtype=torch.float64).flatten()
    if lambda_tensor.numel() < 1:
        raise ValueError("lambdas must contain at least one factor")
    if not torch.isfinite(lambda_tensor).all():
        raise ValueError("lambdas must be finite")
    if (lambda_tensor.abs() >= 1).any():
        raise ValueError("each lambda must lie strictly between -1 and 1")
    if lambda_tensor.numel() > d:
        raise ValueError(f"need number of factors <= d; got {lambda_tensor.numel()} > {d}")
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if n_seq < 1:
        raise ValueError(f"n_seq must be positive; got {n_seq}")
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2; got {seq_len}")

    n_factors = lambda_tensor.numel()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if emissions is None:
        fixed_emissions = orthonormal_rows(n_factors, d, generator=generator)
    else:
        fixed_emissions = _validated_dictionary(
            emissions,
            n_rows=n_factors,
            d=d,
            name="emissions",
        )

    initial = 2 * torch.randint(0, 2, (n_seq, n_factors), generator=generator) - 1
    uniforms = torch.rand(n_seq, seq_len - 1, n_factors, generator=generator)
    stay_probability = (1.0 + lambda_tensor).view(1, 1, n_factors) / 2.0
    multipliers = torch.where(
        uniforms < stay_probability,
        torch.ones((), dtype=torch.long),
        -torch.ones((), dtype=torch.long),
    )
    states = torch.empty(n_seq, seq_len, n_factors, dtype=torch.long)
    states[:, 0] = initial
    for time_index in range(1, seq_len):
        states[:, time_index] = states[:, time_index - 1] * multipliers[:, time_index - 1]

    clean = states.to(torch.float64) @ fixed_emissions
    noise = torch.randn(n_seq, seq_len, d, generator=generator, dtype=torch.float64)
    x = clean + float(sigma) * noise
    return FactorialHMMBatch(
        x=x,
        states=states,
        emissions=fixed_emissions,
        lambdas=lambda_tensor.clone(),
    )


def generate_factorial_hmm_splits(
    *,
    lambdas: Sequence[float] | torch.Tensor,
    d: int,
    sigma: float,
    seq_len: int,
    split_sizes: Mapping[str, int],
    split_seeds: Mapping[str, int],
    emission_seed: int,
) -> dict[str, FactorialHMMBatch]:
    """Generate independent HMM episode splits with shared emissions."""

    _validate_split_spec(split_sizes, split_seeds)
    lambda_tensor = torch.as_tensor(lambdas, dtype=torch.float64).flatten()
    emission_generator = torch.Generator(device="cpu").manual_seed(emission_seed)
    emissions = orthonormal_rows(lambda_tensor.numel(), d, generator=emission_generator)
    return {
        name: generate_factorial_hmm(
            lambdas=lambda_tensor,
            d=d,
            sigma=sigma,
            n_seq=size,
            seq_len=seq_len,
            seed=split_seeds[name],
            emissions=emissions,
        )
        for name, size in split_sizes.items()
    }


def generate_narrowband_sources(
    *,
    frequencies: Sequence[float] | torch.Tensor = DEFAULT_NARROWBAND_FREQUENCIES,
    d: int,
    sigma: float,
    n_seq: int,
    seq_len: int,
    seed: int,
    emissions: torch.Tensor | None = None,
    amplitude_range: tuple[float, float] = (0.75, 1.25),
    min_frequency_separation: float = 1.0 / 16,
) -> NarrowbandSourceBatch:
    """Generate a simultaneous mixture of independent narrowband causes.

    Each source ``j`` has its own carrier ``f_j`` in cycles/sample, random
    episode phase ``phi_j``, positive random amplitude ``a_j``, and orthogonal
    quadrature emission plane ``(u_j, v_j)``:

    ``x_t = sum_j a_j [cos(2 pi f_j (t+1/2) + phi_j) u_j
                       + sin(2 pi f_j (t+1/2) + phi_j) v_j] + noise``.

    The two quadratures are one *cause* at one carrier frequency.  They make
    its representation phase-invariant; they do not combine distinct
    frequencies.  The default carriers occupy separated exact Fourier bins
    for both ``T=16`` and ``T=32``.
    """

    frequency_tensor = torch.as_tensor(frequencies, dtype=torch.float64).flatten()
    if frequency_tensor.numel() < 2:
        raise ValueError("frequencies must contain at least two simultaneous sources")
    if not torch.isfinite(frequency_tensor).all():
        raise ValueError("frequencies must be finite")
    if ((frequency_tensor <= 0) | (frequency_tensor >= 0.5)).any():
        raise ValueError("frequencies must lie strictly between DC and Nyquist")
    if min_frequency_separation <= 0:
        raise ValueError("min_frequency_separation must be positive")
    sorted_frequencies = frequency_tensor.sort().values
    separations = sorted_frequencies[1:] - sorted_frequencies[:-1]
    if (separations + 1e-12 < min_frequency_separation).any():
        raise ValueError(
            "frequencies are not sufficiently separated: "
            f"minimum required gap is {min_frequency_separation}"
        )
    n_sources = frequency_tensor.numel()
    if 2 * n_sources > d:
        raise ValueError(
            f"need two orthonormal emission directions per source; got 2 * {n_sources} > d={d}"
        )
    if sigma < 0:
        raise ValueError(f"sigma must be non-negative; got {sigma}")
    if n_seq < 1:
        raise ValueError(f"n_seq must be positive; got {n_seq}")
    if seq_len < 2:
        raise ValueError(f"seq_len must be at least 2; got {seq_len}")
    amplitude_low, amplitude_high = map(float, amplitude_range)
    if not 0 < amplitude_low <= amplitude_high:
        raise ValueError(
            f"amplitude_range must be finite, positive, and ordered; got {amplitude_range}"
        )
    if not math.isfinite(amplitude_low) or not math.isfinite(amplitude_high):
        raise ValueError(
            f"amplitude_range must be finite, positive, and ordered; got {amplitude_range}"
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    if emissions is None:
        fixed_emissions = orthonormal_rows(
            2 * n_sources,
            d,
            generator=generator,
        ).reshape(n_sources, 2, d)
    else:
        if emissions.shape != (n_sources, 2, d):
            raise ValueError(
                f"emissions must have shape ({n_sources}, 2, {d}); got {tuple(emissions.shape)}"
            )
        fixed_emissions = _validated_dictionary(
            emissions.reshape(2 * n_sources, d),
            n_rows=2 * n_sources,
            d=d,
            name="emissions",
        ).reshape(n_sources, 2, d)

    phases = (
        2.0
        * math.pi
        * torch.rand(
            n_seq,
            n_sources,
            generator=generator,
            dtype=torch.float64,
        )
    )
    amplitude_uniform = torch.rand(
        n_seq,
        n_sources,
        generator=generator,
        dtype=torch.float64,
    )
    amplitudes = amplitude_low + (amplitude_high - amplitude_low) * amplitude_uniform
    time = torch.arange(seq_len, dtype=torch.float64) + 0.5
    angles = (
        2.0 * math.pi * time[None, :, None] * frequency_tensor[None, None, :] + phases[:, None, :]
    )
    unit_quadratures = torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)
    states = amplitudes[:, None, :, None] * unit_quadratures
    clean = torch.einsum("ntjc,jcd->ntd", states, fixed_emissions)
    noise = torch.randn(n_seq, seq_len, d, generator=generator, dtype=torch.float64)
    x = clean + float(sigma) * noise
    return NarrowbandSourceBatch(
        x=x,
        states=states,
        phases=phases,
        amplitudes=amplitudes,
        emissions=fixed_emissions,
        frequencies=frequency_tensor.clone(),
    )


def generate_narrowband_source_splits(
    *,
    frequencies: Sequence[float] | torch.Tensor = DEFAULT_NARROWBAND_FREQUENCIES,
    d: int,
    sigma: float,
    seq_len: int,
    split_sizes: Mapping[str, int],
    split_seeds: Mapping[str, int],
    emission_seed: int,
    amplitude_range: tuple[float, float] = (0.75, 1.25),
    min_frequency_separation: float = 1.0 / 16,
) -> dict[str, NarrowbandSourceBatch]:
    """Generate independent narrowband episodes with one shared dictionary."""

    _validate_split_spec(split_sizes, split_seeds)
    frequency_tensor = torch.as_tensor(frequencies, dtype=torch.float64).flatten()
    emission_generator = torch.Generator(device="cpu").manual_seed(emission_seed)
    emissions = orthonormal_rows(
        2 * frequency_tensor.numel(),
        d,
        generator=emission_generator,
    ).reshape(frequency_tensor.numel(), 2, d)
    return {
        name: generate_narrowband_sources(
            frequencies=frequency_tensor,
            d=d,
            sigma=sigma,
            n_seq=size,
            seq_len=seq_len,
            seed=split_seeds[name],
            emissions=emissions,
            amplitude_range=amplitude_range,
            min_frequency_separation=min_frequency_separation,
        )
        for name, size in split_sizes.items()
    }


def _validate_split_spec(
    split_sizes: Mapping[str, int],
    split_seeds: Mapping[str, int],
) -> None:
    if not split_sizes:
        raise ValueError("split_sizes must not be empty")
    if set(split_sizes) != set(split_seeds):
        raise ValueError("split_sizes and split_seeds must have identical names")
    if len(set(split_seeds.values())) != len(split_seeds):
        raise ValueError("every split must use a distinct episode seed")
    if any(size < 1 for size in split_sizes.values()):
        raise ValueError("every split size must be positive")


def theoretical_hmm_psd(
    lambdas: Sequence[float] | torch.Tensor,
    angular_frequencies: Sequence[float] | torch.Tensor,
) -> torch.Tensor:
    """Return the stationary two-state HMM power spectral density.

    For autocovariance ``gamma(tau) = lambda**abs(tau)``,

    ``S(omega) = (1 - lambda**2) / (1 + lambda**2 - 2 lambda cos(omega))``.

    The result has shape ``(n_factors, n_frequencies)``.
    """

    lambda_tensor = torch.as_tensor(lambdas, dtype=torch.float64).flatten()
    omega = torch.as_tensor(angular_frequencies, dtype=torch.float64).flatten()
    if lambda_tensor.numel() < 1 or omega.numel() < 1:
        raise ValueError("lambdas and angular_frequencies must be non-empty")
    if (lambda_tensor.abs() >= 1).any():
        raise ValueError("each lambda must lie strictly between -1 and 1")
    lam = lambda_tensor[:, None]
    return (1.0 - lam.square()) / (1.0 + lam.square() - 2.0 * lam * torch.cos(omega)[None, :])


def dct_basis(seq_len: int) -> torch.Tensor:
    """Return the orthonormal DCT-II basis with shape ``(seq_len, seq_len)``."""

    if seq_len < 1:
        raise ValueError(f"seq_len must be positive; got {seq_len}")
    time = torch.arange(seq_len, dtype=torch.float64) + 0.5
    frequency = torch.arange(seq_len, dtype=torch.float64)[:, None]
    basis = math.sqrt(2.0 / seq_len) * torch.cos(math.pi * frequency * time / seq_len)
    basis[0] = 1.0 / math.sqrt(seq_len)
    return basis


def expected_dct_energy(
    lambdas: Sequence[float] | torch.Tensor,
    seq_len: int,
) -> torch.Tensor:
    """Expected squared DCT coefficients for unit-variance HMM states.

    Returns one row per factor and one column per DCT-II frequency.  Every row
    sums to ``seq_len`` by Parseval's identity.
    """

    lambda_tensor = torch.as_tensor(lambdas, dtype=torch.float64).flatten()
    if lambda_tensor.numel() < 1:
        raise ValueError("lambdas must not be empty")
    if (lambda_tensor.abs() >= 1).any():
        raise ValueError("each lambda must lie strictly between -1 and 1")
    basis = dct_basis(seq_len)
    index = torch.arange(seq_len)
    lag = (index[:, None] - index[None, :]).abs()
    rows = []
    for value in lambda_tensor:
        covariance = value.pow(lag)
        transformed = basis @ covariance @ basis.T
        rows.append(torch.diagonal(transformed))
    return torch.stack(rows)


def expected_dct_band_energy(
    lambdas: Sequence[float] | torch.Tensor,
    seq_len: int,
    bands: Sequence[tuple[int, int]],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Aggregate expected DCT energy into half-open index bands.

    Bands may omit frequencies but may not overlap.  With ``normalize=True``,
    each factor is normalized over the included bands, so rows sum to one.
    """

    if not bands:
        raise ValueError("bands must not be empty")
    occupied: set[int] = set()
    for start, stop in bands:
        if not 0 <= start < stop <= seq_len:
            raise ValueError(f"invalid half-open band ({start}, {stop}) for seq_len={seq_len}")
        indices = set(range(start, stop))
        if occupied & indices:
            raise ValueError("DCT bands must not overlap")
        occupied.update(indices)

    energy = expected_dct_energy(lambdas, seq_len)
    band_energy = torch.stack(
        [energy[:, start:stop].sum(dim=1) for start, stop in bands],
        dim=1,
    )
    if normalize:
        band_energy = band_energy / band_energy.sum(dim=1, keepdim=True)
    return band_energy
