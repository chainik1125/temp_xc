from __future__ import annotations

import itertools
import math

import pytest
import torch

from experiments.power_spectrum.code.controlled_tasks import (
    DEFAULT_NARROWBAND_FREQUENCIES,
    dct_basis,
    evaluate_polynomial,
    expected_dct_band_energy,
    expected_dct_energy,
    generate_factorial_hmm,
    generate_factorial_hmm_splits,
    generate_narrowband_source_splits,
    generate_narrowband_sources,
    generate_shamir_clock,
    generate_shamir_splits,
    is_prime,
    recover_leading_coefficient,
    theoretical_hmm_psd,
)


def test_shamir_clock_is_deterministic_and_uses_fixed_alphabet() -> None:
    kwargs = {
        "h": 2,
        "q": 7,
        "d": 12,
        "sigma": 0.1,
        "n_seq": 16,
        "seq_len": 8,
        "seed": 4,
    }
    first = generate_shamir_clock(**kwargs)
    second = generate_shamir_clock(**kwargs)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.symbols, second.symbols)
    assert torch.equal(first.secret, second.secret)
    assert torch.equal(first.alphabet, second.alphabet)

    fixed = first.alphabet.clone()
    resampled = generate_shamir_clock(**{**kwargs, "seed": 5, "alphabet": fixed})
    assert torch.equal(resampled.alphabet, fixed)
    assert torch.equal(fixed, first.alphabet)
    assert not torch.equal(resampled.coefficients, first.coefficients)
    assert torch.allclose(fixed @ fixed.T, torch.eye(7, dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("q", [1, 4, 9, 15])
def test_shamir_clock_rejects_nonprime_field(q: int) -> None:
    assert not is_prime(q)
    with pytest.raises(ValueError, match="prime"):
        generate_shamir_clock(
            h=1,
            q=q,
            d=16,
            sigma=0.0,
            n_seq=2,
            seq_len=2,
            seed=0,
        )


@pytest.mark.parametrize("h,q", [(1, 5), (2, 5), (3, 7), (4, 11)])
def test_shamir_threshold_recovers_secret_for_various_h(h: int, q: int) -> None:
    batch = generate_shamir_clock(
        h=h,
        q=q,
        d=q + 3,
        sigma=0.0,
        n_seq=128,
        seq_len=h + 2,
        seed=10 + h,
    )
    recovered = recover_leading_coefficient(batch.symbols[:, : h + 1], q)
    assert torch.equal(recovered, batch.secret)
    assert torch.equal(batch.x, batch.alphabet[batch.symbols])


def test_shamir_short_prefix_has_exact_perfect_privacy() -> None:
    """Enumerate all nuisance coefficients: each short prefix has equal counts."""

    h = 3
    q = 5
    prefix_len = h
    t = torch.arange(prefix_len).unsqueeze(0)
    conditional_counts: list[torch.Tensor] = []
    for secret in range(q):
        nuisance = torch.tensor(
            list(itertools.product(range(q), repeat=h)),
            dtype=torch.long,
        )
        secrets = torch.full((nuisance.shape[0], 1), secret, dtype=torch.long)
        coefficients = torch.cat([nuisance, secrets], dim=1)
        symbols = evaluate_polynomial(coefficients.unsqueeze(1), t, q)
        keys = sum(symbols[:, index] * q**index for index in range(prefix_len))
        conditional_counts.append(torch.bincount(keys, minlength=q**prefix_len))
    for counts in conditional_counts[1:]:
        assert torch.equal(counts, conditional_counts[0])
    assert torch.equal(conditional_counts[0], torch.ones(q**prefix_len, dtype=torch.long))


def test_shamir_splits_share_only_alphabet_and_require_distinct_seeds() -> None:
    split_sizes = {"train": 9, "probe": 7, "eval": 5}
    split_seeds = {"train": 101, "probe": 202, "eval": 303}
    splits = generate_shamir_splits(
        h=2,
        q=7,
        d=10,
        sigma=0.05,
        seq_len=6,
        split_sizes=split_sizes,
        split_seeds=split_seeds,
        alphabet_seed=77,
    )
    assert {name: batch.x.shape[0] for name, batch in splits.items()} == split_sizes
    assert torch.equal(splits["train"].alphabet, splits["probe"].alphabet)
    assert torch.equal(splits["train"].alphabet, splits["eval"].alphabet)
    assert not torch.equal(
        splits["train"].coefficients[:5],
        splits["eval"].coefficients,
    )

    with pytest.raises(ValueError, match="distinct"):
        generate_shamir_splits(
            h=2,
            q=7,
            d=10,
            sigma=0.0,
            seq_len=4,
            split_sizes={"train": 2, "eval": 2},
            split_seeds={"train": 1, "eval": 1},
            alphabet_seed=0,
        )


def test_factorial_hmm_is_deterministic_and_emissions_are_orthonormal() -> None:
    kwargs = {
        "lambdas": (0.85, 0.0, -0.85),
        "d": 8,
        "sigma": 0.1,
        "n_seq": 32,
        "seq_len": 40,
        "seed": 12,
    }
    first = generate_factorial_hmm(**kwargs)
    second = generate_factorial_hmm(**kwargs)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.emissions, second.emissions)
    assert first.states.shape == (32, 40, 3)
    assert set(torch.unique(first.states).tolist()) == {-1, 1}
    assert torch.allclose(
        first.emissions @ first.emissions.T,
        torch.eye(3, dtype=torch.float64),
        atol=1e-10,
    )


def test_factorial_hmm_empirical_lag_one_matches_specified_lambdas() -> None:
    lambdas = torch.tensor([0.9, 0.35, 0.0, -0.4, -0.9])
    batch = generate_factorial_hmm(
        lambdas=lambdas,
        d=7,
        sigma=0.0,
        n_seq=512,
        seq_len=96,
        seed=8,
    )
    products = batch.states[:, 1:] * batch.states[:, :-1]
    empirical = products.to(torch.float64).mean(dim=(0, 1))
    assert torch.allclose(empirical, lambdas.to(torch.float64), atol=0.015)
    # With zero noise, projection onto the known orthonormal emissions exactly
    # recovers every latent trajectory.
    projected = batch.x @ batch.emissions.T
    assert torch.allclose(projected, batch.states.to(torch.float64), atol=1e-10)


def test_hmm_splits_share_emissions_but_resample_episode_trajectories() -> None:
    splits = generate_factorial_hmm_splits(
        lambdas=(0.8, -0.8),
        d=5,
        sigma=0.1,
        seq_len=20,
        split_sizes={"train": 12, "probe": 9, "eval": 6},
        split_seeds={"train": 11, "probe": 22, "eval": 33},
        emission_seed=99,
    )
    assert torch.equal(splits["train"].emissions, splits["probe"].emissions)
    assert torch.equal(splits["train"].emissions, splits["eval"].emissions)
    assert not torch.equal(
        splits["train"].states[:6],
        splits["eval"].states,
    )


def test_hmm_psd_and_dct_energy_separate_low_and_high_frequency_modes() -> None:
    lambdas = (0.9, 0.0, -0.9)
    psd = theoretical_hmm_psd(lambdas, (0.0, math.pi))
    assert torch.all(psd > 0)
    assert psd[0, 0] > 100 * psd[0, 1]
    assert psd[2, 1] > 100 * psd[2, 0]
    assert torch.allclose(psd[1], torch.ones(2, dtype=torch.float64))

    energy = expected_dct_energy(lambdas, seq_len=32)
    assert torch.allclose(
        energy.sum(dim=1),
        torch.full((3,), 32.0, dtype=torch.float64),
        atol=1e-9,
    )
    bands = expected_dct_band_energy(
        lambdas,
        seq_len=32,
        bands=((0, 8), (8, 24), (24, 32)),
    )
    assert torch.allclose(bands.sum(dim=1), torch.ones(3, dtype=torch.float64))
    assert bands[0, 0] > bands[0, 2]
    assert bands[2, 2] > bands[2, 0]
    assert torch.allclose(
        bands[1],
        torch.tensor([0.25, 0.5, 0.25], dtype=torch.float64),
        atol=1e-10,
    )


def test_empirical_dct_energy_tracks_theory_for_positive_and_negative_modes() -> None:
    lambdas = (0.8, -0.8)
    batch = generate_factorial_hmm(
        lambdas=lambdas,
        d=4,
        sigma=0.0,
        n_seq=2048,
        seq_len=24,
        seed=19,
    )
    coefficients = torch.einsum(
        "kt,ntj->nkj",
        dct_basis(24),
        batch.states.to(torch.float64),
    )
    empirical = coefficients.square().mean(dim=0).T
    theory = expected_dct_energy(lambdas, 24)
    relative_error = (empirical - theory).abs().mean() / theory.mean()
    assert relative_error < 0.06
    assert empirical[0, :6].sum() > empirical[0, -6:].sum()
    assert empirical[1, -6:].sum() > empirical[1, :6].sum()


@pytest.mark.parametrize("bad_lambda", [-1.0, 1.0, 1.1, float("nan")])
def test_factorial_hmm_rejects_invalid_lambda(bad_lambda: float) -> None:
    with pytest.raises(ValueError, match="lambda"):
        generate_factorial_hmm(
            lambdas=(bad_lambda,),
            d=2,
            sigma=0.0,
            n_seq=2,
            seq_len=3,
            seed=0,
        )


def test_default_narrowband_carriers_are_exact_separated_bins_at_t16_and_t32() -> None:
    frequencies = torch.tensor(DEFAULT_NARROWBAND_FREQUENCIES, dtype=torch.float64)
    assert torch.allclose(
        frequencies * 16,
        torch.tensor([1.0, 3.0, 5.0, 7.0], dtype=torch.float64),
    )
    assert torch.allclose(
        frequencies * 32,
        torch.tensor([2.0, 6.0, 10.0, 14.0], dtype=torch.float64),
    )
    assert torch.all(torch.diff(frequencies) >= 1.0 / 16)


def test_narrowband_sources_are_deterministic_simultaneous_and_recoverable() -> None:
    kwargs = {
        "d": 12,
        "sigma": 0.0,
        "n_seq": 24,
        "seq_len": 32,
        "seed": 14,
    }
    first = generate_narrowband_sources(**kwargs)
    second = generate_narrowband_sources(**kwargs)
    assert torch.equal(first.x, second.x)
    assert torch.equal(first.states, second.states)
    assert torch.equal(first.phases, second.phases)
    assert torch.equal(first.amplitudes, second.amplitudes)
    assert torch.equal(first.activity, second.activity)
    assert first.activity.all()
    assert first.states.shape == (24, 32, 4, 2)
    assert first.emissions.shape == (4, 2, 12)

    flat_emissions = first.emissions.reshape(8, 12)
    assert torch.allclose(
        flat_emissions @ flat_emissions.T,
        torch.eye(8, dtype=torch.float64),
        atol=1e-10,
    )
    projected = torch.einsum("ntd,jcd->ntjc", first.x, first.emissions)
    assert torch.allclose(projected, first.states, atol=1e-10)
    # Every source is active at every time: quadrature norm equals its
    # independently sampled strictly positive episode amplitude.
    assert torch.allclose(
        first.states.norm(dim=-1),
        first.amplitudes[:, None, :].expand(-1, 32, -1),
        atol=1e-10,
    )


def test_sparse_repeated_frequency_dictionary_has_exact_episode_support() -> None:
    frequencies = (1.0 / 8,) * 4 + (2.0 / 8,) * 4 + (3.0 / 8,) * 4
    batch = generate_narrowband_sources(
        frequencies=frequencies,
        d=24,
        sigma=0.0,
        n_seq=48,
        seq_len=16,
        seed=23,
        active_sources_per_episode=3,
        allow_repeated_frequencies=True,
    )
    assert batch.activity.shape == (48, 12)
    assert torch.equal(
        batch.activity.sum(dim=1),
        torch.full((48,), 3),
    )
    assert torch.equal(
        batch.states.norm(dim=-1) > 0,
        batch.activity[:, None, :].expand(-1, 16, -1),
    )
    projected = torch.einsum("ntd,jcd->ntjc", batch.x, batch.emissions)
    assert torch.allclose(projected, batch.states, atol=1e-10)


def test_sparse_narrowband_dictionary_rejects_invalid_support() -> None:
    with pytest.raises(ValueError, match="active_sources_per_episode"):
        generate_narrowband_sources(
            frequencies=(1.0 / 8, 1.0 / 8),
            d=4,
            sigma=0.0,
            n_seq=2,
            seq_len=8,
            seed=0,
            active_sources_per_episode=3,
            allow_repeated_frequencies=True,
        )
    with pytest.raises(ValueError, match="sufficiently separated"):
        generate_narrowband_sources(
            frequencies=(0.125, 0.125, 0.15),
            d=6,
            sigma=0.0,
            n_seq=2,
            seq_len=8,
            seed=0,
            active_sources_per_episode=2,
            allow_repeated_frequencies=True,
            min_frequency_separation=0.05,
        )


def test_each_narrowband_cause_has_one_distinct_fourier_peak() -> None:
    batch = generate_narrowband_sources(
        d=10,
        sigma=0.0,
        n_seq=32,
        seq_len=64,
        seed=7,
    )
    spectrum = torch.fft.rfft(batch.states, dim=1)
    power = spectrum.abs().square().sum(dim=(0, 3)).T
    expected_bins = (batch.frequencies * 64).round().to(torch.long)
    assert torch.equal(power.argmax(dim=1), expected_bins)
    peak_fraction = power.gather(1, expected_bins[:, None]).squeeze(1) / power.sum(dim=1)
    assert torch.all(peak_fraction > 1.0 - 1e-12)
    assert torch.unique(expected_bins).numel() == batch.frequencies.numel()


def test_narrowband_splits_share_dictionary_not_episode_phase() -> None:
    splits = generate_narrowband_source_splits(
        d=10,
        sigma=0.05,
        seq_len=32,
        split_sizes={"train": 11, "probe": 7, "eval": 5},
        split_seeds={"train": 101, "probe": 202, "eval": 303},
        emission_seed=44,
    )
    assert torch.equal(splits["train"].emissions, splits["probe"].emissions)
    assert torch.equal(splits["train"].emissions, splits["eval"].emissions)
    assert not torch.equal(splits["train"].phases[:5], splits["eval"].phases)


@pytest.mark.parametrize(
    ("frequencies", "match"),
    [
        ((0.1,), "at least two"),
        ((0.0, 0.2), "DC and Nyquist"),
        ((0.1, 0.15), "sufficiently separated"),
    ],
)
def test_narrowband_generator_rejects_non_advantage_frequency_grids(
    frequencies: tuple[float, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        generate_narrowband_sources(
            frequencies=frequencies,
            d=8,
            sigma=0.0,
            n_seq=2,
            seq_len=16,
            seed=0,
        )
