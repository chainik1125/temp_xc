"""Correlation matrix construction and PSD enforcement."""

import random

import torch


def _fix_correlation_matrix(
    matrix: torch.Tensor, min_eigenval: float = 1e-6
) -> torch.Tensor:
    """Clip eigenvalues to ensure positive semi-definiteness.

    Eigendecomposes the matrix, clamps negative eigenvalues, reconstructs,
    and renormalizes the diagonal to 1.
    """
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=min_eigenval)
    fixed_matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
    diag_vals = torch.diag(fixed_matrix)
    fixed_matrix = fixed_matrix / torch.sqrt(
        diag_vals.unsqueeze(0) * diag_vals.unsqueeze(1)
    )
    fixed_matrix.fill_diagonal_(1.0)
    return fixed_matrix


def create_correlation_matrix(
    num_features: int,
    correlations: dict[tuple[int, int], float] | None = None,
    default_correlation: float = 0.0,
) -> torch.Tensor:
    """Build a correlation matrix from explicit pairwise correlations.

    Args:
        num_features: Number of features.
        correlations: Dict mapping (i, j) pairs to correlation values.
        default_correlation: Default off-diagonal correlation.

    Returns:
        PSD correlation matrix of shape (num_features, num_features).
    """
    matrix = torch.eye(num_features) + default_correlation * (
        1 - torch.eye(num_features)
    )
    if correlations is not None:
        for (i, j), corr in correlations.items():
            matrix[i, j] = corr
            matrix[j, i] = corr

    eigenvals = torch.linalg.eigvals(matrix)
    if torch.any(eigenvals.real < -1e-6):
        print("Warning: Correlation matrix is not positive semi-definite!")
        print(f"Minimum eigenvalue: {eigenvals.real.min()}")
        print("Fixing matrix to be positive semi-definite...")
        matrix = _fix_correlation_matrix(matrix)
    return matrix


def generate_random_correlation_matrix(
    num_features: int,
    positive_ratio: float = 0.5,
    correlation_strength_range: tuple[float, float] = (0.3, 0.8),
    sparsity: float = 0.3,
    seed: int | None = None,
) -> torch.Tensor:
    """Generate a random PSD correlation matrix.

    Args:
        num_features: Number of features.
        positive_ratio: Fraction of non-zero correlations that are positive.
        correlation_strength_range: (min, max) absolute correlation strength.
        sparsity: Fraction of pairs that are uncorrelated (zero).
        seed: Random seed for reproducibility.

    Returns:
        PSD correlation matrix of shape (num_features, num_features).
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    matrix = torch.eye(num_features)
    pairs = [(i, j) for i in range(num_features) for j in range(i + 1, num_features)]
    total_pairs = len(pairs)
    if total_pairs == 0:
        return matrix

    num_sparse = int(total_pairs * sparsity)
    num_correlated = total_pairs - num_sparse
    if num_correlated == 0:
        return matrix

    correlated_pairs = random.sample(pairs, num_correlated)
    num_positive = int(num_correlated * positive_ratio)
    min_strength, max_strength = correlation_strength_range

    for i, (pair_i, pair_j) in enumerate(correlated_pairs):
        sign = 1 if i < num_positive else -1
        strength = random.uniform(min_strength, max_strength)
        correlation = sign * strength
        matrix[pair_i, pair_j] = correlation
        matrix[pair_j, pair_i] = correlation

    # Ensure PSD via eigenvalue clipping
    eigenvals, eigenvecs = torch.linalg.eigh(matrix)
    eigenvals = torch.clamp(eigenvals, min=1e-6)
    matrix = eigenvecs @ torch.diag(eigenvals) @ eigenvecs.T
    diag_sqrt = torch.sqrt(torch.diag(matrix))
    matrix = matrix / (diag_sqrt.unsqueeze(0) * diag_sqrt.unsqueeze(1))
    matrix.fill_diagonal_(1.0)
    return matrix
