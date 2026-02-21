"""Gradient-descent orthogonalization of embedding vectors.

Produces a set of unit vectors with pairwise cosine similarity close to a
target value, following the approach in Chanin et al. (2025).
"""

import torch


def orthogonalize(
    num_vectors: int,
    vector_len: int,
    target_cos_sim: float = 0.0,
    num_steps: int = 1000,
    lr: float = 0.01,
) -> torch.Tensor:
    """Optimize a set of vectors to have pairwise cosine similarity near target.

    Uses Adam to minimize the squared deviation of all pairwise dot products
    from the target, plus a penalty to keep vectors unit-norm.

    Args:
        num_vectors: Number of vectors to produce.
        vector_len: Dimensionality of each vector.
        target_cos_sim: Desired pairwise cosine similarity.
        num_steps: Optimization steps.
        lr: Learning rate for Adam.

    Returns:
        Tensor of shape (num_vectors, vector_len) with unit-norm rows.
    """
    embeddings = torch.randn(num_vectors, vector_len)
    embeddings /= embeddings.norm(p=2, dim=1, keepdim=True)
    embeddings.requires_grad_(True)

    optimizer = torch.optim.Adam([embeddings], lr=lr)

    for _ in range(num_steps):
        optimizer.zero_grad()

        dot_products = embeddings @ embeddings.T
        diff = dot_products - target_cos_sim
        diff.fill_diagonal_(0)
        loss = diff.pow(2).sum()
        loss += num_vectors * (dot_products.diag() - 1).pow(2).sum()

        loss.backward()
        optimizer.step()

    embeddings = (
        (embeddings / embeddings.norm(p=2, dim=1, keepdim=True)).detach().clone()
    )
    embeddings.requires_grad_(False)
    return embeddings.detach().clone()
