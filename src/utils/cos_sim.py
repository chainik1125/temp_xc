import torch


def cos_sims(mat1: torch.Tensor, mat2: torch.Tensor) -> torch.Tensor:
    """Pairwise cosine similarity between columns of mat1 and mat2.

    Args:
        mat1: Tensor of shape (d, n1).
        mat2: Tensor of shape (d, n2).

    Returns:
        Tensor of shape (n1, n2) with cosine similarities.
    """
    mat1_normed = mat1 / mat1.norm(dim=0, keepdim=True)
    mat2_normed = mat2 / mat2.norm(dim=0, keepdim=True)
    return mat1_normed.T @ mat2_normed
