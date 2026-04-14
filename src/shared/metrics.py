"""Metrics for evaluating SAE quality.

Implements c_dec (decoder pairwise cosine similarity), decoder-feature
cosine similarity, variance explained, and latent-to-feature matching.
"""

import torch
from sae_lens import TopKTrainingSAE


def decoder_pairwise_cosine_similarity(sae: TopKTrainingSAE) -> float:
    """Compute c_dec: mean absolute pairwise cosine similarity of decoder columns.

    c_dec = (1 / C(h,2)) * sum_{i<j} |cos(W_dec_i, W_dec_j)|

    Lower values indicate more orthogonal (less mixed) decoder latents.
    Minimized at the correct L0.

    Args:
        sae: Trained SAE.

    Returns:
        Scalar c_dec value.
    """
    # W_dec is (d_sae, d_in) — rows are decoder vectors
    W_dec = sae.W_dec.data  # (d_sae, d_in)
    W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)
    cos_sim_matrix = W_dec_normed @ W_dec_normed.T  # (d_sae, d_sae)

    h = cos_sim_matrix.shape[0]
    # Extract upper triangle (excluding diagonal)
    mask = torch.triu(torch.ones(h, h, device=cos_sim_matrix.device), diagonal=1).bool()
    upper_vals = cos_sim_matrix[mask].abs()

    return upper_vals.mean().item()


def decoder_feature_cosine_similarity(
    sae: TopKTrainingSAE,
    feature_directions: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between each SAE decoder latent and each true feature.

    Used to produce heatmaps showing which true features each SAE latent has
    learned (or mixed).

    Args:
        sae: Trained SAE.
        feature_directions: Ground-truth feature direction matrix,
            shape (num_features, hidden_dim).

    Returns:
        Tensor of shape (d_sae, num_features) with cosine similarities.
    """
    # W_dec: (d_sae, d_in) — rows are decoder vectors
    W_dec = sae.W_dec.data  # (d_sae, d_in)
    # feature_directions: (num_feats, hidden_dim) — rows are feature vectors
    features = feature_directions.to(W_dec.device)  # (num_feats, d_in)

    W_dec_normed = W_dec / W_dec.norm(dim=1, keepdim=True)
    features_normed = features / features.norm(dim=1, keepdim=True)

    # (d_sae, num_feats)
    return (W_dec_normed @ features_normed.T).detach().cpu()


def variance_explained(
    true_activations: torch.Tensor,
    reconstructed_activations: torch.Tensor,
) -> float:
    """Compute variance explained: 1 - MSE(true, recon) / Var(true).

    Args:
        true_activations: Ground truth activations.
        reconstructed_activations: SAE reconstructions.

    Returns:
        Scalar variance explained value.
    """
    mse = (true_activations - reconstructed_activations).pow(2).mean()
    var = true_activations.var()
    return (1 - mse / var).item()


def match_sae_latents_to_features(
    cos_sim_matrix: torch.Tensor,
) -> torch.Tensor:
    """Reorder SAE latent indices to best match true feature indices.

    For each true feature, finds the SAE latent with highest absolute cosine
    similarity and returns the permutation.

    Args:
        cos_sim_matrix: Shape (d_sae, num_features) cosine similarity matrix.

    Returns:
        Permutation tensor of length d_sae mapping new index -> old index.
    """
    d_sae, num_features = cos_sim_matrix.shape
    abs_cos = cos_sim_matrix.abs()

    perm = []
    used = set()
    for feat_idx in range(num_features):
        scores = abs_cos[:, feat_idx].clone()
        for u in used:
            scores[u] = -1
        best = scores.argmax().item()
        perm.append(best)
        used.add(best)

    # Add any remaining latents not yet assigned
    for i in range(d_sae):
        if i not in used:
            perm.append(i)

    return torch.tensor(perm)
