"""Initialize an SAE to match ground-truth feature directions."""

import torch
from sae_lens import TopKTrainingSAE


def init_sae_to_features(
    sae: TopKTrainingSAE,
    feature_directions: torch.Tensor,
    noise_level: float = 0.0,
) -> None:
    """Set SAE encoder/decoder weights to ground-truth feature directions.

    This creates the "ground-truth SAE" described in Chanin et al.:
    W_enc = F^T, W_dec = F, b_enc = 0, b_dec = 0.

    Args:
        sae: The SAE to initialize (modified in-place).
        feature_directions: Feature direction matrix, shape (num_features, hidden_dim).
        noise_level: Optional Gaussian noise std to add to weights.
    """
    features = feature_directions.clone()

    if noise_level > 0:
        features = features + noise_level * torch.randn_like(features)
        features = features / features.norm(dim=1, keepdim=True)

    with torch.no_grad():
        sae.W_enc.data = features.T.to(sae.W_enc.device)  # (d_in, d_sae)
        sae.W_dec.data = features.to(sae.W_dec.device)  # (d_sae, d_in)
        sae.b_enc.data.zero_()
        sae.b_dec.data.zero_()
