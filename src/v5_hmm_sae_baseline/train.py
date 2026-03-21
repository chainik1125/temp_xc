"""Training loop for the TopK SAE."""

import torch

from src.utils.logging import log
from src.v5_hmm_sae_baseline.metrics import feature_recovery_score
from src.v5_hmm_sae_baseline.sae import TopKSAE


def train_sae(
    sae: TopKSAE,
    x_flat: torch.Tensor,
    ground_truth_features: torch.Tensor,
    n_epochs: int = 50,
    lr: float = 1e-4,
    batch_size: int = 256,
    log_every: int = 10,
    rng: torch.Generator | None = None,
) -> dict:
    """Train a TopK SAE on flattened observation vectors.

    Args:
        sae: The TopK SAE model.
        x_flat: Observation vectors of shape (N, d_input).
        ground_truth_features: Ground-truth features of shape (k, d_input).
        n_epochs: Number of training epochs.
        lr: Learning rate for Adam.
        batch_size: Training batch size.
        log_every: Log every N epochs.
        rng: Optional torch RNG for reproducible shuffling.

    Returns:
        Dict with final metrics and training history:
            recon_loss, l0, auc, mean_max_cos_sim,
            frac_recovered_90, frac_recovered_80,
            history: list of per-epoch dicts with recon_loss and l0.
    """
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    N = x_flat.shape[0]
    history: list[dict] = []

    for epoch in range(n_epochs):
        # Shuffle data
        perm = torch.randperm(N, generator=rng)
        epoch_loss = 0.0
        epoch_l0 = 0.0
        n_batches = 0

        for start in range(0, N, batch_size):
            idx = perm[start : start + batch_size]
            batch = x_flat[idx]

            _, _, loss_dict = sae(batch)
            loss = loss_dict["recon_loss"]

            optimizer.zero_grad()
            loss.backward()
            sae.remove_parallel_grads()
            optimizer.step()
            sae.normalize_decoder()

            epoch_loss += loss_dict["recon_loss"].item()
            epoch_l0 += loss_dict["l0"].item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        avg_l0 = epoch_l0 / n_batches
        history.append({"epoch": epoch + 1, "recon_loss": avg_loss, "l0": avg_l0})

        if (epoch + 1) % log_every == 0:
            log(
                "train",
                f"epoch {epoch + 1}/{n_epochs}",
                recon_loss=avg_loss,
                l0=avg_l0,
            )

    # Final evaluation using Andre's threshold-sweep AUC
    with torch.no_grad():
        _, _, final_loss_dict = sae(x_flat[:min(N, 4096)])
        recovery = feature_recovery_score(sae.W_dec.data, ground_truth_features)

    results = {
        "recon_loss": final_loss_dict["recon_loss"].item(),
        "l0": final_loss_dict["l0"].item(),
        "auc": recovery["auc"],
        "mean_max_cos_sim": recovery["mean_max_cos_sim"],
        "frac_recovered_90": recovery["frac_recovered_90"],
        "frac_recovered_80": recovery["frac_recovered_80"],
        "history": history,
    }
    log("eval", "final metrics", recon_loss=results["recon_loss"],
        l0=results["l0"], auc=results["auc"],
        mmcs=results["mean_max_cos_sim"],
        r90=results["frac_recovered_90"])
    return results
