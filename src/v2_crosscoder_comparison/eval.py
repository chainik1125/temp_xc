"""Evaluation for crosscoder comparison experiment."""

from collections.abc import Callable
from dataclasses import asdict, dataclass

import torch

from src.utils.logging import log
from src.v2_crosscoder_comparison.architectures.base import BaseArchitecture
from src.v2_crosscoder_comparison.architectures.crosscoder import Crosscoder
from src.v2_crosscoder_comparison.architectures.naive_sae import NaiveSAE
from src.v2_crosscoder_comparison.architectures.stacked_sae import StackedSAE


@dataclass
class EvalResult:
    """Evaluation results for a single architecture run."""

    arch_type: str
    rho: float
    top_k: int
    seed: int
    l0: float
    mse: float
    fvu: float  # fraction of variance unexplained
    mean_max_cos_sim: float
    dead_latent_frac: float

    def to_dict(self) -> dict:
        return asdict(self)


def _cosine_similarity_matrix(
    W_dec: torch.Tensor,
    feature_directions: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine similarity between decoder vectors and true feature directions.

    Args:
        W_dec: Decoder weights, shape (d_sae, d_model).
        feature_directions: True features, shape (num_features, d_model).

    Returns:
        Cosine similarity matrix, shape (d_sae, num_features).
    """
    W_normed = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-8)
    F_normed = feature_directions / (feature_directions.norm(dim=-1, keepdim=True) + 1e-8)
    return W_normed @ F_normed.T


def evaluate(
    arch: BaseArchitecture,
    generate_hidden_fn: Callable[[int], torch.Tensor],
    feature_directions: torch.Tensor,
    arch_type: str,
    rho: float,
    top_k: int,
    seed: int,
    n_eval_samples: int = 100_000,
    batch_size: int = 4096,
) -> EvalResult:
    """Evaluate an architecture on reconstruction quality and feature recovery.

    Args:
        arch: Trained architecture.
        generate_hidden_fn: Hidden activation generator.
        feature_directions: True feature directions, shape (num_features, hidden_dim).
        arch_type: Architecture type string.
        rho: Cross-position correlation used.
        top_k: TopK value used.
        seed: Random seed used.
        n_eval_samples: Number of evaluation samples.
        batch_size: Evaluation batch size.

    Returns:
        EvalResult with all metrics.
    """
    device = feature_directions.device

    all_mse = []
    all_var = []
    all_l0 = []
    feature_fired = None

    n_processed = 0
    with torch.no_grad():
        while n_processed < n_eval_samples:
            current_batch = min(batch_size, n_eval_samples - n_processed)
            hidden = generate_hidden_fn(current_batch).to(device)
            recon = arch.forward(hidden)

            mse = (hidden - recon).pow(2).sum(dim=(-2, -1)).mean()
            var = hidden.var()
            all_mse.append(mse)
            all_var.append(var)

            # L0 and dead latent tracking
            if isinstance(arch, Crosscoder):
                z = arch.encode(hidden)  # (batch, d_sae)
                if feature_fired is None:
                    feature_fired = torch.zeros(z.shape[-1], device=device)
                feature_fired += (z > 0).float().sum(dim=0)
                all_l0.append((z > 0).float().sum(dim=-1).mean())
            elif isinstance(arch, NaiveSAE):
                batch, n_pos, d = hidden.shape
                flat = hidden.reshape(batch * n_pos, d)
                z = arch.sae.encode(flat)
                if feature_fired is None:
                    feature_fired = torch.zeros(z.shape[-1], device=device)
                feature_fired += (z > 0).float().sum(dim=0)
                all_l0.append((z > 0).float().sum(dim=-1).mean())
            elif isinstance(arch, StackedSAE):
                total_l0 = 0.0
                for t in range(2):
                    z_t = arch.saes[t].encode(hidden[:, t, :])
                    if feature_fired is None:
                        feature_fired = torch.zeros(z_t.shape[-1], device=device)
                    feature_fired += (z_t > 0).float().sum(dim=0)
                    total_l0 += (z_t > 0).float().sum(dim=-1).mean()
                all_l0.append(total_l0 / 2)

            n_processed += current_batch

    avg_mse = torch.stack(all_mse).mean().item()
    avg_var = torch.stack(all_var).mean().item()
    fvu = avg_mse / avg_var if avg_var > 0 else float("inf")
    avg_l0 = torch.stack(all_l0).mean().item()
    d_sae = feature_fired.shape[0]
    dead_latent_frac = (feature_fired == 0).float().mean().item()

    # Feature recovery: mean max cosine similarity
    mean_max_cos_sim = _compute_mean_max_cos_sim(arch, feature_directions)

    result = EvalResult(
        arch_type=arch_type,
        rho=rho,
        top_k=top_k,
        seed=seed,
        l0=avg_l0,
        mse=avg_mse,
        fvu=fvu,
        mean_max_cos_sim=mean_max_cos_sim,
        dead_latent_frac=dead_latent_frac,
    )

    log(
        "eval",
        arch_type,
        mean_max_cos_sim=mean_max_cos_sim,
        dead_latents=f"{int(dead_latent_frac * d_sae)}/{d_sae}",
        fvu=fvu,
        l0=avg_l0,
    )

    return result


def _compute_mean_max_cos_sim(
    arch: BaseArchitecture,
    feature_directions: torch.Tensor,
) -> float:
    """Compute mean max cosine similarity between decoder and true features.

    For each true feature, find the decoder vector with highest absolute cosine
    similarity, then average across features.

    For crosscoder: uses W_dec[:, t, :] per position, takes max across positions.
    """
    if isinstance(arch, Crosscoder):
        # W_dec: (d_sae, n_positions, d_model)
        W_dec = arch.get_decoder_weights()
        max_cos_sims = []
        for t in range(W_dec.shape[1]):
            W_dec_t = W_dec[:, t, :]  # (d_sae, d_model)
            cos_mat = _cosine_similarity_matrix(W_dec_t, feature_directions)
            max_cos_per_feat = cos_mat.abs().max(dim=0).values  # (num_features,)
            max_cos_sims.append(max_cos_per_feat)
        # Take max across positions for each feature
        stacked = torch.stack(max_cos_sims, dim=0)  # (n_positions, num_features)
        best_per_feat = stacked.max(dim=0).values  # (num_features,)
        return best_per_feat.mean().item()

    elif isinstance(arch, StackedSAE):
        W_decs = arch.get_decoder_weights()  # list of (d_sae, d_model)
        max_cos_sims = []
        for W_dec in W_decs:
            cos_mat = _cosine_similarity_matrix(W_dec, feature_directions)
            max_cos_per_feat = cos_mat.abs().max(dim=0).values
            max_cos_sims.append(max_cos_per_feat)
        stacked = torch.stack(max_cos_sims, dim=0)
        best_per_feat = stacked.max(dim=0).values
        return best_per_feat.mean().item()

    elif isinstance(arch, NaiveSAE):
        W_dec = arch.get_decoder_weights()  # (d_sae, d_model)
        cos_mat = _cosine_similarity_matrix(W_dec, feature_directions)
        max_cos_per_feat = cos_mat.abs().max(dim=0).values
        return max_cos_per_feat.mean().item()

    else:
        raise ValueError(f"Unknown architecture type: {type(arch)}")


def compute_pareto_frontier(results: list[EvalResult]) -> list[EvalResult]:
    """Filter results to Pareto-optimal points (L0 vs MSE).

    A point is Pareto-optimal if no other point has both lower L0 and lower MSE.

    Args:
        results: List of evaluation results.

    Returns:
        Pareto-optimal subset, sorted by L0.
    """
    sorted_results = sorted(results, key=lambda r: r.l0)
    pareto = []
    best_mse = float("inf")

    for r in sorted_results:
        if r.mse < best_mse:
            pareto.append(r)
            best_mse = r.mse

    return pareto
