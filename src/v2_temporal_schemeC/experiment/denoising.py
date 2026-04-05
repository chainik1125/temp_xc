"""Denoising metrics for HMM experiments (Exp 1c family).

Reusable functions for computing single-latent correlation and extracting
per-position latent activations from windowed models.
"""

import numpy as np
import torch

from src.v2_temporal_schemeC.feature_recovery import cos_sims


def pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation between two 1-D tensors."""
    x = x.float()
    y = y.float()
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = (xm.pow(2).sum() * ym.pow(2).sum()).sqrt()
    if den < 1e-12:
        return 0.0
    return (num / den).item()


@torch.no_grad()
def compute_global_recovery(
    spec,
    model,
    eval_x: torch.Tensor,
    true_features: torch.Tensor,
    eval_support: torch.Tensor,
    eval_hidden: torch.Tensor,
    *,
    num_features: int,
    seq_len: int,
    dict_width: int,
    device: torch.device,
    batch_size: int = 256,
) -> dict:
    """Compute per-feature local and global correlation for denoising analysis.

    For each true feature i, finds the best-matching latent j (highest
    |cos(decoder_j, f_i)|), then computes Pearson correlation of latent j's
    activations with the noisy observation s_i (local) and hidden state h_i
    (global).

    Args:
        spec: ModelSpec instance.
        model: Trained model.
        eval_x: Eval data, shape (n_eval, seq_len, d).
        true_features: Ground truth feature directions, shape (k, d).
        eval_support: Noisy observed support, shape (n_eval, k, seq_len).
        eval_hidden: True hidden states, shape (n_eval, k, seq_len).
        num_features: Number of ground truth features (k).
        seq_len: Sequence length (T).
        dict_width: Dictionary width (d_sae).
        device: Torch device.
        batch_size: Batch size for forward passes.

    Returns:
        Dict with per-feature correlations and means.
    """
    model.eval()
    n_eval = eval_x.shape[0]

    # Get decoder directions and match to true features
    if spec.n_decoder_positions is None:
        dd = spec.decoder_directions(model).to(device)
    else:
        dds = [spec.decoder_directions(model, pos=p).to(device)
               for p in range(spec.n_decoder_positions)]
        dd = torch.stack(dds).mean(dim=0)

    tf = true_features.T.to(device)
    sims = cos_sims(dd, tf).abs()
    best_latent = sims.argmax(dim=0)

    # Extract per-position latent activations
    if spec.data_format == "seq":
        all_z = []
        for s in range(0, n_eval, batch_size):
            x_batch = eval_x[s:min(s + batch_size, n_eval)]
            _, inter = model(x_batch)
            z = inter["novel_codes"] + inter["pred_codes"]
            all_z.append(z.cpu())
        all_z = torch.cat(all_z, dim=0)

        sup = eval_support.permute(0, 2, 1).float()
        hid = eval_hidden.permute(0, 2, 1).float()

        local_corrs, global_corrs = [], []
        for feat_i in range(num_features):
            j = best_latent[feat_i].item()
            z_j = all_z[:, :, j].reshape(-1)
            s_i = sup[:, :, feat_i].reshape(-1)
            h_i = hid[:, :, feat_i].reshape(-1)
            local_corrs.append(pearson(z_j, s_i))
            global_corrs.append(pearson(z_j, h_i))

    elif spec.data_format == "window":
        T_win = spec.n_decoder_positions
        all_z_per_pos = torch.zeros(n_eval, seq_len, dict_width)
        counts = torch.zeros(n_eval, seq_len)

        for t_start in range(seq_len - T_win + 1):
            windows = eval_x[:, t_start:t_start + T_win, :]
            for s in range(0, n_eval, batch_size):
                w = windows[s:min(s + batch_size, n_eval)]
                bs = w.shape[0]
                if hasattr(model, 'encode'):
                    z = model.encode(w).cpu()
                    for t_off in range(T_win):
                        pos = t_start + t_off
                        all_z_per_pos[s:s + bs, pos] += z
                        counts[s:s + bs, pos] += 1
                else:
                    _, _, z = model(w)
                    z = z.cpu()
                    for t_off in range(T_win):
                        pos = t_start + t_off
                        all_z_per_pos[s:s + bs, pos] += z[:, t_off]
                        counts[s:s + bs, pos] += 1

        all_z_per_pos /= counts.unsqueeze(-1).clamp(min=1)

        sup = eval_support.permute(0, 2, 1).float()
        hid = eval_hidden.permute(0, 2, 1).float()

        local_corrs, global_corrs = [], []
        for feat_i in range(num_features):
            j = best_latent[feat_i].item()
            z_j = all_z_per_pos[:, :, j].reshape(-1)
            s_i = sup[:, :, feat_i].reshape(-1)
            h_i = hid[:, :, feat_i].reshape(-1)
            local_corrs.append(pearson(z_j, s_i))
            global_corrs.append(pearson(z_j, h_i))

    else:
        raise ValueError(f"Unknown data_format: {spec.data_format}")

    local_corrs = np.array(local_corrs)
    global_corrs = np.array(global_corrs)

    return {
        "local_corrs": local_corrs.tolist(),
        "global_corrs": global_corrs.tolist(),
        "mean_local": float(np.mean(local_corrs)),
        "mean_global": float(np.mean(global_corrs)),
        "denoising_frac": float(np.mean(global_corrs > local_corrs)),
    }
