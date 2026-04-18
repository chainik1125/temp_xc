"""Denoising metrics for HMM experiments (Exp 1c family).

Reusable functions for computing single-latent correlation, linear probes,
and extracting per-position latent activations from windowed models.
"""

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from src.eval.feature_recovery import cos_sims


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


# ── Latent extraction ──

@torch.no_grad()
def extract_latents_tfa(
    model, eval_x: torch.Tensor, dict_width: int, batch_size: int = 256,
) -> np.ndarray:
    """Extract per-token latent activations from TFA. Returns (n_tokens, d_sae)."""
    all_z = []
    for s in range(0, eval_x.shape[0], batch_size):
        x = eval_x[s:s + batch_size]
        _, inter = model(x)
        z = inter["novel_codes"] + inter["pred_codes"]
        all_z.append(z.cpu())
    return torch.cat(all_z, dim=0).reshape(-1, dict_width).numpy()


@torch.no_grad()
def extract_latents_windowed(
    model, eval_x: torch.Tensor, T_win: int, dict_width: int,
    seq_len: int, is_crosscoder: bool, batch_size: int = 256,
) -> np.ndarray:
    """Extract per-position latent activations from windowed models.

    Averages z across overlapping windows for each position.
    Returns (n_tokens, d_sae).
    """
    n_eval = eval_x.shape[0]
    z_sum = torch.zeros(n_eval, seq_len, dict_width)
    counts = torch.zeros(n_eval, seq_len)

    for t_start in range(seq_len - T_win + 1):
        windows = eval_x[:, t_start:t_start + T_win, :]
        for s in range(0, n_eval, batch_size):
            w = windows[s:s + batch_size]
            bs = w.shape[0]
            if is_crosscoder:
                z = model.encode(w).cpu()
                for t_off in range(T_win):
                    z_sum[s:s + bs, t_start + t_off] += z
                    counts[s:s + bs, t_start + t_off] += 1
            else:
                _, _, z = model(w)
                z = z.cpu()
                for t_off in range(T_win):
                    z_sum[s:s + bs, t_start + t_off] += z[:, t_off]
                    counts[s:s + bs, t_start + t_off] += 1

    z_sum /= counts.unsqueeze(-1).clamp(min=1)
    return z_sum.reshape(-1, dict_width).numpy()


# ── Linear probe ──

def run_linear_probes(
    z: np.ndarray,
    support: torch.Tensor,
    hidden_states: torch.Tensor,
    num_features: int,
    alpha: float = 1.0,
) -> dict:
    """Train Ridge probes z->s_i and z->h_i for each feature.

    Args:
        z: (n_tokens, d_sae) latent activations.
        support: (n_eval, k, T) observed binary support.
        hidden_states: (n_eval, k, T) true hidden states.
        num_features: Number of ground truth features (k).
        alpha: Ridge regularization.

    Returns:
        Dict with per-feature and mean R² for local and global probes.
    """
    sup = support.permute(0, 2, 1).reshape(-1, num_features).numpy()
    hid = hidden_states.permute(0, 2, 1).reshape(-1, num_features).numpy()

    n = z.shape[0]
    split = int(0.8 * n)
    z_train, z_test = z[:split], z[split:]
    sup_train, sup_test = sup[:split], sup[split:]
    hid_train, hid_test = hid[:split], hid[split:]

    local_r2s, global_r2s = [], []
    for feat_i in range(num_features):
        probe_local = Ridge(alpha=alpha)
        probe_local.fit(z_train, sup_train[:, feat_i])
        local_r2s.append(r2_score(sup_test[:, feat_i],
                                   probe_local.predict(z_test)))

        probe_global = Ridge(alpha=alpha)
        probe_global.fit(z_train, hid_train[:, feat_i])
        global_r2s.append(r2_score(hid_test[:, feat_i],
                                    probe_global.predict(z_test)))

    return {
        "local_r2": local_r2s,
        "global_r2": global_r2s,
        "mean_local_r2": float(np.mean(local_r2s)),
        "mean_global_r2": float(np.mean(global_r2s)),
        "ratio": float(np.mean(global_r2s)) / max(float(np.mean(local_r2s)), 1e-12),
    }


# ── Single-target correlation (for coupled features) ──

def compute_correlation_against_targets(
    latent_activations: np.ndarray,
    target_sequences: torch.Tensor,
    decoder_directions: torch.Tensor,
    target_features: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute per-feature Pearson correlation between latents and targets.

    Matches each target feature to the best decoder column, then computes
    Pearson(z_j, target_i) for each matched pair.

    Args:
        latent_activations: (n_tokens, d_sae) from extract_latents_*.
        target_sequences: (n_eval, n_targets, T) binary target sequences.
        decoder_directions: (d, d_sae) decoder weight matrix.
        target_features: (n_targets, d) ground truth directions to match against.
        device: torch device for cosine similarity.

    Returns:
        Dict with per-feature correlations and mean.
    """
    dd = decoder_directions.to(device)
    tf = target_features.T.to(device)  # (d, n_targets)
    sims = cos_sims(dd, tf).abs()      # (d_sae, n_targets)
    best_latent = sims.argmax(dim=0)   # (n_targets,)

    n_targets = target_sequences.shape[1]
    targets = target_sequences.permute(0, 2, 1).reshape(-1, n_targets).float().numpy()

    corrs = []
    for i in range(n_targets):
        j = best_latent[i].item()
        z_j = latent_activations[:, j]
        t_i = targets[:, i]
        corrs.append(_np_pearson(z_j, t_i))

    corrs = np.array(corrs)
    return {
        "corrs": corrs.tolist(),
        "mean": float(np.mean(corrs)),
    }


def _np_pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = np.sqrt((xm**2).sum() * (ym**2).sum())
    if den < 1e-12:
        return 0.0
    return float(num / den)


def run_linear_probes_general(
    z: np.ndarray,
    targets: torch.Tensor,
    num_targets: int,
    alpha: float = 1.0,
) -> dict:
    """Train Ridge probes z -> target_i for each target.

    Args:
        z: (n_tokens, d_sae) latent activations.
        targets: (n_eval, n_targets, T) target sequences.
        num_targets: number of targets.
        alpha: Ridge regularization.

    Returns:
        Dict with per-target R² and mean.
    """
    t = targets.permute(0, 2, 1).reshape(-1, num_targets).numpy()
    n = z.shape[0]
    split = int(0.8 * n)
    z_train, z_test = z[:split], z[split:]
    t_train, t_test = t[:split], t[split:]

    r2s = []
    for i in range(num_targets):
        probe = Ridge(alpha=alpha)
        probe.fit(z_train, t_train[:, i])
        r2s.append(r2_score(t_test[:, i], probe.predict(z_test)))

    return {
        "r2": r2s,
        "mean_r2": float(np.mean(r2s)),
    }
