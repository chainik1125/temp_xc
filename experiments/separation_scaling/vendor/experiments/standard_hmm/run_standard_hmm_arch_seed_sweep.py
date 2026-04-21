from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from sae_day.data import HMMComponent, MESS3_SEPARATED_COMPONENTS
from sae_day.sae import MatryoshkaTemporalCrosscoder, TemporalCrosscoder, TopKSAE


@dataclass
class StandardHMMDataset:
    tokens: torch.Tensor
    observations: torch.Tensor
    sequence_omegas: torch.Tensor
    posterior_omegas: torch.Tensor
    mode: str
    vocab_mode: str


def slugify(name: str) -> str:
    return name.lower().replace(" ", "_")


def cpu_state_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu() for key, value in model.state_dict().items()}


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def train_topk(
    sae: TopKSAE,
    flat_obs: torch.Tensor,
    n_steps: int,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    device = next(sae.parameters()).device
    n_total = flat_obs.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        x = flat_obs[idx].to(device)
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            sae.normalize_decoder()
        losses.append(loss.item())
    return losses


def train_temporal_model(
    model: TemporalCrosscoder,
    flat_windows: torch.Tensor,
    n_steps: int,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    n_total = flat_windows.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        x = flat_windows[idx].to(device)
        loss, _ = model.compute_loss(x)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            model.normalize_decoder()
        losses.append(loss.item())
    return losses


def make_causal_windows(observations: torch.Tensor, window_size: int) -> torch.Tensor:
    windows = observations.unfold(1, window_size, 1)
    return windows.permute(0, 1, 3, 2).contiguous()


def encode_all_sae(model: TopKSAE, flat_obs: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, flat_obs.shape[0], chunk_size):
            _, z = model(flat_obs[i : i + chunk_size].to(device))
            chunks.append(z.cpu())
    return torch.cat(chunks, dim=0).numpy()


def encode_all_temporal_final_latents(
    model: TemporalCrosscoder,
    flat_windows: torch.Tensor,
    final_position: int = 0,
    chunk_size: int = 2048,
) -> np.ndarray:
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, flat_windows.shape[0], chunk_size):
            _, z = model(flat_windows[i : i + chunk_size].to(device))
            start = final_position * model.d_sae
            end = start + model.d_sae
            chunks.append(z[:, start:end].cpu())
    return torch.cat(chunks, dim=0).numpy()


def fit_linear_probe_r2(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_sequences: int,
    samples_per_sequence: int,
    seed: int = 42,
    ridge: float = 1e-8,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    train_seq = perm[:split]

    seq_is_train = np.zeros(n_sequences, dtype=bool)
    seq_is_train[train_seq] = True
    flat_train = np.repeat(seq_is_train, samples_per_sequence)
    flat_test = ~flat_train

    x_train = x_all[flat_train].astype(np.float64, copy=False)
    y_train = y_all[flat_train].astype(np.float64, copy=False)
    x_test = x_all[flat_test].astype(np.float64, copy=False)
    y_test = y_all[flat_test].astype(np.float64, copy=False)

    x_mean = x_train.mean(axis=0, keepdims=True)
    y_mean = y_train.mean(axis=0, keepdims=True)
    xc = x_train - x_mean
    yc = y_train - y_mean

    xtx = xc.T @ xc
    xty = xc.T @ yc
    xtx.flat[:: xtx.shape[0] + 1] += ridge
    beta = np.linalg.solve(xtx, xty)

    pred = (x_test - x_mean) @ beta + y_mean
    sse = ((y_test - pred) ** 2).sum(axis=0)
    sst = ((y_test - y_test.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - sse / np.clip(sst, 1e-12, None)
    return {
        "per_component_r2": [float(v) for v in r2],
        "mean_r2": float(r2.mean()),
    }


def compute_single_feature_probe_scores(
    z_all: np.ndarray,
    y_all: np.ndarray,
    n_sequences: int,
    samples_per_sequence: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    train_seq = perm[:split]

    seq_is_train = np.zeros(n_sequences, dtype=bool)
    seq_is_train[train_seq] = True
    flat_train = np.repeat(seq_is_train, samples_per_sequence)
    flat_test = ~flat_train

    x_train = z_all[flat_train].astype(np.float64, copy=False)
    y_train = y_all[flat_train].astype(np.float64, copy=False)
    x_test = z_all[flat_test].astype(np.float64, copy=False)
    y_test = y_all[flat_test].astype(np.float64, copy=False)

    n_train = x_train.shape[0]
    n_test = x_test.shape[0]

    x_mean = x_train.mean(axis=0)
    y_mean = y_train.mean(axis=0)
    x_center = x_train - x_mean
    y_center = y_train - y_mean
    x_var = (x_center**2).mean(axis=0)
    cov = (x_center.T @ y_center) / n_train

    slopes = cov / np.clip(x_var[:, None], 1e-12, None)
    intercepts = y_mean[None, :] - slopes * x_mean[:, None]

    sum_x = x_test.sum(axis=0)
    sum_x2 = (x_test**2).sum(axis=0)
    sum_y = y_test.sum(axis=0)
    sum_y2 = (y_test**2).sum(axis=0)
    sum_xy = x_test.T @ y_test

    sse = (
        sum_y2[None, :]
        - 2.0 * slopes * sum_xy
        - 2.0 * intercepts * sum_y[None, :]
        + (slopes**2) * sum_x2[:, None]
        + 2.0 * slopes * intercepts * sum_x[:, None]
        + n_test * (intercepts**2)
    )
    y_mean_test = sum_y / n_test
    sst = sum_y2 - n_test * (y_mean_test**2)
    r2 = 1.0 - sse / np.clip(sst[None, :], 1e-12, None)

    x_mean_test = sum_x / n_test
    x_var_test = sum_x2 / n_test - x_mean_test**2
    y_var_test = sum_y2 / n_test - y_mean_test**2
    cov_test = sum_xy / n_test - x_mean_test[:, None] * y_mean_test[None, :]
    corr = cov_test / np.sqrt(
        np.clip(x_var_test[:, None], 1e-12, None) * np.clip(y_var_test[None, :], 1e-12, None)
    )
    return r2, corr


def summarize_single_feature_probe(
    z_all: np.ndarray,
    y_all: np.ndarray,
    n_sequences: int,
    samples_per_sequence: int,
    top_k: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    r2, corr = compute_single_feature_probe_scores(
        z_all,
        y_all,
        n_sequences=n_sequences,
        samples_per_sequence=samples_per_sequence,
        seed=seed,
    )
    best_component = np.argmax(r2, axis=1)
    best_r2 = r2[np.arange(r2.shape[0]), best_component]
    best_corr = corr[np.arange(r2.shape[0]), best_component]
    best_feature_per_component = np.argmax(r2, axis=0)
    best_r2_per_component = r2[best_feature_per_component, np.arange(r2.shape[1])]
    best_corr_per_component = corr[best_feature_per_component, np.arange(corr.shape[1])]
    joint_mean_r2 = r2.mean(axis=1)
    best_joint_feature = int(np.argmax(joint_mean_r2))
    best_joint_component_r2 = r2[best_joint_feature]
    best_joint_component_corr = corr[best_joint_feature]
    active_frac = (z_all > 0).mean(axis=0)
    order = np.argsort(-best_r2)[:top_k]

    top_features = []
    for rank, feat_idx in enumerate(order, start=1):
        top_features.append(
            {
                "rank": rank,
                "feature": int(feat_idx),
                "target_component": int(best_component[feat_idx]),
                "heldout_r2": float(best_r2[feat_idx]),
                "heldout_corr": float(best_corr[feat_idx]),
                "active_frac": float(active_frac[feat_idx]),
            }
        )
    return {
        "best_feature_r2": float(best_r2.max()),
        "mean_best_r2": float(best_r2.mean()),
        "per_component_best_feature": [int(v) for v in best_feature_per_component],
        "per_component_best_r2": [float(v) for v in best_r2_per_component],
        "per_component_best_corr": [float(v) for v in best_corr_per_component],
        "best_joint_feature": best_joint_feature,
        "best_joint_mean_r2": float(joint_mean_r2[best_joint_feature]),
        "best_joint_component_r2": [float(v) for v in best_joint_component_r2],
        "best_joint_component_corr": [float(v) for v in best_joint_component_corr],
        "top_features": top_features,
    }


def evaluate_representation(
    x_all: np.ndarray,
    y_all: np.ndarray,
    n_sequences: int,
    samples_per_sequence: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "single_feature_probe": summarize_single_feature_probe(
            x_all,
            y_all,
            n_sequences=n_sequences,
            samples_per_sequence=samples_per_sequence,
            seed=seed,
        ),
        "linear_probe": fit_linear_probe_r2(
            x_all,
            y_all,
            n_sequences=n_sequences,
            samples_per_sequence=samples_per_sequence,
            seed=seed,
        ),
    }


ARCH_CONFIGS: dict[str, dict[str, Any]] = {
    "TopK SAE": {
        "family": "topk",
        "n_sequences": 1000,
        "seq_len": 300,
        "dict_size": 64,
        "k": 4,
        "sae_steps": 2000,
    },
    "TXC": {
        "family": "txc",
        "n_sequences": 1000,
        "seq_len": 600,
        "dict_size": 64,
        "k": 1,
        "window_size": 30,
        "temporal_steps": 3000,
    },
    "MatryoshkaTXC": {
        "family": "mattxc",
        "n_sequences": 400,
        "seq_len": 400,
        "dict_size": 128,
        "k": 1,
        "window_size": 60,
        "fixed_k_total": 10,
        "inner_weight": 40.0,
        "temporal_steps": 1500,
    },
    "MultiLayerCrosscoder": {
        "family": "mlxc",
        "n_sequences": 400,
        "seq_len": 400,
        "dict_size": 64,
        "k": 1,  # k_per_layer; total = L * k if fixed_k_total is not set
        "fixed_k_total": 8,
        "temporal_steps": 2000,
    },
    "MatryoshkaMultiLayerCrosscoder": {
        "family": "matmlxc",
        "n_sequences": 400,
        "seq_len": 400,
        "dict_size": 128,
        "k": 1,
        "fixed_k_total": 10,
        "inner_weight": 10.0,
        "temporal_steps": 1500,
    },
    # Temporal Matryoshka BatchTopK SAE — port from
    # github.com/AI4LIFE-GROUP/temporal-saes (Apache 2.0).
    # Single-position SAE trained on (x_t, x_{t+1}) pairs with a temporal
    # regulariser that pushes adjacent-activation features to agree,
    # weighted by the cosine similarity of the raw activations.
    "Temporal BatchTopK SAE": {
        "family": "tsae",
        "n_sequences": 1000,
        "seq_len": 200,
        "dict_size": 128,
        "k": 10,                        # BatchTopK: total active features per batch = k * batch_size
        "group_fractions": [0.5, 0.5],  # matryoshka split
        "group_weights": [0.5, 0.5],
        "temporal_alpha": 0.1,
        "temporal": True,
        "temporal_steps": 2000,
        "lr": 3e-4,
    },
    "TFA": {
        "family": "tfa",
        "n_sequences": 1000,
        "seq_len": 128,
        "dict_size": 128,
        "k": 10,
        "sae_diff_type": "topk",
        "n_heads": 4,
        "n_attn_layers": 1,
        "bottleneck_factor": 1,
        "tied_weights": True,
        "use_pos_encoding": False,
        "tfa_steps": 3000,
        "batch_size": 64,
        "lr": 1e-3,
        "min_lr": 9e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 200,
    },
    "TFA-pos": {
        "family": "tfa",
        "n_sequences": 1000,
        "seq_len": 128,
        "dict_size": 128,
        "k": 10,
        "sae_diff_type": "topk",
        "n_heads": 4,
        "n_attn_layers": 1,
        "bottleneck_factor": 1,
        "tied_weights": True,
        "use_pos_encoding": True,
        "tfa_steps": 3000,
        "batch_size": 64,
        "lr": 1e-3,
        "min_lr": 9e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 200,
    },
}

VARIANTS: dict[str, dict[str, str]] = {
    "mixed_shared": {
        "mode": "mixed_initial_belief",
        "vocab_mode": "shared",
        "label": "Mixed Init Belief + Shared Vocab",
    },
    "single_shared": {
        "mode": "single_component",
        "vocab_mode": "shared",
        "label": "Single Component + Shared Vocab",
    },
    "single_distinct": {
        "mode": "single_component",
        "vocab_mode": "distinct",
        "label": "Single Component + Distinct Vocab",
    },
}


def _sample_dirichlet_rows(alpha: torch.Tensor, n_sequences: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    samples = rng.gamma(
        shape=np.asarray(alpha, dtype=np.float64),
        scale=1.0,
        size=(n_sequences, alpha.numel()),
    )
    samples /= samples.sum(axis=1, keepdims=True)
    return torch.tensor(samples, dtype=torch.float32)


def build_global_operators(components: list[HMMComponent], vocab_mode: str) -> torch.Tensor:
    dims = [comp.d for comp in components]
    total_d = sum(dims)
    if vocab_mode == "shared":
        v_total = components[0].V
        ops = torch.zeros(v_total, total_d, total_d, dtype=torch.float32)
        state_offset = 0
        for comp in components:
            for v in range(comp.V):
                ops[v, state_offset : state_offset + comp.d, state_offset : state_offset + comp.d] = (
                    comp.transition_matrices[v]
                )
            state_offset += comp.d
        return ops
    if vocab_mode == "distinct":
        v_total = sum(comp.V for comp in components)
        ops = torch.zeros(v_total, total_d, total_d, dtype=torch.float32)
        state_offset = 0
        token_offset = 0
        for comp in components:
            for v in range(comp.V):
                ops[token_offset + v, state_offset : state_offset + comp.d, state_offset : state_offset + comp.d] = (
                    comp.transition_matrices[v]
                )
            token_offset += comp.V
            state_offset += comp.d
        return ops
    raise ValueError(f"Unknown vocab_mode: {vocab_mode}")


def block_mass(eta: torch.Tensor, dims: list[int]) -> torch.Tensor:
    masses = []
    offset = 0
    for d in dims:
        masses.append(eta[:, offset : offset + d].sum(dim=1))
        offset += d
    return torch.stack(masses, dim=1)


def generate_standard_hmm_data(
    *,
    components: list[HMMComponent],
    omega_prior: list[float],
    n_sequences: int,
    seq_len: int,
    mode: str,
    vocab_mode: str,
    seed: int,
    concentration: float,
) -> StandardHMMDataset:
    gen = torch.Generator().manual_seed(seed)
    ops = build_global_operators(components, vocab_mode)
    dims = [comp.d for comp in components]
    total_d = sum(dims)
    k_total = len(components)
    v_total = ops.shape[0]
    prior = torch.tensor(omega_prior, dtype=torch.float32)

    if mode == "mixed_initial_belief":
        sequence_omegas = _sample_dirichlet_rows(prior * concentration, n_sequences=n_sequences, seed=seed)
        eta = torch.zeros(n_sequences, total_d, dtype=torch.float32)
        state_offset = 0
        for block_idx, d in enumerate(dims):
            eta[:, state_offset : state_offset + d] = sequence_omegas[:, block_idx : block_idx + 1] / d
            state_offset += d
    elif mode == "single_component":
        chosen = torch.multinomial(prior.expand(n_sequences, -1), 1, generator=gen).squeeze(1)
        sequence_omegas = torch.nn.functional.one_hot(chosen, num_classes=k_total).float()
        eta = torch.zeros(n_sequences, total_d, dtype=torch.float32)
        state_offset = 0
        for block_idx, d in enumerate(dims):
            mask = chosen == block_idx
            if mask.any():
                eta[mask, state_offset : state_offset + d] = 1.0 / d
            state_offset += d
    else:
        raise ValueError(f"Unknown mode: {mode}")

    tokens = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    posterior_omegas = torch.zeros(n_sequences, seq_len, k_total, dtype=torch.float32)
    ones = torch.ones(total_d, dtype=torch.float32)

    for t in range(seq_len):
        probs = torch.einsum("nd,vde,e->nv", eta, ops, ones)
        probs = probs.clamp(min=1e-12)
        probs = probs / probs.sum(dim=1, keepdim=True)
        tok = torch.multinomial(probs, 1, generator=gen).squeeze(1)
        tokens[:, t] = tok

        selected_ops = ops[tok]
        new_unnorm = torch.bmm(eta.unsqueeze(1), selected_ops).squeeze(1)
        eta = new_unnorm / new_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-12)
        posterior_omegas[:, t] = block_mass(eta, dims)

    observations = torch.nn.functional.one_hot(tokens, num_classes=v_total).float()
    return StandardHMMDataset(
        tokens=tokens,
        observations=observations,
        sequence_omegas=sequence_omegas,
        posterior_omegas=posterior_omegas,
        mode=mode,
        vocab_mode=vocab_mode,
    )


def evaluate_topk(
    seed: int,
    cfg: dict[str, Any],
    dataset: StandardHMMDataset,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    observations = dataset.observations
    sequence_omega = dataset.sequence_omegas

    flat_obs = observations.reshape(-1, observations.shape[-1])
    target = (
        sequence_omega[:, None, :]
        .expand(cfg["n_sequences"], cfg["seq_len"], sequence_omega.shape[-1])
        .reshape(-1, sequence_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    topk = TopKSAE(d_in=observations.shape[-1], d_sae=cfg["dict_size"], k=cfg["k"]).to(device)
    losses = train_topk(topk, flat_obs, n_steps=cfg["sae_steps"], seed=seed)
    z_all = encode_all_sae(topk, flat_obs)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=cfg["n_sequences"],
        samples_per_sequence=cfg["seq_len"],
        seed=seed,
    )
    run = {
        "seed": seed,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "TopK SAE",
                "seed": seed,
                "config": cfg,
                "dataset": {
                    "mode": dataset.mode,
                    "vocab_mode": dataset.vocab_mode,
                    "n_sequences": int(dataset.tokens.shape[0]),
                    "seq_len": int(dataset.tokens.shape[1]),
                    "d_vocab": int(dataset.observations.shape[-1]),
                },
                "run": run,
                "state_dict": cpu_state_dict(topk),
            },
        )
    return run


def evaluate_txc_family(
    seed: int,
    cfg: dict[str, Any],
    dataset: StandardHMMDataset,
    device: torch.device,
    *,
    matryoshka: bool,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    observations = dataset.observations
    sequence_omega = dataset.sequence_omegas

    window_size = cfg["window_size"]
    windows = make_causal_windows(observations, window_size)
    n_sequences, n_windows, _, d_in = windows.shape
    flat_windows = windows.reshape(-1, window_size, d_in)
    target = (
        sequence_omega[:, None, :]
        .expand(n_sequences, n_windows, sequence_omega.shape[-1])
        .reshape(-1, sequence_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if matryoshka:
        widths = [w for w in [8, 16, 32, 64, cfg["dict_size"]] if w <= cfg["dict_size"]]
        model = MatryoshkaTemporalCrosscoder(
            d_in=d_in,
            d_sae=cfg["dict_size"],
            T=window_size,
            k_per_pos=cfg["k"],
            k_total=cfg.get("fixed_k_total"),
            matryoshka_widths=widths,
            inner_weight=cfg["inner_weight"],
        ).to(device)
    else:
        model = TemporalCrosscoder(
            d_in=d_in,
            d_sae=cfg["dict_size"],
            T=window_size,
            k_per_pos=cfg["k"],
            k_total=cfg.get("fixed_k_total"),
        ).to(device)

    losses = train_temporal_model(model, flat_windows, n_steps=cfg["temporal_steps"], seed=seed)
    z_all = encode_all_temporal_final_latents(model, flat_windows, final_position=0)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_sequences,
        samples_per_sequence=n_windows,
        seed=seed,
    )
    run = {
        "seed": seed,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "MatryoshkaTXC" if matryoshka else "TXC",
                "seed": seed,
                "config": cfg,
                "dataset": {
                    "mode": dataset.mode,
                    "vocab_mode": dataset.vocab_mode,
                    "n_sequences": int(dataset.tokens.shape[0]),
                    "seq_len": int(dataset.tokens.shape[1]),
                    "d_vocab": int(dataset.observations.shape[-1]),
                },
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run


def summarize_runs(runs: list[dict[str, Any]]) -> dict[str, float]:
    single = [run["metrics"]["single_feature_probe"]["best_feature_r2"] for run in runs]
    linear = [run["metrics"]["linear_probe"]["mean_r2"] for run in runs]
    return {
        "single_feature_mean_r2": float(np.mean(single)),
        "single_feature_std_r2": float(np.std(single, ddof=1)) if len(single) > 1 else 0.0,
        "linear_mean_r2": float(np.mean(linear)),
        "linear_std_r2": float(np.std(linear, ddof=1)) if len(linear) > 1 else 0.0,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--concentration", type=float, default=10.0)
    parser.add_argument("--save-checkpoints", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "checkpoints",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path(__file__).parent / "outputs" / "standard_hmm_arch_seed_sweep.json",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    if args.save_checkpoints:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=== Standard HMM Architecture Seed Sweep ===")
    print(f"Device={device}")

    all_results: dict[str, Any] = {
        "target": "SequenceOmega",
        "architectures": ARCH_CONFIGS,
        "seeds": args.seeds,
        "variants": {},
    }

    for variant_name, variant_cfg in VARIANTS.items():
        print(f"\n=== Variant: {variant_name} ===")
        variant_record: dict[str, Any] = {
            "mode": variant_cfg["mode"],
            "vocab_mode": variant_cfg["vocab_mode"],
            "label": variant_cfg["label"],
            "architectures": {},
        }
        for arch_name, arch_cfg in ARCH_CONFIGS.items():
            print(f"\n--- {arch_name} ---")
            runs = []
            for seed in args.seeds:
                print(f"seed={seed}")
                dataset = generate_standard_hmm_data(
                    components=MESS3_SEPARATED_COMPONENTS,
                    omega_prior=[0.4, 0.35, 0.25],
                    n_sequences=arch_cfg["n_sequences"],
                    seq_len=arch_cfg["seq_len"],
                    mode=variant_cfg["mode"],
                    vocab_mode=variant_cfg["vocab_mode"],
                    seed=seed,
                    concentration=args.concentration,
                )
                checkpoint_path = None
                if args.save_checkpoints:
                    checkpoint_path = args.checkpoint_dir / variant_name / slugify(arch_name) / f"seed_{seed}.pt"
                if arch_cfg["family"] == "topk":
                    run = evaluate_topk(seed, arch_cfg, dataset, device, checkpoint_path=checkpoint_path)
                elif arch_cfg["family"] == "txc":
                    run = evaluate_txc_family(
                        seed,
                        arch_cfg,
                        dataset,
                        device,
                        matryoshka=False,
                        checkpoint_path=checkpoint_path,
                    )
                elif arch_cfg["family"] == "mattxc":
                    run = evaluate_txc_family(
                        seed,
                        arch_cfg,
                        dataset,
                        device,
                        matryoshka=True,
                        checkpoint_path=checkpoint_path,
                    )
                else:
                    raise ValueError(f"Unknown family: {arch_cfg['family']}")
                single = run["metrics"]["single_feature_probe"]["best_feature_r2"]
                linear = run["metrics"]["linear_probe"]["mean_r2"]
                print(f"  single={single:.4f}  linear={linear:.4f}")
                runs.append(run)

            variant_record["architectures"][arch_name] = {
                "config": arch_cfg,
                "runs": runs,
                "summary": summarize_runs(runs),
            }
        all_results["variants"][variant_name] = variant_record

    args.output_json.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved {args.output_json}")


if __name__ == "__main__":
    main()
