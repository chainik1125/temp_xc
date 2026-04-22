"""Evaluator for single-position MatryoshkaSAE — the "no-window × matryoshka"
cell missing from Dmitry's separation_scaling sweep.

Mirrors the signature + return shape of
`evaluate_topk_on_activations` in his vendored
`run_transformer_standard_hmm_arch_sweep.py`, so the driver's dispatch
can call it like any other arch family and its output drops into
`results/cell_delta_*/results.json` with the same schema.

Only new ingredient: training loop using `MatryoshkaSAE.compute_loss`
(nested-reconstruction-loss weighted by `inner_weight`) instead of
plain MSE. Everything else — activation shaping, evaluation via
`evaluate_representation`, checkpoint saving — is identical to the
TopK path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

# Imports from Dmitry's vendored code — resolved via PYTHONPATH at runtime
# (see run_ablation.sh).
from sae_day.sae import MatryoshkaSAE  # type: ignore[import-not-found]
from run_standard_hmm_arch_seed_sweep import (  # type: ignore[import-not-found]
    evaluate_representation,
)
from run_transformer_standard_hmm_arch_sweep import (  # type: ignore[import-not-found]
    cpu_state_dict,
    save_checkpoint,
)


def train_matsae(
    sae: MatryoshkaSAE,
    flat_obs: torch.Tensor,
    n_steps: int,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
) -> list[float]:
    """Mirror of `train_topk` but calls `MatryoshkaSAE.compute_loss`.

    Returns the list of total-loss values per step (same contract as
    `train_topk`'s return, so the caller can pull `losses[-1]` as
    `final_loss`).
    """
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    device = next(sae.parameters()).device
    n_total = flat_obs.shape[0]
    losses: list[float] = []

    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        x = flat_obs[idx].to(device)
        total_loss, _info = sae.compute_loss(x)
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            sae._normalize_decoder()
        losses.append(float(total_loss.item()))
    return losses


def encode_all_matsae(model: MatryoshkaSAE, flat_obs: torch.Tensor, chunk_size: int = 4096) -> np.ndarray:
    """Batched encode, returning a (N, d_sae) numpy array."""
    device = next(model.parameters()).device
    model.eval()
    chunks = []
    with torch.no_grad():
        for i in range(0, flat_obs.shape[0], chunk_size):
            _, z = model(flat_obs[i : i + chunk_size].to(device))
            chunks.append(z.cpu())
    return torch.cat(chunks, dim=0).numpy()


def evaluate_matsae_on_activations(
    seed: int,
    arch_cfg: dict[str, Any],
    train_acts: torch.Tensor,
    eval_acts: torch.Tensor,
    eval_omega: torch.Tensor,
    d_model: int,
    device: torch.device,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    """Same contract as Dmitry's evaluate_topk_on_activations."""
    n_train = min(arch_cfg["n_sequences"], train_acts.shape[0])
    n_eval = min(arch_cfg["n_sequences"], eval_acts.shape[0])
    seq_len = min(arch_cfg["seq_len"], train_acts.shape[1], eval_acts.shape[1])

    arch_train_acts = train_acts[:n_train, :seq_len, :].contiguous()
    arch_eval_acts = eval_acts[:n_eval, :seq_len, :].contiguous()
    arch_eval_omega = eval_omega[:n_eval]

    flat_train = arch_train_acts.reshape(-1, d_model)
    flat_eval = arch_eval_acts.reshape(-1, d_model)
    target = (
        arch_eval_omega[:, None, :]
        .expand(n_eval, seq_len, arch_eval_omega.shape[-1])
        .reshape(-1, arch_eval_omega.shape[-1])
        .numpy()
    )

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = MatryoshkaSAE(
        d_in=d_model,
        d_sae=arch_cfg["dict_size"],
        k=arch_cfg["k"],
        matryoshka_widths=arch_cfg.get("matryoshka_widths"),
        inner_weight=arch_cfg.get("inner_weight", 1.0),
    ).to(device)

    losses = train_matsae(
        model,
        flat_train,
        n_steps=arch_cfg["sae_steps"],
        batch_size=arch_cfg.get("batch_size", 256),
        lr=arch_cfg.get("lr", 3e-4),
        seed=seed,
    )

    z_all = encode_all_matsae(model, flat_eval)
    metrics = evaluate_representation(
        z_all,
        target,
        n_sequences=n_eval,
        samples_per_sequence=seq_len,
        seed=seed,
    )

    run = {
        "seed": seed,
        "n_train_sequences": n_train,
        "n_eval_sequences": n_eval,
        "effective_seq_len": seq_len,
        "final_loss": float(losses[-1]),
        "metrics": metrics,
    }
    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            {
                "kind": "MatryoshkaSAE (no-window)",
                "seed": seed,
                "config": arch_cfg,
                "activation_shape": [int(v) for v in arch_train_acts.shape],
                "run": run,
                "state_dict": cpu_state_dict(model),
            },
        )
    return run
