"""Evaluation utilities for the aliased paired-feature benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch.nn as nn
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from ..metrics import compute_nmse, feature_recovery
from ..models.base import ModelOutput, TemporalAE
from .aliased_data import AliasedBatch


@dataclass
class AliasedEvalMetrics:
    nmse: float
    novel_l0: float
    pred_l0: float
    total_l0: float
    auc: float
    r_at_90: float
    probe_auc: float
    corr_local: float
    corr_pred: float
    delta: float
    n_informative: int

    def to_dict(self) -> dict:
        return asdict(self)


def _l0_sums(model: TemporalAE, out: ModelOutput) -> tuple[float, float, float]:
    if "novel_codes" in out.aux and "pred_codes" in out.aux:
        novel_mask = (out.aux["novel_codes"] > 0).float()
        pred_mask = (out.aux["pred_codes"].abs() > 1e-8).float()
        total_mask = torch.maximum(novel_mask, pred_mask)
        return (
            novel_mask.sum().item(),
            pred_mask.sum().item(),
            total_mask.sum().item(),
        )

    metric_latents = model.latents_for_metrics(out)
    total = (metric_latents != 0).float().sum().item()
    return total, 0.0, total


def _bias_free_reconstruction(
    model: TemporalAE,
    x_hat: torch.Tensor,
    *,
    pos: int | None = None,
) -> torch.Tensor:
    bias = model.reconstruction_bias(pos)
    if bias is None:
        return x_hat
    if x_hat.dim() == 3:
        return x_hat - bias.view(1, 1, -1)
    return x_hat - bias.view(1, -1)


def _pair_alignment_sums(
    bias_free_x_hat: torch.Tensor,
    meta: AliasedBatch,
    pair_dirs: torch.Tensor,
) -> tuple[float, float, int]:
    coeffs = torch.einsum("btd,gid->btgi", bias_free_x_hat, pair_dirs)
    norms = coeffs.norm(dim=-1).clamp(min=1e-8)

    current_bits = (meta.current_feature_idx.transpose(1, 2) % 2).unsqueeze(-1)
    next_bits = (meta.next_feature_idx.transpose(1, 2) % 2).unsqueeze(-1)
    local = coeffs.gather(-1, current_bits).squeeze(-1).abs() / norms
    pred = coeffs.gather(-1, next_bits).squeeze(-1).abs() / norms

    informative = meta.informative_mask.transpose(1, 2)
    local_sum = local[informative].sum().item()
    pred_sum = pred[informative].sum().item()
    informative_n = int(informative.sum().item())

    return local_sum, pred_sum, informative_n


class _LinearProbe(nn.Module):
    """Per-position linear probe from latents to visible support."""

    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)


def _train_and_eval_probe(
    latents_train: torch.Tensor,
    support_train: torch.Tensor,
    latents_test: torch.Tensor,
    support_test: torch.Tensor,
    *,
    n_steps: int,
    lr: float,
    batch_size: int,
) -> float:
    device = latents_train.device
    probe = _LinearProbe(latents_train.shape[-1], support_train.shape[-1]).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    n_train = latents_train.shape[0]
    batch_size = min(batch_size, n_train)
    probe.train()
    for _ in range(n_steps):
        idx = torch.randint(n_train, (batch_size,), device=device)
        logits = probe(latents_train[idx])
        loss = F.binary_cross_entropy_with_logits(logits, support_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        logits = probe(latents_test)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        targets = support_test.detach().cpu().numpy()

    aucs = []
    for feature_idx in range(targets.shape[-1]):
        target = targets[:, feature_idx]
        if target.sum() == 0 or target.sum() == len(target):
            continue
        aucs.append(roc_auc_score(target, probs[:, feature_idx]))
    return float(sum(aucs) / len(aucs)) if aucs else 0.0


def _position_probe_auc(
    latents_by_pos: list[list[torch.Tensor]],
    support_by_pos: list[list[torch.Tensor]],
    *,
    n_steps: int,
    lr: float,
    batch_size: int,
) -> float:
    generator = torch.Generator().manual_seed(0)
    pos_aucs = []

    for pos_latents, pos_support in zip(latents_by_pos, support_by_pos):
        if not pos_latents:
            continue
        latents = torch.cat(pos_latents, dim=0)
        support = torch.cat(pos_support, dim=0)
        n = latents.shape[0]
        if n < 2:
            continue

        perm = torch.randperm(n, generator=generator, device=latents.device)
        split = max(1, int(0.7 * n))
        if split >= n:
            split = n - 1
        train_idx = perm[:split]
        test_idx = perm[split:]
        if test_idx.numel() == 0:
            continue

        auc = _train_and_eval_probe(
            latents[train_idx],
            support[train_idx],
            latents[test_idx],
            support[test_idx],
            n_steps=n_steps,
            lr=lr,
            batch_size=batch_size,
        )
        pos_aucs.append(auc)

    return float(sum(pos_aucs) / len(pos_aucs)) if pos_aucs else 0.0


def _collect_probe_inputs(
    model: TemporalAE,
    eval_batch: AliasedBatch,
    *,
    eval_chunk_size: int,
) -> tuple[list[list[torch.Tensor]], list[list[torch.Tensor]]]:
    if model.n_positions is None:
        n_positions = eval_batch.x.shape[1]
        latents_by_pos = [[] for _ in range(n_positions)]
        support_by_pos = [[] for _ in range(n_positions)]

        with torch.no_grad():
            for start in range(0, eval_batch.x.shape[0], eval_chunk_size):
                chunk = eval_batch.sequence_slice(start, start + eval_chunk_size)
                out = model(chunk.x)
                latents = out.latents
                for pos in range(n_positions):
                    latents_by_pos[pos].append(latents[:, pos, :])
                    support_by_pos[pos].append(chunk.support[:, :, pos])
        return latents_by_pos, support_by_pos

    n_positions = model.n_positions
    latents_by_pos = [[] for _ in range(n_positions)]
    support_by_pos = [[] for _ in range(n_positions)]

    with torch.no_grad():
        for seq_start in range(0, eval_batch.x.shape[0], eval_chunk_size):
            seq_chunk = eval_batch.sequence_slice(seq_start, seq_start + eval_chunk_size)
            for time_start in range(eval_batch.x.shape[1] - n_positions + 1):
                window = seq_chunk.time_slice(time_start, time_start + n_positions)
                out = model(window.x)
                latents = out.latents
                for pos in range(n_positions):
                    latents_by_pos[pos].append(latents[:, pos, :])
                    support_by_pos[pos].append(window.support[:, :, pos])
    return latents_by_pos, support_by_pos


def _feature_recovery_for_model(model: TemporalAE, true_features: torch.Tensor) -> tuple[float, float]:
    if model.n_positions is None:
        recovery = feature_recovery(model.decoder_directions(), true_features)
        return recovery["auc"], recovery["r_at_90"]

    aucs = []
    r90s = []
    for pos in range(model.n_positions):
        recovery = feature_recovery(model.decoder_directions(pos), true_features)
        aucs.append(recovery["auc"])
        r90s.append(recovery["r_at_90"])
    return float(sum(aucs) / len(aucs)), float(sum(r90s) / len(r90s))


def evaluate_aliased_model(
    model: TemporalAE,
    eval_batch: AliasedBatch,
    true_features: torch.Tensor,
    *,
    eval_chunk_size: int = 256,
    probe_steps: int = 200,
    probe_lr: float = 1e-3,
    probe_batch_size: int = 256,
) -> AliasedEvalMetrics:
    model.eval()
    pair_dirs = true_features.view(-1, 2, true_features.shape[-1]).to(eval_batch.x.device)

    sum_se = 0.0
    sum_signal = 0.0
    sum_novel_l0 = 0.0
    sum_pred_l0 = 0.0
    sum_total_l0 = 0.0
    n_tokens = 0
    local_sum = 0.0
    pred_sum = 0.0
    informative_n = 0

    with torch.no_grad():
        if model.n_positions is None:
            for start in range(0, eval_batch.x.shape[0], eval_chunk_size):
                chunk = eval_batch.sequence_slice(start, start + eval_chunk_size)
                out = model(chunk.x)
                sum_se += (chunk.x - out.x_hat).pow(2).sum().item()
                sum_signal += chunk.x.pow(2).sum().item()
                novel_l0, pred_l0, total_l0 = _l0_sums(model, out)
                sum_novel_l0 += novel_l0
                sum_pred_l0 += pred_l0
                sum_total_l0 += total_l0
                n_tokens += chunk.x.shape[0] * chunk.x.shape[1]

                bias_free = _bias_free_reconstruction(model, out.x_hat)
                ls, ps, inf_n = _pair_alignment_sums(bias_free, chunk, pair_dirs)
                local_sum += ls
                pred_sum += ps
                informative_n += inf_n
        else:
            T = model.n_positions
            for seq_start in range(0, eval_batch.x.shape[0], eval_chunk_size):
                seq_chunk = eval_batch.sequence_slice(seq_start, seq_start + eval_chunk_size)
                for time_start in range(eval_batch.x.shape[1] - T + 1):
                    window = seq_chunk.time_slice(time_start, time_start + T)
                    out = model(window.x)
                    sum_se += (window.x - out.x_hat).pow(2).sum().item()
                    sum_signal += window.x.pow(2).sum().item()
                    novel_l0, pred_l0, total_l0 = _l0_sums(model, out)
                    sum_novel_l0 += novel_l0
                    sum_pred_l0 += pred_l0
                    sum_total_l0 += total_l0
                    n_tokens += window.x.shape[0] * window.x.shape[1]

                    bias_free = _bias_free_reconstruction(model, out.x_hat)
                    ls, ps, inf_n = _pair_alignment_sums(bias_free, window, pair_dirs)
                    local_sum += ls
                    pred_sum += ps
                    informative_n += inf_n

    auc, r_at_90 = _feature_recovery_for_model(model, true_features)
    latents_by_pos, support_by_pos = _collect_probe_inputs(
        model, eval_batch, eval_chunk_size=eval_chunk_size
    )
    probe_auc = _position_probe_auc(
        latents_by_pos,
        support_by_pos,
        n_steps=probe_steps,
        lr=probe_lr,
        batch_size=probe_batch_size,
    )

    corr_local = local_sum / max(informative_n, 1)
    corr_pred = pred_sum / max(informative_n, 1)
    return AliasedEvalMetrics(
        nmse=sum_se / max(sum_signal, 1e-8),
        novel_l0=sum_novel_l0 / max(n_tokens, 1),
        pred_l0=sum_pred_l0 / max(n_tokens, 1),
        total_l0=sum_total_l0 / max(n_tokens, 1),
        auc=auc,
        r_at_90=r_at_90,
        probe_auc=probe_auc,
        corr_local=corr_local,
        corr_pred=corr_pred,
        delta=corr_pred - corr_local,
        n_informative=informative_n,
    )
