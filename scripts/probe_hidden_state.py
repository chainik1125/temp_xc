"""Probe from model latents to ground-truth HMM hidden states.

For each model and rho value:
  1. Train the model (SAE, TXCDR, PerFeat, PerFeat-C)
  2. Generate eval data with matched ground-truth support
  3. Extract model latents on eval data
  4. Train a per-position linear probe: latents[:,t,:] -> support[:,:,t]
  5. Report probe AUC (how well latents recover the hidden Markov state)

For the XC, also probe from pre-TopK activations (richer than post-TopK).
"""

from __future__ import annotations

import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "src")

from temporal_bench.config import DataConfig, TrainConfig
from temporal_bench.data.markov import generate_markov_support
from temporal_bench.data.pipeline import DataPipeline
from temporal_bench.models.per_feature_temporal import PerFeatureTemporalAE
from temporal_bench.models.temporal_crosscoder import TemporalCrosscoder
from temporal_bench.models.topk_sae import TopKSAE
from temporal_bench.train import train


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def generate_eval_with_support(
    pipeline: DataPipeline,
    n_sequences: int,
    T: int,
    rho: float,
    seed: int = 9999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate eval data and matching ground-truth binary support.

    Returns:
        x: (n_sequences, T, d_model) activations
        support: (n_sequences, n_features, T) binary hidden states
    """
    cfg = pipeline.config
    eval_gen = torch.Generator().manual_seed(seed)
    support = generate_markov_support(
        cfg.n_features, T, cfg.pi, rho,
        n_sequences=n_sequences, generator=eval_gen,
    )
    x = pipeline.toy_model.embed(
        support, cfg.magnitude_mean, cfg.magnitude_std, generator=eval_gen,
    )
    return x.to(pipeline.device), support.to(pipeline.device)


def extract_latents(model, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Extract latent representations from a model.

    Returns dict of name -> (B, T, m) or (B, m) tensors.
    """
    model.eval()
    with torch.no_grad():
        out = model(x)
    reps = {"latents": out.latents}  # (B, T, m) for all models

    # For the XC, also extract pre-TopK activations
    if isinstance(model, TemporalCrosscoder):
        with torch.no_grad():
            pre = torch.einsum("btd,tdm->bm", x, model.W_enc) + model.b_enc
            pre = F.relu(pre)
        reps["pre_topk"] = pre  # (B, m) — before sparsification

    model.train()
    return reps


class LinearProbe(nn.Module):
    """Per-position linear probe: latents[:,t,:] -> support[:,:,t]."""

    def __init__(self, d_in: int, n_features: int):
        super().__init__()
        self.linear = nn.Linear(d_in, n_features)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)  # (B, n_features) logits


def train_and_eval_probe(
    latents_train: torch.Tensor,
    support_train: torch.Tensor,
    latents_test: torch.Tensor,
    support_test: torch.Tensor,
    n_steps: int = 2000,
    lr: float = 1e-3,
) -> float:
    """Train a linear probe and return test AUC.

    Args:
        latents_train: (N, m) input features for probe
        support_train: (N, n_features) binary targets
        latents_test: (N_test, m)
        support_test: (N_test, n_features)

    Returns:
        AUC averaged over features (macro).
    """
    device = latents_train.device
    m = latents_train.shape[1]
    n_features = support_train.shape[1]

    probe = LinearProbe(m, n_features).to(device)
    optimizer = torch.optim.Adam(probe.parameters(), lr=lr)

    # Train
    probe.train()
    n_train = latents_train.shape[0]
    batch_size = min(256, n_train)

    for step in range(n_steps):
        idx = torch.randint(n_train, (batch_size,), device=device)
        logits = probe(latents_train[idx])
        loss = F.binary_cross_entropy_with_logits(logits, support_train[idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Eval
    probe.eval()
    with torch.no_grad():
        logits = probe(latents_test)
        probs = torch.sigmoid(logits).cpu().numpy()
        targets = support_test.cpu().numpy()

    # Macro AUC across features (skip features that are all-0 or all-1)
    aucs = []
    for k in range(n_features):
        if targets[:, k].sum() == 0 or targets[:, k].sum() == len(targets):
            continue
        aucs.append(roc_auc_score(targets[:, k], probs[:, k]))

    return float(sum(aucs) / len(aucs)) if aucs else 0.0


def probe_model(
    reps: dict[str, torch.Tensor],
    support: torch.Tensor,
    T: int,
    n_features: int,
) -> dict[str, float]:
    """Train probes for all representations of a model.

    Args:
        reps: dict of name -> tensor from extract_latents
        support: (N, n_features, T) ground truth binary states
        T: window length
        n_features: number of features

    Returns:
        dict of "{rep_name}" -> AUC
    """
    N = support.shape[0]
    n_train = int(0.7 * N)

    results = {}

    for rep_name, rep in reps.items():
        # Handle shared (B, m) vs per-position (B, T, m) latents
        if rep.dim() == 2:
            # Shared latent (XC pre_topk): (B, m)
            # Probe: can this shared vector predict per-position states?
            # Average AUC across positions
            pos_aucs = []
            for t in range(T):
                s_t = support[:, :, t]  # (N, n_features)
                auc = train_and_eval_probe(
                    rep[:n_train], s_t[:n_train],
                    rep[n_train:], s_t[n_train:],
                )
                pos_aucs.append(auc)
            results[rep_name] = sum(pos_aucs) / len(pos_aucs)
        else:
            # Per-position latents (B, T, m)
            pos_aucs = []
            for t in range(T):
                lat_t = rep[:, t, :]  # (N, m)
                s_t = support[:, :, t]  # (N, n_features)
                auc = train_and_eval_probe(
                    lat_t[:n_train], s_t[:n_train],
                    lat_t[n_train:], s_t[n_train:],
                )
                pos_aucs.append(auc)
            results[rep_name] = sum(pos_aucs) / len(pos_aucs)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU'}")

    data_cfg = DataConfig()
    train_cfg = TrainConfig(n_steps=30_000, eval_every=30_000, batch_size=64)
    pipeline = DataPipeline(data_cfg, device=device)

    T = 5
    k = 5
    n_eval = 2000
    rho_values = [0.0, 0.5, 1.0]

    # Header
    print(f"\n{'rho':>5}  {'Model':>12}  {'Loss':>7}  {'MMCS':>5}  "
          f"{'L0':>5}  {'FeatRec':>7}  {'ProbeAUC':>9}  {'PreTopK':>9}  {'Time':>5}")
    print("-" * 90)

    for rho in rho_values:
        # Generate eval data with ground truth support
        x_eval, support_eval = generate_eval_with_support(
            pipeline, n_eval, T, rho, seed=9999,
        )

        # Define models
        models = {
            "SAE": TopKSAE(
                d_in=data_cfg.d_model, d_sae=data_cfg.n_features, k=k,
            ),
            "PerFeat": PerFeatureTemporalAE(
                d_in=data_cfg.d_model, d_sae=data_cfg.n_features,
                T=T, k=k,
            ),
        }

        for model_name, model in models.items():
            t0 = time.time()
            model = model.to(device)

            # Train
            data_fn = lambda bs: pipeline.sample_windows(bs, T, rho)
            history = train(
                model, data_fn, train_cfg,
                eval_data=x_eval, true_features=pipeline.true_features,
                silent=True,
            )

            # Standard metrics
            final = history[-1] if history else None
            loss = final.nmse if final else -1
            mmcs = final.mean_max_cos if final else -1
            l0 = final.l0 if final else -1
            feat_rec = final.r_at_90 if final else -1

            # Extract latents and probe
            reps = extract_latents(model, x_eval)
            probe_results = probe_model(
                reps, support_eval, T, data_cfg.n_features,
            )

            probe_auc = probe_results.get("latents", -1)
            pre_topk_auc = probe_results.get("pre_topk", -1)

            elapsed = time.time() - t0

            pre_topk_str = f"{pre_topk_auc:.3f}" if pre_topk_auc >= 0 else "  —"
            print(
                f"{rho:5.1f}  {model_name:>12}  {loss:7.4f}  "
                f"{mmcs:5.3f}  {l0:5.1f}  {feat_rec:7.2f}  "
                f"{probe_auc:9.3f}  {pre_topk_str:>9}  {elapsed:5.0f}s"
            )

        print()


if __name__ == "__main__":
    main()
