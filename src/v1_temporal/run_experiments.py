"""v1 experiments: per-token SAE on temporal data with causal mixing.

Four experiments testing how causal mean-pooling mixing (gamma) affects
per-token SAE feature recovery, using only Chanin et al. metrics.

Experiment A: L0 sweep at gamma=0.3
Experiment B: Gamma sweep at correct L0 (key experiment)
Experiment C: Mixing x correlations (2x2)
Experiment D: Large-scale sanity check
"""

import json
import os
import time

import torch

from src.shared.configs import TrainingConfig
from src.shared.correlation import create_correlation_matrix
from src.shared.eval_sae import eval_sae
from src.shared.metrics import (
    decoder_feature_cosine_similarity,
    decoder_pairwise_cosine_similarity,
    match_sae_latents_to_features,
)
from src.shared.plotting import plot_cdec_vs_l0, plot_decoder_feature_heatmap
from src.shared.train_sae import create_sae, train_sae
from src.utils.seed import set_seed
from src.v1_temporal.temporal_data_generation import generate_temporal_batch
from src.v1_temporal.temporal_toy_model import TemporalToyModel

RESULTS_BASE = os.path.join(os.path.dirname(__file__), "results")


def global_local_recovery(
    sae, feature_directions, n_global, n_local,
):
    """Compute mean heatmap diagonal for global and local features separately.

    Uses existing shared metrics: decoder_feature_cosine_similarity +
    match_sae_latents_to_features.

    Returns:
        dict with recovery_global, recovery_local, recovery_mean, cdec, cos_sim_matrix
    """
    cos_sim = decoder_feature_cosine_similarity(sae, feature_directions)
    perm = match_sae_latents_to_features(cos_sim)
    reordered = cos_sim[perm]  # (d_sae, num_features) reordered

    num_features = n_global + n_local
    diag = torch.tensor([reordered[i, i].abs().item() for i in range(num_features)])

    recovery_global = diag[:n_global].mean().item() if n_global > 0 else float("nan")
    recovery_local = diag[n_global:num_features].mean().item() if n_local > 0 else float("nan")
    recovery_mean = diag[:num_features].mean().item()

    cdec = decoder_pairwise_cosine_similarity(sae)

    return {
        "recovery_global": recovery_global,
        "recovery_local": recovery_local,
        "recovery_mean": recovery_mean,
        "cdec": cdec,
        "cos_sim_reordered": reordered,
    }


def train_and_evaluate(
    n_global, n_local, hidden_dim, seq_len, gamma, k, p,
    global_corr_matrix=None, local_corr_matrix=None,
    mean_magnitudes=None, std_magnitudes=None,
    total_samples=10_000_000, seed=42, device=None,
):
    """Full pipeline: build model, train SAE, evaluate."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(seed)
    num_features = n_global + n_local

    model = TemporalToyModel(
        num_features, hidden_dim, gamma=gamma, ortho_num_steps=1000,
    )
    model = model.to(device)

    global_probs = torch.tensor([p] * n_global)
    local_probs = torch.tensor([p] * n_local)
    if mean_magnitudes is None:
        mean_magnitudes = torch.ones(num_features)
    if std_magnitudes is None:
        std_magnitudes = torch.zeros(num_features)

    def generate_flattened(batch_size):
        n_sequences = batch_size // seq_len
        feats = generate_temporal_batch(
            n_sequences, seq_len, global_probs, local_probs,
            global_corr_matrix=global_corr_matrix,
            local_corr_matrix=local_corr_matrix,
            mean_magnitudes=mean_magnitudes,
            std_magnitudes=std_magnitudes,
            device=device,
        )
        hidden = model(feats)
        return hidden.reshape(-1, hidden_dim)

    true_l0 = p * num_features

    training_cfg = TrainingConfig(
        k=k, d_sae=num_features,
        total_training_samples=total_samples,
        batch_size=4096, seed=seed,
    )

    sae = create_sae(hidden_dim, num_features, k=k, device=device)
    sae = train_sae(sae, generate_flattened, training_cfg, device)

    eval_result = eval_sae(sae, generate_flattened, n_samples=50_000, true_l0=true_l0)

    metrics = global_local_recovery(sae, model.feature_directions, n_global, n_local)
    metrics["ve"] = eval_result.ve

    return metrics, sae, model


def experiment_a(device):
    """Experiment A: L0 sweep with mixing (gamma=0.3)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT A: L0 Sweep with Mixing (gamma=0.3)")
    print("=" * 60)

    results_dir = os.path.join(RESULTS_BASE, "exp_a_l0_sweep")
    os.makedirs(results_dir, exist_ok=True)

    n_global, n_local = 5, 5
    hidden_dim, seq_len, gamma, p = 20, 4, 0.3, 0.4
    true_l0 = p * (n_global + n_local)
    k_values = [1, 2, 3, 4, 5, 6, 7, 8]
    seeds = [42, 43, 44]

    cdec_results = {k: [] for k in k_values}
    summary = {}

    for k in k_values:
        all_metrics = []
        for seed in seeds:
            metrics, sae, model = train_and_evaluate(
                n_global, n_local, hidden_dim, seq_len, gamma, k, p,
                seed=seed, device=device,
            )
            cdec_results[k].append(metrics["cdec"])
            all_metrics.append(metrics)
            print(f"  k={k}, seed={seed}: cdec={metrics['cdec']:.4f} "
                  f"rec_g={metrics['recovery_global']:.3f} "
                  f"rec_l={metrics['recovery_local']:.3f}")

        means = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in ["cdec", "recovery_global", "recovery_local", "recovery_mean"]
        }
        summary[k] = means

        # Save heatmap for representative cases
        if k in [2, 4, 6]:
            cos_sim = all_metrics[0]["cos_sim_reordered"]
            plot_decoder_feature_heatmap(
                cos_sim, f"k={k}, gamma={gamma}",
                os.path.join(results_dir, f"heatmap_k{k}"),
            )

    print("\n  Summary (means across seeds):")
    for k in k_values:
        s = summary[k]
        print(f"    k={k}: cdec={s['cdec']:.4f} rec_g={s['recovery_global']:.3f} rec_l={s['recovery_local']:.3f}")

    plot_cdec_vs_l0(cdec_results, true_l0, os.path.join(results_dir, "cdec_vs_l0"))

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def experiment_b(device):
    """Experiment B: Gamma sweep at correct L0 (key experiment)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT B: Gamma Sweep at Correct L0")
    print("=" * 60)

    results_dir = os.path.join(RESULTS_BASE, "exp_b_gamma_sweep")
    os.makedirs(results_dir, exist_ok=True)

    n_global, n_local = 5, 5
    hidden_dim, seq_len, p = 20, 4, 0.4
    true_l0 = p * (n_global + n_local)
    k = true_l0
    gamma_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
    seeds = [42, 43, 44]

    summary = {}

    for gamma in gamma_values:
        all_metrics = []
        for seed in seeds:
            metrics, sae, model = train_and_evaluate(
                n_global, n_local, hidden_dim, seq_len, gamma, k, p,
                seed=seed, device=device,
            )
            all_metrics.append(metrics)
            print(f"  gamma={gamma}, seed={seed}: cdec={metrics['cdec']:.4f} "
                  f"rec_g={metrics['recovery_global']:.3f} "
                  f"rec_l={metrics['recovery_local']:.3f}")

        means = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in ["cdec", "recovery_global", "recovery_local", "recovery_mean"]
        }
        summary[gamma] = means

        # Save heatmap for each gamma
        cos_sim = all_metrics[0]["cos_sim_reordered"]
        g_str = str(gamma).replace(".", "p")
        plot_decoder_feature_heatmap(
            cos_sim, f"gamma={gamma}, k={k}",
            os.path.join(results_dir, f"heatmap_gamma_{g_str}"),
        )

    print("\n  Summary (means across seeds):")
    for gamma in gamma_values:
        s = summary[gamma]
        print(f"    gamma={gamma}: cdec={s['cdec']:.4f} rec_g={s['recovery_global']:.3f} rec_l={s['recovery_local']:.3f}")

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    # Plot recovery vs gamma
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    g_vals = [summary[g]["recovery_global"] for g in gamma_values]
    l_vals = [summary[g]["recovery_local"] for g in gamma_values]
    ax.plot(gamma_values, g_vals, "o-", label="Global features", color="steelblue")
    ax.plot(gamma_values, l_vals, "s-", label="Local features", color="coral")
    ax.set_xlabel("Mixing strength (gamma)")
    ax.set_ylabel("Mean |heatmap diagonal|")
    ax.set_title("Feature recovery vs mixing strength")
    ax.legend()
    ax.set_ylim(0, 1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "recovery_vs_gamma.png"), dpi=150)
    plt.close()

    return summary


def experiment_c(device):
    """Experiment C: Mixing x correlations (2x2 design)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT C: Mixing x Correlations")
    print("=" * 60)

    results_dir = os.path.join(RESULTS_BASE, "exp_c_mixing_corr")
    os.makedirs(results_dir, exist_ok=True)

    n_global, n_local = 5, 5
    hidden_dim, seq_len, p = 20, 4, 0.4
    true_l0 = p * (n_global + n_local)
    k = true_l0
    seeds = [42, 43, 44]

    # Correlation matrix: +0.4 between global feature 0 and features 1-4
    corr_entries = {(0, i): 0.4 for i in range(1, n_global)}
    global_corr = create_correlation_matrix(n_global, corr_entries)

    conditions = {
        "gamma0_no_corr": {"gamma": 0.0, "global_corr_matrix": None},
        "gamma0_with_corr": {"gamma": 0.0, "global_corr_matrix": global_corr},
        "gamma03_no_corr": {"gamma": 0.3, "global_corr_matrix": None},
        "gamma03_with_corr": {"gamma": 0.3, "global_corr_matrix": global_corr},
    }

    summary = {}

    for name, cfg in conditions.items():
        all_metrics = []
        for seed in seeds:
            metrics, sae, model = train_and_evaluate(
                n_global, n_local, hidden_dim, seq_len,
                gamma=cfg["gamma"], k=k, p=p,
                global_corr_matrix=cfg["global_corr_matrix"],
                seed=seed, device=device,
            )
            all_metrics.append(metrics)
            print(f"  {name}, seed={seed}: cdec={metrics['cdec']:.4f} "
                  f"rec_g={metrics['recovery_global']:.3f} "
                  f"rec_l={metrics['recovery_local']:.3f}")

        means = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in ["cdec", "recovery_global", "recovery_local", "recovery_mean", "ve"]
        }
        summary[name] = means

        cos_sim = all_metrics[0]["cos_sim_reordered"]
        plot_decoder_feature_heatmap(
            cos_sim, name.replace("_", " "),
            os.path.join(results_dir, f"heatmap_{name}"),
        )

    print("\n  Summary (means across seeds):")
    for name in conditions:
        s = summary[name]
        print(f"    {name}: cdec={s['cdec']:.4f} rec_g={s['recovery_global']:.3f} "
              f"rec_l={s['recovery_local']:.3f} ve={s['ve']:.3f}")

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def experiment_d(device):
    """Experiment D: Large-scale sanity check."""
    print("\n" + "=" * 60)
    print("EXPERIMENT D: Large-Scale L0 Sweep (gamma=0.3)")
    print("=" * 60)

    results_dir = os.path.join(RESULTS_BASE, "exp_d_large_scale")
    os.makedirs(results_dir, exist_ok=True)

    n_global, n_local = 25, 25
    num_features = n_global + n_local
    hidden_dim, seq_len, gamma = 100, 4, 0.3
    seeds = [42, 43, 44]

    # Power-law firing probabilities
    probs = torch.linspace(0.345, 0.05, num_features)
    p_global = probs[:n_global]
    p_local = probs[n_global:]
    true_l0 = probs.sum().item()
    print(f"  True L0: {true_l0:.2f}")

    # Magnitude noise
    mean_magnitudes = torch.ones(num_features)
    std_magnitudes = torch.full((num_features,), 0.15)

    k_values = [2, 4, 6, 8, 10, 12, 14, 17, 20]

    cdec_results = {k: [] for k in k_values}
    summary = {}

    for k in k_values:
        all_metrics = []
        for seed in seeds:
            set_seed(seed)

            model = TemporalToyModel(
                num_features, hidden_dim, gamma=gamma, ortho_num_steps=1000,
            )
            model = model.to(device)

            def generate_flattened(batch_size, _model=model, _p_g=p_global,
                                   _p_l=p_local, _mm=mean_magnitudes,
                                   _sm=std_magnitudes, _sl=seq_len):
                n_sequences = batch_size // _sl
                feats = generate_temporal_batch(
                    n_sequences, _sl, _p_g, _p_l,
                    mean_magnitudes=_mm, std_magnitudes=_sm,
                    device=device,
                )
                hidden = _model(feats)
                return hidden.reshape(-1, hidden_dim)

            training_cfg = TrainingConfig(
                k=k, d_sae=num_features,
                total_training_samples=10_000_000,
                batch_size=4096, seed=seed,
            )

            sae = create_sae(hidden_dim, num_features, k=k, device=device)
            sae = train_sae(sae, generate_flattened, training_cfg, device)

            metrics = global_local_recovery(sae, model.feature_directions, n_global, n_local)
            eval_result = eval_sae(sae, generate_flattened, n_samples=50_000, true_l0=true_l0)
            metrics["ve"] = eval_result.ve

            cdec_results[k].append(metrics["cdec"])
            all_metrics.append(metrics)
            print(f"  k={k}, seed={seed}: cdec={metrics['cdec']:.4f} "
                  f"rec_g={metrics['recovery_global']:.3f} "
                  f"rec_l={metrics['recovery_local']:.3f}")

        means = {
            key: sum(m[key] for m in all_metrics) / len(all_metrics)
            for key in ["cdec", "recovery_global", "recovery_local", "recovery_mean"]
        }
        summary[k] = means

        if k in [4, 10, 17]:
            cos_sim = all_metrics[0]["cos_sim_reordered"]
            plot_decoder_feature_heatmap(
                cos_sim, f"k={k}, gamma={gamma}, large",
                os.path.join(results_dir, f"heatmap_k{k}"),
            )

    print("\n  Summary (means across seeds):")
    for k in k_values:
        s = summary[k]
        print(f"    k={k:>2}: cdec={s['cdec']:.4f} rec_g={s['recovery_global']:.3f} rec_l={s['recovery_local']:.3f}")

    plot_cdec_vs_l0(cdec_results, true_l0, os.path.join(results_dir, "cdec_vs_l0"))

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"v1 Experiments (with causal mixing)")
    print(f"Device: {device}")

    t0 = time.time()

    times = {}
    t_start = time.time()
    experiment_a(device)
    times["A"] = time.time() - t_start

    t_start = time.time()
    experiment_b(device)
    times["B"] = time.time() - t_start

    t_start = time.time()
    experiment_c(device)
    times["C"] = time.time() - t_start

    t_start = time.time()
    experiment_d(device)
    times["D"] = time.time() - t_start

    total = time.time() - t0
    print(f"\n{'=' * 60}")
    for name, t in times.items():
        print(f"  Experiment {name}: {t/60:.1f} min")
    print(f"  Total: {total/60:.1f} min")
    print(f"Results in {RESULTS_BASE}/exp_{{a,b,c,d}}_*/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
