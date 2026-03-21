"""Main experiment: sweep (lambda, gamma) pairs with HMM data + TopK SAE.

Sweeps gamma from 0 (i.i.d.) to 1 (MC) at fixed mu=0.05 by varying q and
(p_A, p_B). The MC case (gamma=1) serves as the sanity check.
"""

import json
from pathlib import Path

import torch

from src.utils.logging import log, log_sweep
from src.utils.seed import set_seed
from src.v5_hmm_sae_baseline.hmm_data import (
    HMMDataConfig,
    generate_hmm_dataset,
    hmm_autocorrelation_amplitude,
    hmm_marginal_sparsity,
    hmm_theoretical_autocorrelation,
)
from src.v5_hmm_sae_baseline.metrics import (
    compute_empirical_autocorrelation,
    compute_pooled_autocorrelation,
)
from src.v5_hmm_sae_baseline.sae import TopKSAE
from src.v5_hmm_sae_baseline.train import train_sae

# --- Constants ---
SEED = 42
K = 10
D = 64
T = 128
N_SEQUENCES = 200
MAX_LAG = 30

LAMBDA_VALUES = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]

# Sub-experiment A: sweep gamma at fixed mu=0.05, varying q and (p_A, p_B)
# gamma = q(1-q)(p_B - p_A)^2 / [mu(1-mu)]
# To get a wide gamma range we vary q alongside p_A/p_B.
EMISSION_CONFIGS_A = [
    {"p_A": 0.05, "p_B": 0.05, "q": 0.5,  "label": "gamma=0.000"},   # i.i.d.
    {"p_A": 0.00, "p_B": 0.10, "q": 0.5,  "label": "gamma=0.053"},   # q=0.5
    {"p_A": 0.00, "p_B": 0.25, "q": 0.2,  "label": "gamma=0.211"},   # q=0.2
    {"p_A": 0.00, "p_B": 0.50, "q": 0.1,  "label": "gamma=0.474"},   # q=0.1
    {"p_A": 0.00, "p_B": 1.00, "q": 0.05, "label": "gamma=1.000"},   # MC case
]


# SAE hyperparameters
D_INPUT = D
N_LATENTS = 64
SAE_K = 1
N_EPOCHS = 300
LR = 1e-4
BATCH_SIZE = 256

RESULTS_DIR = Path("results/v5_hmm_sae_baseline")


def run_single(
    lam: float,
    q: float,
    p_A: float,
    p_B: float,
    label: str,
) -> dict:
    """Run a single (lambda, emission) configuration.

    Returns dict with all metrics and validation results.
    """
    set_seed(SEED)

    config = HMMDataConfig.from_reset_process_hmm(
        lam=lam, q=q, p_A=p_A, p_B=p_B,
        k=K, d=D, T=T, n_sequences=N_SEQUENCES, seed=SEED,
    )

    # Theoretical quantities
    mu_theory = hmm_marginal_sparsity(config)
    gamma = hmm_autocorrelation_amplitude(config)
    theory_autocorr = hmm_theoretical_autocorrelation(config, MAX_LAG)

    # Generate dataset
    dataset = generate_hmm_dataset(config)

    # Validate marginal sparsity
    empirical_mu = dataset["support"].mean().item()
    mu_error = abs(empirical_mu - mu_theory)

    # Validate autocorrelation (both per-chain and pooled estimators)
    empirical_autocorr = compute_empirical_autocorrelation(
        dataset["support"], MAX_LAG
    )
    pooled_autocorr = compute_pooled_autocorrelation(
        dataset["support"], MAX_LAG
    )

    # Train SAE — reseed immediately before construction so SAE init is
    # identical across configs regardless of how much RNG data generation consumed
    x_flat = dataset["x"].reshape(-1, D)  # (N_SEQUENCES * T, D)
    torch.manual_seed(SEED)
    sae = TopKSAE(D_INPUT, N_LATENTS, SAE_K)
    train_results = train_sae(
        sae, x_flat, dataset["features"],
        n_epochs=N_EPOCHS, lr=LR, batch_size=BATCH_SIZE, log_every=10,
    )

    result = {
        "lam": lam,
        "q": q,
        "p_A": p_A,
        "p_B": p_B,
        "label": label,
        "mu_theory": mu_theory,
        "mu_empirical": empirical_mu,
        "mu_error": mu_error,
        "gamma": gamma,
        "theory_autocorr": theory_autocorr.tolist(),
        "empirical_autocorr": empirical_autocorr.tolist(),
        "pooled_autocorr": pooled_autocorr.tolist(),
        "recon_loss": train_results["recon_loss"],
        "l0": train_results["l0"],
        "auc": train_results["auc"],
        "mean_max_cos_sim": train_results["mean_max_cos_sim"],
        "frac_recovered_90": train_results["frac_recovered_90"],
        "frac_recovered_80": train_results["frac_recovered_80"],
        "history": train_results["history"],
    }

    log(
        "result",
        f"lam={lam}, {label}",
        mu_err=mu_error,
        gamma=gamma,
        auc=train_results["auc"],
        recon=train_results["recon_loss"],
        r90=train_results["frac_recovered_90"],
    )

    return result


def main() -> None:
    """Run the full experiment sweep."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results: list[dict] = []

    configs = [
        (lam, ec)
        for lam in LAMBDA_VALUES
        for ec in EMISSION_CONFIGS_A
    ]
    total = len(configs)

    log("info", "starting sweep", n_configs=total)

    for i, (lam, ec) in enumerate(configs, 1):
        log_sweep(i, total, lam=lam, label=ec["label"])
        result = run_single(lam, ec["q"], ec["p_A"], ec["p_B"], ec["label"])
        all_results.append(result)

    # Save results
    results_path = RESULTS_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log("done", f"saved results to {results_path}")

    # Print summary table
    log("summary", "results")
    for r in all_results:
        print(
            f"  lam={r['lam']:.1f} | {r['label']:>15s} | "
            f"mu_err={r['mu_error']:.4f} | "
            f"auc={r['auc']:.3f} | "
            f"mmcs={r['mean_max_cos_sim']:.3f} | "
            f"r90={r['frac_recovered_90']:.2f} | "
            f"recon={r['recon_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
