"""Quick smoke test for statistics module on real SAE features."""

import numpy as np
from bill.temporal_autocorrelation.config import ExperimentConfig
from bill.temporal_autocorrelation.activations import load_model_and_sae, extract_sae_features_batch
from bill.temporal_autocorrelation.data import load_tokenized_sequences
from bill.temporal_autocorrelation.statistics import (
    FeatureStatistics,
    compute_shuffled_baseline,
    ljung_box_test,
)

config = ExperimentConfig(num_sequences=2, batch_size=2)

print("Loading model and SAE...")
model, sae = load_model_and_sae(config, device="cpu")

print(f"Tokenizing {config.num_sequences} sequences...")
tokens = load_tokenized_sequences(config, model)

print("Extracting features...")
acts = extract_sae_features_batch(model, sae, tokens, config.hook_point)
print(f"Feature acts shape: {acts.shape}")

# --- FeatureStatistics ---
print("\n--- FeatureStatistics (incremental) ---")
stats = FeatureStatistics(num_features=acts.shape[2], max_lag=config.max_lag)
for i in range(acts.shape[0]):
    stats.update(acts[i], min_activations=config.min_activations_for_autocorr)
    print(f"  Updated with sequence {i}")
stats.finalize()

valid_mag = ~np.isnan(stats.mean_magnitude_when_active)
valid_ac = ~np.isnan(stats.mean_autocorrelation[:, 0])
print(f"Features with valid magnitude: {valid_mag.sum()}")
print(f"Features with valid lag-1 AC:  {valid_ac.sum()}")
print(f"Activation frequency: min={stats.activation_frequency.min():.4f}, "
      f"median={np.median(stats.activation_frequency):.4f}, "
      f"max={stats.activation_frequency.max():.4f}")
print(f"Mean magnitude (when active): min={np.nanmin(stats.mean_magnitude_when_active):.4f}, "
      f"median={np.nanmedian(stats.mean_magnitude_when_active):.4f}, "
      f"max={np.nanmax(stats.mean_magnitude_when_active):.4f}")
print(f"Mean lag-1 AC: min={np.nanmin(stats.mean_autocorrelation[:, 0]):.4f}, "
      f"median={np.nanmedian(stats.mean_autocorrelation[:, 0]):.4f}, "
      f"max={np.nanmax(stats.mean_autocorrelation[:, 0]):.4f}")

# --- Shuffled baseline ---
print("\n--- Shuffled Baseline (first sequence) ---")
rng = np.random.default_rng(42)
shuffled_ac = compute_shuffled_baseline(
    acts[0], config.max_lag, config.min_activations_for_autocorr, rng
)
valid_shuf = ~np.isnan(shuffled_ac[:, 0])
print(f"Features with valid shuffled lag-1 AC: {valid_shuf.sum()}")
if valid_shuf.any():
    shuf_lag1 = shuffled_ac[valid_shuf, 0]
    print(f"Shuffled lag-1 AC: mean={shuf_lag1.mean():.4f}, std={shuf_lag1.std():.4f}")

    # Compare to real
    real_ac_seq0 = stats.mean_autocorrelation[valid_shuf, 0]
    real_valid = ~np.isnan(real_ac_seq0)
    if real_valid.any():
        print(f"Real lag-1 AC (same features): mean={real_ac_seq0[real_valid].mean():.4f}")

# --- Ljung-Box (on a small subset of features to keep it fast) ---
print("\n--- Ljung-Box Test (first sequence, first 500 features) ---")
pvals = ljung_box_test(acts[0, :, :500], config.max_lag)
valid_p = ~np.isnan(pvals)
print(f"Features with valid p-values: {valid_p.sum()} / 500")
if valid_p.any():
    p = pvals[valid_p]
    print(f"P-values: min={p.min():.2e}, median={np.median(p):.2e}, max={p.max():.2e}")
    print(f"Significant at p<0.01: {(p < 0.01).sum()} / {valid_p.sum()}")
    print(f"Significant at p<0.05: {(p < 0.05).sum()} / {valid_p.sum()}")

print("\nOK")
