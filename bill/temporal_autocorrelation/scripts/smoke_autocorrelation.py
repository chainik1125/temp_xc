"""Quick smoke test for autocorrelation on real SAE features."""

import numpy as np
from bill.temporal_autocorrelation.config import ExperimentConfig
from bill.temporal_autocorrelation.activations import load_model_and_sae, extract_sae_features_batch
from bill.temporal_autocorrelation.data import load_tokenized_sequences
from bill.temporal_autocorrelation.autocorrelation import compute_autocorrelations_vectorized

config = ExperimentConfig(num_sequences=2, batch_size=2)

print("Loading model and SAE...")
model, sae = load_model_and_sae(config, device="cpu")

print(f"Tokenizing {config.num_sequences} sequences...")
tokens = load_tokenized_sequences(config, model)

print("Extracting features...")
acts = extract_sae_features_batch(model, sae, tokens, config.hook_point)
print(f"Feature acts shape: {acts.shape}")

# Run autocorrelation on first sequence
print(f"\nComputing autocorrelation on first sequence (T={config.seq_length}, D={acts.shape[2]})...")
ac = compute_autocorrelations_vectorized(
    acts[0],  # [T, D]
    max_lag=config.max_lag,
    min_activations=config.min_activations_for_autocorr,
)
print(f"Result shape: {ac.shape}")

valid = ~np.isnan(ac[:, 0])
print(f"Features with valid lag-1 autocorrelation: {valid.sum()} / {ac.shape[0]}")

if valid.any():
    lag1 = ac[valid, 0]
    print(f"\nLag-1 autocorrelation stats (valid features):")
    print(f"  Mean:   {lag1.mean():.4f}")
    print(f"  Median: {np.median(lag1):.4f}")
    print(f"  Std:    {lag1.std():.4f}")
    print(f"  Min:    {lag1.min():.4f}")
    print(f"  Max:    {lag1.max():.4f}")

    # Show decay across lags for the top-5 most autocorrelated features
    top5_idx = np.argsort(lag1)[-5:][::-1]
    feature_ids = np.where(valid)[0][top5_idx]
    print(f"\nTop 5 features by lag-1 autocorrelation:")
    for fid in feature_ids:
        lags = ac[fid]
        lag_str = ", ".join(f"{l:.3f}" for l in lags)
        num_active = np.count_nonzero(acts[0, :, fid])
        print(f"  Feature {fid:5d}: lags 1-{config.max_lag} = [{lag_str}], active {num_active}/{config.seq_length} positions")

print("OK")
