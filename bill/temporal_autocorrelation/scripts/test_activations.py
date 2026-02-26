"""Quick smoke test for model/SAE loading and feature extraction on real text."""

from bill.temporal_autocorrelation.config import ExperimentConfig
from bill.temporal_autocorrelation.activations import load_model_and_sae, extract_sae_features_batch
from bill.temporal_autocorrelation.data import load_tokenized_sequences

config = ExperimentConfig(num_sequences=2, batch_size=2)

print(f"Loading model and SAE...")
model, sae = load_model_and_sae(config, device="cpu")
print(f"Model: {config.model_name}, SAE: {config.sae_release} / {config.sae_id}")
print(f"SAE num features: {sae.cfg.d_sae}")

print(f"\nTokenizing {config.num_sequences} sequences of length {config.seq_length}...")
tokens = load_tokenized_sequences(config, model)
print(f"Tokens shape: {tokens.shape}")
print(f"Decoded start of first sequence:\n  {model.tokenizer.decode(tokens[0, :50])!r}")

print(f"\nExtracting features...")
acts = extract_sae_features_batch(model, sae, tokens, config.hook_point)

print(f"Output shape: {acts.shape}")
print(f"Dtype: {acts.dtype}")
print(f"Min: {acts.min():.4f}, Max: {acts.max():.4f}")
print(f"Fraction nonzero: {(acts > 0).mean():.4f}")
print(f"Mean nonzero activation: {acts[acts > 0].mean():.4f}")
print("OK")
