"""Quick smoke test for data loading."""

from transformer_lens import HookedTransformer

from bill.temporal_autocorrelation.config import ExperimentConfig
from bill.temporal_autocorrelation.data import load_tokenized_sequences

config = ExperimentConfig(num_sequences=4)
print(f"Loading model: {config.model_name}")
model = HookedTransformer.from_pretrained(config.model_name)

print(f"Tokenizing {config.num_sequences} sequences of length {config.seq_length}...")
tokens = load_tokenized_sequences(config, model)

print(f"Shape: {tokens.shape}")
print(f"Dtype: {tokens.dtype}")
print(f"First sequence, first 20 tokens: {tokens[0, :20].tolist()}")
print(f"Decoded start of first sequence:\n  {model.tokenizer.decode(tokens[0, :50])!r}")
print(f"Decoded start of last sequence:\n  {model.tokenizer.decode(tokens[-1, :50])!r}")
print("OK")
