import random

import torch
from datasets import load_dataset
from transformer_lens import HookedTransformer
from tqdm import tqdm

from bill.temporal_autocorrelation.config import ExperimentConfig


def load_tokenized_sequences(
    config: ExperimentConfig,
    model: HookedTransformer,
    seed: int = 42,
) -> torch.Tensor:
    """Stream OpenWebText, tokenize, and return sequences of fixed length.

    For documents longer than seq_length, a random contiguous window is
    extracted (rather than always taking the first seq_length tokens).
    Documents shorter than seq_length are skipped.

    Returns:
        Tensor of shape [num_sequences, seq_length] (int64, on CPU).
    """
    rng = random.Random(seed)
    tokenizer = model.tokenizer
    bos_token_id = tokenizer.bos_token_id

    dataset = load_dataset(config.dataset_name, split="train", streaming=True)

    sequences = []
    for example in tqdm(dataset, desc="Tokenizing", total=config.num_sequences):
        token_ids = tokenizer.encode(example["text"])
        # Prepend BOS
        token_ids = [bos_token_id] + token_ids

        if len(token_ids) < config.seq_length:
            continue

        # Random offset into the document
        max_start = len(token_ids) - config.seq_length
        start = rng.randint(0, max_start)
        sequences.append(token_ids[start : start + config.seq_length])

        if len(sequences) >= config.num_sequences:
            break

    return torch.tensor(sequences, dtype=torch.long)
