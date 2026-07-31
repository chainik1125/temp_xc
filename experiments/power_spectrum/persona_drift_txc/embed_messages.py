"""Embed latest user messages with Qwen3-Embedding-0.6B as in the paper."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from experiments.power_spectrum.persona_drift_txc.protocol import (
    EXPERIMENT_ROOT,
    iter_jsonl,
    load_config,
)


def last_token_pool(
    last_hidden_states: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Official Qwen3-Embedding Transformers pooling recipe."""
    left_padding = bool(attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    rows = torch.arange(batch_size, device=last_hidden_states.device)
    return last_hidden_states[rows, sequence_lengths]


@torch.inference_mode()
def embed_messages(
    *,
    metadata_path: Path,
    output_path: Path,
    batch_size: int,
    max_length: int,
) -> None:
    config = load_config()
    metadata = list(iter_jsonl(metadata_path))
    model_name = config["embedding_model"]
    revision = config["embedding_model_revision"]
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        revision=revision,
        padding_side="left",
    )
    model = AutoModel.from_pretrained(
        model_name,
        revision=revision,
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    messages = [message for record in metadata for message in record["user_messages"]]
    batches: list[torch.Tensor] = []
    for start in range(0, len(messages), batch_size):
        tokenized = tokenizer(
            messages[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(model.device)
        outputs = model(**tokenized)
        pooled = last_token_pool(outputs.last_hidden_state, tokenized["attention_mask"])
        batches.append(F.normalize(pooled.float(), p=2, dim=1).cpu())
        print(f"[embed] {min(start + batch_size, len(messages))}/{len(messages)}")
    embeddings = torch.cat(batches).reshape(
        len(metadata), int(config["turns_per_conversation"]), -1
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "embeddings": embeddings,
            "conversation_ids": [record["conversation_id"] for record in metadata],
            "model": model_name,
            "pooling": "last_token_l2_normalized",
        },
        output_path,
    )
    print(f"[embed] wrote {output_path} shape={tuple(embeddings.shape)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--metadata",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations" / "metadata.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "user_embeddings.pt",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embed_messages(
        metadata_path=args.metadata,
        output_path=args.output,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
