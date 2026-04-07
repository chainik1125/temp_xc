#!/usr/bin/env python3
"""
autointerp.py — Automated interpretability pipeline using Claude Haiku.

Finds the best StackedSAE and best TXCDR checkpoints (lowest final loss),
extracts highest-activating examples for their top features, and sends them
to Claude Haiku for short natural-language explanations.

Usage:
    python autointerp.py                          # auto-select best models
    python autointerp.py --checkpoint path.pt     # interpret a specific checkpoint
    python autointerp.py --top-features 20        # fewer features
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import StackedSAE, TemporalCrosscoder

from config import (
    D_SAE, LAYER_SPECS, DEVICE, SEED,
    CACHE_DIR, CHECKPOINT_DIR, LOG_DIR, VIZ_DIR,
    AUTOINTERP_MODEL, AUTOINTERP_MAX_EXAMPLES,
    AUTOINTERP_TOP_FEATURES, AUTOINTERP_BATCH_SIZE,
)


def find_best_checkpoints(log_dir: str, checkpoint_dir: str) -> dict[str, dict]:
    """Find best StackedSAE and best TXCDR checkpoint by lowest final loss.

    Returns dict with keys "stacked_sae" and "txcdr", each mapping to
    {path, model, layer, k, T, loss}.
    """
    summary_path = os.path.join(log_dir, "sweep_summary.json")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(
            f"No sweep summary at {summary_path}. Run sweep.py first."
        )

    with open(summary_path) as f:
        summary = json.load(f)

    best: dict[str, dict] = {}
    for row in summary:
        model_type = row["model"]
        loss = row["final_loss"]
        ckpt_name = f"{model_type}__{row['layer']}__k{row['k']}__T{row['T']}.pt"
        ckpt_path = os.path.join(checkpoint_dir, ckpt_name)

        if not os.path.exists(ckpt_path):
            continue

        if model_type not in best or loss < best[model_type]["loss"]:
            best[model_type] = {
                "path": ckpt_path,
                "model": model_type,
                "layer": row["layer"],
                "k": row["k"],
                "T": row["T"],
                "loss": loss,
            }

    return best


def load_model_from_checkpoint(info: dict) -> torch.nn.Module:
    """Load a model from checkpoint info dict."""
    d_act = LAYER_SPECS[info["layer"]]["d_act"]
    if info["model"] == "stacked_sae":
        model = StackedSAE(d_in=d_act, d_sae=D_SAE, T=info["T"], k=info["k"])
    else:
        model = TemporalCrosscoder(d_in=d_act, d_sae=D_SAE, T=info["T"], k=info["k"])

    state = torch.load(info["path"], map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def get_top_activating_examples(
    model: torch.nn.Module,
    model_type: str,
    layer_key: str,
    T: int,
    n_features: int,
    n_examples: int,
    cache_dir: str,
) -> dict[int, list[dict]]:
    """For each of the top `n_features` most-active features, find the
    `n_examples` highest-activating windows.

    Returns: {feature_idx: [{activation, window_start, chain_idx}, ...]}
    """
    act_path = os.path.join(cache_dir, f"{layer_key}.npy")
    data = np.load(act_path, mmap_mode="r")
    num_chains = data.shape[0]
    seq_length = data.shape[1]
    n_windows = seq_length // T

    # Accumulate activation statistics across a sample of chains
    sample_chains = min(500, num_chains)
    chain_indices = np.random.choice(num_chains, size=sample_chains, replace=False)

    # Track: for each feature, keep top-n_examples activations
    from heapq import heappush, heapreplace

    feature_heaps: dict[int, list] = {}  # feature_idx -> min-heap of (activation, chain, window)

    with torch.no_grad():
        for ci in tqdm(chain_indices, desc="Scanning for top activations"):
            chain = torch.from_numpy(data[ci].copy()).float().unsqueeze(0)  # (1, L, d)
            for w in range(n_windows):
                start = w * T
                window = chain[:, start : start + T, :]
                if model_type == "stacked_sae":
                    _, _, u = model(window)
                    # Use mean activation across positions
                    acts = u.squeeze(0).mean(dim=0).numpy()  # (h,)
                else:
                    _, _, z = model(window)
                    acts = z.squeeze(0).numpy()  # (h,)

                for feat_idx in range(len(acts)):
                    val = acts[feat_idx]
                    if val <= 0:
                        continue
                    entry = (float(val), int(ci), int(start))
                    if feat_idx not in feature_heaps:
                        feature_heaps[feat_idx] = []
                    heap = feature_heaps[feat_idx]
                    if len(heap) < n_examples:
                        heappush(heap, entry)
                    elif val > heap[0][0]:
                        heapreplace(heap, entry)

    # Select top features by total activation mass
    feature_total = {
        idx: sum(e[0] for e in heap)
        for idx, heap in feature_heaps.items()
    }
    top_features = sorted(feature_total, key=feature_total.get, reverse=True)[:n_features]

    results: dict[int, list[dict]] = {}
    for idx in top_features:
        examples = sorted(feature_heaps[idx], key=lambda x: -x[0])
        results[idx] = [
            {"activation": e[0], "chain_idx": e[1], "window_start": e[2]}
            for e in examples
        ]

    return results


def get_token_context(
    chain_idx: int, window_start: int, T: int,
    tokenizer, token_ids: np.ndarray, context_window: int = 10,
) -> str:
    """Get the text context around a window position."""
    start = max(0, window_start - context_window)
    end = min(token_ids.shape[1], window_start + T + context_window)
    tokens = token_ids[chain_idx, start:end]
    text = tokenizer.decode(tokens, skip_special_tokens=True)

    # Mark the actual window
    pre_tokens = token_ids[chain_idx, start:window_start]
    window_tokens = token_ids[chain_idx, window_start:window_start + T]
    post_tokens = token_ids[chain_idx, window_start + T:end]

    pre = tokenizer.decode(pre_tokens, skip_special_tokens=True)
    window = tokenizer.decode(window_tokens, skip_special_tokens=True)
    post = tokenizer.decode(post_tokens, skip_special_tokens=True)

    return f"{pre}>>>{window}<<<{post}"


def interpret_features_with_claude(
    feature_examples: dict[int, list[str]],
    model_label: str,
    batch_size: int = AUTOINTERP_BATCH_SIZE,
) -> dict[int, str]:
    """Send feature examples to Claude Haiku for interpretation.

    Args:
        feature_examples: {feature_idx: [context_string, ...]}
        model_label: e.g. "stacked_sae_mid_res_k50_T10"

    Returns: {feature_idx: interpretation_string}
    """
    import anthropic

    client = anthropic.Anthropic()
    interpretations: dict[int, str] = {}

    feature_indices = list(feature_examples.keys())

    for batch_start in tqdm(
        range(0, len(feature_indices), batch_size),
        desc=f"Interpreting {model_label}",
    ):
        batch_indices = feature_indices[batch_start : batch_start + batch_size]

        for feat_idx in batch_indices:
            examples = feature_examples[feat_idx]
            examples_text = "\n".join(
                f"  {i+1}. {ex}" for i, ex in enumerate(examples)
            )

            prompt = (
                f"You are analyzing features learned by a sparse autoencoder "
                f"trained on a language model's internal activations.\n\n"
                f"Below are the top-activating text contexts for feature #{feat_idx}. "
                f"The text between >>> and <<< is the window where this feature fires "
                f"most strongly.\n\n"
                f"Examples:\n{examples_text}\n\n"
                f"In 1-2 sentences, describe what concept or pattern this feature "
                f"appears to detect. Be specific and concrete."
            )

            response = client.messages.create(
                model=AUTOINTERP_MODEL,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            interpretations[feat_idx] = response.content[0].text

    return interpretations


def main():
    parser = argparse.ArgumentParser(description="Autointerp pipeline with Claude Haiku")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint to interpret")
    parser.add_argument("--model-type", type=str, default=None,
                        choices=["stacked_sae", "txcdr"])
    parser.add_argument("--layer", type=str, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)
    parser.add_argument("--top-features", type=int, default=AUTOINTERP_TOP_FEATURES)
    parser.add_argument("--max-examples", type=int, default=AUTOINTERP_MAX_EXAMPLES)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    output_dir = args.output_dir or os.path.join(VIZ_DIR, "autointerp")
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    from transformers import AutoTokenizer
    from config import MODEL_NAME
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Load token IDs
    token_path = os.path.join(CACHE_DIR, "token_ids.npy")
    if not os.path.exists(token_path):
        raise FileNotFoundError(f"Token IDs not found at {token_path}. Run cache_activations.py first.")
    token_ids = np.load(token_path)

    # Determine which checkpoints to interpret
    if args.checkpoint:
        if not all([args.model_type, args.layer, args.k, args.T]):
            raise ValueError(
                "When using --checkpoint, also provide --model-type, --layer, --k, --T"
            )
        models_to_interp = {
            args.model_type: {
                "path": args.checkpoint,
                "model": args.model_type,
                "layer": args.layer,
                "k": args.k,
                "T": args.T,
                "loss": 0.0,
            }
        }
    else:
        print("Finding best checkpoints by lowest loss...")
        models_to_interp = find_best_checkpoints(LOG_DIR, CHECKPOINT_DIR)
        for key, info in models_to_interp.items():
            print(f"  Best {key}: {info['layer']} k={info['k']} T={info['T']} loss={info['loss']:.4f}")

    if not models_to_interp:
        print("ERROR: No checkpoints found. Run sweep.py with --save-checkpoints first.")
        sys.exit(1)

    all_results: dict[str, dict] = {}

    for model_type, info in models_to_interp.items():
        label = f"{info['model']}_{info['layer']}_k{info['k']}_T{info['T']}"
        print(f"\n{'=' * 60}")
        print(f"  Interpreting: {label}")
        print(f"{'=' * 60}")

        # Load model
        model = load_model_from_checkpoint(info)

        # Get top activating examples
        top_examples = get_top_activating_examples(
            model, info["model"], info["layer"], info["T"],
            n_features=args.top_features,
            n_examples=args.max_examples,
            cache_dir=CACHE_DIR,
        )
        print(f"  Found activating examples for {len(top_examples)} features")

        # Convert to text contexts
        feature_contexts: dict[int, list[str]] = {}
        for feat_idx, examples in top_examples.items():
            contexts = []
            for ex in examples:
                ctx = get_token_context(
                    ex["chain_idx"], ex["window_start"], info["T"],
                    tokenizer, token_ids,
                )
                contexts.append(ctx)
            feature_contexts[feat_idx] = contexts

        # Interpret with Claude
        interpretations = interpret_features_with_claude(feature_contexts, label)

        # Save results
        result = {
            "model_info": info,
            "interpretations": {
                str(k): {
                    "interpretation": v,
                    "top_activation": top_examples[k][0]["activation"],
                    "num_activating_windows": len(top_examples[k]),
                }
                for k, v in interpretations.items()
            },
        }
        all_results[label] = result

        # Save per-model results
        result_path = os.path.join(output_dir, f"{label}_interp.json")
        with open(result_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {result_path}")

        # Print sample interpretations
        print(f"\n  Sample interpretations:")
        for feat_idx, interp in list(interpretations.items())[:5]:
            print(f"    Feature {feat_idx}: {interp}")

    # Save combined results
    combined_path = os.path.join(output_dir, "all_interpretations.json")
    with open(combined_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll interpretations saved to {combined_path}")


if __name__ == "__main__":
    main()
