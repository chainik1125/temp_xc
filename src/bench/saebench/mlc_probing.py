"""Multi-hook sparse-probing eval for MLC (layer-wise crosscoder).

SAEBench's stock `run_eval_single_dataset` hooks a single layer;
MLC takes simultaneous activations from a layer window
`{10, 11, 12, 13, 14}` and needs all 5 collected per forward pass.
This module forks the minimum pieces of SAEBench's probing pipeline
to support that.

Design decisions (see plan § 11):
  - Fork scope: collection + encode → everything downstream (mean-pool,
    feature selection, sklearn probe) reuses SAEBench utilities.
  - Text → labels → splits: reuses SAEBench's `dataset_utils` so the
    8-dataset probing suite + train/test splits + class balance match
    SAE / TempXC runs exactly.
  - Hook scheme: HookedTransformer's `run_with_hooks` with a
    `names_filter` covering all 5 resid_post hook points.
  - Caching: single `.pt` per (dataset, class) keyed by layer-stack —
    path differs from stock SAEBench's cache to avoid collisions.
  - Output: one JSONL record per (task, k) combination, matching
    `probing_runner.py`'s schema, so downstream aggregation works
    identically across architectures.

Usage:
    from src.bench.saebench.mlc_probing import run_mlc_probing
    run_mlc_probing(
        ckpt_path="results/saebench/ckpts/mlc__gemma-2-2b__l10-14__k100__protA__seed42.pt",
        protocol="A",
        aggregation="mean",
        output_jsonl="results/saebench/results/mlc_protA.jsonl",
        device="cuda:0",
    )
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.bench.saebench.aggregation import AggregationName, aggregate
from src.bench.saebench.configs import (
    CKPT_DIR,
    CONTEXT_LENGTH,
    D_MODEL,
    D_SAE,
    MLC_LAYERS,
    ProtocolName,
    RESULTS_DIR,
    SAEBENCH_ARTIFACTS_DIR,
    SUBJECT_MODEL,
    PROBING_K_VALUES,
)
from src.bench.saebench.matching_protocols import protocol_k


def _load_mlc(
    ckpt_path: str,
    k: int,
    n_layers: int,
    device: torch.device,
):
    """Instantiate LayerCrosscoder + load checkpoint."""
    from src.bench.architectures.mlc import LayerCrosscoderSpec

    spec = LayerCrosscoderSpec(n_layers=n_layers)
    model = spec.create(d_in=D_MODEL, d_sae=D_SAE, k=k, device=device)
    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return spec, model


def _collect_multi_layer_activations(
    llm,
    tokens: torch.Tensor,               # (B, L) token ids
    hook_names: list[str],              # 5 resid_post hook points
    mask_bos_pad: bool,
    bos_token_id: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Run `llm` forward on `tokens`, collect activations at all
    `hook_names` simultaneously, stack into `(B, L, n_layers, d_model)`.

    Returns float tensor on LLM's device. Zeroes out BOS/pad positions
    if mask_bos_pad is True (matches SAEBench's standard path).
    """
    captured: dict[str, torch.Tensor] = {}

    def make_hook(name: str):
        def hook_fn(acts, hook):
            captured[name] = acts.detach()
        return hook_fn

    fwd_hooks = [(name, make_hook(name)) for name in hook_names]
    with torch.no_grad():
        llm.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type=None)

    stacked = torch.stack([captured[n] for n in hook_names], dim=2)  # (B, L, n_layers, d_model)

    if mask_bos_pad:
        mask = ((tokens != bos_token_id) & (tokens != pad_token_id)).unsqueeze(-1).unsqueeze(-1)
        stacked = stacked * mask.to(stacked.dtype)

    return stacked.float()


def run_mlc_probing(
    ckpt_path: str,
    protocol: ProtocolName,
    aggregation: AggregationName,
    output_jsonl: str,
    device: str = "cuda:0",
    random_seed: int = 42,
    n_layers: int = 5,
    k_values: tuple[int, ...] = PROBING_K_VALUES,
    artifacts_path: str = SAEBENCH_ARTIFACTS_DIR,
    batch_size: int = 32,
) -> dict:
    """Fork of SAEBench's sparse_probing.run_eval for MLC.

    Collects multi-layer Gemma activations, encodes through MLC with
    the chosen aggregation, mean-pools over L (matching SAEBench's
    convention), trains sklearn logistic-regression probes using
    SAEBench's `probe_training.train_probe_on_activations`.

    Emits one JSONL record per (task, k) to `output_jsonl`.
    """
    # Deferred SAEBench imports — sidecar venv.
    from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
    from sae_bench.evals.sparse_probing import probe_training
    from sae_bench.sae_bench_utils import (
        activation_collection,
        dataset_utils,
        general_utils,
    )
    from transformer_lens import HookedTransformer

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    torch_device = torch.device(device)

    # 1. Load MLC
    k = protocol_k("mlc", protocol)
    spec, model = _load_mlc(ckpt_path, k, n_layers, torch_device)

    # 2. Load Gemma via HookedTransformer (same path as SAEBench)
    cfg = SparseProbingEvalConfig(
        random_seed=random_seed,
        model_name=SUBJECT_MODEL,
    )
    cfg.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[SUBJECT_MODEL]
    cfg.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[SUBJECT_MODEL]
    cfg.k_values = list(k_values)

    print(f"[mlc_probing] loading {SUBJECT_MODEL} on {device}...")
    dtype = getattr(torch, cfg.llm_dtype)
    llm = HookedTransformer.from_pretrained_no_processing(
        cfg.model_name,
        dtype=dtype,
    ).to(torch_device)
    llm.eval()

    # Hook points for all n_layers
    hook_names = [f"blocks.{l}.hook_resid_post" for l in MLC_LAYERS[:n_layers]]
    print(f"[mlc_probing] hooks: {hook_names}")

    # 3. For each dataset, collect acts → encode via MLC → probe
    results_per_task: list[dict] = []
    run_id = f"mlc__prot{protocol}__agg{aggregation}"

    for dataset_name in cfg.dataset_names:
        t_ds = time.time()
        print(f"\n[mlc_probing] ── dataset: {dataset_name} ──")

        # Pull labeled train/test splits (same call SAEBench uses)
        train_data, test_data = dataset_utils.get_multi_label_train_test_data(
            dataset_name=dataset_name,
            train_set_size=cfg.probe_train_set_size,
            test_set_size=cfg.probe_test_set_size,
            random_seed=cfg.random_seed,
        )

        class_names = sorted(train_data.keys())
        per_class_test_accs: dict[int, list[float]] = {kv: [] for kv in k_values}

        for class_idx, class_name in enumerate(class_names):
            train_pos_texts = train_data[class_name]
            test_pos_texts = test_data[class_name]
            # Negative pool: all other classes combined
            train_neg_texts = [
                t for c, texts in train_data.items() if c != class_name for t in texts
            ]
            test_neg_texts = [
                t for c, texts in test_data.items() if c != class_name for t in texts
            ]

            def encode_texts(pos_texts, neg_texts):
                all_texts = list(pos_texts) + list(neg_texts)
                labels = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
                # Tokenize in batches, hook Gemma, encode through MLC
                feats_out = []
                for i in range(0, len(all_texts), batch_size):
                    chunk = all_texts[i : i + batch_size]
                    # SAEBench's SparseProbingEvalConfig doesn't carry prepend_bos /
                    # context_length; the stock pipeline hardcodes them. Mirror that
                    # here. Gemma-2's tokenizer requires BOS, and our pre-registered
                    # context length is 128 (configs.CONTEXT_LENGTH).
                    toks = llm.to_tokens(chunk, prepend_bos=True)
                    toks = toks[:, :CONTEXT_LENGTH]
                    # pad token id for gemma-2-2b
                    pad_id = getattr(llm.tokenizer, "pad_token_id", 0) or 0
                    bos_id = getattr(llm.tokenizer, "bos_token_id", 2) or 2

                    stacked = _collect_multi_layer_activations(
                        llm, toks, hook_names,
                        mask_bos_pad=True, bos_token_id=bos_id, pad_token_id=pad_id,
                    )  # (B, L, n_layers, d_model)

                    # Encode via MLC: treat the layer axis like TempXC's T
                    with torch.no_grad():
                        B, L, NL, D = stacked.shape
                        # _encode_window treats the 2nd axis as the "window" axis
                        flat = stacked.reshape(B * L, NL, D)
                        z_flat = spec._encode_window(model, flat)  # (B*L, NL, d_sae)
                        z = z_flat.view(B, L, NL, -1)

                    # For MLC, aggregation operates on the layer axis.
                    # Our aggregate() function treats the 3rd axis as "T",
                    # so we pass (B, L, NL, d_sae) directly.
                    z_agg = aggregate(z, aggregation)  # (B, L, d_sae_eff)

                    # Mean-pool over non-zero (non-BOS/pad) positions
                    token_mask = (toks != bos_id) & (toks != pad_id)
                    token_mask = token_mask.unsqueeze(-1).float()
                    pooled = (z_agg * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
                    feats_out.append(pooled.cpu())
                feats = torch.cat(feats_out, dim=0)  # (N, d_sae_eff)
                return feats, labels

            train_feats, train_labels = encode_texts(train_pos_texts, train_neg_texts)
            test_feats, test_labels = encode_texts(test_pos_texts, test_neg_texts)

            # For each k, select top-k features + fit probe + measure accuracy
            for kv in k_values:
                # Class-separation score: mean difference between classes
                pos_mean = train_feats[train_labels == 1].mean(dim=0)
                neg_mean = train_feats[train_labels == 0].mean(dim=0)
                score = (pos_mean - neg_mean).abs()
                top_k_feat_idx = score.topk(kv).indices.numpy()

                X_train = train_feats[:, top_k_feat_idx].numpy()
                X_test = test_feats[:, top_k_feat_idx].numpy()

                acc = probe_training.train_probe_on_activations(
                    X_train, train_labels, X_test, test_labels,
                )
                per_class_test_accs[kv].append(float(acc))

        # Average across classes for each k
        ds_result = {"dataset_name": dataset_name}
        for kv in k_values:
            if per_class_test_accs[kv]:
                ds_result[f"sae_top_{kv}_test_accuracy"] = float(
                    np.mean(per_class_test_accs[kv])
                )
        results_per_task.append(ds_result)
        print(f"  [{dataset_name}] done in {time.time() - t_ds:.1f}s; "
              f"top-5 acc = {ds_result.get('sae_top_5_test_accuracy', 'N/A')}")

    # 4. Write JSONL
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    n_written = 0
    params = sum(p.numel() for p in model.parameters())
    ckpt_basename = os.path.basename(ckpt_path)
    with open(output_jsonl, "a") as fout:
        for row in results_per_task:
            for kv in k_values:
                acc_key = f"sae_top_{kv}_test_accuracy"
                if acc_key not in row:
                    continue
                rec = {
                    "architecture": "mlc",
                    "t": n_layers,
                    "matching_protocol": protocol,
                    "aggregation": aggregation,
                    "task": row["dataset_name"],
                    "k": kv,
                    "accuracy": row[acc_key],
                    "param_count": params,
                    "training_flops": None,
                    "checkpoint_path": ckpt_path,
                    "checkpoint_basename": ckpt_basename,
                    "saebench_run_id": run_id,
                }
                fout.write(json.dumps(rec) + "\n")
                n_written += 1

    del model, llm
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "n_records_written": n_written,
        "n_tasks": len(results_per_task),
    }
