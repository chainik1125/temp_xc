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


# ─── Per-example prediction persistence (item 8) ────────────────────────

PREDICTIONS_DIR = "results/saebench/predictions"


def _write_predictions(
    run_id: str,
    dataset_name: str,
    rows: list[dict],
    example_id_to_text: dict[str, str],
) -> str:
    """Write per-example predictions to predictions/<run_id>/<task>.jsonl.

    Storage-split so text isn't duplicated across every (class, k):
      <task>.jsonl       — one row per (example_id, class, k): pred, prob, label
      <task>__texts.jsonl — one row per example_id: the raw text (shared)

    Downstream confusion-matrix analysis joins on example_id. Total
    footprint drops from ~3 GB to ~70 MB per (task, arch, aggregation).
    See eval_infra_lessons.md item 8.
    """
    safe_task = dataset_name.replace("/", "_")
    out_dir = Path(PREDICTIONS_DIR) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    preds_path = out_dir / f"{safe_task}.jsonl"
    with open(preds_path, "w") as fout:
        for r in rows:
            fout.write(json.dumps(r) + "\n")

    texts_path = out_dir / f"{safe_task}__texts.jsonl"
    if not texts_path.exists():  # text mapping is task-scoped, write once
        with open(texts_path, "w") as fout:
            for eid, text in example_id_to_text.items():
                fout.write(json.dumps({"example_id": eid, "text": text}) + "\n")
    return str(preds_path)


def _sanity_check_persistence(
    run_id: str,
    dataset_name: str,
    expected: dict,
    k_values: tuple[int, ...],
    rtol: float = 1e-10,
) -> None:
    """Recompute aggregate accuracy from persisted per-example predictions,
    assert it matches the probing loop's reported aggregate to machine
    precision. Catches silent persistence-layer drift — the "prediction
    row misaligned with label" class of bug.

    `expected` is the ds_result dict with sae_top_K_test_accuracy keys.
    """
    safe_task = dataset_name.replace("/", "_")
    path = Path(PREDICTIONS_DIR) / run_id / f"{safe_task}.jsonl"
    if not path.exists():
        raise AssertionError(
            f"item 8 sanity check: predictions JSONL missing for "
            f"{run_id}/{safe_task}. Expected at {path}."
        )
    rows = [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]
    # Per-class accuracy = mean(pred==label) over test examples for that class
    # at each k. Averaged across classes per k, should match the expected dict.
    from collections import defaultdict
    per_class: dict[int, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )  # [k][class] -> list of (1 if correct else 0)
    for r in rows:
        per_class[r["k"]][r["class"]].append(int(r["pred"] == r["label"]))
    for kv in k_values:
        key = f"sae_top_{kv}_test_accuracy"
        if key not in expected:
            continue
        per_class_accs = [
            np.mean(v) for v in per_class[kv].values() if v
        ]
        if not per_class_accs:
            raise AssertionError(
                f"item 8: no persisted predictions for k={kv} on "
                f"{dataset_name}."
            )
        recomputed = float(np.mean(per_class_accs))
        reported = float(expected[key])
        if abs(recomputed - reported) > rtol * max(1.0, abs(reported)):
            raise AssertionError(
                f"item 8 FAILED: persistence drift on {dataset_name} at k={kv}. "
                f"reported={reported:.12f}, recomputed={recomputed:.12f}, "
                f"|Δ|={abs(recomputed - reported):.3e}. This means the "
                f"persisted per-example predictions don't reproduce the "
                f"aggregate the probe loop reported — label alignment or "
                f"dataset ordering has drifted. See "
                f"docs/aniket/bench_harness/bug_ledger_check.md § Item 8."
            )


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
    convention), trains sklearn logistic-regression probes one-vs-rest
    per class (SAEBench's train_probe_on_activations takes a
    dict[class, Tensor] multi-class interface; we inline sklearn
    instead to keep the MLC path purely binary).

    Emits one JSONL record per (task, k) to `output_jsonl`.
    """
    # Deferred SAEBench imports — sidecar venv.
    from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
    from sae_bench.sae_bench_utils import (
        activation_collection,
        dataset_utils,
        general_utils,
    )
    from transformer_lens import HookedTransformer

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    torch_device = torch.device(device)
    t_start = time.time()

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

        # Encode every unique text ONCE, keyed by (split, class_idx_within_class).
        # Previously we re-encoded each text ~N_classes times (each text serving
        # as a negative for every other class). At 28 classes that was a 28×
        # Gemma-forward overhead per dataset. Now: encode once, per-class
        # label-masks are cheap.
        def encode_flat(texts):
            feats_out = []
            for i in range(0, len(texts), batch_size):
                chunk = texts[i : i + batch_size]
                toks = llm.to_tokens(chunk, prepend_bos=True)
                toks = toks[:, :CONTEXT_LENGTH]
                pad_id = getattr(llm.tokenizer, "pad_token_id", 0) or 0
                bos_id = getattr(llm.tokenizer, "bos_token_id", 2) or 2

                stacked = _collect_multi_layer_activations(
                    llm, toks, hook_names,
                    mask_bos_pad=True, bos_token_id=bos_id, pad_token_id=pad_id,
                )  # (B, L, n_layers, d_model)

                with torch.no_grad():
                    B, L, NL, D = stacked.shape
                    flat = stacked.reshape(B * L, NL, D)
                    z_flat = spec._encode_window(model, flat)
                    z = z_flat.view(B, L, NL, -1)

                z_agg = aggregate(z, aggregation)
                token_mask = (toks != bos_id) & (toks != pad_id)
                token_mask = token_mask.unsqueeze(-1).float()
                pooled = (z_agg * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1)
                feats_out.append(pooled.cpu())
            return torch.cat(feats_out, dim=0)  # (N, d_sae_eff)

        # Flatten into one long list per split; remember where each class lives.
        def flatten_with_index(data_dict, class_names):
            all_texts: list[str] = []
            class_ids: list[int] = []
            for ci, c in enumerate(class_names):
                texts = data_dict[c]
                all_texts.extend(texts)
                class_ids.extend([ci] * len(texts))
            return all_texts, np.array(class_ids, dtype=np.int64)

        train_texts_all, train_cls = flatten_with_index(train_data, class_names)
        test_texts_all, test_cls = flatten_with_index(test_data, class_names)
        print(f"  encoding {len(train_texts_all)} train + "
              f"{len(test_texts_all)} test texts through MLC "
              f"(aggregation={aggregation})...")
        t_enc = time.time()
        train_feats = encode_flat(train_texts_all)  # (N_train, d_sae_eff)
        test_feats = encode_flat(test_texts_all)    # (N_test,  d_sae_eff)
        print(f"  encode took {time.time() - t_enc:.1f}s")

        # Per-example prediction persistence (see item 8 + bug_ledger_check.md).
        # Storage-split: predictions store (example_id, class, k, pred, prob,
        # label) — text goes to a sibling <task>__texts.jsonl keyed by
        # example_id. Avoids N_classes × N_k × N_test copies of the text.
        per_example_preds: list[dict] = []
        # Build example_id → text map once per dataset. example_id is the test
        # row's position in test_texts_all, stable across all (class, k).
        example_id_to_text: dict[str, str] = {
            f"{dataset_name}__test_{i}":
                (t if isinstance(t, str) else str(t))
            for i, t in enumerate(test_texts_all)
        }

        from sklearn.linear_model import LogisticRegression
        for class_idx, class_name in enumerate(class_names):
            train_labels = (train_cls == class_idx).astype(np.int64)
            test_labels = (test_cls == class_idx).astype(np.int64)

            for kv in k_values:
                # Class-separation score on the pre-computed features.
                pos_mean = train_feats[train_labels == 1].mean(dim=0)
                neg_mean = train_feats[train_labels == 0].mean(dim=0)
                score = (pos_mean - neg_mean).abs()
                top_k_feat_idx = score.topk(kv).indices.numpy()

                X_train = train_feats[:, top_k_feat_idx].numpy()
                X_test = test_feats[:, top_k_feat_idx].numpy()

                probe = LogisticRegression(max_iter=1000, random_state=random_seed)
                probe.fit(X_train, train_labels)
                acc = probe.score(X_test, test_labels)
                per_class_test_accs[kv].append(float(acc))

                # Capture per-example predictions for the TEST split. example_id
                # is the test row's position in test_texts_all — stable across
                # all (class, k) iterations so downstream joins work.
                test_preds = probe.predict(X_test)
                test_probs = probe.predict_proba(X_test)[:, 1]  # P(class_name)
                for eid_idx, (p, prob, y) in enumerate(
                    zip(test_preds, test_probs, test_labels)
                ):
                    per_example_preds.append({
                        "example_id": f"{dataset_name}__test_{eid_idx}",
                        "class": class_name,
                        "k": int(kv),
                        "pred": int(p),
                        "prob": float(prob),
                        "label": int(y),
                    })

        # Average across classes for each k
        ds_result = {"dataset_name": dataset_name}
        for kv in k_values:
            if per_class_test_accs[kv]:
                ds_result[f"sae_top_{kv}_test_accuracy"] = float(
                    np.mean(per_class_test_accs[kv])
                )
        results_per_task.append(ds_result)
        # Persist this dataset's per-example predictions (preds + sibling text map)
        _write_predictions(run_id, dataset_name, per_example_preds, example_id_to_text)
        # Item 8: sanity-check that persisted aggregates recompute cleanly.
        _sanity_check_persistence(
            run_id=run_id, dataset_name=dataset_name,
            expected=ds_result, k_values=k_values,
        )
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
        "elapsed_sec": time.time() - t_start,
    }
