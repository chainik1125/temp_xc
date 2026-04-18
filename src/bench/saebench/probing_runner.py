"""Wrapper around SAEBench's sparse-probing eval.

Takes a trained checkpoint + architecture spec + aggregation + protocol,
wraps the model in `SAEBenchAdapter`, runs SAEBench's `run_eval`, parses
the output JSON, and emits one JSONL record per
(arch, T, protocol, aggregation, task, k) combination into our
cross-run JSONL file.

Expects SAEBench installed in a sidecar env — sparse_probing's strict
dependency pins (numpy<2, datasets<4, sae_lens^6.22.2) conflict with
our main env. The invoking shell script activates the right venv.

See docs/aniket/experiments/sparse_probing/plan.md § 11 for the output
schema this emits.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path

import torch

from src.bench.architectures.base import ArchSpec
from src.bench.saebench.configs import (
    ArchName,
    CKPT_DIR,
    D_MODEL,
    D_SAE,
    HOOK_NAME,
    LAYER,
    PROBING_K_VALUES,
    ProtocolName,
    RESULTS_DIR,
    SAEBENCH_ARTIFACTS_DIR,
    SUBJECT_MODEL,
    ckpt_name,
)
from src.bench.saebench.aggregation import AggregationName
from src.bench.saebench.saebench_wrapper import SAEBenchAdapter


def _load_arch_and_model(
    arch: ArchName,
    ckpt_path: str,
    protocol: ProtocolName,
    t: int,
    device: torch.device,
):
    """Instantiate the right ArchSpec + model, load the checkpoint.

    Resolves the architecture's training-time TopK k from (arch, protocol, t)
    via `protocol_k`. This is load-bearing: TopKSAE / TemporalCrosscoder
    keep `k` as a plain Python int, not in `state_dict`, so if we instantiate
    with the wrong k the loaded weights work but `encode()` runs `topk(k, ...)`
    at the wrong sparsity — k=0 silently returns all zeros.
    """
    from src.bench.architectures.topk_sae import TopKSAESpec
    from src.bench.architectures.crosscoder import CrosscoderSpec
    from src.bench.architectures.mlc import LayerCrosscoderSpec
    from src.bench.saebench.matching_protocols import protocol_k

    k = protocol_k(arch, protocol, t=t)

    if arch == "sae":
        spec = TopKSAESpec()
        model = spec.create(d_in=D_MODEL, d_sae=D_SAE, k=k, device=device)
    elif arch == "tempxc":
        spec = CrosscoderSpec(T=t)
        model = spec.create(d_in=D_MODEL, d_sae=D_SAE, k=k, device=device)
    elif arch == "mlc":
        # MLC's own architecture loads fine; t here is reinterpreted as
        # n_layers (the layer window width). The probing path through
        # SAEBench isn't wired yet — see plan § 11.
        spec = LayerCrosscoderSpec(n_layers=t)
        model = spec.create(d_in=D_MODEL, d_sae=D_SAE, k=k, device=device)
    else:
        raise ValueError(f"unknown arch: {arch}")

    state = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return spec, model


def _param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def run_probing(
    arch: ArchName,
    ckpt_path: str,
    protocol: ProtocolName,
    t: int,
    aggregation: AggregationName,
    output_jsonl: str,
    k: int,
    k_values: tuple[int, ...] = PROBING_K_VALUES,
    device: str = "cuda:0",
    random_seed: int = 42,
    artifacts_path: str = SAEBENCH_ARTIFACTS_DIR,
    force_rerun: bool = True,
    shuffle_seed: int | None = None,
) -> dict:
    """Run SAEBench sparse_probing on one (checkpoint, aggregation) config.

    `force_rerun` defaults to True because our `run_id` embeds a
    placeholder `k=0` (k is swept inside SAEBench), so preflight and
    real-eval collide on the same cache key. Re-running is cheap since
    SAEBench's activation_collection caches Gemma activations to
    `artifacts_path` independently of the probe training.

    Side effects:
      - Runs SAEBench (which itself runs the Gemma forward pass + caches
        LLM activations to `artifacts_path`).
      - Appends one JSONL record per (task, k) combination to
        `output_jsonl`.

    Returns a dict with aggregate accuracy + FLOPs stats, useful for the
    orchestrator's summary print.
    """
    # Skip-if-JSONL-records-exist: if this (arch, protocol, T, aggregation)
    # tuple is already fully represented in the output JSONL from a prior
    # run (crash mid-sweep, restart), don't re-run. Saves ~1–2 hours across
    # a restarted eval phase. Delete the JSONL to force re-eval.
    if os.path.exists(output_jsonl):
        try:
            with open(output_jsonl) as fin:
                existing = [json.loads(line) for line in fin if line.strip()]
            hit = [
                r for r in existing
                if r.get("architecture") == arch
                and r.get("matching_protocol") == protocol
                and r.get("t") == t
                and r.get("aggregation") == aggregation
                and r.get("shuffle_seed") == shuffle_seed  # ordered vs shuffled distinct
            ]
            if hit:
                shuf_str = "shuffled" if shuffle_seed is not None else "ordered"
                print(
                    f"  SKIP — {len(hit)} records already exist for "
                    f"{arch} prot={protocol} T={t} agg={aggregation} ({shuf_str}). "
                    f"Delete {output_jsonl} to force re-eval."
                )
                shuf_tag = "__shuffled" if shuffle_seed is not None else "__ordered"
                return {
                    "run_id": f"{arch}__prot{protocol}__T{t}__agg{aggregation}{shuf_tag}",
                    "n_records_written": 0,
                    "elapsed_sec": 0.0,
                    "skipped": True,
                }
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARN: could not pre-check {output_jsonl} ({e}); running anyway")

    # SAEBench imports are deferred because they pull in numpy<2 +
    # sae_lens + transformer_lens — cost we only pay when actually
    # running an eval.
    import sae_bench.evals.sparse_probing.main as sp
    from sae_bench.evals.sparse_probing.eval_config import (
        SparseProbingEvalConfig,
    )
    from sae_bench.sae_bench_utils import (
        general_utils,
        activation_collection,
    )

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    torch_device = torch.device(device)

    # 1. Instantiate our architecture + load weights. The `k` arg is a
    # placeholder (probing k is swept inside SAEBench); the SAE's
    # training-time TopK k is resolved from (arch, protocol, t).
    spec, model = _load_arch_and_model(arch, ckpt_path, protocol, t, torch_device)

    # 2. MLC requires multi-hook collection; dispatch to the forked
    #    probing path for MLC. SAE and TempXC use SAEBench's stock flow.
    if arch == "mlc":
        from src.bench.saebench.mlc_probing import run_mlc_probing
        return run_mlc_probing(
            ckpt_path=ckpt_path,
            protocol=protocol,
            aggregation=aggregation,
            output_jsonl=output_jsonl,
            device=device,
            random_seed=random_seed,
            n_layers=t,  # t is reinterpreted as n_layers for MLC
            k_values=k_values,
            artifacts_path=artifacts_path,
            shuffle_seed=shuffle_seed,
        )

    # 2. Wrap in SAEBench-compatible adapter (SAE / TempXC)
    adapter = SAEBenchAdapter(
        arch=arch, spec=spec, model=model,
        t=t, aggregation=aggregation,
        shuffle_seed=shuffle_seed,
    )

    # 3. Configure SAEBench sparse-probing. Suffix the run_id with
    # shuffle status so ordered and shuffled runs don't collide in the
    # SAEBench output cache (same bug class as B11 w/ k=0 placeholder).
    shuffle_tag = "__shuffled" if shuffle_seed is not None else "__ordered"
    run_id = (
        f"{arch}__prot{protocol}__T{t}__agg{aggregation}__k{k}{shuffle_tag}"
    )
    cfg = SparseProbingEvalConfig(
        random_seed=random_seed,
        model_name=SUBJECT_MODEL,
    )
    cfg.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[SUBJECT_MODEL]
    cfg.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[SUBJECT_MODEL]
    cfg.k_values = list(k_values)
    # Lower SAE-encode batch size so (B, L, T*d_sae) output fits on H100
    # when full_window aggregation is in play. At T=20 full_window the
    # output is 20*d_sae=368k features, and the default 125 gives
    # 125 * 128 * 368640 * 4 = 23 GB, OOMing the ~10 GB we have free
    # after Gemma + SAEBench's cached activations are resident.
    cfg.sae_batch_size = 16

    selected_saes = [(run_id, adapter)]

    output_folder = os.path.join(RESULTS_DIR, "saebench_json", run_id)
    os.makedirs(output_folder, exist_ok=True)

    # 4. Run SAEBench eval (blocking; runs Gemma forward + probing)
    t0 = time.time()
    sp.run_eval(
        cfg,
        selected_saes,
        torch_device,
        output_folder,
        force_rerun=force_rerun,
        clean_up_activations=False,
        save_activations=True,
        artifacts_path=artifacts_path,
    )
    elapsed = time.time() - t0

    # 5. Parse SAEBench's JSON output and emit our JSONL records
    saebench_json_files = list(Path(output_folder).glob("*_eval_results.json"))
    if not saebench_json_files:
        raise RuntimeError(
            f"SAEBench did not write output JSON under {output_folder}; "
            f"check the run's stderr."
        )
    sb_out = json.loads(saebench_json_files[0].read_text())

    # 6. Flatten SAEBench's nested output into our JSONL schema
    params = _param_count(model)
    ckpt_basename = os.path.basename(ckpt_path)

    per_task_rows = sb_out.get("eval_result_details", [])
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    n_records_written = 0
    with open(output_jsonl, "a") as fout:
        for row in per_task_rows:
            dataset_name = row["dataset_name"]
            for k_val in k_values:
                acc_key = f"sae_top_{k_val}_test_accuracy"
                if acc_key not in row or row[acc_key] is None:
                    continue
                record = {
                    "architecture": arch,
                    "t": t,
                    "matching_protocol": protocol,
                    "aggregation": aggregation,
                    "task": dataset_name,
                    "k": k_val,
                    "accuracy": float(row[acc_key]),
                    "param_count": params,
                    "training_flops": None,  # filled by orchestrator from training log
                    "checkpoint_path": ckpt_path,
                    "checkpoint_basename": ckpt_basename,
                    "saebench_run_id": run_id,
                    "elapsed_sec": round(elapsed, 1),
                    "shuffle_seed": shuffle_seed,  # None → ordered run
                }
                fout.write(json.dumps(record) + "\n")
                n_records_written += 1

    del model, adapter
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "run_id": run_id,
        "n_records_written": n_records_written,
        "elapsed_sec": elapsed,
        "saebench_json_path": str(saebench_json_files[0]),
    }
