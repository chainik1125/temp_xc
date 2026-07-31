"""Generate Qwen conversations and extract per-turn activations with reference code.

The target-model generation delegates to ``assistant_axis.generate_response``.
Activation extraction delegates to ``ActivationExtractor``,
``ConversationEncoder``, and ``SpanMapper`` from the pinned reference checkout.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download

from experiments.power_spectrum.persona_drift_txc.protocol import (
    EXPERIMENT_ROOT,
    append_jsonl,
    config_digest,
    file_sha256,
    iter_jsonl,
    load_config,
    project_axis,
    validate_script_record,
    write_json,
)


def _reference_imports() -> dict[str, Any]:
    try:
        from assistant_axis import (
            VLLMGenerator,
            generate_response,
            load_axis,
            project,
        )
        from assistant_axis.internals import (
            ActivationExtractor,
            ConversationEncoder,
            ProbingModel,
            SpanMapper,
        )
    except ImportError as error:
        raise RuntimeError(
            "The pinned assistant-axis checkout is required. Set PYTHONPATH to "
            "safety-research/assistant-axis@a98961956072224eaf244eb289d6c01700b63795."
        ) from error
    return {
        "ActivationExtractor": ActivationExtractor,
        "ConversationEncoder": ConversationEncoder,
        "ProbingModel": ProbingModel,
        "SpanMapper": SpanMapper,
        "VLLMGenerator": VLLMGenerator,
        "generate_response": generate_response,
        "load_axis": load_axis,
        "project": project,
    }


def _load_probing_model(config: dict[str, Any]) -> Any:
    reference = _reference_imports()
    return reference["ProbingModel"](
        _subject_model_source(config),
        device="cuda:0" if torch.cuda.is_available() else None,
        dtype=torch.bfloat16,
    )


def _subject_model_source(config: dict[str, Any]) -> str:
    return snapshot_download(
        repo_id=config["subject_model"],
        revision=config["subject_model_revision"],
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_conversations(
    *,
    pm: Any,
    scripts_path: Path,
    output_path: Path,
    limit: int | None,
) -> None:
    config = load_config()
    generation = config["generation"]
    scripts = list(iter_jsonl(scripts_path))
    for record in scripts:
        validate_script_record(record, int(config["turns_per_conversation"]))

    completed = {row["conversation_id"] for row in iter_jsonl(output_path)}
    pending = [row for row in scripts if row["conversation_id"] not in completed]
    if limit is not None:
        pending = pending[:limit]

    reference = _reference_imports()
    generate_response = reference["generate_response"]
    domain_index = {domain: index for index, domain in enumerate(config["domains"])}

    for script in pending:
        seed = (
            int(generation["base_seed"])
            + 10_000 * domain_index[script["domain"]]
            + 100 * int(script["persona_id"])
            + int(script["topic_id"])
        )
        _seed_everything(seed)
        conversation: list[dict[str, str]] = []
        response_token_counts: list[int] = []
        for turn, user_message in enumerate(script["user_messages"]):
            conversation.append({"role": "user", "content": user_message})
            response = generate_response(
                pm.model,
                pm.tokenizer,
                conversation,
                max_new_tokens=int(generation["max_new_tokens"]),
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                do_sample=bool(generation["do_sample"]),
            ).strip()
            if not response:
                raise RuntimeError(f"{script['conversation_id']}: empty response at turn {turn}")
            conversation.append({"role": "assistant", "content": response})
            response_token_counts.append(
                len(pm.tokenizer(response, add_special_tokens=False)["input_ids"])
            )

        append_jsonl(
            output_path,
            {
                **script,
                "model": config["subject_model"],
                "generation_seed": seed,
                "response_token_counts": response_token_counts,
                "conversation": conversation,
            },
        )
        print(
            f"[generate] {script['conversation_id']} "
            f"turns={len(script['user_messages'])} "
            f"response_tokens={sum(response_token_counts)}",
            flush=True,
        )


def generate_conversations_vllm(
    *,
    scripts_path: Path,
    output_path: Path,
    limit: int | None,
    batch_size: int,
) -> None:
    """Generate multi-turn conversations with the reference batched generator."""
    if batch_size < 1:
        raise ValueError("generation batch size must be positive")
    config = load_config()
    generation = config["generation"]
    scripts = sorted(iter_jsonl(scripts_path), key=lambda row: row["conversation_id"])
    for record in scripts:
        validate_script_record(record, int(config["turns_per_conversation"]))
    completed = {row["conversation_id"] for row in iter_jsonl(output_path)}
    if limit is not None:
        scripts = scripts[:limit]
    pending_count = sum(row["conversation_id"] not in completed for row in scripts)
    if not pending_count:
        print("[generate-vllm] all requested conversations already exist", flush=True)
        return

    reference = _reference_imports()
    generator = reference["VLLMGenerator"](
        model_name=_subject_model_source(config),
        max_model_len=4096,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.92,
        temperature=float(generation["temperature"]),
        max_tokens=int(generation["max_new_tokens"]),
        top_p=float(generation["top_p"]),
    )
    generator.load()
    tokenizer = generator.llm.get_tokenizer()
    domain_index = {domain: index for index, domain in enumerate(config["domains"])}

    total_batches = math.ceil(len(scripts) / batch_size)
    saved = 0
    for batch_start in range(0, len(scripts), batch_size):
        # Regenerate a partially completed canonical batch in full. This keeps
        # prompt ordering and seeds identical after interruption; rows already
        # present are simply not appended a second time.
        batch = scripts[batch_start : batch_start + batch_size]
        batch_pending = [row for row in batch if row["conversation_id"] not in completed]
        if not batch_pending:
            continue
        conversations: list[list[dict[str, str]]] = [[] for _ in batch]
        response_token_counts: list[list[int]] = [[] for _ in batch]
        canonical_batch_index = batch_start // batch_size
        batch_seed = int(generation["base_seed"]) + batch_start
        generator.sampling_params.seed = batch_seed
        for turn in range(int(config["turns_per_conversation"])):
            for conversation, script in zip(conversations, batch, strict=True):
                conversation.append({"role": "user", "content": script["user_messages"][turn]})
            responses = generator.generate_batch(conversations)
            for conversation, response, counts in zip(
                conversations,
                responses,
                response_token_counts,
                strict=True,
            ):
                cleaned = response.strip()
                if not cleaned:
                    raise RuntimeError(f"empty vLLM response in batch={batch_start}, turn={turn}")
                conversation.append({"role": "assistant", "content": cleaned})
                counts.append(len(tokenizer(cleaned, add_special_tokens=False)["input_ids"]))
            print(
                f"[generate-vllm] batch={batch_start // batch_size + 1}/"
                f"{total_batches} turn={turn + 1}/"
                f"{config['turns_per_conversation']}",
                flush=True,
            )

        for script, conversation, counts in zip(
            batch, conversations, response_token_counts, strict=True
        ):
            if script["conversation_id"] in completed:
                continue
            conversation_seed = (
                int(generation["base_seed"])
                + 10_000 * domain_index[script["domain"]]
                + 100 * int(script["persona_id"])
                + int(script["topic_id"])
            )
            append_jsonl(
                output_path,
                {
                    **script,
                    "model": config["subject_model"],
                    "generation_backend": "assistant_axis.VLLMGenerator",
                    "generation_seed": batch_seed,
                    "canonical_generation_batch": canonical_batch_index,
                    "legacy_conversation_seed": conversation_seed,
                    "response_token_counts": counts,
                    "conversation": conversation,
                },
            )
            completed.add(script["conversation_id"])
            saved += 1
        print(
            f"[generate-vllm] saved={saved}/{pending_count}",
            flush=True,
        )


def _load_axis(config: dict[str, Any]) -> torch.Tensor:
    reference = _reference_imports()
    axis_path = hf_hub_download(
        repo_id=config["assistant_axis_repo"],
        filename=config["assistant_axis_file"],
        revision=config["assistant_axis_revision"],
        repo_type="dataset",
    )
    return reference["load_axis"](axis_path)


def extract_activations(
    *,
    pm: Any,
    conversations_path: Path,
    output_root: Path,
    batch_size: int,
    max_length: int,
    limit: int | None,
) -> None:
    config = load_config()
    layer = int(config["monitor_layer"])
    records = list(iter_jsonl(conversations_path))
    records.sort(key=lambda row: row["conversation_id"])
    if limit is not None:
        records = records[:limit]
    chunks = output_root / "chunks"
    chunks.mkdir(parents=True, exist_ok=True)

    reference = _reference_imports()
    encoder = reference["ConversationEncoder"](pm.tokenizer, pm.model_name)
    extractor = reference["ActivationExtractor"](pm, encoder)
    mapper = reference["SpanMapper"](pm.tokenizer)
    axis = _load_axis(config)
    chat_kwargs = {"enable_thinking": bool(config["generation"]["enable_thinking"])}

    for batch_start in range(0, len(records), batch_size):
        batch_records = records[batch_start : batch_start + batch_size]
        pending = [
            row for row in batch_records if not (chunks / f"{row['conversation_id']}.pt").exists()
        ]
        if not pending:
            continue
        conversations = [row["conversation"] for row in pending]
        batch_activations, batch_metadata = extractor.batch_conversations(
            conversations,
            layer=[layer],
            max_length=max_length,
            **chat_kwargs,
        )
        _ids, batch_spans, _span_metadata = encoder.build_batch_turn_spans(
            conversations, **chat_kwargs
        )
        mapped = mapper.map_spans(batch_activations, batch_spans, batch_metadata)

        for row, all_turns in zip(pending, mapped, strict=True):
            expected_turns = len(row["user_messages"])
            assistant_turns = all_turns[1::2, 0, :].detach().cpu()
            if assistant_turns.shape != (expected_turns, pm.hidden_size):
                raise RuntimeError(
                    f"{row['conversation_id']}: assistant activation shape "
                    f"{tuple(assistant_turns.shape)} != {(expected_turns, pm.hidden_size)}"
                )
            scores = project_axis(assistant_turns, axis, layer=layer)

            # Byte-for-byte formula agreement with the reference projector.
            reference_scores = torch.tensor(
                [
                    reference["project"](turn, axis, layer=layer, normalize=True)
                    for turn in assistant_turns
                ],
                dtype=torch.float32,
            )
            maximum_difference = float((scores - reference_scores).abs().max())
            if maximum_difference > 1e-4:
                raise RuntimeError(
                    f"{row['conversation_id']}: vectorized/reference projection "
                    f"difference={maximum_difference}"
                )
            torch.save(
                {
                    "conversation_id": row["conversation_id"],
                    "activations": assistant_turns.to(torch.bfloat16),
                    "axis_scores": scores.float(),
                    "reference_projection_max_abs_diff": maximum_difference,
                    "actual_token_length": batch_metadata["actual_lengths"][pending.index(row)],
                    "truncated_token_length": batch_metadata["truncated_lengths"][
                        pending.index(row)
                    ],
                },
                chunks / f"{row['conversation_id']}.pt",
            )
            print(
                f"[extract] {row['conversation_id']} "
                f"axis_first={float(scores[0]):.3f} axis_last={float(scores[-1]):.3f}",
                flush=True,
            )
        del batch_activations, mapped
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@torch.inference_mode()
def validate_reference_transcripts(
    *,
    pm: Any,
    reference_root: Path,
    output_root: Path,
    max_length: int,
) -> None:
    """Reproduce the paper's qualitative capped/unsteered trajectory ordering."""
    config = load_config()
    layer = int(config["monitor_layer"])
    transcript_root = reference_root / "transcripts" / "case_studies" / "qwen-3-32b"
    reference = _reference_imports()
    encoder = reference["ConversationEncoder"](pm.tokenizer, pm.model_name)
    extractor = reference["ActivationExtractor"](pm, encoder)
    mapper = reference["SpanMapper"](pm.tokenizer)
    axis = _load_axis(config)
    chat_kwargs = {"enable_thinking": bool(config["generation"]["enable_thinking"])}
    scenarios = ("jailbreak", "delusion", "selfharm")
    curves: dict[str, list[float]] = {}

    for scenario in scenarios:
        for condition in ("unsteered", "capped"):
            candidates = [
                transcript_root / f"{scenario}_{condition}.json",
                transcript_root / f"{scenario}_{condition}.json.json",
            ]
            path = next((candidate for candidate in candidates if candidate.exists()), None)
            if path is None:
                raise FileNotFoundError(f"missing reference transcript for {scenario}/{condition}")
            with path.open() as handle:
                conversation = json.load(handle)["conversation"]
            batch_activations, batch_metadata = extractor.batch_conversations(
                [conversation],
                layer=[layer],
                max_length=max_length,
                **chat_kwargs,
            )
            _ids, spans, _span_metadata = encoder.build_batch_turn_spans(
                [conversation], **chat_kwargs
            )
            mapped = mapper.map_spans(batch_activations, spans, batch_metadata)[0]
            assistant_turns = mapped[1::2, 0, :].detach().cpu()
            scores = project_axis(assistant_turns, axis, layer=layer)
            for turn, score in zip(assistant_turns, scores, strict=True):
                reference_score = reference["project"](turn, axis, layer=layer, normalize=True)
                if abs(float(score) - reference_score) > 1e-4:
                    raise RuntimeError("reference projector formula mismatch")
            curves[f"{scenario}_{condition}"] = scores.tolist()
            del batch_activations, mapped

    comparisons = {}
    for scenario in scenarios:
        unsteered = np.asarray(curves[f"{scenario}_unsteered"])
        capped = np.asarray(curves[f"{scenario}_capped"])
        comparisons[scenario] = {
            "unsteered_mean": float(unsteered.mean()),
            "capped_mean": float(capped.mean()),
            "capped_minus_unsteered_mean": float(capped.mean() - unsteered.mean()),
            "unsteered_final": float(unsteered[-1]),
            "capped_final": float(capped[-1]),
            "capped_minus_unsteered_final": float(capped[-1] - unsteered[-1]),
        }
        if comparisons[scenario]["capped_minus_unsteered_mean"] <= 0:
            raise RuntimeError(
                f"{scenario}: released capped transcript does not project above "
                "the unsteered transcript on average"
            )

    output_root.mkdir(parents=True, exist_ok=True)
    write_json(
        output_root / "reference_transcript_check.json",
        {
            "assistant_axis_code_commit": config["assistant_axis_code_commit"],
            "monitor_layer": layer,
            "curves": curves,
            "comparisons": comparisons,
        },
    )
    figure, axes = plt.subplots(1, len(scenarios), figsize=(11.2, 3.2), sharey=True)
    for axis_plot, scenario in zip(axes, scenarios, strict=True):
        for condition, color in (("unsteered", "#D55E00"), ("capped", "#0072B2")):
            values = curves[f"{scenario}_{condition}"]
            axis_plot.plot(
                np.arange(1, len(values) + 1),
                values,
                marker="o",
                markersize=2.5,
                label=condition,
                color=color,
            )
        axis_plot.axhline(
            float(config["safe_threshold"]),
            color="black",
            linestyle="--",
            linewidth=0.8,
        )
        axis_plot.set_title(scenario)
        axis_plot.set_xlabel("Assistant turn")
        axis_plot.grid(alpha=0.2)
    axes[0].set_ylabel("Assistant Axis projection (layer 32)")
    axes[-1].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output_root / "reference_transcript_check.png", dpi=220)
    figure.savefig(output_root / "reference_transcript_check.pdf")
    plt.close(figure)


def pack_activations(
    *,
    conversations_path: Path,
    output_root: Path,
) -> None:
    config = load_config()
    records = sorted(iter_jsonl(conversations_path), key=lambda row: row["conversation_id"])
    chunks = output_root / "chunks"
    activations: list[torch.Tensor] = []
    axis_scores: list[torch.Tensor] = []
    metadata: list[dict[str, Any]] = []
    reference_differences: list[float] = []
    actual_token_lengths: list[int] = []
    truncated_token_lengths: list[int] = []
    for record in records:
        path = chunks / f"{record['conversation_id']}.pt"
        if not path.exists():
            raise FileNotFoundError(f"missing activation chunk: {path}")
        chunk = torch.load(path, map_location="cpu", weights_only=False)
        activations.append(chunk["activations"])
        axis_scores.append(chunk["axis_scores"])
        reference_differences.append(chunk["reference_projection_max_abs_diff"])
        actual_token_lengths.append(int(chunk["actual_token_length"]))
        truncated_token_lengths.append(int(chunk["truncated_token_length"]))
        metadata.append(
            {
                key: record[key]
                for key in (
                    "conversation_id",
                    "domain",
                    "persona_id",
                    "persona",
                    "topic_id",
                    "topic",
                    "split",
                    "user_messages",
                    "response_token_counts",
                    "generation_seed",
                )
            }
        )
    tensor_path = output_root / "turn_activations.pt"
    torch.save(
        {
            "activations": torch.stack(activations),
            "axis_scores": torch.stack(axis_scores),
            "conversation_ids": [row["conversation_id"] for row in metadata],
        },
        tensor_path,
    )
    metadata_path = output_root / "metadata.jsonl"
    with metadata_path.open("w") as handle:
        for row in metadata:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    write_json(
        output_root / "activation_manifest.json",
        {
            "protocol_version": config["protocol_version"],
            "config_sha256": config_digest(config),
            "subject_model": config["subject_model"],
            "monitor_layer": config["monitor_layer"],
            "assistant_axis_repo": config["assistant_axis_repo"],
            "assistant_axis_file": config["assistant_axis_file"],
            "assistant_axis_code_commit": config["assistant_axis_code_commit"],
            "n_conversations": len(records),
            "turns_per_conversation": int(config["turns_per_conversation"]),
            "activation_shape": list(torch.stack(activations).shape),
            "reference_projection_max_abs_diff": max(reference_differences),
            "conversation_file_sha256": file_sha256(conversations_path),
            "activation_file_sha256": file_sha256(tensor_path),
            "metadata_file_sha256": file_sha256(metadata_path),
            "actual_token_length_min": min(actual_token_lengths),
            "actual_token_length_max": max(actual_token_lengths),
            "n_truncated_conversations": sum(
                actual != truncated
                for actual, truncated in zip(
                    actual_token_lengths,
                    truncated_token_lengths,
                    strict=True,
                )
            ),
        },
    )
    print(f"[pack] wrote {tensor_path}", flush=True)


def release_model(pm: Any) -> None:
    pm.close()
    del pm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "phase",
        choices=(
            "reference",
            "generate",
            "generate-vllm",
            "extract",
            "collect",
            "all",
            "pack",
        ),
    )
    parser.add_argument(
        "--scripts",
        type=Path,
        default=EXPERIMENT_ROOT / "data" / "user_scripts.jsonl",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "qwen_conversations.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=EXPERIMENT_ROOT / "artifacts" / "activations",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--generation-batch-size", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=Path(os.environ.get("ASSISTANT_AXIS_ROOT", "/workspace/assistant-axis")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.phase == "pack":
        pack_activations(
            conversations_path=args.conversations,
            output_root=args.output_root,
        )
        return
    if args.phase == "generate-vllm":
        generate_conversations_vllm(
            scripts_path=args.scripts,
            output_path=args.conversations,
            limit=args.limit,
            batch_size=args.generation_batch_size,
        )
        return

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    pm = _load_probing_model(load_config())
    try:
        if args.phase in {"reference", "all"}:
            validate_reference_transcripts(
                pm=pm,
                reference_root=args.reference_root,
                output_root=args.output_root,
                max_length=args.max_length,
            )
        if args.phase in {"generate", "collect", "all"}:
            generate_conversations(
                pm=pm,
                scripts_path=args.scripts,
                output_path=args.conversations,
                limit=args.limit,
            )
        if args.phase in {"extract", "collect", "all"}:
            extract_activations(
                pm=pm,
                conversations_path=args.conversations,
                output_root=args.output_root,
                batch_size=args.batch_size,
                max_length=args.max_length,
                limit=args.limit,
            )
    finally:
        release_model(pm)
    if args.phase in {"collect", "all"} and args.limit is None:
        pack_activations(
            conversations_path=args.conversations,
            output_root=args.output_root,
        )


if __name__ == "__main__":
    main()
