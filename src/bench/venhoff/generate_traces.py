"""Reasoning-trace generation for the Venhoff pipeline.

Generates `<think>...</think>` traces from DeepSeek-R1-Distill-Llama-8B
(or equivalent thinking model) on MMLU-Pro examples. Primary engine is
vLLM — fastest batched generation on 8B. Transformers fallback exists
for laptops / CI.

Resume: if `traces.json` + sidecar metadata already exists and matches
the current config hash, the run is skipped. Use `--force` to rebuild.

Adapted from Venhoff `generate-responses/generate_responses.py` at
upstream commit `49a7f731ce693d813b9ae9a414f1739b992dbcef`. Differences:
  - `model_registry.py` drives model identity (no hardcoded HF paths)
  - output schema captures thinking_process + answer/answer_index for
    downstream scoring without re-parsing
  - no MAX_TOKENS_IN_INPUT prompt-length error: we just truncate the
    dataset-level prompt instead (tokenizer's own behavior)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.bench.model_registry import get_model_config
from src.bench.venhoff.dataset import MMLUProExample, load_mmlu_pro
from src.bench.venhoff.paths import ArtifactPaths, RunIdentity, can_resume, write_with_metadata
from src.bench.venhoff.responses import extract_thinking_process

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("venhoff.generate_traces")


DEFAULT_MAX_TOKENS = 1000
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0


@dataclass(frozen=True)
class Trace:
    """One reasoning trace produced by the subject model."""

    question_id: str
    category: str
    prompt: str
    full_response: str
    thinking_process: str
    answer: str
    answer_index: int


def generate_traces_vllm(
    model_name: str,
    examples: list[MMLUProExample],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    tensor_parallel_size: int | None,
    dtype: str,
) -> list[Trace]:
    """vLLM engine path — preferred on GPU."""
    from vllm import LLM, SamplingParams  # deferred import: vllm is heavy
    from transformers import AutoTokenizer

    cfg = get_model_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)

    # Apply the chat template (thinking models expect this).
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": ex.prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in examples
    ]

    llm = LLM(
        model=cfg.hf_path,
        tensor_parallel_size=tensor_parallel_size or 1,
        dtype=dtype,
        seed=seed,
    )
    sampling = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
    )
    outputs = llm.generate(prompts, sampling)

    traces: list[Trace] = []
    for ex, out in zip(examples, outputs):
        full_response = out.outputs[0].text
        traces.append(
            Trace(
                question_id=ex.question_id,
                category=ex.category,
                prompt=ex.prompt,
                full_response=full_response,
                thinking_process=extract_thinking_process(full_response),
                answer=ex.answer,
                answer_index=ex.answer_index,
            )
        )
    return traces


def generate_traces_transformers(
    model_name: str,
    examples: list[MMLUProExample],
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
    dtype: str,
    batch_size: int = 4,
) -> list[Trace]:
    """Transformers-.generate() fallback — slow but universally works."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = get_model_config(model_name)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}.get(
        dtype, torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(cfg.hf_path, torch_dtype=torch_dtype, device_map="auto")
    model.eval()

    torch.manual_seed(seed)

    traces: list[Trace] = []
    for i in range(0, len(examples), batch_size):
        chunk = examples[i : i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": ex.prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for ex in chunk
        ]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            out_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                top_p=top_p,
                pad_token_id=tokenizer.pad_token_id,
            )
        for ex, in_ids, full_ids in zip(chunk, inputs.input_ids, out_ids):
            new_tokens = full_ids[in_ids.shape[0]:]
            full_response = tokenizer.decode(new_tokens, skip_special_tokens=True)
            traces.append(
                Trace(
                    question_id=ex.question_id,
                    category=ex.category,
                    prompt=ex.prompt,
                    full_response=full_response,
                    thinking_process=extract_thinking_process(full_response),
                    answer=ex.answer,
                    answer_index=ex.answer_index,
                )
            )
    return traces


def generate(
    paths: ArtifactPaths,
    model_name: str,
    n_traces: int,
    engine: str = "vllm",
    max_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    tensor_parallel_size: int | None = None,
    dtype: str = "bfloat16",
    force: bool = False,
) -> Path:
    """Produce the traces artifact; return its path.

    Skips generation if a valid cached artifact is found (resume). Call
    with `force=True` to rebuild.
    """
    paths.ensure_dirs()
    out = paths.traces_json

    config = {
        "stage": "generate_traces",
        "model": model_name,
        "dataset": paths.identity.dataset,
        "split": paths.identity.dataset_split,
        "n_traces": n_traces,
        "seed": paths.identity.seed,
        "engine": engine,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "dtype": dtype,
    }

    if not force and can_resume(out, config):
        log.info("resume: traces already cached at %s", out)
        return out

    examples = list(
        load_mmlu_pro(
            split=paths.identity.dataset_split,
            limit=n_traces,
            seed=paths.identity.seed,
            shuffle=True,
        )
    )
    log.info("loaded %d MMLU-Pro examples; generating via %s", len(examples), engine)

    if engine == "vllm":
        traces = generate_traces_vllm(
            model_name=model_name,
            examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=paths.identity.seed,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
        )
    elif engine == "transformers":
        traces = generate_traces_transformers(
            model_name=model_name,
            examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=paths.identity.seed,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown engine: {engine!r}")

    payload = json.dumps([asdict(t) for t in traces], indent=2)
    write_with_metadata(out, payload, config)
    log.info("wrote %d traces to %s", len(traces), out)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--root", type=Path, default=Path("results/venhoff_eval"))
    p.add_argument("--model", default="deepseek-r1-distill-llama-8b")
    p.add_argument("--dataset", default="mmlu-pro")
    p.add_argument("--split", default="test")
    p.add_argument("--n-traces", type=int, default=1000)
    p.add_argument("--layer", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--engine", default="vllm", choices=["vllm", "transformers"])
    p.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--tensor-parallel-size", type=int, default=None)
    p.add_argument("--force", action="store_true", help="rebuild even if cached")
    args = p.parse_args(argv)

    paths = ArtifactPaths(
        root=args.root,
        identity=RunIdentity(
            model=args.model,
            dataset=args.dataset,
            dataset_split=args.split,
            n_traces=args.n_traces,
            layer=args.layer,
            seed=args.seed,
        ),
    )
    generate(
        paths=paths,
        model_name=args.model,
        n_traces=args.n_traces,
        engine=args.engine,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
