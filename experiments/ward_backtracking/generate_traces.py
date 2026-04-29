"""Run DeepSeek-R1-Distill on the 300 seed prompts.

Wraps the existing `src.bench.venhoff.generate_traces` machinery but with
our own prompt list (from seed_prompts.py) instead of MATH500/MMLU-Pro.

Output schema matches `Trace` dataclass in the venhoff module exactly so
downstream activation collection is a drop-in.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward.generate_traces")


@dataclass(frozen=True)
class _PromptExample:
    """Mimics the .question_id / .category / .prompt / .answer / .answer_index
    contract of MMLUProExample/MATH500Example so the venhoff generator can
    consume it without modification."""

    question_id: str
    category: str
    prompt: str
    answer: str = ""           # n/a — open-ended generation, no grader
    answer_index: int = -1
    subject: str = ""          # mirrors MATH500Example for _make_trace fallback


def _load_examples(prompts_path: Path) -> list[_PromptExample]:
    items = json.loads(prompts_path.read_text())
    return [
        _PromptExample(
            question_id=item["id"],
            category=item["category"],
            prompt=item["prompt"],
            subject=item["category"],
        )
        for item in items
    ]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--engine", default=None, choices=["vllm", "transformers"])
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())

    prompts_path = Path(cfg["paths"]["prompts"])
    out_path = Path(cfg["paths"]["traces"])
    if out_path.exists() and not args.force:
        existing = json.loads(out_path.read_text())
        log.info("[info] resume | %d traces exist at %s — pass --force to regenerate", len(existing), out_path)
        return 0

    examples = _load_examples(prompts_path)
    log.info("[data] %d prompts loaded from %s", len(examples), prompts_path)

    engine = args.engine or cfg["trace_gen"]["engine"]
    max_tokens = args.max_tokens or int(cfg["trace_gen"]["max_tokens"])
    temperature = float(cfg["trace_gen"]["temperature"])
    dtype = cfg["trace_gen"]["dtype"]

    # Reuse the venhoff generator paths verbatim — same schema/output.
    if engine == "vllm":
        from src.bench.venhoff.generate_traces import generate_traces_vllm
        traces = generate_traces_vllm(
            model_name="deepseek-r1-distill-llama-8b",
            examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=args.seed,
            tensor_parallel_size=1,
            dtype=dtype,
        )
    elif engine == "transformers":
        from src.bench.venhoff.generate_traces import generate_traces_transformers
        traces = generate_traces_transformers(
            model_name="deepseek-r1-distill-llama-8b",
            examples=examples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=1.0,
            seed=args.seed,
            dtype=dtype,
        )
    else:
        raise ValueError(f"unknown engine {engine!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps([asdict(t) for t in traces], indent=2))
    log.info("[done] saved %d traces | path=%s", len(traces), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
