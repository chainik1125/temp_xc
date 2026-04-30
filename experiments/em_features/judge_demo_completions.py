"""Take a demo.json (output of gen_steered_demo.py), run the Gemini judge on
every (prompt, α) completion, and write back per-rollout align/coh scores.

    uv run python -m experiments.em_features.judge_demo_completions \\
        --demo /root/em_features/results/steering_demo_feat4563.json \\
        --out  /root/em_features/results/steering_demo_feat4563.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EM_FEATURES = Path("/root/em_features")
for p in (str(REPO_ROOT), str(EM_FEATURES)):
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--demo", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--judge_model", default="gemini-3.1-flash-lite-preview")
    p.add_argument("--judge_temperature", type=float, default=0.5)
    p.add_argument("--max_concurrent", type=int, default=12)
    return p.parse_args()


def main():
    args = parse_args()
    from experiments.em_features.gemini_judge import evaluate_generations_with_gemini

    d = json.loads(args.demo.read_text())
    alphas = d["meta"]["alphas"]

    # Build a flat list of generations to judge: list of {question, answer}
    flat_generations = []
    flat_index = []   # (prompt_idx, alpha_str)
    for prompt_idx, p in enumerate(d["prompts"]):
        q = p["question"]
        for a in alphas:
            aKey = f"{a:+.2f}"
            text = p["completions_by_alpha"].get(aKey, "")
            flat_generations.append({"question": q, "answer": text})
            flat_index.append((prompt_idx, aKey))

    print(f"judging {len(flat_generations)} (prompt, α) cells × 2 (align+coh) calls...",
          flush=True)
    align_scores, coh_scores = asyncio.run(
        evaluate_generations_with_gemini(
            flat_generations,
            model_name=args.judge_model,
            max_concurrent=args.max_concurrent,
            temperature=args.judge_temperature,
        )
    )
    print(f"got {sum(1 for s in align_scores if s is not None)}/{len(align_scores)} valid align scores",
          flush=True)

    # Write per-rollout scores back into demo.json
    if "rollout_scores_by_alpha" not in d:
        d["rollout_scores_by_alpha"] = {}
    # We use a per-prompt structure: prompts[i].judge_by_alpha[α] = {align, coh}
    for prompt in d["prompts"]:
        prompt.setdefault("judge_by_alpha", {})

    for (prompt_idx, aKey), align, coh in zip(flat_index, align_scores, coh_scores):
        d["prompts"][prompt_idx]["judge_by_alpha"][aKey] = {
            "align": align, "coh": coh,
        }

    d["meta"]["judged"] = True
    d["meta"]["judge_model"] = args.judge_model
    args.out.write_text(json.dumps(d, indent=2))
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
