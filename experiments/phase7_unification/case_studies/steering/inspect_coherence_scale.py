"""Sanity-check the Sonnet coherence grade scale.

Pulls steering generations + grades from a broad pool of arch+protocol
results, buckets them by coherence grade (0/1/2/3), samples N examples
per bucket, and prints them with (success, coherence, concept, strength)
labels. Lets us eyeball whether the grades match human intuition.

Run: TQDM_DISABLE=1 .venv/bin/python -m \\
    experiments.phase7_unification.case_studies.steering.inspect_coherence_scale --n 50
"""
from __future__ import annotations
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")

# Sample from a diverse set of (arch, subdir) pairs to get broad protocol+arch coverage
POOL = [
    ("tsae_paper_k20", "steering_paper_normalised"),
    ("txc_h8_t2_kpos20_shifts2", "steering_paper_normalised"),
    ("txc_h8_t2_kpos20_shifts2", "steering_paper_window_perposition"),
    ("txc_softmaxpool_t2_kpos20", "steering_paper_window_perposition"),
    ("txc_softmaxpool_t3_kpos20", "steering_paper_window_tiled_broadcast"),
    ("txc_bare_antidead_t2_kpos20", "steering_paper_normalised"),
    ("txc_softmax_pool_h8_t2_kpos20_shifts2", "steering_paper_window_tiled_broadcast"),
    ("txc_maxpool_t2_kpos20", "steering_paper_window_perposition"),
]


def load_pair(arch, subdir):
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists():
        return []
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    if len(gens) != len(grads):
        return []
    out = []
    for gg, rr in zip(gens, grads):
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None:
            continue
        out.append({
            "arch": arch,
            "subdir": subdir,
            "concept": gg["concept_id"],
            "concept_desc": rr.get("concept_desc", ""),
            "strength": gg.get("strength"),
            "s_norm": gg.get("s_norm"),
            "succ": rr["success_grade"],
            "coh": rr["coherence_grade"],
            "prompt": gg["prompt"],
            "text": gg["generated_text"],
        })
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="examples per coherence bucket")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    random.seed(args.seed)

    # Pool all examples across (arch, subdir) pairs
    all_examples = []
    for arch, subdir in POOL:
        all_examples.extend(load_pair(arch, subdir))

    print(f"Loaded {len(all_examples)} graded examples from {len(POOL)} (arch, subdir) pairs.\n")

    by_coh = defaultdict(list)
    for ex in all_examples:
        by_coh[ex["coh"]].append(ex)

    print(f"Coherence-grade distribution: " +
          ", ".join(f"{c}: {len(by_coh.get(c, []))}" for c in [0, 1, 2, 3]))
    print()

    # Sample N per bucket
    out_path = BASE / "plots" / "coherence_scale_inspection.md"
    lines = ["# Coherence-grade sanity check\n",
             f"Random {args.n} examples per bucket (coh ∈ {{0, 1, 2, 3}}) from " +
             f"{len(POOL)} (arch, subdir) pairs.\n",
             ""]
    for coh in [3, 2, 1, 0]:
        bucket = by_coh.get(coh, [])
        if not bucket:
            lines.append(f"\n## coh = {coh} (no examples)\n")
            continue
        sample = random.sample(bucket, min(args.n, len(bucket)))
        lines.append(f"\n## coh = {coh} ({len(sample)} of {len(bucket)} examples)\n")
        for i, ex in enumerate(sample, 1):
            text = ex["text"][:280].replace("\n", " ").strip()
            lines.append(f"### {i}. coh={ex['coh']} succ={ex['succ']} | concept={ex['concept']} | s={ex['strength']:.1f} | arch={ex['arch'][:32]} | proto={ex['subdir'].replace('steering_paper_', '')}")
            lines.append(f"prompt: `{ex['prompt']}`")
            lines.append(f"text: `{text}{'...' if len(ex['text']) > 280 else ''}`")
            lines.append("")
    out_path.write_text("\n".join(lines))
    print(f"Wrote {out_path}")
    print(f"  ({sum(len(by_coh.get(c, [])) for c in [0,1,2,3])} graded examples; sampled {args.n} per bucket)")


if __name__ == "__main__":
    main()
