"""Per-concept-class TXC vs T-SAE breakdown plot.

At each architecture's peak s_norm under family-normalised paper-clamp,
group the 30 concepts into:
  - knowledge / domain (medical, mathematical, historical, code,
    scientific, religious, geographical, financial)
  - discourse / register (dialogue, casual_register, formal_register,
    imperative_form, instructional, question_form, narrative)
  - safety / alignment (harmful_content, deception, refusal_pattern,
    jailbreak_pattern, helpfulness_marker)
  - format / stylistic (poetic, literary, list_format, citation_pattern,
    technical_jargon, code_context)
  - sentiment (positive_emotion, negative_emotion, neutral_factual)

Show mean success per arch per class. The hypothesis: TXC family wins
on knowledge concepts; T-SAE k=20 wins on discourse/safety concepts.

Output:
  results/case_studies/plots/phase7_steering_v2_concept_classes.png
"""
from __future__ import annotations

import argparse
import collections
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

from experiments.phase7_unification._paths import OUT_DIR, banner
from experiments.phase7_unification.case_studies._paths import CASE_STUDIES_DIR


CONCEPT_CLASSES = {
    "knowledge / domain":   {"medical", "mathematical", "historical",
                             "code_context", "scientific", "religious",
                             "geographical", "financial", "programming"},
    "discourse / register": {"dialogue", "casual_register", "formal_register",
                             "imperative_form", "instructional", "question_form",
                             "narrative", "neutral_factual"},
    "safety / alignment":   {"harmful_content", "deception", "refusal_pattern",
                             "jailbreak_pattern", "helpfulness_marker", "legal"},
    "stylistic":            {"poetic", "literary", "list_format", "citation_pattern",
                             "technical_jargon"},
    "sentiment":            {"positive_emotion", "negative_emotion"},
}


ARCHS = [
    ("tsae_paper_k20",                          "T-SAE k=20"),
    ("tsae_paper_k500",                         "T-SAE k=500"),
    ("topk_sae",                                "TopKSAE k=500"),
    ("agentic_txc_02",                          "TXC matryoshka T=5"),
    ("phase57_partB_h8_bare_multidistance_t5",  "H8 multi-dist T=5"),
    ("phase5b_subseq_h8",                       "SubseqH8 T=10"),
]
ARCH_COLOR = {
    "tsae_paper_k20": "#d62728",
    "tsae_paper_k500": "#ff7f0e",
    "topk_sae": "#1f77b4",
    "agentic_txc_02": "#2ca02c",
    "phase57_partB_h8_bare_multidistance_t5": "#e377c2",
    "phase5b_subseq_h8": "#8c564b",
}


def _peak_per_concept(arch_id, subdir):
    """Returns {concept_id: success} at the arch's peak s_norm."""
    p_grades = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    p_gens = CASE_STUDIES_DIR / subdir / arch_id / "generations.jsonl"
    if not p_grades.exists() or not p_gens.exists():
        return {}
    rows = [json.loads(l) for l in p_grades.open()]
    by_s = collections.defaultdict(list)
    for r in rows:
        if r.get("success_grade") is None:
            continue
        by_s[float(r["strength"])].append(r)
    peaks = {s: np.mean([r["success_grade"] for r in pairs]) for s, pairs in by_s.items()}
    if not peaks:
        return {}
    peak_s = max(peaks, key=peaks.get)
    return {r["concept_id"]: r["success_grade"] for r in by_s[peak_s]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "phase7_steering_v2_concept_classes.png")
    args = ap.parse_args()
    banner(__file__)

    # Per arch, per class, mean success at peak
    per_arch_class = {}
    for arch_id, _ in ARCHS:
        per_concept = _peak_per_concept(arch_id, "steering_paper_normalised")
        per_class = {}
        for cls, concept_set in CONCEPT_CLASSES.items():
            scores = [per_concept[c] for c in concept_set if c in per_concept]
            if scores:
                per_class[cls] = float(np.mean(scores))
            else:
                per_class[cls] = 0.0
        per_arch_class[arch_id] = per_class

    classes = list(CONCEPT_CLASSES.keys())
    arch_ids = [a for a, _ in ARCHS]

    fig, ax = plt.subplots(figsize=(13, 6))
    n_archs = len(arch_ids)
    bar_w = 0.8 / n_archs
    x = np.arange(len(classes))

    for i, (arch_id, label) in enumerate(ARCHS):
        ys = [per_arch_class[arch_id][cls] for cls in classes]
        ax.bar(x + (i - (n_archs - 1) / 2) * bar_w, ys, bar_w,
               label=label, color=ARCH_COLOR.get(arch_id, "black"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_ylabel("Mean steering success (Sonnet 4.6, 0-3)")
    ax.set_title(
        "Per-concept-class steering success at each arch's peak s_norm\n"
        "TXC family wins on knowledge / domain concepts; T-SAE k=20 wins on discourse / register",
        fontsize=11,
    )
    ax.set_ylim(0, 3.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"  saved {args.out}")

    print("\n## Per-class mean steering success per arch")
    print()
    print("| arch | " + " | ".join(classes) + " |")
    print("|" + "---|" * (1 + len(classes)))
    for arch_id, label in ARCHS:
        row = [f"{per_arch_class[arch_id][cls]:.2f}" for cls in classes]
        print(f"| `{arch_id}` | " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
