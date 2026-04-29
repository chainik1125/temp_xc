"""Compare TXC at ln1 (input_layernorm) vs TXC at resid_post.

Tests Y's pivot hypothesis: training the same arch (TXCBareAntidead T=5)
at the input_layernorm hook (pre-attention) makes its knowledge-domain
advantage grow vs the resid_post baseline. This is a within-Phase-7
replication target for the collaborator's TinyStories result.

Loads grades.jsonl for both archs, computes per-concept peak success
(max-over-strengths via paper-clamp normalised), aggregates by concept
class, prints a delta table, and writes a comparison bar plot.

Output:
  results/case_studies/plots/ln1_vs_resid_concept_class.png
  results/case_studies/plots/ln1_vs_resid_concept_class.html  (no — just png+thumb)
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
    ("txc_bare_antidead_t5",      "TXC bare T=5 @ resid_post"),
    ("txc_bare_antidead_t5_ln1",  "TXC bare T=5 @ ln1 (pre-attn)"),
]


def _peak_per_concept(arch_id, subdir):
    p_grades = CASE_STUDIES_DIR / subdir / arch_id / "grades.jsonl"
    if not p_grades.exists():
        print(f"  [warn] missing grades for {arch_id}")
        return {}, None
    rows = [json.loads(l) for l in p_grades.open()]
    by_s = collections.defaultdict(list)
    for r in rows:
        if r.get("success_grade") is None:
            continue
        by_s[float(r["strength"])].append(r)
    if not by_s:
        return {}, None
    peaks = {s: np.mean([r["success_grade"] for r in pairs]) for s, pairs in by_s.items()}
    peak_s = max(peaks, key=peaks.get)
    per_concept = {r["concept_id"]: r["success_grade"] for r in by_s[peak_s]}
    return per_concept, peak_s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subdir", default="steering_paper_normalised")
    ap.add_argument("--out", type=Path,
                    default=OUT_DIR / "case_studies" / "plots" / "ln1_vs_resid_concept_class.png")
    args = ap.parse_args()
    banner(__file__)

    per_arch_concept = {}
    per_arch_peak_s = {}
    for arch_id, _ in ARCHS:
        per_arch_concept[arch_id], per_arch_peak_s[arch_id] = _peak_per_concept(arch_id, args.subdir)

    per_arch_class = {}
    for arch_id in per_arch_concept:
        per_class = {}
        for cls, concept_set in CONCEPT_CLASSES.items():
            scores = [per_arch_concept[arch_id][c] for c in concept_set if c in per_arch_concept[arch_id]]
            per_class[cls] = float(np.mean(scores)) if scores else float("nan")
        per_arch_class[arch_id] = per_class

    classes = list(CONCEPT_CLASSES.keys())

    print("\n## Peak s_norm per arch")
    for arch_id, label in ARCHS:
        print(f"  {label:40s}  peak_s_norm = {per_arch_peak_s[arch_id]}")

    print("\n## Per-class mean success at each arch's peak s_norm")
    print()
    print("| concept class | resid_post | ln1 | Δ(ln1−resid) |")
    print("|---|---:|---:|---:|")
    for cls in classes:
        a = per_arch_class["txc_bare_antidead_t5"][cls]
        b = per_arch_class["txc_bare_antidead_t5_ln1"][cls]
        delta = b - a
        print(f"| {cls} | {a:.2f} | {b:.2f} | {delta:+.2f} |")

    print("\n## Per-concept (knowledge / domain) success")
    print()
    print("| concept | resid_post | ln1 | Δ |")
    print("|---|---:|---:|---:|")
    for c in sorted(CONCEPT_CLASSES["knowledge / domain"]):
        a = per_arch_concept["txc_bare_antidead_t5"].get(c, float("nan"))
        b = per_arch_concept["txc_bare_antidead_t5_ln1"].get(c, float("nan"))
        delta = b - a if (np.isfinite(a) and np.isfinite(b)) else float("nan")
        a_s = f"{a:.2f}" if np.isfinite(a) else "—"
        b_s = f"{b:.2f}" if np.isfinite(b) else "—"
        d_s = f"{delta:+.2f}" if np.isfinite(delta) else "—"
        print(f"| {c} | {a_s} | {b_s} | {d_s} |")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(classes))
    bar_w = 0.36
    a_vals = [per_arch_class["txc_bare_antidead_t5"][c] for c in classes]
    b_vals = [per_arch_class["txc_bare_antidead_t5_ln1"][c] for c in classes]
    ax.bar(x - bar_w / 2, a_vals, bar_w, label=ARCHS[0][1], color="#1f77b4", alpha=0.85)
    ax.bar(x + bar_w / 2, b_vals, bar_w, label=ARCHS[1][1], color="#d62728", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=12)
    ax.set_ylabel("Mean steering success at peak s_norm (Sonnet 4.6)")
    ax.set_title(
        "TXC bare T=5: ln1 hook (pre-attn) vs resid_post — per-concept-class\n"
        "Hypothesis: ln1 hook increases TXC's knowledge-domain advantage",
        fontsize=11,
    )
    ax.set_ylim(0, 3.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(args.out))
    plt.close(fig)
    print(f"\n  saved {args.out}")


if __name__ == "__main__":
    main()
