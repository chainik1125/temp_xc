"""Per-concept-class breakdown at multiple coh thresholds.

Strength-uniform peak success within each class (concept group), restricted
to (s_norm, threshold) where the within-class mean coh exceeds threshold.

Outputs:
  results/case_studies/plots/per_class_coh_thresholds.json
  results/case_studies/plots/per_class_coh_thresholds.png

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_per_class_coh_thresholds
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"

GROUPS = {
    "knowledge_domain": ["medical","mathematical","historical","legal","religious","geographical","financial","scientific","code_context"],
    "knowledge_format": ["technical_jargon","citation_pattern","list_format","instructional","programming"],
    "discourse_register": ["formal_register","casual_register"],
    "discourse_safety": ["harmful_content","deception","refusal_pattern","jailbreak_pattern","helpfulness_marker"],
    "discourse_style": ["poetic","literary","narrative"],
    "behavior_form": ["question_form","imperative_form","dialogue"],
    "behavior_emotion": ["positive_emotion","negative_emotion","neutral_factual"],
}

CELLS = [
    ("T-SAE k=20 (anchor)", "blue", [
        ("steering_paper_normalised", "tsae_paper_k20", 42),
    ]),
    ("T=2 H8 PP", "red", [
        ("steering_paper_window_perposition", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_window_perposition_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_window_perposition_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 H8 RE", "darkred", [
        ("steering_paper_normalised", "txc_h8_t2_kpos20_shifts2", 42),
        ("steering_paper_normalised_seed1", "txc_h8_t2_kpos20_shifts2", 1),
        ("steering_paper_normalised_seed2", "txc_h8_t2_kpos20_shifts2", 2),
    ]),
    ("T=2 bare PP", "orange", [
        ("steering_paper_window_perposition", "txc_bare_antidead_t2_kpos20", 42),
        ("steering_paper_window_perposition_seed1", "txc_bare_antidead_t2_kpos20", 1),
        ("steering_paper_window_perposition_seed2", "txc_bare_antidead_t2_kpos20", 2),
    ]),
]

THRESHOLDS = [1.5, 1.75, 2.0]


def load_curve_class_aware(subdir, arch, class_ids):
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists(): return None
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    if len(gens) != len(grads): return None
    by_s = defaultdict(lambda: {"succ": [], "coh": []})
    for gg, rr in zip(gens, grads):
        if gg["concept_id"] not in class_ids: continue
        s = gg.get("s_norm", gg.get("strength"))
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None: continue
        by_s[s]["succ"].append(rr["success_grade"])
        by_s[s]["coh"].append(rr["coherence_grade"])
    out = {}
    for s, d in sorted(by_s.items()):
        if not d["succ"]: continue
        out[s] = (sum(d["succ"]) / len(d["succ"]), sum(d["coh"]) / len(d["coh"]))
    return out


def mean_curve(curves):
    if not curves: return {}
    common = set(curves[0].keys())
    for c in curves[1:]: common &= set(c.keys())
    out = {}
    for s in sorted(common):
        succs = [c[s][0] for c in curves]
        cohs = [c[s][1] for c in curves]
        out[s] = (sum(succs) / len(succs), sum(cohs) / len(cohs))
    return out


def peak_at_threshold(curve, thr):
    eligible = [v[0] for v in curve.values() if v[1] >= thr]
    return max(eligible) if eligible else 0.0


def main():
    data = {}
    for thr in THRESHOLDS:
        data[thr] = {}
        for label, color, specs in CELLS:
            data[thr][label] = {"color": color, "by_class": {}}
            for group, ids in GROUPS.items():
                seeds = [load_curve_class_aware(s, a, ids) for s, a, _sd in specs]
                seeds = [c for c in seeds if c]
                mc = mean_curve(seeds)
                p = peak_at_threshold(mc, thr)
                data[thr][label]["by_class"][group] = p

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    json_path = PLOTS_DIR / "per_class_coh_thresholds.json"
    json_path.write_text(json.dumps(data, indent=2))
    print(f"saved {json_path}")

    # Plot: 3 panels (one per threshold), grouped bars by class, color by cell
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    cell_labels = [c[0] for c in CELLS]
    cell_colors = [c[1] for c in CELLS]
    classes = list(GROUPS.keys())
    n_classes = len(classes)
    n_cells = len(CELLS)
    bar_w = 0.18

    for ti, thr in enumerate(THRESHOLDS):
        ax = axes[ti]
        x = np.arange(n_classes)
        for ci, (label, color, _) in enumerate(CELLS):
            vals = [data[thr][label]["by_class"][cls] for cls in classes]
            offset = (ci - (n_cells - 1) / 2) * bar_w
            ax.bar(x + offset, vals, bar_w, color=color, label=label, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c}\n({len(GROUPS[c])})" for c in classes], rotation=25, ha="right", fontsize=8)
        ax.set_ylabel("peak success")
        ax.set_title(f"coh ≥ {thr}", fontsize=11)
        ax.set_ylim(0, 2.5)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle("Per-concept-class peak success at multiple coherence thresholds (3-seed mean-curve where available)\n"
                 "Strength-uniform: max within-class mean succ at the strength satisfying within-class mean coh ≥ threshold",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out_png = PLOTS_DIR / "per_class_coh_thresholds.png"
    out_thumb = PLOTS_DIR / "per_class_coh_thresholds.thumb.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    fig.savefig(out_thumb, dpi=48, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_png}")


if __name__ == "__main__":
    main()
