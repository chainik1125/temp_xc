"""Per-class signature plot for the 3 mystery archs (Contrastive / MaxPool / OBLIT)
vs T-SAE k=20 anchor — paper figure showing the 'three TXC families have three
distinct per-class signatures unified by sentiment win' finding.

Two coh thresholds: 1.5 (PRREG) and 1.75 (Y's GIGABRAIN). One subplot per coh.

Run: TQDM_DISABLE=1 .venv/bin/python -m \
    experiments.phase7_unification.case_studies.steering.plot_mystery_arch_per_class
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

BASE = Path("/workspace/temp_xc/experiments/phase7_unification/results/case_studies")
PLOTS_DIR = BASE / "plots"

CONCEPT_CLASSES = {
    "knowledge":   {"medical", "mathematical", "historical",
                    "code_context", "scientific", "religious",
                    "geographical", "financial", "programming"},
    "discourse":   {"dialogue", "casual_register", "formal_register",
                    "imperative_form", "instructional", "question_form",
                    "narrative", "neutral_factual"},
    "safety":      {"harmful_content", "deception", "refusal_pattern",
                    "jailbreak_pattern", "helpfulness_marker", "legal"},
    "stylistic":   {"poetic", "literary", "list_format", "citation_pattern",
                    "technical_jargon"},
    "sentiment":   {"positive_emotion", "negative_emotion"},
}


def class_of(cid):
    for cls, ids in CONCEPT_CLASSES.items():
        if cid in ids: return cls
    return "other"


def load_per_concept(subdir, arch):
    g = BASE / subdir / arch / "generations.jsonl"
    r = BASE / subdir / arch / "grades.jsonl"
    if not g.exists() or not r.exists():
        return None
    gens = [json.loads(l) for l in g.open()]
    grads = [json.loads(l) for l in r.open()]
    out = defaultdict(dict)
    for gg, rr in zip(gens, grads):
        if rr.get("success_grade") is None or rr.get("coherence_grade") is None: continue
        s = gg.get("s_norm", gg.get("strength"))
        out[gg["concept_id"]][s] = (rr["success_grade"], rr["coherence_grade"])
    return dict(out)


def avg_per_concept(curves):
    if not curves: return {}
    cids = set()
    for c in curves: cids |= set(c.keys())
    out = {}
    for cid in cids:
        per_s = defaultdict(list)
        for c in curves:
            if cid not in c: continue
            for s, (succ, coh) in c[cid].items():
                per_s[s].append((succ, coh))
        avg = {}
        for s, pairs in per_s.items():
            if len(pairs) == len(curves):
                avg[s] = (sum(p[0] for p in pairs)/len(pairs), sum(p[1] for p in pairs)/len(pairs))
        if avg: out[cid] = avg
    return out


def per_concept_cliff(per_concept, thr):
    out = {}
    for cid, sd in per_concept.items():
        valid = [su for s, (su, co) in sd.items() if co >= thr]
        out[cid] = max(valid) if valid else 0.0
    return out


def per_class_mean(per_concept_cliffs):
    by_cls = defaultdict(list)
    for cid, val in per_concept_cliffs.items():
        by_cls[class_of(cid)].append(val)
    return {cls: float(np.mean(vals)) for cls, vals in by_cls.items() if vals}


CELLS = [
    ("T-SAE k=20 RE", "tsae_paper_k20", "blue", [
        ("steering_paper_normalised", 42),
        ("steering_paper_normalised_seed1", 1),
    ]),
    ("OBLIT RE n=3", "txc_h8_t2_kpos20_shifts2", "darkorange", [
        ("steering_paper_normalised", 42),
        ("steering_paper_normalised_seed1", 1),
        ("steering_paper_normalised_seed2", 2),
    ]),
    ("MaxPool RE n=3", "txc_maxpool_h8_t2_kpos20_shifts2", "magenta", [
        ("steering_paper_normalised", 42),
        ("steering_paper_normalised_seed1", 1),
        ("steering_paper_normalised_seed2", 2),
    ]),
    ("Contrast RE n=3", "txc_contrastive_h8_t2_kpos20_shifts2", "deeppink", [
        ("steering_paper_normalised", 42),
        ("steering_paper_normalised_seed1", 1),
        ("steering_paper_normalised_seed2", 2),
    ]),
    ("Contrast V6 n=3", "txc_contrastive_h8_t2_kpos20_shifts2", "purple", [
        ("steering_paper_window_dec_broadcast", 42),
        ("steering_paper_window_dec_broadcast_seed1", 1),
        ("steering_paper_window_dec_broadcast_seed2", 2),
    ]),
]


def main():
    cells_data = {}
    for label, arch, color, specs in CELLS:
        seeds_pc = [load_per_concept(s, arch) for s, _sd in specs]
        seeds_pc = [c for c in seeds_pc if c]
        if not seeds_pc:
            continue
        cells_data[label] = {"pc": avg_per_concept(seeds_pc), "color": color}

    classes = ["knowledge", "discourse", "safety", "stylistic", "sentiment"]
    cell_labels = ["T-SAE k=20 RE", "OBLIT RE n=3", "MaxPool RE n=3", "Contrast RE n=3", "Contrast V6 n=3"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    for ax, thr in zip(axes, [1.5, 1.75]):
        per_arch_class = {}
        for label in cell_labels:
            if label not in cells_data: continue
            cliffs = per_concept_cliff(cells_data[label]["pc"], thr)
            per_arch_class[label] = per_class_mean(cliffs)

        x = np.arange(len(classes))
        n_archs = len(per_arch_class)
        bar_w = 0.85 / n_archs

        for i, label in enumerate(cell_labels):
            if label not in per_arch_class: continue
            vals = [per_arch_class[label].get(c, 0.0) for c in classes]
            color = cells_data[label]["color"]
            offset = (i - (n_archs - 1) / 2) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=label, color=color,
                          alpha=0.85 if "T-SAE" not in label else 1.0,
                          edgecolor="black" if "T-SAE" in label else None,
                          linewidth=1.5 if "T-SAE" in label else 0)

        ax.set_xticks(x)
        ax.set_xticklabels(classes, rotation=12, fontsize=10)
        ax.set_title(f"Cliff at coh ≥ {thr}", fontsize=12)
        ax.set_ylim(0, 2.5)
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylabel("Mean steering success (n=3 multi-seed mean-curve)", fontsize=10)
        if thr == 1.75:
            ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    fig.suptitle(
        "Three TXC mystery-arch signatures vs T-SAE k=20 — per-concept-class breakdown\n"
        "Sentiment win is universal across TXC archs; other classes split (Contrast→knowledge, MaxPool→stylistic, OBLIT→knowledge)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()

    out = PLOTS_DIR / "mystery_arch_per_class_signature.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    from src.plotting.save_figure import save_figure
    save_figure(fig, str(out))
    plt.close(fig)
    print(f"saved {out}")
    print(f"saved {out.with_suffix('.thumb.png')}")


if __name__ == "__main__":
    main()
