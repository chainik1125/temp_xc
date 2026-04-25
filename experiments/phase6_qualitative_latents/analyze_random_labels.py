"""Content analysis of the top-8 features per arch on concat_random.

Outputs a markdown table comparing what the top-8 features LOOK LIKE
on uncurated text. This is the qualitative evidence for the Phase 6.1
finding that tsae_paper finds real concepts while TXC variants find
boundary patterns.

Reads `results/autointerp/<arch>__seed42__concatrandom__labels.json`
and emits `results/phase61_random_label_comparison.md` — a side-by-
side view of the top-8 labels per arch plus per-feature judge verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"
OUT = REPO / "experiments/phase6_qualitative_latents/results/phase61_random_label_comparison.md"

DISPLAY = {
    "agentic_txc_02": "TXC (baseline)",
    "agentic_txc_02_batchtopk": "TXC+BatchTopK (Cycle F)",
    "agentic_txc_09_auxk": "TXC+AuxK (Cycle A)",
    "agentic_txc_10_bare": "TXC+anti-dead (Track 2)",
    "agentic_txc_11_stack": "TXC+BatchTopK+AuxK (Cycle H)",
    "agentic_mlc_08": "MLC (Phase 5.7)",
    "tsae_ours": "T-SAE (naive port)",
    "tsae_paper": "T-SAE (paper-faithful)",
    "tfa_big": "TFA",
}

ORDER = ["tsae_paper", "agentic_txc_10_bare", "tsae_ours",
         "agentic_mlc_08", "agentic_txc_02_batchtopk", "agentic_txc_02",
         "agentic_txc_09_auxk", "agentic_txc_11_stack", "tfa_big"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top", type=int, default=8, help="features per arch to show")
    args = p.parse_args()

    lines = [
        "# Phase 6.1 — concat_random top-8 label content comparison",
        "",
        f"Seed={args.seed}. Top-{args.top} features per arch (ranked by per-token variance).",
        "Verdict = majority of 2 Haiku judges at temperature=0.",
        "",
        "## By arch",
        "",
    ]

    for arch in ORDER:
        p_json = IN_DIR / f"{arch}__seed{args.seed}__concatrandom__labels.json"
        if not p_json.exists():
            lines.append(f"### {DISPLAY.get(arch, arch)} — MISSING")
            lines.append("")
            continue
        d = json.loads(p_json.read_text())
        m = d["metrics"]
        lines.append(f"### {DISPLAY.get(arch, arch)}")
        lines.append(
            f"*Metrics: {m['semantic_count']}/{m['N']} semantic, "
            f"coverage {m['passage_coverage_count']}/{m['n_passages']}, "
            f"judge disagreement {m['judge_disagreement_rate']:.2f}*"
        )
        lines.append("")
        lines.append("| rank | feat_idx | peak | verdict | label |")
        lines.append("|---|---|---|---|---|")
        for i, f in enumerate(d["features"][: args.top], start=1):
            ver = f["judge"]["verdict"]
            if not f["judge"].get("agree", True):
                ver = f"⚠ {ver}"
            lines.append(
                f"| {i} | {f['feature_idx']} | "
                f"{f.get('peak_passage', '?')} | {ver} | "
                f"{f['label']} |"
            )
        lines.append("")

    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT} ({sum(1 for line in lines if line.startswith('|'))-1} rows)")


if __name__ == "__main__":
    main()
