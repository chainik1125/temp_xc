"""Assemble all per-(arch, seed, concat) autointerp cells into the
§9.5 Phase 6.1 headline table (and a wider per-cell matrix) for
summary.md.

Reads `results/autointerp/*__seed*__concat*__labels.json`, aggregates
per (arch × concat) across seeds (mean ± stderr), and emits markdown
fragments: one headline table (triangle archs) + one full matrix.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
IN_DIR = REPO / "experiments/phase6_qualitative_latents/results/autointerp"

ARCH_ORDER = [
    "agentic_txc_02_batchtopk",         # Cycle F (TXC-based winner)
    "agentic_txc_12_bare_batchtopk",    # 2x2 cell (may replace Cycle F)
    "tsae_paper",                       # paper baseline
    "tfa_big",                          # TFA baseline
    "agentic_txc_10_bare",              # Track 2 (TXC w/ full anti-dead, no BatchTopK)
    "agentic_mlc_08",                   # Phase 5.7 MLC winner
    "agentic_txc_11_stack",             # Cycle H (F + AuxK)
    "agentic_txc_09_auxk",              # Cycle A (AuxK only)
    "agentic_txc_02",                   # Phase 5 baseline
    "tsae_ours",                        # naive T-SAE port (control)
]


def _stderr(a: np.ndarray) -> float:
    return float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0


def load_cells():
    cells = []
    for p in sorted(IN_DIR.glob("*__seed*__concat*__labels.json")):
        d = json.loads(p.read_text())
        cells.append(d)
    return cells


def aggregate(cells):
    """Return {(arch, concat) -> {n_seeds, seeds, sem_mean, sem_se, ...}}"""
    by_key = defaultdict(list)
    for c in cells:
        by_key[(c["arch"], c["concat"])].append(c)
    agg = {}
    for (arch, concat), rows in by_key.items():
        sem = np.array([r["metrics"]["semantic_count"] for r in rows], dtype=float)
        cov = np.array([r["metrics"]["passage_coverage_count"] for r in rows], dtype=float)
        ent = np.array([r["metrics"]["passage_coverage_entropy"] for r in rows], dtype=float)
        dis = np.array([r["metrics"]["judge_disagreement_rate"] for r in rows], dtype=float)
        P = rows[0]["metrics"]["n_passages"]
        N = rows[0]["metrics"]["N"]
        agg[(arch, concat)] = {
            "n_seeds": len(rows),
            "seeds": sorted(int(r["seed"]) for r in rows),
            "N": N, "P": P,
            "sem_mean": float(sem.mean()), "sem_se": _stderr(sem),
            "cov_mean": float(cov.mean()), "cov_se": _stderr(cov),
            "ent_mean": float(ent.mean()), "ent_se": _stderr(ent),
            "dis_mean": float(dis.mean()),
        }
    return agg


def format_table(agg):
    out = []
    out.append("## Phase 6.1 headline: triangle parity table\n")
    out.append(f"Autointerp at **N=32**, multi-judge (2 Haiku prompts), "
               f"passage-coverage diagnostic (k/P).\n")
    out.append("### Triangle (3-seed where available)\n")
    out.append("| arch (family) | concat | n_seeds | /32 sem (mean ± se) | "
               "cov k/P (mean ± se) | cov entropy | judge disagree |")
    out.append("|---|---|---|---|---|---|---|")
    family = {
        "agentic_txc_02_batchtopk": "TXC (Cycle F)",
        "agentic_txc_12_bare_batchtopk": "TXC (2x2 cell)",
        "tsae_paper": "T-SAE baseline",
        "tfa_big": "TFA baseline",
    }
    triangle_archs = [a for a in family if any(a == k[0] for k in agg)]
    for arch in triangle_archs:
        for concat in ("A", "B", "random"):
            k = (arch, concat)
            if k not in agg:
                continue
            m = agg[k]
            out.append(
                f"| {family[arch]} | {concat} | {m['n_seeds']} | "
                f"{m['sem_mean']:.1f}/{m['N']} ± {m['sem_se']:.2f} | "
                f"{m['cov_mean']:.1f}/{m['P']} ± {m['cov_se']:.2f} | "
                f"{m['ent_mean']:.2f} ± {m['ent_se']:.2f} | "
                f"{m['dis_mean']:.2f} |"
            )
    out.append("")
    out.append("### Full 9-arch matrix at seed=42\n")
    out.append("| arch | concat | /32 sem | cov k/P | cov ent | disagree |")
    out.append("|---|---|---|---|---|---|")
    for arch in ARCH_ORDER:
        for concat in ("A", "B", "random"):
            k = (arch, concat)
            if k not in agg:
                continue
            m = agg[k]
            # Show seed-42 row if present; otherwise skip (aggregate row
            # would duplicate the triangle table).
            if 42 not in m["seeds"]:
                continue
            sem_str = (f"{int(m['sem_mean'])}" if m["n_seeds"] == 1
                       else f"{m['sem_mean']:.1f}")
            cov_str = (f"{int(m['cov_mean'])}" if m["n_seeds"] == 1
                       else f"{m['cov_mean']:.1f}")
            out.append(
                f"| {arch} | {concat} | "
                f"{sem_str}/{m['N']} | "
                f"{cov_str}/{m['P']} | "
                f"{m['ent_mean']:.2f} | "
                f"{m['dis_mean']:.2f} |"
            )
    return "\n".join(out) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str,
                   default=str(IN_DIR / "phase61_headline.md"))
    args = p.parse_args()

    cells = load_cells()
    if not cells:
        print("no cells under", IN_DIR)
        return
    agg = aggregate(cells)
    md = format_table(agg)
    Path(args.out).write_text(md)
    print(f"wrote {args.out} ({len(cells)} cells, "
          f"{len(agg)} (arch, concat) pairs)")
    # Preview
    print("\n---\n" + md[:2000])


if __name__ == "__main__":
    main()
