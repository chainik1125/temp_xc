"""Write a markdown summary of one or more frontier_sweep.py JSON outputs.

    uv run python -m experiments.em_features.write_frontier_summary \
        --sweep SAE=results/sae_extended.json \
        --sweep TXC=results/txc.json \
        [--sweep MLC=results/mlc.json] \
        --baseline_align 64.2 --baseline_coh 84.9 \
        --out docs/dmitry/results/em_features/summary_sae_vs_txc_vs_mlc.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", action="append", required=True,
                   help="LABEL=path/to/frontier_sweep.json")
    p.add_argument("--baseline_align", type=float, default=None,
                   help="bad-model baseline alignment at α=0 (for Δ columns)")
    p.add_argument("--baseline_coh", type=float, default=None,
                   help="bad-model baseline coherence at α=0")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--title", default="Coherence / suppression frontier summary")
    return p.parse_args()


def load_sweep(path: Path) -> list[dict]:
    data = json.loads(Path(path).expanduser().read_text())
    rows = [r for r in data["rows"]
            if r.get("mean_alignment") is not None and r.get("mean_coherence") is not None]
    rows.sort(key=lambda r: r["alpha"])
    return rows


def method_summary(label: str, rows: list[dict], baseline_align, baseline_coh) -> list[str]:
    if not rows:
        return [f"### {label}", "No judged rows found.", ""]
    best = max(rows, key=lambda r: r["mean_alignment"])
    worst = min(rows, key=lambda r: r["mean_alignment"])
    lines = [f"### {label}",
             "",
             f"- **peak alignment**: {best['mean_alignment']:.2f} at α={best['alpha']:+.2f}"
             f" (coherence {best['mean_coherence']:.2f})",
             f"- **worst alignment**: {worst['mean_alignment']:.2f} at α={worst['alpha']:+.2f}"
             f" (coherence {worst['mean_coherence']:.2f})",
             f"- feature set: k={rows[0].get('k')} (selection={rows[0].get('selection')})",
             "",
             "| α | alignment | coherence | Δalign | Δcoh |",
             "|---:|---:|---:|---:|---:|"]
    for r in rows:
        da = f"{r['mean_alignment'] - baseline_align:+.2f}" if baseline_align is not None else "—"
        dc = f"{r['mean_coherence'] - baseline_coh:+.2f}" if baseline_coh is not None else "—"
        lines.append(f"| {r['alpha']:+.2f} | {r['mean_alignment']:.2f} | {r['mean_coherence']:.2f} | {da} | {dc} |")
    lines.append("")
    return lines


def main():
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    sweeps: list[tuple[str, list[dict]]] = []
    for spec in args.sweep:
        label, path = spec.split("=", 1)
        sweeps.append((label, load_sweep(Path(path))))

    lines: list[str] = [
        "---",
        "author: Dmitry",
        "date: 2026-04-23",
        "tags:",
        "  - results",
        "  - in-progress",
        "---",
        "",
        f"## {args.title}",
        "",
        "Qwen-2.5-7B-Instruct *bad-medical* (PEFT adapter vs base), layer 15,"
        " k=10 bundled feature steering via `ActivationSteerer(intervention_type=\"addition\")`.",
        " Scores from the em-features longform + OpenAI-judge loop.",
        "",
        f"Baseline (bad-medical, α=0): alignment={args.baseline_align}, coherence={args.baseline_coh}",
        "",
    ]

    for label, rows in sweeps:
        lines.extend(method_summary(label, rows, args.baseline_align, args.baseline_coh))

    # Cross-method headline
    headline = ["## Headline"]
    peaks = []
    for label, rows in sweeps:
        if rows:
            best = max(rows, key=lambda r: r["mean_alignment"])
            peaks.append((label, best))
    if peaks:
        for label, best in peaks:
            headline.append(
                f"- {label}: peak align **{best['mean_alignment']:.2f}** at α={best['alpha']:+.2f}, "
                f"coherence {best['mean_coherence']:.2f}"
            )
        best_overall = max(peaks, key=lambda lp: lp[1]["mean_alignment"])
        headline.append("")
        headline.append(f"**Winner (peak alignment): {best_overall[0]}** — "
                        f"{best_overall[1]['mean_alignment']:.2f}.")
    lines.extend(headline)

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
