"""Analyze + plot Venhoff benchmark JSONs.

For each arch (sae / tempxc / mlc) and each run-suffix (e.g. single-cell
vs grid-guardrail), reads the trusted aggregate accuracies from the
JSON, computes Gap Recovery, and emits:

  1. Console summary table
  2. CSV of all rows
  3. Bar chart comparing thinking / base / hybrid per (arch, suffix)
  4. Gap-Recovery dot plot vs Venhoff's 3.5% baseline

Confirmed via inspection 2026-04-27:

  - benchmark_results JSON's `tasks[].model_answers.hybrid_model` is a
    SINGLE string per task (not per-cell). hybrid_token.py's internal
    guardrail picks one cell per token and writes the resulting answer.
    So there is no per-(coef × window) cell breakdown to extract; the
    grid run's reported hybrid_acc is the guardrail's aggregate output.
  - hybrid_stats_<base>_<dataset>.json `accuracies` field also stores
    the same aggregate (no per-cell breakdown either).
  - Re-grading from raw answer strings via math_verify is unreliable
    here because Venhoff's traces are byte-level-BPE encoded (Ġ for
    space etc.); the trusted numbers are the ones hybrid_token.py
    reports at the end of its run, which we just consume.

Usage:

    python experiments/venhoff_paper_run/analyze_grid.py \\
        --json-dir /workspace/spar-temporal-crosscoders/vendor/thinking-llms-interp/hybrid/results/ \\
        --out-dir experiments/venhoff_paper_run/results/

Run via wrapper: bash experiments/venhoff_paper_run/run_analysis.sh
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

VENHOFF_BASELINE_PCT = 3.5  # Llama-3.1-8B MATH500 cell, paper Table 2.


def load_arch_run(json_path: Path) -> dict | None:
    """Read one benchmark JSON. Returns dict with thinking/base/hybrid
    accuracies (as fractions 0-1) and Gap Recovery, or None if not found.
    """
    if not json_path.exists():
        return None
    data = json.loads(json_path.read_text())
    acc = data.get("results", {}).get("accuracy") or {}
    if not acc:
        return None
    # Numbers in the JSON are percentages (0-100). Convert to fractions.
    t = float(acc.get("thinking_model", 0)) / 100.0
    b = float(acc.get("base_model", 0)) / 100.0
    h = float(acc.get("hybrid_model", 0)) / 100.0
    n_tasks = data.get("metadata", {}).get("n_tasks", len(data.get("tasks", [])))
    gr = (h - b) / (t - b) if t > b else None
    return {
        "thinking_acc": t,
        "base_acc": b,
        "hybrid_acc": h,
        "gap_recovery": gr,
        "n_tasks": n_tasks,
    }


def render_summary_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    lines = [f"{'arch':<10} {'suffix':<10} {'n':>4}  {'thinking':>9} {'base':>8} {'hybrid':>8}  {'GR':>9}  {'vs 3.5%':>9}"]
    lines.append("-" * 78)
    for r in rows:
        gr = r["gap_recovery"]
        gr_pct = gr * 100 if gr is not None else None
        gr_str = f"{gr_pct:+.1f}%" if gr_pct is not None else "undef"
        vs_str = f"{gr_pct - VENHOFF_BASELINE_PCT:+.1f}pp" if gr_pct is not None else "-"
        lines.append(
            f"{r['arch']:<10} {r['suffix']:<10} {r['n_tasks']:>4d}  "
            f"{r['thinking_acc']*100:>8.1f}% {r['base_acc']*100:>7.1f}% {r['hybrid_acc']*100:>7.1f}%  "
            f"{gr_str:>9}  {vs_str:>9}"
        )
    return "\n".join(lines)


def render_bar_chart(rows: list[dict], out_path: Path) -> None:
    """Grouped bar chart: per (arch, suffix), bars for thinking / base / hybrid."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed — skipping bar chart", file=sys.stderr)
        return
    if not rows:
        return

    labels = [f"{r['arch']}\n{r['suffix']}" for r in rows]
    thinking = [r["thinking_acc"] * 100 for r in rows]
    base = [r["base_acc"] * 100 for r in rows]
    hybrid = [r["hybrid_acc"] * 100 for r in rows]

    x = np.arange(len(rows))
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(8, 1.3 * len(rows)), 5))
    ax.bar(x - width, thinking, width, label="thinking", color="#4C72B0")
    ax.bar(x, base, width, label="base", color="#888888")
    ax.bar(x + width, hybrid, width, label="hybrid", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Venhoff MATH500 accuracies per (arch × run)")
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok] bar chart saved: {out_path}")


def render_gr_dotplot(rows: list[dict], out_path: Path) -> None:
    """Gap-Recovery dot plot vs Venhoff's 3.5% reference line."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib not installed — skipping GR dot plot", file=sys.stderr)
        return

    valid = [r for r in rows if r["gap_recovery"] is not None]
    if not valid:
        print("[warn] no rows with defined GR — skipping", file=sys.stderr)
        return

    labels = [f"{r['arch']}\n{r['suffix']}" for r in valid]
    grs = [r["gap_recovery"] * 100 for r in valid]
    arches = [r["arch"] for r in valid]
    color_map = {"sae": "#888888", "tempxc": "#4C72B0", "mlc": "#DD8452"}
    colors = [color_map.get(a, "#000000") for a in arches]

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(valid)), 5))
    ax.scatter(range(len(valid)), grs, c=colors, s=180, zorder=3)
    for i, gr in enumerate(grs):
        ax.text(i, gr + 2 if gr >= 0 else gr - 4, f"{gr:+.1f}%",
                ha="center", fontsize=9)
    ax.axhline(VENHOFF_BASELINE_PCT, color="green", linestyle="--", alpha=0.6,
               label=f"Venhoff baseline ({VENHOFF_BASELINE_PCT}%)")
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.3)
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Gap Recovery (%)")
    ax.set_title("Venhoff MATH500 Gap Recovery — best hybrid acc vs base/thinking spread")
    ax.legend(loc="best")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[ok] GR dot plot saved: {out_path}")


def write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["arch", "suffix", "n_tasks", "thinking_acc", "base_acc",
              "hybrid_acc", "gap_recovery_pct", "vs_venhoff_pp"]
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for r in rows:
            gr_pct = r["gap_recovery"] * 100 if r["gap_recovery"] is not None else ""
            vs = (gr_pct - VENHOFF_BASELINE_PCT) if isinstance(gr_pct, float) else ""
            w.writerow([
                r["arch"], r["suffix"], r["n_tasks"],
                f"{r['thinking_acc']:.3f}", f"{r['base_acc']:.3f}",
                f"{r['hybrid_acc']:.3f}",
                f"{gr_pct:.2f}" if isinstance(gr_pct, float) else "",
                f"{vs:.2f}" if isinstance(vs, float) else "",
            ])
    print(f"[ok] csv saved: {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True,
                    help="Directory containing benchmark_results_<base>_<dataset>(_<arch>(_<suffix>)?).json files")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--base-short", default="llama-3.1-8b")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--arches", nargs="+", default=["sae", "tempxc", "mlc"])
    ap.add_argument("--suffixes", nargs="+", default=["", "_grid"],
                    help="Filename suffixes to scan. '' = no suffix; '_grid' = grid-guardrail run.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for arch in args.arches:
        for sfx in args.suffixes:
            json_path = (
                args.json_dir
                / f"benchmark_results_{args.base_short}_{args.dataset}_{arch}{sfx}.json"
            )
            data = load_arch_run(json_path)
            if data is None:
                continue
            rows.append({
                "arch": arch,
                "suffix": sfx.lstrip("_") or "single",
                **data,
            })
            print(f"[info] loaded: arch={arch} suffix={sfx or '(none)'} GR={data['gap_recovery']*100:+.1f}%"
                  if data['gap_recovery'] is not None
                  else f"[info] loaded: arch={arch} suffix={sfx or '(none)'} GR=undef")

    if not rows:
        print("[error] no JSONs found", file=sys.stderr)
        return 2

    print()
    print("=" * 78)
    print(render_summary_table(rows))
    print("=" * 78)
    print()

    write_csv(rows, args.out_dir / "summary.csv")
    render_bar_chart(rows, args.out_dir / "accuracies_bar.png")
    render_gr_dotplot(rows, args.out_dir / "gap_recovery.png")

    # Full structured dump.
    full = {
        "venhoff_baseline_pct": VENHOFF_BASELINE_PCT,
        "rows": rows,
    }
    out_json = args.out_dir / "analysis.json"
    out_json.write_text(json.dumps(full, indent=2, default=str))
    print(f"[ok] structured dump: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
