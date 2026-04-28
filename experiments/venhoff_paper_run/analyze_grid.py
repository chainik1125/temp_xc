"""Analyze + plot the Venhoff grid-sweep benchmark JSONs.

Reads the per-arch saved benchmark JSONs from a directory, recomputes
per-cell Gap Recovery from per-task model answers (re-grading via
math_verify), and produces:

  1. A per-arch (coef × token_window) heatmap of Gap Recovery
  2. A summary CSV with best-cell GR per arch
  3. A console-printable table

Why not trust `results.accuracy.hybrid_model` directly: that field
is one aggregate (probably last cell or mean), not the best cell.
For the paper claim "best cell GR vs Venhoff's 3.5%" we need per-cell
re-grading from the raw `tasks[].model_answers.hybrid_model` data.

Usage (laptop or pod, after `_<arch>_grid.json` files are saved):

    python experiments/venhoff_paper_run/analyze_grid.py \\
        --json-dir /workspace/spar-temporal-crosscoders/vendor/thinking-llms-interp/hybrid/results/ \\
        --out-dir experiments/venhoff_paper_run/results/

The hybrid_token.py output schema (from inspecting one of these JSONs):

    {
      "metadata": {"base_model", "thinking_model", "n_tasks"},
      "results": {"accuracy": {...}, "correct_count": {...}},
      "tasks": [
          {
              "question", "correct_answer",
              "model_answers": {"base_model", "thinking_model", "hybrid_model"}
          }
      ]
    }

The `hybrid_model` field is whatever the LAST cell wrote (or first,
or fixed — it's not arch-vs-grid distinguished). To get per-cell, we
would need the underlying detailed file or per-task per-cell data,
which Venhoff stores in `hybrid/results/detailed/hybrid_stats_*.json`
or in the rolling jsonl. This script tries multiple schemas:

  1. Per-cell field in benchmark JSON (preferred if present)
  2. Detailed file lookup
  3. Top-line single-number fallback (no grid breakdown possible)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterable


# Re-implementation of grading we use elsewhere — avoids needing
# repo PYTHONPATH so this script works on the pod or laptop.
_BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def extract_boxed(text: str | None) -> str | None:
    if not text:
        return None
    matches = _BOXED_RE.findall(text)
    return matches[-1].strip() if matches else None


def _normalize_answer(s: str) -> str:
    s = s.strip()
    for tok in ("$", "\\,", "\\;", "\\!", "\\ ", " "):
        s = s.replace(tok, "")
    s = s.replace("\\left", "").replace("\\right", "")
    return s.lower()


def is_correct(predicted: str | None, reference: str) -> bool:
    if predicted is None:
        return False
    try:
        from math_verify import parse, verify
        return bool(verify(parse(f"${predicted}$"), parse(f"${reference}$")))
    except ImportError:
        return _normalize_answer(predicted) == _normalize_answer(reference)
    except Exception:
        return _normalize_answer(predicted) == _normalize_answer(reference)


def _grade_field(tasks: list[dict], model_field: str) -> tuple[int, int, list[bool]]:
    """Re-grade `tasks[].model_answers[model_field]` against `correct_answer`.

    Returns (n_correct, n_total, per_task_outcomes).
    """
    flags: list[bool] = []
    for t in tasks:
        ref = str(t.get("correct_answer", "")).strip()
        if not ref:
            continue
        ans = (t.get("model_answers") or {}).get(model_field) or ""
        pred = extract_boxed(ans)
        flags.append(bool(is_correct(pred, ref)))
    return sum(flags), len(flags), flags


def analyze_arch_top_level(json_path: Path) -> dict:
    """Extract aggregate accuracies from one arch's benchmark JSON.

    This gets the top-line numbers (matches what the run printed at
    the end). Per-cell breakdown is NOT in this file's schema — see
    `try_per_cell` for that.
    """
    data = json.loads(json_path.read_text())
    acc = data.get("results", {}).get("accuracy", {})
    cc = data.get("results", {}).get("correct_count", {})
    n_tasks = data.get("metadata", {}).get("n_tasks", len(data.get("tasks", [])))

    # Re-grade locally for sanity-check vs reported accuracy.
    tasks = data.get("tasks", [])
    th_corr, th_n, th_flags = _grade_field(tasks, "thinking_model")
    bs_corr, bs_n, bs_flags = _grade_field(tasks, "base_model")
    hy_corr, hy_n, hy_flags = _grade_field(tasks, "hybrid_model")

    return {
        "n_tasks": n_tasks,
        "reported_accuracy_pct": acc,  # what hybrid_token.py printed
        "regraded": {
            "thinking_acc": th_corr / th_n if th_n else 0.0,
            "base_acc": bs_corr / bs_n if bs_n else 0.0,
            "hybrid_acc": hy_corr / hy_n if hy_n else 0.0,
            "thinking_correct_count": th_corr,
            "base_correct_count": bs_corr,
            "hybrid_correct_count": hy_corr,
        },
        "per_task_outcomes": {
            "thinking": th_flags,
            "base": bs_flags,
            "hybrid": hy_flags,
        },
    }


def try_per_cell(arch: str, detailed_dir: Path, base_short: str = "llama-3.1-8b",
                 dataset: str = "math500") -> list[dict] | None:
    """Try to extract per-(coef, window) cell accuracies from the detailed
    hybrid_stats file. Returns a list of dicts with keys
    {coefficient, token_window, hybrid_correct_count, hybrid_acc} if found,
    None otherwise.

    Schema observed: hybrid_stats_<base_short>_<dataset>.json has a
    `task_stats` list, where each task has a `cell_results` dict keyed
    by "(coef, window)" → {accuracy/predicted/correct}. Adjust as needed.
    """
    candidate_paths = [
        detailed_dir / f"hybrid_stats_{base_short}_{dataset}.json",
        detailed_dir / f"hybrid_stats_{base_short}_{dataset}_{arch}.json",
    ]
    for p in candidate_paths:
        if not p.exists():
            continue
        try:
            d = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        # Inspect schema heuristically.
        # Possible keys: 'task_stats', 'per_cell', 'cells', 'results_by_cell'.
        if "per_cell_accuracy" in d:
            return [
                {
                    "coefficient": c.get("coefficient"),
                    "token_window": c.get("token_window"),
                    "hybrid_acc": c.get("accuracy"),
                }
                for c in d["per_cell_accuracy"]
            ]
        if "cells" in d and isinstance(d["cells"], list):
            return d["cells"]
        # Try task-level → cell-level extraction.
        if "task_stats" in d and isinstance(d["task_stats"], list):
            cells: dict[tuple[float, int], list[bool]] = {}
            for t in d["task_stats"]:
                ref = str(t.get("correct_answer", "")).strip()
                if not ref:
                    continue
                cell_res = t.get("cell_results") or t.get("cells") or {}
                for key, val in cell_res.items() if isinstance(cell_res, dict) else []:
                    # key format guess: "(0.5, -15)" or "0.5_-15"
                    m = re.match(r"\(?\s*([\-\d\.]+)\s*[,_]\s*([\-\d]+)\s*\)?", str(key))
                    if not m:
                        continue
                    c = float(m.group(1))
                    w = int(m.group(2))
                    pred = extract_boxed((val or {}).get("predicted") or (val or {}).get("response") or "")
                    cells.setdefault((c, w), []).append(bool(is_correct(pred, ref)))
            return [
                {
                    "coefficient": c,
                    "token_window": w,
                    "hybrid_correct_count": sum(flags),
                    "n_total": len(flags),
                    "hybrid_acc": sum(flags) / len(flags) if flags else 0.0,
                }
                for (c, w), flags in cells.items()
            ]
    return None


def gap_recovery(thinking: float, base: float, hybrid: float) -> float | None:
    if thinking <= base:
        return None
    return (hybrid - base) / (thinking - base)


def render_heatmap(per_arch_cells: dict, out_path: Path,
                   thinking_acc: dict, base_acc: dict) -> None:
    """Plot a (coef × window) Gap Recovery heatmap per arch.

    per_arch_cells: {arch_name: list[{coefficient, token_window, hybrid_acc}]}
    thinking_acc, base_acc: {arch_name: float} for GR denominator.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("[warn] matplotlib/numpy not installed — skipping heatmap", file=sys.stderr)
        return

    arches = sorted(per_arch_cells.keys())
    if not arches:
        print("[warn] no per-arch data — skipping heatmap", file=sys.stderr)
        return

    fig, axes = plt.subplots(1, len(arches), figsize=(5 * len(arches), 4), sharey=True)
    if len(arches) == 1:
        axes = [axes]

    # Collect all unique coefs and windows.
    all_cells = [c for arch_cells in per_arch_cells.values() for c in arch_cells]
    coefs = sorted({float(c["coefficient"]) for c in all_cells if c.get("coefficient") is not None})
    windows = sorted({int(c["token_window"]) for c in all_cells if c.get("token_window") is not None}, reverse=True)

    for ax, arch in zip(axes, arches):
        cells = per_arch_cells[arch]
        t = thinking_acc.get(arch)
        b = base_acc.get(arch)
        if t is None or b is None or t <= b:
            ax.set_title(f"{arch} (GR undef)")
            ax.axis("off")
            continue
        grid = np.full((len(windows), len(coefs)), np.nan)
        for c in cells:
            try:
                ci = coefs.index(float(c["coefficient"]))
                wi = windows.index(int(c["token_window"]))
                grid[wi, ci] = (c["hybrid_acc"] - b) / (t - b) * 100
            except (KeyError, ValueError, TypeError):
                continue
        im = ax.imshow(grid, cmap="RdBu", vmin=-100, vmax=100, aspect="auto")
        ax.set_xticks(range(len(coefs)))
        ax.set_xticklabels([f"{c:g}" for c in coefs], fontsize=9)
        ax.set_yticks(range(len(windows)))
        ax.set_yticklabels([str(w) for w in windows], fontsize=9)
        ax.set_xlabel("coefficient")
        ax.set_ylabel("token_window")
        ax.set_title(f"{arch} — GR (%)")
        # Annotate cells.
        for wi in range(grid.shape[0]):
            for ci in range(grid.shape[1]):
                v = grid[wi, ci]
                if np.isnan(v):
                    continue
                ax.text(ci, wi, f"{v:.0f}", ha="center", va="center",
                        color="white" if abs(v) > 50 else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Venhoff MATH500 Gap Recovery per (arch × coef × window)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    print(f"[ok] heatmap saved: {out_path}")


def render_summary_table(per_arch: dict, out_csv: Path) -> str:
    """Produce a console + CSV summary."""
    rows = [["arch", "n_tasks", "thinking_acc", "base_acc",
             "best_cell_hybrid_acc", "best_cell_coef", "best_cell_window",
             "best_GR_pct", "venhoff_baseline_pct"]]
    lines = []
    for arch in sorted(per_arch.keys()):
        x = per_arch[arch]
        t = x["regraded"]["thinking_acc"]
        b = x["regraded"]["base_acc"]
        cells = x.get("per_cell") or []
        if cells:
            best = max(cells, key=lambda c: c.get("hybrid_acc", 0))
            h = best.get("hybrid_acc", 0)
            gr = gap_recovery(t, b, h)
            rows.append([
                arch, x["n_tasks"], f"{t:.3f}", f"{b:.3f}",
                f"{h:.3f}", best.get("coefficient"), best.get("token_window"),
                f"{gr * 100:.1f}" if gr is not None else "undef",
                "3.5",
            ])
            lines.append(
                f"{arch:>8}  n={x['n_tasks']:<3}  thinking={t:.1%} base={b:.1%}  "
                f"best=(c={best.get('coefficient')}, w={best.get('token_window')}, h={h:.1%})  "
                f"GR={gr * 100:+.1f}%" if gr is not None else f"{arch}: GR undef"
            )
        else:
            h = x["regraded"]["hybrid_acc"]
            gr = gap_recovery(t, b, h)
            rows.append([
                arch, x["n_tasks"], f"{t:.3f}", f"{b:.3f}",
                f"{h:.3f}", "(no per-cell)", "(no per-cell)",
                f"{gr * 100:.1f}" if gr is not None else "undef",
                "3.5",
            ])
            lines.append(
                f"{arch:>8}  n={x['n_tasks']:<3}  thinking={t:.1%} base={b:.1%}  "
                f"hybrid_aggregate={h:.1%}  GR={gr * 100:+.1f}% (no per-cell available)"
                if gr is not None else f"{arch}: GR undef"
            )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        csv.writer(f).writerows(rows)
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-dir", type=Path, required=True,
                    help="Directory containing benchmark_results_<base>_<dataset>_<arch>.json files")
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="Output directory for plots + CSV")
    ap.add_argument("--base-short", default="llama-3.1-8b")
    ap.add_argument("--dataset", default="math500")
    ap.add_argument("--suffix", default="grid",
                    help="Filename suffix to look for (e.g. 'grid', 'singlecell'). "
                         "Default 'grid'. Files: benchmark_results_<base>_<dataset>_<arch>_<suffix>.json")
    ap.add_argument("--arches", nargs="+", default=["sae", "tempxc", "mlc"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    per_arch: dict = {}
    per_arch_cells: dict = {}
    thinking_acc: dict = {}
    base_acc: dict = {}

    detailed_dir = args.json_dir / "detailed"
    for arch in args.arches:
        json_path = (
            args.json_dir
            / f"benchmark_results_{args.base_short}_{args.dataset}_{arch}_{args.suffix}.json"
        )
        if not json_path.exists():
            # Fall back to no-suffix file.
            json_path = (
                args.json_dir
                / f"benchmark_results_{args.base_short}_{args.dataset}_{arch}.json"
            )
        if not json_path.exists():
            print(f"[warn] no JSON for arch={arch} — skipping", file=sys.stderr)
            continue
        print(f"[info] reading {json_path.name}")
        info = analyze_arch_top_level(json_path)
        cells = try_per_cell(arch, detailed_dir, args.base_short, args.dataset)
        if cells:
            print(f"[info]   {len(cells)} cells found in detailed file")
            info["per_cell"] = cells
            per_arch_cells[arch] = cells
        else:
            print(f"[warn]   no per-cell data — falling back to top-line aggregate")
        per_arch[arch] = info
        thinking_acc[arch] = info["regraded"]["thinking_acc"]
        base_acc[arch] = info["regraded"]["base_acc"]

    if not per_arch:
        print("[error] no arch data loaded", file=sys.stderr)
        return 2

    # Console + CSV summary.
    summary_str = render_summary_table(per_arch, args.out_dir / "summary.csv")
    print()
    print("=" * 80)
    print(summary_str)
    print("=" * 80)

    # Heatmap (if per-cell data is available).
    if per_arch_cells:
        render_heatmap(
            per_arch_cells,
            args.out_dir / "gap_recovery_heatmap.png",
            thinking_acc,
            base_acc,
        )
    else:
        print("[note] no per-cell data — skipping heatmap. To enable per-cell"
              " analysis, ensure hybrid_stats_*.json contains task_stats with"
              " cell_results, or modify try_per_cell() in this script.")

    # Dump full structured analysis for downstream notebooks.
    full_dump = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "per_arch": {
            arch: {
                "n_tasks": info["n_tasks"],
                "reported_accuracy_pct": info["reported_accuracy_pct"],
                "regraded": info["regraded"],
                "per_cell": info.get("per_cell"),
            }
            for arch, info in per_arch.items()
        },
    }
    out_json = args.out_dir / "analysis.json"
    out_json.write_text(json.dumps(full_dump, indent=2, default=str))
    print(f"[ok] full analysis: {out_json}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
