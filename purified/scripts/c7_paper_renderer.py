"""Render paper-ready C7 backtracking results: tables, figures, and a single
markdown bundle that embeds everything the camera-ready paper needs.

Output layout (under ``--output-dir``)::

    c7_paper_results.md
    c7_paper_assets/
        delta_gc_vs_magnitude.png
        peak_delta_gc_bar.png
        pr_auc_vs_S.png
        pr_auc_S8_bar.png
        nmse_vs_step.png
        l0_vs_step.png
        dead_vs_step.png
        delta_gc_vs_magnitude_log.png

Inputs (read-only)::

    purified/results/leaderboard.jsonl
    purified/results/runs/<eval_key>/metrics.json (optional extras)
    purified/checkpoints/<train_key>/snapshots/eval_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from temp_bench.config import (  # noqa: E402
    checkpoint_dir,
    purified_root,
)

log = logging.getLogger("c7_paper_renderer")

PAPER_ARCH_ORDER = ["topk_sae", "tsae_paper", "mlc", "txc_base", "txc_pro"]
PAPER_ARCH_LABEL = {
    "topk_sae":   "TopK SAE",
    "tsae_paper": "T-SAE",
    "mlc":        "MLC",
    "txc_base":   "TXC-base",
    "txc_pro":    "TXC-pro",
    "stacked_sae": "Stacked SAE",
    "tfa":        "TFA",
}
PAPER_ARCH_COLOR = {
    "topk_sae":   "#4C72B0",
    "tsae_paper": "#55A868",
    "mlc":        "#C44E52",
    "txc_base":   "#8172B2",
    "txc_pro":    "#CCB974",
    "stacked_sae": "#64B5CD",
    "tfa":        "#777777",
}

# Per-cell colors: each (arch, bs) gets a distinct hue so the
# legend has no duplicates and overlapping curves don't share colors.
# Two TXC archs each get two cells (bs=256 + bs=1024); per-token
# baselines run only at bs=1024.
PAPER_CELL_COLOR = {
    ("txc_base",   256):  "#8172B2",  # light purple
    ("txc_base",   1024): "#4F3A78",  # dark purple
    ("txc_pro",    256):  "#E8A33D",  # light orange
    ("txc_pro",    1024): "#A36B0F",  # dark orange
    ("topk_sae",   1024): "#4C72B0",  # blue
    ("tsae_paper", 1024): "#55A868",  # green
    ("mlc",        1024): "#C44E52",  # red
    ("stacked_sae",1024): "#64B5CD",  # cyan
    ("tfa",        1024): "#777777",  # grey
}


def cell_color(arch: str, bs: int | None) -> str:
    """Return a distinct color per (arch, bs) cell."""
    bs_int = int(bs) if bs is not None else 0
    return PAPER_CELL_COLOR.get((arch, bs_int)) or PAPER_ARCH_COLOR.get(arch, "#333333")
S_GRID = (1, 2, 4, 8, 16, 32)
MAG_KEY_RE = re.compile(r"^delta_gc_mag_([+-]?\d+(?:\.\d+)?)$")


# ── Loading ────────────────────────────────────────────────────────────


def load_c7_rows() -> list[dict]:
    lb_path = purified_root() / "results" / "leaderboard.jsonl"
    if not lb_path.exists():
        return []
    rows = []
    for line in lb_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("component") != "c7":
            continue
        rows.append(r)
    return rows


def latest_per_cell(rows: list[dict]) -> list[dict]:
    """Keep the most recent row per (arch, train_key, eval_key).

    A single trained SAE (one ``train_key``) can produce MULTIPLE
    leaderboard rows when re-evaluated with different magnitude grids
    (canonical vs extended-mags follow-up; each gets its own
    ``eval_key`` because the magnitude list is part of ``eval_cfg``).
    Keeping every (arch, train_key, eval_key) row lets the Δgc plot
    merge canonical + extended evals into one combined curve per cell.
    """
    by_key: dict[tuple, dict] = {}
    for r in rows:
        key = (r["arch"], r["train_key"], r["eval_key"])
        cur = by_key.get(key)
        if cur is None or r["ts"] > cur["ts"]:
            by_key[key] = r
    out = list(by_key.values())
    out.sort(key=lambda r: (
        PAPER_ARCH_ORDER.index(r["arch"]) if r["arch"] in PAPER_ARCH_ORDER else 99,
        r.get("eval_cfg", {}).get("batch_size", 0),
        -ord(r["ts"][0]) if r["ts"] else 0,
    ))
    return out


def _is_extended(r: dict) -> bool:
    return bool(r.get("eval_cfg", {}).get("_extended_mags"))


def canonical_rows(rows: list[dict]) -> list[dict]:
    """Drop extended-mags follow-up rows (peak / PR-AUC tables only show
    canonical evals; extended rows have a magnitude-restricted peak that
    isn't comparable to the canonical headline)."""
    return [r for r in rows if not _is_extended(r)]


def merge_mag_curves(rows: list[dict]) -> list[tuple[str, dict, dict[float, float]]]:
    """Group rows by (arch, train_key) and merge per-magnitude Δgc data
    across all eval_keys (canonical + extended). Returns a list of
    (cell_id, representative_row, {mag: Δgc}) tuples — one entry per cell.

    For overlapping magnitudes (e.g. m=0 appears in both canonical and
    extended evals), the canonical value wins because the canonical
    eval's per-question baseline is the headline reference.
    """
    by_cell: dict[tuple, list[dict]] = {}
    for r in rows:
        cell = (r["arch"], r["train_key"])
        by_cell.setdefault(cell, []).append(r)
    out = []
    for cell, cell_rows in by_cell.items():
        # Canonical first so its mag=0 baseline wins for shared mags.
        cell_rows = sorted(cell_rows, key=lambda r: (_is_extended(r), r["ts"]))
        merged: dict[float, float] = {}
        for r in cell_rows:
            for mag, delta in parse_mag_metrics(r["metrics"]):
                merged.setdefault(mag, delta)
        # Pick a representative row for label / color: prefer canonical.
        rep = next((r for r in cell_rows if not _is_extended(r)), cell_rows[0])
        out.append((f"{cell[0]}|{cell[1][:8]}", rep, merged))
    return out


def parse_mag_metrics(metrics: dict) -> list[tuple[float, float]]:
    out = []
    for k, v in metrics.items():
        m = MAG_KEY_RE.match(k)
        if m and isinstance(v, (int, float)):
            out.append((float(m.group(1)), float(v)))
    out.sort()
    return out


def load_probe_log(train_key: str) -> list[dict]:
    p = checkpoint_dir(train_key) / "snapshots" / "eval_log.jsonl"
    if not p.exists():
        return []
    rows = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    rows.sort(key=lambda r: r["step"])
    return rows


def cell_label(r: dict) -> str:
    arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
    bs = _bs_from_row(r) or "?"
    n_steps = _n_steps_from_row(r) or "?"
    return f"{arch} (bs={bs}, n_steps={n_steps})"


def cell_short_label(r: dict) -> str:
    arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
    bs = _bs_from_row(r)
    return f"{arch} bs={bs if bs is not None else '?'}"


_TRAIN_CFG_CACHE: dict[str, dict] = {}


def _train_cfg(r: dict) -> dict:
    """Resolve the row's training_cfg by reading checkpoints/<train_key>/config.json
    (the leaderboard schema does not embed training_cfg at the top level)."""
    tk = r.get("train_key")
    if not tk:
        return {}
    if tk in _TRAIN_CFG_CACHE:
        return _TRAIN_CFG_CACHE[tk]
    cfg_path = checkpoint_dir(tk) / "config.json"
    cfg: dict = {}
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text()).get("training_cfg") or {}
        except (json.JSONDecodeError, OSError):
            cfg = {}
    _TRAIN_CFG_CACHE[tk] = cfg
    return cfg


def _bs_from_row(r: dict) -> int | None:
    bs = _train_cfg(r).get("batch_size")
    if bs is None:
        bs = r.get("eval_cfg", {}).get("batch_size")
    return int(bs) if bs is not None else None


def _n_steps_from_row(r: dict) -> int | None:
    ns = _train_cfg(r).get("n_steps")
    return int(ns) if ns is not None else None


# ── Plotting ───────────────────────────────────────────────────────────


def plot_delta_gc_vs_magnitude(rows: list[dict], out_path: Path) -> None:
    """Plot Δgc vs magnitude per cell, merging canonical + extended-mags
    evals for the same (arch, train_key) into a single curve so the
    appendix figure shows the full magnitude span the cell was evaluated
    at."""
    plt.figure(figsize=(8.5, 5.0))
    seen_labels: set[str] = set()
    for _cell_id, rep, merged in merge_mag_curves(rows):
        if not merged:
            continue
        mags = sorted(merged.keys())
        deltas = [merged[m] for m in mags]
        color = cell_color(rep["arch"], _bs_from_row(rep))
        label = cell_short_label(rep)
        if label in seen_labels:
            label = f"{label} [{rep['train_key'][:6]}]"
        seen_labels.add(label)
        plt.plot(mags, deltas, marker="o", markersize=4, linewidth=1.6,
                 color=color, linestyle="-", label=label)
    plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.xlabel("Steering magnitude $m$")
    plt.ylabel(r"$\Delta gc(a, m)$  (per-question baseline at $m=0$)")
    plt.title(r"Inducement curves: $\Delta gc$ vs steering magnitude")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_peak_delta_gc_bar(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["metrics"].get("delta_gc_peak", 0.0))
    labels = [cell_short_label(r) for r in rows]
    # Disambiguate duplicate labels by appending a train_key suffix.
    seen: dict[str, int] = {}
    final_labels = []
    for r, lab in zip(rows, labels):
        if lab in seen:
            final_labels.append(f"{lab} [{r['train_key'][:6]}]")
        else:
            final_labels.append(lab)
        seen[lab] = seen.get(lab, 0) + 1
    labels = final_labels
    peaks = [r["metrics"].get("delta_gc_peak", 0.0) for r in rows]
    colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    plt.figure(figsize=(8.5, max(3.5, 0.4 * len(rows) + 1.5)))
    bars = plt.barh(labels, peaks, color=colors, alpha=0.85)
    for r, bar, peak in zip(rows, bars, peaks):
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f" {peak:.3f} (m={peak_mag:+g})" if peak_mag is not None
                 else f" {peak:.3f}",
                 va="center", fontsize=9)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel(r"Peak $\Delta gc$ across magnitudes")
    plt.title(r"Peak $\Delta gc$ per cell")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_auc_vs_S(rows: list[dict], out_path: Path) -> None:
    plt.figure(figsize=(8.0, 5.0))
    seen_labels: set[str] = set()
    for r in rows:
        ys = []
        for S in S_GRID:
            v = r["metrics"].get(f"pr_auc_S{S}")
            ys.append(float(v) if v is not None else None)
        if all(y is None for y in ys):
            continue
        valid_x = [x for x, y in zip(S_GRID, ys) if y is not None]
        valid_y = [y for y in ys if y is not None]
        color = cell_color(r["arch"], _bs_from_row(r))
        label = cell_short_label(r)
        if label in seen_labels:
            label = f"{label} [{r['train_key'][:6]}]"
        seen_labels.add(label)
        plt.plot(valid_x, valid_y, marker="o", color=color, linestyle="-",
                 linewidth=1.6, label=label)
    plt.xscale("log", base=2)
    plt.xticks(list(S_GRID), [str(s) for s in S_GRID])
    plt.xlabel("Top-$S$ probe features")
    plt.ylabel("PR-AUC (5-fold GroupKFold by question)")
    plt.title("Sparse-probe PR-AUC for backtracking-sentence detection")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_pr_auc_S8_bar(rows: list[dict], out_path: Path) -> None:
    rows = sorted(rows, key=lambda r: r["metrics"].get("pr_auc_S8", 0.0))
    labels = [cell_short_label(r) for r in rows]
    seen: dict[str, int] = {}
    final_labels = []
    for r, lab in zip(rows, labels):
        if lab in seen:
            final_labels.append(f"{lab} [{r['train_key'][:6]}]")
        else:
            final_labels.append(lab)
        seen[lab] = seen.get(lab, 0) + 1
    labels = final_labels
    aucs = [r["metrics"].get("pr_auc_S8", 0.0) for r in rows]
    colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    plt.figure(figsize=(8.0, max(3.5, 0.4 * len(rows) + 1.5)))
    bars = plt.barh(labels, aucs, color=colors, alpha=0.85)
    for bar, auc in zip(bars, aucs):
        plt.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f" {auc:.3f}", va="center", fontsize=9)
    plt.axvline(0.12, color="grey", linestyle=":",
                label="positive-class prior (0.12)")
    plt.xlabel("PR-AUC at $S = 8$")
    plt.title("Detection PR-AUC at $S = 8$ per cell")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _enumerate_probe_cells() -> list[tuple[str, int, str, list[dict]]]:
    """Walk all checkpoints/<train_key>/ dirs that have eval_log.jsonl. Returns
    (arch, bs, train_key, log_rows) for every in-flight or completed cell at
    the headline n_steps=300_000 config (older 30K sprint cells are excluded
    so the convergence plots match the leaderboard rows we report)."""
    out = []
    cp_root = purified_root() / "checkpoints"
    if not cp_root.exists():
        return out
    for tk_dir in sorted(cp_root.iterdir()):
        if not tk_dir.is_dir():
            continue
        eval_log_path = tk_dir / "snapshots" / "eval_log.jsonl"
        cfg_path = tk_dir / "config.json"
        if not (eval_log_path.exists() and cfg_path.exists()):
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        # Filter to c7 cells: subject model is the Llama datasource.
        if cfg.get("datasource") != "llama_3_1_8b_base_l10_ward_nousmirror":
            continue
        tcfg = cfg.get("training_cfg") or {}
        # Filter to the headline training config so old sprint cells (e.g.
        # n_steps=20_000 / 30_000) don't appear in the convergence plots.
        if tcfg.get("n_steps") != 300_000:
            continue
        arch = cfg.get("arch")
        bs = tcfg.get("batch_size", 0)
        log_rows = []
        for line in eval_log_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                log_rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
        log_rows.sort(key=lambda r: r["step"])
        if log_rows:
            out.append((arch, int(bs), tk_dir.name, log_rows))
    return out


def plot_probe_curves(rows: list[dict], out_dir: Path) -> dict[str, Path]:
    """One PNG per metric: NMSE / L0 / dead, all cells overlaid.

    Iterates over every checkpoint dir with an eval_log.jsonl so in-flight
    cells appear too (not just completed leaderboard rows).
    """
    series = {}
    seen_label_count: dict[str, int] = {}
    for arch, bs, tk, log_rows in _enumerate_probe_cells():
        base_label = f"{PAPER_ARCH_LABEL.get(arch, arch)} bs={bs}"
        # Append a train_key suffix only if the same (arch, bs) has
        # multiple cells (so legend labels are unique without being noisy
        # in the common single-cell case).
        n = seen_label_count.get(base_label, 0)
        seen_label_count[base_label] = n + 1
        label = base_label if n == 0 else f"{base_label} [{tk[:6]}]"
        series[(arch, bs, tk)] = (label, log_rows, arch)
    out = {}
    metric_specs = [
        ("nmse", "NMSE on held-out batch", "nmse_vs_step"),
        ("l0",   "$\\ell_0$ density on held-out batch",  "l0_vs_step"),
        ("dead", "Dead features (out of $d_{\\mathrm{SAE}}$)", "dead_vs_step"),
    ]
    # Two scales per metric: linear (default) and log-log. Both are
    # rendered so the appendix can pick whichever reads better, and
    # the linear one stays available for readers who prefer it.
    scale_specs = [
        ("",       "linear", "linear"),
        ("_loglog", "log",   "log"),
    ]
    for metric, ylabel, fname_stem in metric_specs:
        for fname_suffix, x_scale, y_scale in scale_specs:
            plt.figure(figsize=(8.5, 5.0))
            for (arch, bs, _tk), (lab, log_rows, arch_name) in sorted(series.items()):
                steps = [x["step"] for x in log_rows]
                vals = [x.get(metric) for x in log_rows]
                # log-log axes can't render zero or negative values; drop them.
                if x_scale == "log":
                    valid = [(s, v) for s, v in zip(steps, vals)
                             if v is not None and v > 0 and s > 0]
                else:
                    valid = [(s, v) for s, v in zip(steps, vals) if v is not None]
                if not valid:
                    continue
                xs, ys = zip(*valid)
                color = cell_color(arch_name, bs)
                plt.plot(xs, ys, color=color, linestyle="-",
                         linewidth=1.4, label=lab)
            plt.xlabel("Training step")
            plt.ylabel(ylabel)
            plt.xscale(x_scale)
            plt.yscale(y_scale)
            scale_tag = " (log-log)" if x_scale == "log" else ""
            plt.title(
                f"{ylabel} vs training step (held-out probe every 100 steps){scale_tag}"
            )
            plt.legend(fontsize=8, loc="best", ncol=2)
            plt.grid(alpha=0.3, which="both")
            plt.tight_layout()
            out_path = out_dir / f"{fname_stem}{fname_suffix}.png"
            plt.savefig(out_path, dpi=150)
            plt.close()
            out[f"{metric}{fname_suffix}"] = out_path
    return out


# ── Markdown bundle ────────────────────────────────────────────────────


def fmt_num(v, precision=3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{precision}f}"
    return str(v)


def _table_headline(rows: list[dict]) -> str:
    headers = ["Arch", "bs", "n_steps", "Peak $\\Delta gc$", "Peak mag",
               "PR-AUC@8", "PR-AUC@32", "n_judge", "ts (UTC)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            str(_bs_from_row(r) or "?"),
            f"{_n_steps_from_row(r) or '?':,}" if _n_steps_from_row(r) else "?",
            fmt_num(m.get("delta_gc_peak")),
            fmt_num(m.get("delta_gc_peak_magnitude"), precision=1) + (
                "" if m.get("delta_gc_peak_magnitude") is None else ""),
            fmt_num(m.get("pr_auc_S8")),
            fmt_num(m.get("pr_auc_S32")),
            f"{int(m.get('n_judge_calls', 0))}",
            r["ts"][:19].replace("T", " "),
        ]) + " |")
    return "\n".join(lines)


def _table_pr_auc_full(rows: list[dict]) -> str:
    headers = ["Arch (cell)"] + [f"$S\\!=\\!{s}$" for s in S_GRID]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        row = [cell_short_label(r)]
        for s in S_GRID:
            row.append(fmt_num(m.get(f"pr_auc_S{s}")))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _table_per_magnitude(rows: list[dict]) -> str:
    """Per-magnitude Δgc table — one row per (arch, train_key) cell with
    canonical + extended-mags merged."""
    cells = merge_mag_curves(rows)
    all_mags = sorted({m for _, _, merged in cells for m in merged.keys()})
    headers = ["Arch (cell)"] + [f"{m:+g}" for m in all_mags]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _cell_id, rep, merged in cells:
        row = [cell_short_label(rep)]
        for mag in all_mags:
            row.append(fmt_num(merged.get(mag)))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_markdown(rows: list[dict], *, canonical: list[dict] | None = None,
                   assets_rel: str, out_path: Path) -> None:
    if canonical is None:
        canonical = canonical_rows(rows)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    n_cells = len(rows)
    n_extended = len(rows) - len(canonical)

    archs_done = sorted({r["arch"] for r in canonical})
    archs_expected = ["txc_base", "txc_pro", "topk_sae", "tsae_paper", "mlc"]
    missing = [a for a in archs_expected if a not in archs_done]

    md = []
    md.append("# C7 — Backtracking case study: results bundle (autogenerated)")
    md.append("")
    md.append(f"_Last updated: {now}._  "
              f"_Source: `purified/results/leaderboard.jsonl` filtered to `component == 'c7'`._")
    md.append("")
    md.append(f"**Cells in leaderboard:** {n_cells} ({n_extended} extended-mags follow-up).  "
              f"**Architectures present:** {', '.join(PAPER_ARCH_LABEL.get(a, a) for a in archs_done) if archs_done else '(none yet)'}.  "
              f"**Still pending:** {', '.join(PAPER_ARCH_LABEL.get(a, a) for a in missing) if missing else '(all five present)'}.")
    md.append("")
    md.append("This file is regenerated automatically as new leaderboard rows land. ")
    md.append("It is paired with the prose section in `purified/docs/components/c7_section.tex`; ")
    md.append("the numbers + figures here are the inputs to the `[TBD]` placeholders there.")
    md.append("")

    md.append("## Headline numbers")
    md.append("")
    if canonical:
        md.append(_table_headline(canonical))
    else:
        md.append("_(No canonical cells yet.)_")
    md.append("")
    md.append(f"![Peak Δgc per cell]({assets_rel}/peak_delta_gc_bar.png)")
    md.append("")

    md.append("## Inducement curves: $\\Delta gc$ vs steering magnitude")
    md.append("")
    md.append("Solid lines are batch-size 1024 cells (headline); dashed lines are the ")
    md.append("additional batch-size 256 cells run for the two TXC architectures (see ")
    md.append("`app:c7-bs-256` in the section TeX).")
    md.append("")
    md.append(f"![Δgc vs magnitude]({assets_rel}/delta_gc_vs_magnitude.png)")
    md.append("")

    md.append("## Detection PR-AUC")
    md.append("")
    md.append("Sparse-probe PR-AUC at top-$S$ feature counts ($5$-fold GroupKFold by question, ")
    md.append("$\\ell_1$-regularised logistic regression, $C = 1$). Positive-class prior $\\approx 0.12$.")
    md.append("")
    md.append(f"![PR-AUC at S=8]({assets_rel}/pr_auc_S8_bar.png)")
    md.append("")
    md.append(f"![PR-AUC vs S]({assets_rel}/pr_auc_vs_S.png)")
    md.append("")
    md.append("### Full PR-AUC table")
    md.append("")
    md.append(_table_pr_auc_full(canonical) if canonical else "_(No canonical cells yet.)_")
    md.append("")

    md.append("## Per-magnitude $\\Delta gc$")
    md.append("")
    md.append("Per-question-baselined Δgc at each magnitude in the locked grid ")
    md.append("$\\mathcal{M} = \\{-16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5, 0, +0.5, +1, +2, +3, +4, +5, +6, +7, +8, +10, +12, +16\\}$.")
    md.append("")
    md.append(_table_per_magnitude(rows) if rows else "_(No cells yet.)_")
    md.append("")

    md.append("## Convergence audit (probe every 100 training steps)")
    md.append("")
    md.append("Free probe data captured during training: NMSE, $\\ell_0$ density, and ")
    md.append("dead-feature count on a held-out batch. These plots cover both completed ")
    md.append("and in-flight cells, so curves may extend to different terminal steps.")
    md.append("")
    md.append(f"![NMSE vs training step (linear)]({assets_rel}/nmse_vs_step.png)")
    md.append("")
    md.append(f"![NMSE vs training step (log-log)]({assets_rel}/nmse_vs_step_loglog.png)")
    md.append("")
    md.append(f"![L0 vs training step (linear)]({assets_rel}/l0_vs_step.png)")
    md.append("")
    md.append(f"![L0 vs training step (log-log)]({assets_rel}/l0_vs_step_loglog.png)")
    md.append("")
    md.append(f"![Dead features vs training step (linear)]({assets_rel}/dead_vs_step.png)")
    md.append("")
    md.append(f"![Dead features vs training step (log-log)]({assets_rel}/dead_vs_step_loglog.png)")
    md.append("")

    md.append("## Per-cell metadata")
    md.append("")
    md.append("| Arch | bs | n_steps | seed | train_key | eval_key | ts |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows:
        md.append("| " + " | ".join([
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            str(_bs_from_row(r) or "?"),
            f"{_n_steps_from_row(r) or '?':,}" if _n_steps_from_row(r) else "?",
            str(r["seed"]),
            r["train_key"][:12],
            r["eval_key"][:12],
            r["ts"][:19].replace("T", " "),
        ]) + " |")
    md.append("")

    md.append("---")
    md.append("")
    md.append("_Plots and tables are regenerated by ")
    md.append("`purified/scripts/c7_paper_renderer.py`. Numbers are not hand-edited; ")
    md.append("if a value looks wrong, fix the source data and re-run the renderer._")

    out_path.write_text("\n".join(md))


# ── Driver ─────────────────────────────────────────────────────────────


def main(*, output_dir: Path) -> None:
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "c7_paper_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)
    md_path = output_dir / "c7_paper_results.md"

    all_rows = latest_per_cell(load_c7_rows())
    # Filter to headline config (n_steps == 300_000); older 30K cells are
    # reference-only and live in docs/components/c7.md, not the paper bundle.
    rows = [r for r in all_rows if _n_steps_from_row(r) == 300_000]
    log.info("[c7_paper] loaded %d c7 cells (%d at the 300K headline config)",
             len(all_rows), len(rows))

    canonical = canonical_rows(rows)

    if rows:
        # Δgc-vs-mag: merge canonical + extended per cell so the curve
        # spans the full evaluated range.
        plot_delta_gc_vs_magnitude(rows, assets_dir / "delta_gc_vs_magnitude.png")
        # Peak / PR-AUC: canonical only — the extended row's "peak"
        # is just the max over its 5-mag subset and isn't comparable.
        plot_peak_delta_gc_bar(canonical, assets_dir / "peak_delta_gc_bar.png")
        plot_pr_auc_vs_S(canonical, assets_dir / "pr_auc_vs_S.png")
        plot_pr_auc_S8_bar(canonical, assets_dir / "pr_auc_S8_bar.png")
        plot_probe_curves(rows, assets_dir)

    write_markdown(rows, canonical=canonical, assets_rel="c7_paper_assets",
                   out_path=md_path)
    log.info("[c7_paper] wrote %s", md_path)


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Destination directory (will contain c7_paper_results.md "
                         "and c7_paper_assets/).")
    args = ap.parse_args()
    main(output_dir=args.output_dir)


if __name__ == "__main__":
    cli()
