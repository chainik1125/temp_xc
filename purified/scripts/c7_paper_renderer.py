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
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Paper-ready style — no embedded titles (captions live in the .tex),
# slightly larger fonts, tight grid, no top/right spines.
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.dpi": 110,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "lines.linewidth": 1.8,
})

from temp_bench.config import (  # noqa: E402
    checkpoint_dir,
    purified_root,
)

# Branch ref to pull the unified c7 leaderboard from in --unified mode.
UNIFIED_BRANCH = "origin/final"

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
        r["_source"] = "ours"
        rows.append(r)
    return rows


def _git_show(branch: str, path: str) -> str | None:
    """Read ``path`` from ``branch`` via ``git show``; None on error."""
    try:
        return subprocess.check_output(
            ["git", "show", f"{branch}:{path}"],
            cwd=str(purified_root().parent),
            stderr=subprocess.DEVNULL,
        ).decode()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def load_unified_c7_rows(branch: str = UNIFIED_BRANCH) -> list[dict]:
    """Pull the unified c7 leaderboard rows from ``branch`` (read-only).

    Each row gets ``_source="unified"`` so downstream filters can distinguish.
    Returns [] if the branch / file isn't accessible.
    """
    text = _git_show(branch, "purified/results/leaderboard.jsonl")
    if text is None:
        log.warning("[c7_paper] could not read %s:purified/results/"
                    "leaderboard.jsonl — proceeding without unified rows",
                    branch)
        return []
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("component") != "c7":
            continue
        r["_source"] = "unified"
        rows.append(r)
    return rows


def _unified_train_cfg(train_key: str, branch: str = UNIFIED_BRANCH) -> dict:
    txt = _git_show(branch, f"purified/checkpoints/{train_key}/config.json")
    if txt is None:
        return {}
    try:
        return json.loads(txt).get("training_cfg") or {}
    except json.JSONDecodeError:
        return {}


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


# ── Bootstrap CI helpers (NeurIPS-checklist sweep) ───


def _judge_outputs_path(eval_key: str) -> Path:
    from temp_bench.config import run_dir
    return run_dir(eval_key) / "judge_outputs.jsonl"


def _judge_outputs_path_for_row(r: dict) -> Path:
    """Resolve the judge_outputs.jsonl path for a leaderboard row.

    Pre-existing skew: ``runner.run_cell`` computes ``eval_key`` over the
    full ``eval_cfg`` (including underscore-prefixed flags such as
    ``_extended_mags``), but ``my_eval_fn`` in ``experiments/c7_backtracking/run.py``
    strips underscore-prefixed keys before it derives the *workspace*
    eval_key. So extended-mags rows have a leaderboard ``eval_key`` that
    points at an empty workspace, while the actual judge file lives at a
    different (stripped-hash) workspace. This helper tries the leaderboard
    eval_key first, then falls back to the stripped-hash key by re-deriving
    it the way ``my_eval_fn`` does.
    """
    p = _judge_outputs_path(r["eval_key"])
    if p.exists():
        return p
    # Fallback: re-derive eval_key with underscore-prefixed keys stripped
    # (matches purified/experiments/c7_backtracking/run.py:_hash_eval_cfg).
    try:
        from temp_bench.config import compute_eval_key
        ec = r.get("eval_cfg", {}) or {}
        stripped = {
            k: v for k, v in ec.items()
            if not k.startswith("_") and k not in (
                "feature_mining_acts", "sentence_acts",
                "sentence_labels", "sentence_qids",
            )
        }
        alt_key = compute_eval_key(
            train_key=r["train_key"],
            eval_protocol_version=r.get("eval_protocol_version", "1.0.0"),
            eval_cfg=stripped,
        )
        return _judge_outputs_path(alt_key)
    except Exception:
        return p


def _per_qid_dgc_for_row(r: dict) -> dict[float, list[float]]:
    """Recompute per-question Δgc per magnitude for one leaderboard row,
    by reading its run_dir's judge_outputs.jsonl.

    Returns ``{magnitude: [Δgc per question]}`` where Δgc(qid, m) =
    gc(qid, m) - gc(qid, 0). Empty dict if judge file missing or no
    mag=0 baseline available.
    """
    p = _judge_outputs_path_for_row(r)
    if not p.exists():
        return {}
    arch = r["arch"]
    seed = int(r.get("seed", 0))
    panel_means: dict[tuple[str, float], list[int]] = defaultdict(list)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("arch") != arch:
            continue
        if int(row.get("seed", 0)) != seed:
            continue
        label = row.get("label", -1)
        if label is None or int(label) < 0:
            continue
        panel_means[(row["transcript_id"], float(row["magnitude"]))].append(int(label))
    if not panel_means:
        return {}
    panel = {k: float(np.mean(v)) for k, v in panel_means.items()}
    baseline = {qid: panel[(qid, 0.0)]
                for (qid, _m) in panel
                if (qid, 0.0) in panel}
    if not baseline:
        return {}
    out: dict[float, list[float]] = defaultdict(list)
    for (qid, mag), val in panel.items():
        if qid not in baseline:
            continue
        out[float(mag)].append(val - baseline[qid])
    return dict(out)


def _bootstrap_mean_ci(values: list[float], *, n_boot: int = 1000,
                       ci: float = 0.95, seed: int = 42
                       ) -> tuple[float, float, float]:
    """Bootstrap (mean, ci_lo, ci_hi) over ``values``. Returns
    ``(mean, lo, hi)`` with the percentile interval."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (0.0, 0.0, 0.0)
    rng = np.random.default_rng(int(seed))
    n = arr.size
    boots = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = arr[idx].mean()
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(boots, alpha))
    hi = float(np.quantile(boots, 1.0 - alpha))
    return (float(arr.mean()), lo, hi)


def _per_qid_gc_for_row(r: dict) -> dict[float, list[float]]:
    """Recompute per-question raw genuine-count per magnitude for one
    leaderboard row, by reading its run_dir's judge_outputs.jsonl.

    Returns ``{magnitude: [gc(qid, m) per question]}``. No baseline
    subtraction. Empty dict if judge file missing.
    """
    p = _judge_outputs_path_for_row(r)
    if not p.exists():
        return {}
    arch = r["arch"]
    seed = int(r.get("seed", 0))
    panel_means: dict[tuple[str, float], list[int]] = defaultdict(list)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("arch") != arch:
            continue
        if int(row.get("seed", 0)) != seed:
            continue
        label = row.get("label", -1)
        if label is None or int(label) < 0:
            continue
        panel_means[(row["transcript_id"], float(row["magnitude"]))].append(int(label))
    out: dict[float, list[float]] = defaultdict(list)
    for (qid, mag), vs in panel_means.items():
        out[float(mag)].append(float(np.mean(vs)))
    return dict(out)


def per_mag_dgc_with_ci(r: dict, *, n_boot: int = 1000, ci: float = 0.95,
                        ) -> dict[float, tuple[float, float, float]]:
    """For one leaderboard row, compute per-magnitude (mean, lo, hi)
    of Δgc bootstrapped over the cohort questions. Returns empty dict
    if the row's judge file is unavailable or has no usable rows."""
    raw = _per_qid_dgc_for_row(r)
    out: dict[float, tuple[float, float, float]] = {}
    for mag, deltas in raw.items():
        out[mag] = _bootstrap_mean_ci(deltas, n_boot=n_boot, ci=ci,
                                       seed=int(r.get("seed", 42)))
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
    (the leaderboard schema does not embed training_cfg at the top level).

    For unified rows (``_source == "unified"``), falls back to ``git show
    origin/final:purified/checkpoints/<train_key>/config.json`` since
    the checkpoint file does not exist on the local pod.
    """
    tk = r.get("train_key")
    if not tk:
        return {}
    if tk in _TRAIN_CFG_CACHE:
        return _TRAIN_CFG_CACHE[tk]
    cfg: dict = {}
    cfg_path = checkpoint_dir(tk) / "config.json"
    if cfg_path.exists():
        try:
            cfg = json.loads(cfg_path.read_text()).get("training_cfg") or {}
        except (json.JSONDecodeError, OSError):
            cfg = {}
    if not cfg and r.get("_source") == "unified":
        cfg = _unified_train_cfg(tk)
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
    """Plot Δgc vs magnitude per cell with bootstrap 95% CI shaded
    bands. Merges canonical + extended-mags evals for the same
    (arch, train_key) into a single curve. Bootstrap is over the 61
    cohort questions per (arch, magnitude) panel; n_boot=1000."""
    plt.figure(figsize=(9.5, 4.8))
    seen_labels: set[str] = set()

    # Build a per-cell {mag → (mean, lo, hi)} for bootstrap shading by
    # walking each underlying leaderboard row (canonical + extended)
    # for the (arch, train_key) cell, computing CIs from its
    # judge_outputs.jsonl, and merging by magnitude. For overlapping
    # magnitudes (mag=0 in both canonical + extended evals), keep
    # canonical.
    by_cell_rows: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_cell_rows[(r["arch"], r["train_key"])].append(r)

    for cell_key, cell_rows in by_cell_rows.items():
        # Canonical first so its mag=0 baseline wins for shared mags.
        cell_rows = sorted(cell_rows, key=lambda r: (_is_extended(r), r["ts"]))
        merged_ci: dict[float, tuple[float, float, float]] = {}
        for r in cell_rows:
            for mag, ci in per_mag_dgc_with_ci(r).items():
                merged_ci.setdefault(mag, ci)
        if not merged_ci:
            continue
        rep = next((r for r in cell_rows if not _is_extended(r)), cell_rows[0])
        mags = sorted(merged_ci.keys())
        means = [merged_ci[m][0] for m in mags]
        los = [merged_ci[m][1] for m in mags]
        his = [merged_ci[m][2] for m in mags]
        color = cell_color(rep["arch"], _bs_from_row(rep))
        label = cell_short_label(rep)
        if label in seen_labels:
            label = f"{label} [{rep['train_key'][:6]}]"
        seen_labels.add(label)
        plt.plot(mags, means, marker="o", markersize=4, linewidth=1.6,
                 color=color, linestyle="-", label=label)
        plt.fill_between(mags, los, his, color=color, alpha=0.18,
                         linewidth=0)

    ax = plt.gca()
    ax.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    ax.set_xlabel(r"steering magnitude $m$")
    ax.set_ylabel(r"$\Delta gc(a, m)$")
    # Move legend outside the right margin so the curves never get covered.
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
              frameon=False, ncol=1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_peak_delta_gc_bar(rows: list[dict], out_path: Path) -> None:
    """Bar chart of peak Δgc per cell, with bootstrap-CI error bars
    derived from the cohort-level Δgc distribution at each cell's
    peak magnitude."""
    rows = sorted(rows, key=lambda r: r["metrics"].get("delta_gc_peak", 0.0))
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
    peaks = [r["metrics"].get("delta_gc_peak", 0.0) for r in rows]
    colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]

    # Bootstrap 95% CI half-widths at the cell's peak magnitude.
    err_low: list[float] = []
    err_high: list[float] = []
    for r in rows:
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        if peak_mag is None:
            err_low.append(0.0); err_high.append(0.0)
            continue
        per_qid = _per_qid_dgc_for_row(r)
        deltas = per_qid.get(float(peak_mag), [])
        mean, lo, hi = _bootstrap_mean_ci(
            deltas, n_boot=1000, ci=0.95,
            seed=int(r.get("seed", 42)),
        )
        err_low.append(max(0.0, mean - lo))
        err_high.append(max(0.0, hi - mean))

    fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.4 * len(rows) + 1.2)))
    bars = ax.barh(labels, peaks, color=colors, alpha=0.9,
                   xerr=[err_low, err_high], capsize=3,
                   error_kw={"elinewidth": 1.0, "ecolor": "#222"})
    # Reserve right margin for value labels so they stay inside the plot box.
    x_max = max(p + h for p, h in zip(peaks, err_high))
    pad = max(0.05, 0.08 * x_max)
    ax.set_xlim(min(0, min(p - h for p, h in zip(peaks, err_low)) - 0.02),
                x_max + pad * 3)
    for r, bar, peak, eh in zip(rows, bars, peaks, err_high):
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        ax.text(peak + eh + pad * 0.3,
                bar.get_y() + bar.get_height() / 2,
                f"{peak:.2f}  ($m={peak_mag:+g}$)" if peak_mag is not None
                else f"{peak:.2f}",
                va="center", ha="left", fontsize=9, color="#222")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel(r"peak $\Delta gc$  (mean lift over $m{=}0$ baseline)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gc_at_peak_paired(rows: list[dict], out_path: Path) -> None:
    """Headline figure: per-question genuine-backtracking counts, paired
    unsteered vs. optimal-steering bars per cell, sorted descending by
    gc_at_peak. Bootstrap 95% CIs over the 61 cohort questions.

    Vertical-bar variant — paper-team request. Caption-ready:
    no embedded title, value labels stay inside the plot box.
    """
    rows = [r for r in rows if r["metrics"].get("gc_at_peak_mag") is not None]
    if not rows:
        return
    # Descending — best cell on the left, like a paper leaderboard.
    rows = sorted(rows, key=lambda r: -r["metrics"].get("gc_at_peak_mag", 0.0))
    labels = [cell_short_label(r) for r in rows]
    seen: dict[str, int] = {}
    final_labels = []
    for r, lab in zip(rows, labels):
        if lab in seen:
            final_labels.append(f"{lab}\n[{r['train_key'][:6]}]")
        else:
            final_labels.append(lab)
        seen[lab] = seen.get(lab, 0) + 1
    labels = final_labels

    base_means: list[float] = []
    base_lo: list[float] = []
    base_hi: list[float] = []
    peak_means: list[float] = []
    peak_lo: list[float] = []
    peak_hi: list[float] = []
    peak_mags: list[float | None] = []
    for r in rows:
        per_qid = _per_qid_gc_for_row(r)
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        peak_mags.append(float(peak_mag) if peak_mag is not None else None)
        seed = int(r.get("seed", 42))
        bm, blo, bhi = _bootstrap_mean_ci(
            per_qid.get(0.0, []), n_boot=1000, ci=0.95, seed=seed)
        peak_vals = per_qid.get(float(peak_mag), []) if peak_mag is not None else []
        pm, plo, phi = _bootstrap_mean_ci(
            peak_vals, n_boot=1000, ci=0.95, seed=seed + 1)
        base_means.append(bm); base_lo.append(blo); base_hi.append(bhi)
        peak_means.append(pm); peak_lo.append(plo); peak_hi.append(phi)

    n = len(rows)
    x_pos = np.arange(n)
    bar_w = 0.4

    fig, ax = plt.subplots(figsize=(max(6.0, 1.05 * n + 1.5), 4.6))
    base_color = "#bbbbbb"
    peak_colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    base_err_lo = [m - lo for m, lo in zip(base_means, base_lo)]
    base_err_hi = [hi - m for m, hi in zip(base_means, base_hi)]
    peak_err_lo = [m - lo for m, lo in zip(peak_means, peak_lo)]
    peak_err_hi = [hi - m for m, hi in zip(peak_means, peak_hi)]

    ax.bar(x_pos - bar_w / 2, base_means, width=bar_w,
           color=base_color, alpha=0.9, label=r"unsteered ($m{=}0$)",
           yerr=[base_err_lo, base_err_hi], capsize=3,
           error_kw={"elinewidth": 1.0, "ecolor": "#444"})
    bars_peak = ax.bar(x_pos + bar_w / 2, peak_means, width=bar_w,
                       color=peak_colors, alpha=0.95,
                       label=r"optimal steering (peak $m$)",
                       yerr=[peak_err_lo, peak_err_hi], capsize=3,
                       error_kw={"elinewidth": 1.0, "ecolor": "#222"})

    # Value labels above each peak bar — small, inside the plot box.
    y_max = max(hi for hi in peak_hi)
    headroom = max(0.18, 0.12 * y_max)
    for bar, pm_val, pm_mag, hi in zip(bars_peak, peak_means, peak_mags, peak_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + headroom * 0.18,
                f"{pm_val:.2f}\n$m={pm_mag:+g}$" if pm_mag is not None
                else f"{pm_val:.2f}",
                ha="center", va="bottom", fontsize=8.5,
                color="#222")

    ax.set_xticks(x_pos)
    # Two-line labels: arch on top, batch size below — readable at any width.
    short_labels = []
    for r in rows:
        bs = _bs_from_row(r)
        arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
        short_labels.append(f"{arch}\n($bs{{=}}{bs}$)")
    ax.set_xticklabels(short_labels, rotation=0, fontsize=10)
    ax.set_ylabel(r"genuine backtracks per question, $gc(a, m)$")
    ax.set_ylim(0, y_max + headroom + 0.10)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(loc="upper right", framealpha=0.95, frameon=True, ncol=1)
    ax.grid(axis="x", visible=False)
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_gc_at_peak_paired_compact(rows: list[dict], out_path: Path) -> None:
    """Compact variant of plot_gc_at_peak_paired sized for the headline
    two-panel figure (text-comparison panel on the left, bar chart on the
    right, both at ~half the page width). Smaller fonts, rotated x-tick
    labels, tighter margins."""
    rows = [r for r in rows if r["metrics"].get("gc_at_peak_mag") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -r["metrics"].get("gc_at_peak_mag", 0.0))
    base_means: list[float] = []
    base_lo: list[float] = []
    base_hi: list[float] = []
    peak_means: list[float] = []
    peak_lo: list[float] = []
    peak_hi: list[float] = []
    peak_mags: list[float | None] = []
    for r in rows:
        per_qid = _per_qid_gc_for_row(r)
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        peak_mags.append(float(peak_mag) if peak_mag is not None else None)
        seed = int(r.get("seed", 42))
        bm, blo, bhi = _bootstrap_mean_ci(
            per_qid.get(0.0, []), n_boot=1000, ci=0.95, seed=seed)
        peak_vals = per_qid.get(float(peak_mag), []) if peak_mag is not None else []
        pm, plo, phi = _bootstrap_mean_ci(
            peak_vals, n_boot=1000, ci=0.95, seed=seed + 1)
        base_means.append(bm); base_lo.append(blo); base_hi.append(bhi)
        peak_means.append(pm); peak_lo.append(plo); peak_hi.append(phi)
    n = len(rows)
    x_pos = np.arange(n)
    bar_w = 0.4
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    base_color = "#bbbbbb"
    peak_colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    base_err_lo = [m - lo for m, lo in zip(base_means, base_lo)]
    base_err_hi = [hi - m for m, hi in zip(base_means, base_hi)]
    peak_err_lo = [m - lo for m, lo in zip(peak_means, peak_lo)]
    peak_err_hi = [hi - m for m, hi in zip(peak_means, peak_hi)]
    ax.bar(x_pos - bar_w / 2, base_means, width=bar_w,
           color=base_color, alpha=0.9, label=r"unsteered ($m{=}0$)",
           yerr=[base_err_lo, base_err_hi], capsize=2,
           error_kw={"elinewidth": 0.8, "ecolor": "#444"})
    bars_peak = ax.bar(x_pos + bar_w / 2, peak_means, width=bar_w,
                       color=peak_colors, alpha=0.95,
                       label=r"optimal $m$",
                       yerr=[peak_err_lo, peak_err_hi], capsize=2,
                       error_kw={"elinewidth": 0.8, "ecolor": "#222"})
    y_max = max(hi for hi in peak_hi)
    headroom = max(0.18, 0.12 * y_max)
    for bar, pm_val, pm_mag, hi in zip(bars_peak, peak_means, peak_mags, peak_hi):
        ax.text(bar.get_x() + bar.get_width() / 2, hi + headroom * 0.18,
                f"{pm_val:.2f}", ha="center", va="bottom", fontsize=7,
                color="#222")
    ax.set_xticks(x_pos)
    short_labels = []
    for r in rows:
        bs = _bs_from_row(r)
        arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
        short_labels.append(f"{arch}\n($bs{{=}}{bs}$)")
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"$gc(a, m)$ per question", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


_METRIC_META = {
    "pr_auc":  {"label": "PR-AUC",  "chance": 0.12, "chance_text": r"chance $\approx 0.12$"},
    "roc_auc": {"label": "ROC-AUC", "chance": 0.5,  "chance_text": r"chance $= 0.5$"},
}


def _draw_metric_vs_S(ax, rows: list[dict], metric: str, *,
                      show_legend: bool = True) -> None:
    meta = _METRIC_META[metric]
    seen_labels: set[str] = set()
    for r in rows:
        ys = []
        for S in S_GRID:
            v = r["metrics"].get(f"{metric}_S{S}")
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
        ax.plot(valid_x, valid_y, marker="o", markersize=4, color=color,
                linestyle="-", label=label)
    ax.axhline(meta["chance"], color="grey", linestyle=":", linewidth=1,
               label=meta["chance_text"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(list(S_GRID))
    ax.set_xticklabels([str(s) for s in S_GRID])
    ax.set_xlabel(r"top-$S$ probe features")
    ax.set_ylabel(meta["label"])
    if show_legend:
        ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                  frameon=False, ncol=1)


def plot_pr_auc_vs_S(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _draw_metric_vs_S(ax, rows, "pr_auc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_roc_auc_vs_S(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    _draw_metric_vs_S(ax, rows, "roc_auc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_probe_metrics_vs_S_2panel(rows: list[dict], out_path: Path) -> None:
    """Side-by-side: PR-AUC (left) and ROC-AUC (right) vs top-$S$ probe
    features. Same architectures, same colours, one shared legend on
    the right of the figure so the per-panel chance-line entries are
    the only thing that differs between panels."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6),
                             sharex=True)
    _draw_metric_vs_S(axes[0], rows, "pr_auc",  show_legend=False)
    _draw_metric_vs_S(axes[1], rows, "roc_auc", show_legend=False)
    # Shared per-arch legend (excluding the chance entry, which is
    # panel-specific and stays drawn inside each panel as a dotted line).
    handles, labels = axes[0].get_legend_handles_labels()
    arch_handles = [h for h, l in zip(handles, labels) if "chance" not in l]
    arch_labels  = [l for l in labels if "chance" not in l]
    fig.legend(arch_handles, arch_labels, loc="center left",
               bbox_to_anchor=(0.97, 0.5), frameon=False, fontsize=9)
    fig.tight_layout(rect=[0.0, 0.0, 0.95, 1.0])
    fig.savefig(out_path)
    plt.close(fig)


def _draw_metric_S8_bar(ax, rows: list[dict], metric: str) -> None:
    meta = _METRIC_META[metric]
    rows = sorted(rows, key=lambda r: r["metrics"].get(f"{metric}_S8", 0.0))
    labels = [cell_short_label(r) for r in rows]
    seen: dict[str, int] = {}
    final_labels: list[str] = []
    for r, lab in zip(rows, labels):
        if lab in seen:
            final_labels.append(f"{lab} [{r['train_key'][:6]}]")
        else:
            final_labels.append(lab)
        seen[lab] = seen.get(lab, 0) + 1
    labels = final_labels
    aucs = [r["metrics"].get(f"{metric}_S8", 0.0) for r in rows]
    colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    bars = ax.barh(labels, aucs, color=colors, alpha=0.9)
    x_max = max(aucs) if aucs else 1.0
    pad = max(0.015, 0.06 * x_max)
    ax.set_xlim(0, x_max + pad * 3)
    for bar, auc in zip(bars, aucs):
        ax.text(auc + pad * 0.3, bar.get_y() + bar.get_height() / 2,
                f"{auc:.3f}", va="center", ha="left", fontsize=9, color="#222")
    ax.axvline(meta["chance"], color="grey", linestyle=":", linewidth=1)
    ax.text(meta["chance"], len(rows) - 0.4, "  " + meta["chance_text"],
            va="bottom", ha="left", fontsize=8.5, color="#666")
    ax.set_xlabel(meta["label"] + r" at $S{=}8$")
    ax.grid(axis="y", visible=False)


def plot_pr_auc_S8_bar(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.4 * len(rows) + 1.4)))
    _draw_metric_S8_bar(ax, rows, "pr_auc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()


def plot_pr_auc_S8_bar_compact(rows: list[dict], out_path: Path) -> None:
    """Compact vertical-bar PR-AUC@8 chart matching the figsize, font sizes,
    and x-tick treatment of plot_gc_at_peak_paired_compact so the two can
    be placed side-by-side in the headline 1x3 figure without geometric
    mismatch."""
    rows = [r for r in rows if r["metrics"].get("pr_auc_S8") is not None]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: -r["metrics"]["pr_auc_S8"])
    n = len(rows)
    x_pos = np.arange(n)
    bar_w = 0.6
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    colors = [cell_color(r["arch"], _bs_from_row(r)) for r in rows]
    bars = ax.bar(x_pos, [r["metrics"]["pr_auc_S8"] for r in rows],
                  width=bar_w, color=colors, alpha=0.95)
    # chance line at 0.12 (the positive-class prior); label via legend so
    # the line annotation doesn't sit over the bars.
    ax.axhline(0.12, color="#666", linestyle=":", linewidth=1.0,
               label=r"chance $\approx 0.12$")
    # value labels above each bar
    y_max = max(r["metrics"]["pr_auc_S8"] for r in rows)
    headroom = 0.06 * y_max
    for bar, r in zip(bars, rows):
        ax.text(bar.get_x() + bar.get_width() / 2,
                r["metrics"]["pr_auc_S8"] + headroom * 0.18,
                f"{r['metrics']['pr_auc_S8']:.2f}",
                ha="center", va="bottom", fontsize=7, color="#222")
    ax.set_xticks(x_pos)
    short_labels = []
    for r in rows:
        bs = _bs_from_row(r)
        arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
        short_labels.append(f"{arch}\n($bs{{=}}{bs}$)")
    ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(r"PR-AUC at $S{=}8$", fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, y_max * 1.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.04)
    ax.legend(loc="upper right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_roc_auc_S8_bar(rows: list[dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, max(3.0, 0.4 * len(rows) + 1.4)))
    _draw_metric_S8_bar(ax, rows, "roc_auc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close()


def plot_probe_metrics_S8_bar_2panel(rows: list[dict], out_path: Path) -> None:
    """Side-by-side: PR-AUC@S=8 bar (left) and ROC-AUC@S=8 bar (right).
    Each panel sorts its own bars descending so the within-metric
    ordering is immediately readable; the colours stay tied to the
    same (arch, bs) cell across panels."""
    n = len(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, max(3.2, 0.4 * n + 1.6)))
    _draw_metric_S8_bar(axes[0], rows, "pr_auc")
    _draw_metric_S8_bar(axes[1], rows, "roc_auc")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


# ── Unified-mode helpers (--unified, pulls unified leaderboard) ──────────


def best_per_arch(rows: list[dict]) -> list[dict]:
    """Pick the cell with maximum peak Δgc per architecture.

    Ignores extended-mags rows: their "peak" is over a magnitude-restricted
    subset and isn't comparable to canonical peaks.
    """
    by_arch: dict[str, dict] = {}
    for r in rows:
        if _is_extended(r):
            continue
        arch = r["arch"]
        peak = r.get("metrics", {}).get("delta_gc_peak")
        if peak is None:
            continue
        cur = by_arch.get(arch)
        cur_peak = cur.get("metrics", {}).get("delta_gc_peak") if cur else None
        if cur is None or peak > cur_peak:
            by_arch[arch] = r
    return list(by_arch.values())


def cell_unified_label(r: dict) -> str:
    """Verbose label for unified plots: arch + bs + n_steps + source."""
    arch = PAPER_ARCH_LABEL.get(r["arch"], r["arch"])
    bs = _bs_from_row(r)
    ns = _n_steps_from_row(r)
    src = r.get("_source", "ours")
    pieces = [arch]
    if bs is not None:
        pieces.append(f"bs={bs}")
    if ns is not None:
        pieces.append(f"{ns//1000}K" if ns >= 1000 else f"{ns}")
    if src == "unified":
        pieces.append("[unified]")
    return " ".join(pieces)


def plot_unified_headline(rows: list[dict], out_path: Path) -> None:
    """Best-cell-per-arch Δgc-vs-magnitude — one curve per architecture
    at whatever training config gave it the highest peak. Legend includes
    the source (unified vs ours) and training-config tag for each curve."""
    best = best_per_arch(rows)
    plt.figure(figsize=(8.5, 5.0))
    ordered = sorted(best, key=lambda r: PAPER_ARCH_ORDER.index(r["arch"])
                     if r["arch"] in PAPER_ARCH_ORDER else 99)
    for r in ordered:
        pairs = parse_mag_metrics(r["metrics"])
        if not pairs:
            continue
        mags = [m for m, _ in pairs]
        deltas = [d for _, d in pairs]
        # One curve per arch in the headline figure → use the per-arch
        # color rather than per-cell (we don't want bs-shading here).
        color = PAPER_ARCH_COLOR.get(r["arch"], "#333333")
        plt.plot(mags, deltas, marker="o", markersize=4, linewidth=1.7,
                 color=color, linestyle="-",
                 label=cell_unified_label(r))
    plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.xlabel("Steering magnitude $m$")
    plt.ylabel(r"$\Delta gc(a, m)$  (per-question baseline at $m=0$)")
    plt.title(r"Inducement headline: best cell per architecture (unified)")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_unified_peak_bar(rows: list[dict], out_path: Path) -> None:
    """Bar chart of peak Δgc, best cell per arch, with config annotation."""
    best = best_per_arch(rows)
    best.sort(key=lambda r: r["metrics"].get("delta_gc_peak", 0.0))
    labels = [cell_unified_label(r) for r in best]
    peaks = [r["metrics"].get("delta_gc_peak", 0.0) for r in best]
    colors = [PAPER_ARCH_COLOR.get(r["arch"], "#333333") for r in best]
    plt.figure(figsize=(9.0, max(3.5, 0.45 * len(best) + 1.5)))
    bars = plt.barh(labels, peaks, color=colors, alpha=0.85)
    for r, bar in zip(best, bars):
        peak = r["metrics"].get("delta_gc_peak", 0.0)
        peak_mag = r["metrics"].get("delta_gc_peak_magnitude")
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                 f" {peak:.3f} (m={peak_mag:+g})" if peak_mag is not None
                 else f" {peak:.3f}",
                 va="center", fontsize=9)
    plt.axvline(0, color="black", linewidth=0.5)
    plt.xlabel(r"Peak $\Delta gc$")
    plt.title(r"Peak $\Delta gc$ per architecture (unified, best cell)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_sprint_vs_extended_comparison(rows: list[dict], out_path: Path) -> None:
    """Per-arch overlay of best sprint cell + best extended cell so the
    token-budget sensitivity is visible at a glance.

    Sprint = n_steps < 50K (the unified runs); extended = n_steps >= 100K (ours).
    Only archs that appear in BOTH bins are plotted.
    """
    by_arch: dict[str, dict[str, dict]] = {}
    for r in rows:
        if _is_extended(r):
            continue
        ns = _n_steps_from_row(r)
        if ns is None:
            continue
        arch = r["arch"]
        bin_ = "sprint" if ns < 50_000 else "extended"
        peak = r.get("metrics", {}).get("delta_gc_peak", -float("inf"))
        slot = by_arch.setdefault(arch, {})
        cur = slot.get(bin_)
        cur_peak = cur.get("metrics", {}).get("delta_gc_peak", -float("inf")) if cur else -float("inf")
        if cur is None or peak > cur_peak:
            slot[bin_] = r

    plotted = 0
    plt.figure(figsize=(9.0, 5.5))
    for arch in PAPER_ARCH_ORDER + ["tfa", "stacked_sae"]:
        bins = by_arch.get(arch)
        if not bins or "sprint" not in bins or "extended" not in bins:
            continue
        color = PAPER_ARCH_COLOR.get(arch, "#333333")
        sprint = bins["sprint"]
        extended = bins["extended"]
        sp_pairs = parse_mag_metrics(sprint["metrics"])
        ex_pairs = parse_mag_metrics(extended["metrics"])
        if sp_pairs:
            mags, deltas = zip(*sp_pairs)
            ns = _n_steps_from_row(sprint) or 0
            plt.plot(mags, deltas, color=color, linestyle="--", linewidth=1.4,
                     alpha=0.75, marker="x", markersize=5,
                     label=f"{PAPER_ARCH_LABEL.get(arch, arch)} sprint ({ns//1000}K)")
        if ex_pairs:
            mags, deltas = zip(*ex_pairs)
            ns = _n_steps_from_row(extended) or 0
            plt.plot(mags, deltas, color=color, linestyle="-", linewidth=1.6,
                     marker="o", markersize=4,
                     label=f"{PAPER_ARCH_LABEL.get(arch, arch)} extended ({ns//1000}K)")
        plotted += 1
    if plotted == 0:
        plt.close()
        return
    plt.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.axvline(0, color="black", linewidth=0.5, alpha=0.5)
    plt.xlabel("Steering magnitude $m$")
    plt.ylabel(r"$\Delta gc(a, m)$")
    plt.title(r"Token-budget sensitivity: sprint (dashed) vs extended (solid) per arch")
    plt.legend(fontsize=8, loc="best", ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def _table_unified_headline(rows: list[dict]) -> str:
    """Best-cell-per-arch markdown table for unified rendering."""
    best = best_per_arch(rows)
    best.sort(key=lambda r: -r["metrics"].get("delta_gc_peak", 0.0))
    headers = ["Arch", "Source", "bs", "$n_{\\text{steps}}$",
               "Peak $\\Delta gc$", "Peak mag",
               "PR-AUC@8", "PR-AUC@32"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in best:
        m = r["metrics"]
        ns = _n_steps_from_row(r)
        bs = _bs_from_row(r)
        src = r.get("_source", "ours")
        lines.append("| " + " | ".join([
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            "unified" if src == "unified" else "ours",
            str(bs) if bs is not None else "?",
            f"{ns:,}" if ns is not None else "?",
            fmt_num(m.get("delta_gc_peak")),
            fmt_num(m.get("delta_gc_peak_magnitude"), precision=1),
            fmt_num(m.get("pr_auc_S8")),
            fmt_num(m.get("pr_auc_S32")),
        ]) + " |")
    return "\n".join(lines)


# ── (back to existing single-source helpers) ──────────────────────────


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


def plot_probe_curves_combined(out_dir: Path) -> Path | None:
    """One PNG with three side-by-side panels (NMSE / ℓ0 / dead) sharing a
    single horizontal legend above the row. This is the appendix-figure
    variant — the per-metric individual PNGs (``plot_probe_curves``) are
    still emitted alongside for the markdown bundle and the standalone
    \\autofig path."""
    series = {}
    seen_label_count: dict[str, int] = {}
    for arch, bs, tk, log_rows in _enumerate_probe_cells():
        base_label = f"{PAPER_ARCH_LABEL.get(arch, arch)} bs={bs}"
        n = seen_label_count.get(base_label, 0)
        seen_label_count[base_label] = n + 1
        label = base_label if n == 0 else f"{base_label} [{tk[:6]}]"
        series[(arch, bs, tk)] = (label, log_rows, arch)
    if not series:
        return None
    metric_specs = [
        ("nmse", "NMSE on held-out batch"),
        ("l0",   r"$\ell_0$ density on held-out batch"),
        ("dead", r"Dead features (out of $d_{\mathrm{SAE}}$)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    handles: list = []
    labels: list[str] = []
    for ax, (metric, ylabel) in zip(axes, metric_specs):
        for (arch, bs, _tk), (lab, log_rows, arch_name) in sorted(series.items()):
            steps = [x["step"] for x in log_rows]
            vals = [x.get(metric) for x in log_rows]
            valid = [(s, v) for s, v in zip(steps, vals)
                     if v is not None and v > 0]
            if not valid:
                continue
            xs, ys = zip(*valid)
            color = cell_color(arch_name, bs)
            line, = ax.plot(xs, ys, color=color, linestyle="-",
                            linewidth=1.4, label=lab)
            if lab not in labels:
                handles.append(line)
                labels.append(lab)
        ax.set_xlabel("training step")
        ax.set_ylabel(ylabel)
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    # Single shared legend above the row, laid out horizontally.
    fig.legend(handles, labels, loc="upper center",
               bbox_to_anchor=(0.5, 1.04),
               ncol=len(labels), frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_path = out_dir / "probe_curves_combined.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    # Two scales per metric: linear-x + log-y (default) and log-log.
    # Both use log-y so the high-dynamic-range trajectories (NMSE
    # ~1 → ~0.02; L0 ~32K → ~100; dead-feature counts) are readable.
    # The "linear" variant only differs from the log-log one on the
    # x-axis (training-step axis stays linear).
    scale_specs = [
        ("",       "linear", "log"),
        ("_loglog", "log",   "log"),
    ]
    for metric, ylabel, fname_stem in metric_specs:
        for fname_suffix, x_scale, y_scale in scale_specs:
            plt.figure(figsize=(7.0, 4.4))
            for (arch, bs, _tk), (lab, log_rows, arch_name) in sorted(series.items()):
                steps = [x["step"] for x in log_rows]
                vals = [x.get(metric) for x in log_rows]
                # log axes can't render zero / negative values; drop them.
                # The y-axis is log in both variants; the x-axis only
                # in the log-log variant.
                valid = []
                for s, v in zip(steps, vals):
                    if v is None:
                        continue
                    if y_scale == "log" and v <= 0:
                        continue
                    if x_scale == "log" and s <= 0:
                        continue
                    valid.append((s, v))
                if not valid:
                    continue
                xs, ys = zip(*valid)
                color = cell_color(arch_name, bs)
                plt.plot(xs, ys, color=color, linestyle="-",
                         linewidth=1.4, label=lab)
            ax = plt.gca()
            ax.set_xlabel("training step")
            ax.set_ylabel(ylabel)
            ax.set_xscale(x_scale)
            ax.set_yscale(y_scale)
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5),
                      frameon=False, ncol=1)
            ax.grid(alpha=0.25, which="both")
            plt.tight_layout()
            out_path = out_dir / f"{fname_stem}{fname_suffix}.png"
            plt.savefig(out_path)
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
    headers = ["Arch", "bs", "n_steps", "$gc_{\\text{peak}}$", "$gc_{\\text{base}}$",
               "Peak $\\Delta gc$", "Peak mag",
               "PR-AUC@8", "ROC-AUC@8", "n_judge", "ts (UTC)"]
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        lines.append("| " + " | ".join([
            PAPER_ARCH_LABEL.get(r["arch"], r["arch"]),
            str(_bs_from_row(r) or "?"),
            f"{_n_steps_from_row(r) or '?':,}" if _n_steps_from_row(r) else "?",
            fmt_num(m.get("gc_at_peak_mag")),
            fmt_num(m.get("gc_at_baseline")),
            fmt_num(m.get("delta_gc_peak")),
            fmt_num(m.get("delta_gc_peak_magnitude"), precision=1),
            fmt_num(m.get("pr_auc_S8")),
            fmt_num(m.get("roc_auc_S8")),
            f"{int(m.get('n_judge_calls', 0))}",
            r["ts"][:19].replace("T", " "),
        ]) + " |")
    return "\n".join(lines)


def _table_pr_auc_full(rows: list[dict]) -> str:
    """PR-AUC + ROC-AUC at every S in the grid. PR-AUC is the headline
    detection metric (chance ≈ 0.12, the positive-class prevalence);
    ROC-AUC reported alongside for continuity with prior internal
    exploration that used the older ROC-AUC convention."""
    headers = (["Arch (cell)"]
               + [f"PR@{s}" for s in S_GRID]
               + [f"ROC@{s}" for s in S_GRID])
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        m = r["metrics"]
        row = [cell_short_label(r)]
        for s in S_GRID:
            row.append(fmt_num(m.get(f"pr_auc_S{s}")))
        for s in S_GRID:
            row.append(fmt_num(m.get(f"roc_auc_S{s}")))
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
                   assets_rel: str, out_path: Path,
                   unified_rows: list[dict] | None = None) -> None:
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
    md.append("### Headline figure: genuine backtracks per question (raw counts)")
    md.append("")
    md.append("Direct comparison of the unsteered baseline (gc at magnitude $0$) vs. the ")
    md.append("optimal-magnitude steered cell (gc at peak $m$), per architecture. Bigger ")
    md.append("paired bar = the architecture's mined feature direction actually causes ")
    md.append("more genuine backtracking events when steered. Error bars are bootstrap 95% ")
    md.append("CIs over the 61 cohort questions ($n_{\\text{boot}} = 1000$).")
    md.append("")
    md.append(f"![gc at peak vs baseline (paired)]({assets_rel}/gc_at_peak_paired.png)")
    md.append("")
    md.append("### Mean lift over baseline ($\\Delta gc_{\\text{peak}}$)")
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

    md.append("## Detection: PR-AUC + ROC-AUC")
    md.append("")
    md.append("Sparse-probe PR-AUC and ROC-AUC at top-$S$ feature counts ($5$-fold GroupKFold by ")
    md.append("question, $\\ell_1$-regularised logistic regression, $C = 1$). PR-AUC chance $\\approx 0.12$ ")
    md.append("(positive-class prior); ROC-AUC chance $= 0.5$.")
    md.append("")
    md.append(f"![Detection metrics at S=8 (PR-AUC + ROC-AUC)]({assets_rel}/probe_metrics_S8_bar.png)")
    md.append("")
    md.append(f"![Detection metrics vs S (PR-AUC + ROC-AUC)]({assets_rel}/probe_metrics_vs_S.png)")
    md.append("")
    md.append("Single-metric variants (kept for back-compat with the C7 paper section TeX):")
    md.append("")
    md.append(f"![PR-AUC at S=8]({assets_rel}/pr_auc_S8_bar.png)")
    md.append(f"![ROC-AUC at S=8]({assets_rel}/roc_auc_S8_bar.png)")
    md.append("")
    md.append(f"![PR-AUC vs S]({assets_rel}/pr_auc_vs_S.png)")
    md.append(f"![ROC-AUC vs S]({assets_rel}/roc_auc_vs_S.png)")
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

    if unified_rows is not None:
        n_han = sum(1 for r in unified_rows if r.get("_source") == "unified")
        md.append("## Unified comparison")
        md.append("")
        md.append(f"_Pulled {n_han} additional row(s) from `{UNIFIED_BRANCH}:purified/results/leaderboard.jsonl` "
                  f"(read-only; merged at render time)._")
        md.append("")
        md.append("### Best cell per architecture (across both pods)")
        md.append("")
        md.append(_table_unified_headline(unified_rows) if unified_rows
                  else "_(No unified cells yet.)_")
        md.append("")
        md.append(f"![Best-cell-per-arch Δgc vs magnitude (unified)]"
                  f"({assets_rel}/delta_gc_unified_headline.png)")
        md.append("")
        md.append(f"![Peak Δgc per arch (unified)]"
                  f"({assets_rel}/peak_delta_gc_unified_bar.png)")
        md.append("")
        md.append("### Token-budget sensitivity (sprint vs extended)")
        md.append("")
        md.append("For architectures evaluated at BOTH a sprint config "
                  "(`n_steps < 50K`, the legacy pod) and an extended config "
                  "(`n_steps >= 100K`, our pod), the per-magnitude curves are "
                  "overlaid below. Dashed = sprint, solid = extended.")
        md.append("")
        md.append(f"![Sprint vs extended comparison]"
                  f"({assets_rel}/sprint_vs_extended_comparison.png)")
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


def main(*, output_dir: Path, unified: bool = False) -> None:
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
        plot_gc_at_peak_paired(canonical, assets_dir / "gc_at_peak_paired.png")
        plot_gc_at_peak_paired_compact(canonical, assets_dir / "gc_at_peak_paired_compact.png")
        plot_pr_auc_S8_bar_compact(canonical, assets_dir / "pr_auc_S8_bar_compact.png")
        plot_pr_auc_vs_S(canonical, assets_dir / "pr_auc_vs_S.png")
        plot_roc_auc_vs_S(canonical, assets_dir / "roc_auc_vs_S.png")
        plot_probe_metrics_vs_S_2panel(
            canonical, assets_dir / "probe_metrics_vs_S.png")
        plot_pr_auc_S8_bar(canonical, assets_dir / "pr_auc_S8_bar.png")
        plot_roc_auc_S8_bar(canonical, assets_dir / "roc_auc_S8_bar.png")
        plot_probe_metrics_S8_bar_2panel(
            canonical, assets_dir / "probe_metrics_S8_bar.png")
        plot_probe_curves(rows, assets_dir)
        plot_probe_curves_combined(assets_dir)

    # ── --unified mode: pull the unified c7 leaderboard via git show ─────────
    unified_rows: list[dict] | None = None
    if unified:
        unified_rows_extra = load_unified_c7_rows()
        log.info("[c7_paper] unified mode: %d unified rows, %d ours rows",
                 len(unified_rows_extra), len(all_rows))
        # Merge + dedupe by (arch, train_key, eval_key) — each pod's cells
        # carry distinct train_keys (different training_cfg), so no collision.
        unified_rows = latest_per_cell(all_rows + unified_rows_extra)
        # New plots — write next to the single-source ones; existing
        # plots are unchanged so the live paper-loop diff stays minimal.
        plot_unified_headline(unified_rows,
                              assets_dir / "delta_gc_unified_headline.png")
        plot_unified_peak_bar(unified_rows,
                              assets_dir / "peak_delta_gc_unified_bar.png")
        plot_sprint_vs_extended_comparison(
            unified_rows, assets_dir / "sprint_vs_extended_comparison.png")

    write_markdown(rows, canonical=canonical, assets_rel="c7_paper_assets",
                   out_path=md_path, unified_rows=unified_rows)
    log.info("[c7_paper] wrote %s", md_path)


def _purified_root() -> Path:
    return Path(__file__).resolve().parent.parent


def cli():
    root = _purified_root()
    ap = argparse.ArgumentParser(description=(
        "C7 (Ward Stage B backtracking) paper figure renderer. "
        "Reads from the in-repo leaderboard / checkpoints / runs."
    ))
    ap.add_argument(
        "--output-dir", type=Path,
        default=root / "figs" / "c7",
        help="Destination directory (default: purified/figs/c7/). Will contain "
             "c7_paper_results.md and c7_paper_assets/.",
    )
    ap.add_argument("--unified", action="store_true",
                    help=f"Pull additional c7 leaderboard rows from {UNIFIED_BRANCH} (read-only "
                         "via `git show`) and render additional unified plots.")
    args = ap.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main(output_dir=args.output_dir, unified=args.unified)


if __name__ == "__main__":
    cli()
