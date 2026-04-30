"""Metrics for the Stage B paper-budget hill-climb.

Han's brief defines the objective as "AUC of coherence vs steering" with
a concrete operationalisation: "peak success at coherence ≥ 1.5". Here we
adapt to our backtracking case study:

    primary_metric = peak |kw_rate - baseline_kw|
                     subject to:  max_consecutive_same_word_run ≤ 2
                     across all magnitudes in the steering grid

Why this metric:
  - peak |kw_rate - baseline_kw| because both positive and negative
    steering count (Dmitry's neg-steering ask). Higher = more effective
    direction.
  - max-run ≤ 2 is the coherence floor — at our scale this catches the
    "Wait Wait Wait..." collapse that inflates kw rate without producing
    real backtracking. (Sprint findings: collapse at +16 for our top
    feature.) Coherence ≤ 2 means the model isn't degenerating into
    repeat loops at any magnitude in the cell's curve.
  - peak across magnitudes (not at a fixed magnitude) because different
    cells (arch × hookpoint × k_per_position) may have different ⟨|z|⟩
    scales. The norm-rescaling in b1_steer_eval already controls for
    this somewhat, but peak-across-grid is robust to remaining variation.

Secondary metrics computed alongside:
  - peak_kw_no_coh: the unfiltered peak (sprint's 1.64× claim used this)
  - n_coherent_mags: how many of the magnitudes are coherent
  - best_magnitude: the magnitude where the peak happens
  - direction: 'positive' or 'negative' (sign of best magnitude)
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

WORD_RE = re.compile(r"\w+", re.UNICODE)
KEYWORD_RE = re.compile(r"\b(wait|hmm)\b", re.IGNORECASE)


def _max_repeat_run(text: str) -> int:
    toks = WORD_RE.findall(text.lower())
    if not toks:
        return 0
    best = run = 1
    for a, b in zip(toks, toks[1:]):
        if a == b:
            run += 1; best = max(best, run)
        else:
            run = 1
    return best


def _coh_ok(text: str, threshold: int = 2) -> bool:
    return _max_repeat_run(text) <= threshold


def cell_metric(rows: list[dict],
                source_tags: list[str],
                baseline_kw: float = 0.007,
                coh_threshold: int = 2) -> dict:
    """Compute the hill-climb metric for ONE cell's set of source tags.

    Args:
      rows: a flat list of B1 result rows (each has source, magnitude,
            prompt_id, keyword_rate, text, ...)
      source_tags: only count rows whose source is in this set (the cell's
            mined features × modes)
      baseline_kw: subtracted from every kw_rate to get effect size
      coh_threshold: max consecutive same-word run; cells whose magnitudes
            fail this floor are excluded from the peak

    Returns dict with primary metric + secondaries.
    """
    src_set = set(source_tags)
    by_cell_mag = defaultdict(list)   # (source, mag) -> [rows]
    for r in rows:
        if r["source"] in src_set:
            by_cell_mag[(r["source"], r["magnitude"])].append(r)

    peak = -float("inf")
    peak_no_coh = -float("inf")
    peak_mag = 0.0
    peak_source = None
    n_cells_total = 0
    n_cells_coherent = 0

    for (src, mag), cell_rows in by_cell_mag.items():
        n_cells_total += 1
        if not cell_rows:
            continue
        mean_kw = sum(r["keyword_rate"] for r in cell_rows) / len(cell_rows)
        # Coherence: cell is coherent iff EVERY prompt's text passes the
        # max-run floor. (One degenerate completion sinks the cell.)
        all_coh = all(_coh_ok(r.get("text", ""), coh_threshold) for r in cell_rows)
        if all_coh:
            n_cells_coherent += 1
        effect = abs(mean_kw - baseline_kw)
        if effect > peak_no_coh:
            peak_no_coh = effect
        if all_coh and effect > peak:
            peak = effect
            peak_mag = mag
            peak_source = src

    if peak == -float("inf"):
        peak = 0.0
        peak_mag = 0.0
        peak_source = None
    if peak_no_coh == -float("inf"):
        peak_no_coh = 0.0

    return {
        "primary_kw_at_coh": float(peak),
        "peak_kw_no_coh": float(peak_no_coh),
        "best_magnitude": float(peak_mag),
        "best_source": peak_source,
        "direction": "positive" if peak_mag > 0 else ("negative" if peak_mag < 0 else "none"),
        "n_cells_evaluated": n_cells_total,
        "n_cells_coherent": n_cells_coherent,
        "frac_coherent": (n_cells_coherent / max(1, n_cells_total)),
    }


def load_b1_rows(path: Path | str) -> list[dict]:
    obj = json.loads(Path(path).read_text())
    return obj["rows"]


def merge_b1_jsons(paths: list[Path], out_path: Path) -> int:
    """Merge per-cell B1 JSONs into one canonical file. Dedupes on
    (source, magnitude, prompt_id) so the per-cell DoM rows don't
    multiply.

    Returns the count of unique rows merged.
    """
    seen = set()
    rows = []
    meta = {}
    for p in paths:
        if not Path(p).exists():
            continue
        obj = json.loads(Path(p).read_text())
        meta = obj.get("meta", meta)
        for r in obj["rows"]:
            key = (r["source"], r["magnitude"], r["prompt_id"])
            if key in seen: continue
            seen.add(key); rows.append(r)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"rows": rows, "meta": meta}, indent=2))
    return len(rows)
