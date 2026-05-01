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
                coh_threshold: int = 2,
                sonnet_grades: dict | None = None,
                sonnet_floor: int = 2) -> dict:
    """Compute the hill-climb metric for ONE cell's set of source tags.

    Two coherence floors are evaluated:
      1. Per-word `max_same_word_run ≤ coh_threshold` — fast heuristic.
      2. (Optional) Sonnet 4.6 0–3 grade `≥ sonnet_floor` — catches
         sentence-level degeneration the per-word floor misses.

    The reported `primary_kw_at_coh` uses the per-word floor (legacy /
    backwards-compatible). The new `primary_kw_at_sonnet_coh` uses the
    Sonnet floor when grades are available, else falls back to per-word.

    Args:
      rows: a flat list of B1 result rows (each has source, magnitude,
            prompt_id, keyword_rate, text, ...)
      source_tags: only count rows whose source is in this set (the cell's
            mined features × modes)
      baseline_kw: subtracted from every kw_rate to get effect size
      coh_threshold: max consecutive same-word run; cells whose magnitudes
            fail this floor are excluded from the peak
      sonnet_grades: optional dict keyed by row index (str) → {"grade": int}.
            When provided, the Sonnet-floor metric is also computed.
      sonnet_floor: minimum Sonnet 0-3 grade required to count as coherent
            (default 2: "mostly coherent")

    Returns dict with primary metric + secondaries.
    """
    src_set = set(source_tags)
    by_cell_mag: dict = defaultdict(list)        # (source, mag) -> [(row_idx, row)]
    for idx, r in enumerate(rows):
        if r["source"] in src_set:
            by_cell_mag[(r["source"], r["magnitude"])].append((idx, r))

    peak_run = -float("inf")
    peak_sonnet = -float("inf")
    peak_no_coh = -float("inf")
    peak_run_mag = 0.0; peak_run_src = None
    peak_sonnet_mag = 0.0; peak_sonnet_src = None
    n_cells_total = 0
    n_cells_coh_run = 0
    n_cells_coh_sonnet = 0

    for (src, mag), cell_rows in by_cell_mag.items():
        n_cells_total += 1
        if not cell_rows:
            continue
        mean_kw = sum(r["keyword_rate"] for _, r in cell_rows) / len(cell_rows)
        run_ok = all(_coh_ok(r.get("text", ""), coh_threshold) for _, r in cell_rows)
        if run_ok:
            n_cells_coh_run += 1

        sonnet_ok: bool | None = None
        if sonnet_grades is not None:
            grades_for_cell = []
            for idx, r in cell_rows:
                g = sonnet_grades.get(str(idx))
                if g is None or g.get("grade", -1) < 0:
                    grades_for_cell.append(None)
                else:
                    grades_for_cell.append(int(g["grade"]))
            # Cell is Sonnet-coherent iff EVERY prompt has a valid grade ≥ floor.
            if all(g is not None and g >= sonnet_floor for g in grades_for_cell):
                sonnet_ok = True
                n_cells_coh_sonnet += 1
            else:
                sonnet_ok = False

        effect = abs(mean_kw - baseline_kw)
        if effect > peak_no_coh:
            peak_no_coh = effect
        if run_ok and effect > peak_run:
            peak_run = effect
            peak_run_mag = mag
            peak_run_src = src
        if sonnet_ok and effect > peak_sonnet:
            peak_sonnet = effect
            peak_sonnet_mag = mag
            peak_sonnet_src = src

    if peak_run == -float("inf"):
        peak_run = 0.0; peak_run_mag = 0.0; peak_run_src = None
    if peak_sonnet == -float("inf"):
        peak_sonnet = 0.0; peak_sonnet_mag = 0.0; peak_sonnet_src = None
    if peak_no_coh == -float("inf"):
        peak_no_coh = 0.0

    out = {
        "primary_kw_at_coh": float(peak_run),
        "peak_kw_no_coh": float(peak_no_coh),
        "best_magnitude": float(peak_run_mag),
        "best_source": peak_run_src,
        "direction": "positive" if peak_run_mag > 0 else ("negative" if peak_run_mag < 0 else "none"),
        "n_cells_evaluated": n_cells_total,
        "n_cells_coherent": n_cells_coh_run,
        "frac_coherent": (n_cells_coh_run / max(1, n_cells_total)),
    }
    if sonnet_grades is not None:
        out["primary_kw_at_sonnet_coh"] = float(peak_sonnet)
        out["best_magnitude_sonnet"] = float(peak_sonnet_mag)
        out["best_source_sonnet"] = peak_sonnet_src
        out["direction_sonnet"] = (
            "positive" if peak_sonnet_mag > 0
            else ("negative" if peak_sonnet_mag < 0 else "none")
        )
        out["n_cells_coh_sonnet"] = n_cells_coh_sonnet
        out["frac_coh_sonnet"] = (n_cells_coh_sonnet / max(1, n_cells_total))
    return out


def load_sonnet_grades(grades_path: Path | str) -> dict | None:
    p = Path(grades_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


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
