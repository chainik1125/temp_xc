"""Aggregate a steering wave into per-magnitude curves and a per-source table.

    uv run --no-sync python experiments/backtracking_steering_dsm/analyse.py \
        --wave-dir <dir with rows__*.json + judged__*.json> --out-dir <dir>

Metrics, per the rulings for this study:
  - `gc` = MEAN GENUINE EVENTS PER GENERATION. This is the published quantity
    (c7_headline gc_at_baseline 0.6557377 x 61 = 40 exactly), not a per-prompt
    share.
  - `event_rate` = mean(genuine_count >= 1), the per-prompt share, reported
    alongside because the two can move in opposite directions.
  - delta_gc(alpha) = gc(alpha) - gc(alpha = 0) for the SAME source, so every
    comparison is internal to this run.

Coherence floors, both from the reference:
  - run floor: metrics._coh_ok, max consecutive same-word run <= 2.
  - sonnet floor: grade_sonnet 0-3 grade >= 2.
A magnitude cell counts as coherent only if EVERY prompt at that magnitude
passes, which is metrics.cell_metric's own cell-level rule.

The bootstrap is the prompt-resampling scheme of metrics._bootstrap_ci
(percentile, resample prompt ids with replacement), applied to genuine_count
instead of keyword_rate.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ward_backtracking_txc.metrics import _coh_ok, _max_repeat_run

SONNET_FLOOR = 2


def _mean(xs):
    return (sum(xs) / len(xs)) if xs else float("nan")


def load_wave(wave_dir: Path) -> dict[str, list[dict]]:
    """-> {tag: [row, ...]} with judge fields merged onto each row."""
    out = {}
    for rp in sorted(wave_dir.glob("rows__*.json")):
        tag = rp.name[len("rows__"):-len(".json")]
        obj = json.loads(rp.read_text())
        rows = obj["rows"]
        jp = wave_dir / f"judged__{tag}.json"
        if not jp.exists():
            print(f"[warn] no judgements for {tag}, skipping", file=sys.stderr)
            continue
        judged = json.loads(jp.read_text())
        for i, r in enumerate(rows):
            j = judged.get(str(i), {})
            r["genuine_count"] = int(j.get("genuine_count", -1))
            r["coh_grade"] = int(j.get("grade", -1))
            r["max_repeat_run"] = _max_repeat_run(r.get("text", ""))
            r["coh_ok"] = _coh_ok(r.get("text", ""))
            r["sonnet_ok"] = r["coh_grade"] >= SONNET_FLOOR
        out[tag] = {"rows": rows, "meta": obj["meta"]}
    return out


def curve(rows: list[dict]) -> list[dict]:
    """Per-magnitude aggregate over the eval prompts."""
    by_mag: dict[float, list[dict]] = {}
    for r in rows:
        by_mag.setdefault(r["magnitude"], []).append(r)
    out = []
    for mag in sorted(by_mag):
        cell = by_mag[mag]
        valid = [r for r in cell if r["genuine_count"] >= 0]
        counts = [r["genuine_count"] for r in valid]
        out.append({
            "magnitude": mag,
            "n": len(cell), "n_valid": len(valid),
            "gc": _mean(counts),
            "event_rate": _mean([1.0 if c >= 1 else 0.0 for c in counts]),
            "kw_rate": _mean([r["keyword_rate"] for r in cell]),
            "frac_coh_run": _mean([1.0 if r["coh_ok"] else 0.0 for r in cell]),
            "frac_coh_sonnet": _mean([1.0 if r["sonnet_ok"] else 0.0 for r in cell]),
            "mean_sonnet_grade": _mean([r["coh_grade"] for r in cell
                                        if r["coh_grade"] >= 0]),
            "cell_coh_run": all(r["coh_ok"] for r in cell),
            "cell_coh_sonnet": all(r["sonnet_ok"] for r in cell),
            "mean_words": _mean([r["n_words"] for r in cell]),
            "mean_tokens": _mean([r.get("n_tokens_retok", 0) for r in cell]),
            "max_repeat_run_max": max(r["max_repeat_run"] for r in cell),
        })
    return out


def _boot_delta(rows: list[dict], mag: float, n_resamples: int = 2000,
                seed: int = 42, ci: float = 0.95) -> tuple[float, float]:
    """Percentile CI on gc(mag) - gc(0), resampling prompt ids (metrics
    ::_bootstrap_ci scheme) so the pairing across magnitudes is preserved."""
    by_pid_mag: dict[tuple[str, float], int] = {}
    for r in rows:
        if r["genuine_count"] >= 0:
            by_pid_mag[(r["prompt_id"], r["magnitude"])] = r["genuine_count"]
    pids = sorted({p for p, _ in by_pid_mag})
    pids = [p for p in pids if (p, mag) in by_pid_mag and (p, 0.0) in by_pid_mag]
    if not pids:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    vals = []
    for _ in range(n_resamples):
        samp = [pids[rng.randrange(len(pids))] for _ in pids]
        vals.append(_mean([by_pid_mag[(p, mag)] for p in samp])
                    - _mean([by_pid_mag[(p, 0.0)] for p in samp]))
    vals.sort()
    lo = vals[int((1 - ci) / 2 * n_resamples)]
    hi = vals[int((1 + ci) / 2 * n_resamples) - 1]
    return float(lo), float(hi)


def summarise(tag: str, rows: list[dict], meta: dict) -> dict:
    c = curve(rows)
    base = next((x for x in c if x["magnitude"] == 0.0), None)
    gc0 = base["gc"] if base else float("nan")

    def peak(filt):
        cand = [x for x in c if x["magnitude"] != 0.0 and filt(x)]
        if not cand:
            return None
        return max(cand, key=lambda x: abs(x["gc"] - gc0))

    p_any = peak(lambda x: True)
    p_run = peak(lambda x: x["cell_coh_run"])
    p_son = peak(lambda x: x["cell_coh_sonnet"])
    lo, hi = _boot_delta(rows, p_run["magnitude"]) if p_run else (float("nan"),) * 2
    worst = min((x for x in c if x["magnitude"] != 0.0),
                key=lambda x: x["gc"], default=None)
    # Peak-across-grid is the published metric but it is a max over 24 noisy
    # cells, so it rewards variance. The mean |delta| over the coherent
    # magnitudes is reported next to it as the selection-free companion.
    coh_nonzero = [x for x in c if x["magnitude"] != 0.0 and x["cell_coh_run"]]
    mean_abs = _mean([abs(x["gc"] - gc0) for x in coh_nonzero])

    return {
        "source": tag, "arm": meta["arm"], "arch": meta["arch"],
        "hook": meta["hook"], "mode": meta["mode"],
        "feature_id": meta["feature_id"],
        "gc_baseline": gc0,
        "event_rate_baseline": base["event_rate"] if base else float("nan"),
        "delta_gc_peak_any": (p_any["gc"] - gc0) if p_any else float("nan"),
        "peak_any_magnitude": p_any["magnitude"] if p_any else None,
        "delta_gc_peak_at_coh_run": (p_run["gc"] - gc0) if p_run else float("nan"),
        "peak_run_magnitude": p_run["magnitude"] if p_run else None,
        "peak_run_ci95": [lo, hi],
        "mean_abs_delta_gc_over_coh": mean_abs,
        "delta_gc_peak_at_coh_sonnet": (p_son["gc"] - gc0) if p_son else float("nan"),
        "peak_sonnet_magnitude": p_son["magnitude"] if p_son else None,
        "gc_min": worst["gc"] if worst else float("nan"),
        "gc_min_magnitude": worst["magnitude"] if worst else None,
        "delta_event_rate_at_peak_run":
            (p_run["event_rate"] - base["event_rate"]) if (p_run and base)
            else float("nan"),
        "n_coherent_cells_run": sum(1 for x in c if x["cell_coh_run"]),
        "n_coherent_cells_sonnet": sum(1 for x in c if x["cell_coh_sonnet"]),
        "n_cells": len(c),
        "mean_words_at_baseline": base["mean_words"] if base else float("nan"),
        "mean_words_at_peak_run": p_run["mean_words"] if p_run else float("nan"),
        "steered_vector_norm": meta.get("steered_vector_norm"),
        "curve": c,
    }


def _f(x, nd: int = 2, sign: bool = False) -> str:
    """Numbers that do not exist print as an em dash, not as `nan`.

    An arm with no coherent magnitude cell is a reportable outcome -- the
    denoise-after-steer arm flattening its curve is one of the things this study
    is looking for -- so that row must read as "no coherent cell" rather than as
    a formatting accident.
    """
    if x is None or x != x:
        return "—"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _m(x) -> str:
    return "—" if x is None else f"{x:g}"


def _ci(pair) -> str:
    lo, hi = pair
    if lo is None or lo != lo or hi is None or hi != hi:
        return "—"
    return f"[{lo:+.2f}, {hi:+.2f}]"


def table_md(summaries: list[dict]) -> str:
    head = ("| source | arm | mode | mining score (t) | gc base | "
            "Δgc peak (coh, run) | at α | "
            "95% CI | mean abs Δgc | Δgc peak (coh, Sonnet) | at α | "
            "Δgc peak (no floor) | at α | gc min | event-rate base → peak | "
            "coherent cells (run/Sonnet) |\n")
    head += "|" + "---|" * 16 + "\n"
    body = ""
    # nan sorts unpredictably, so arms with no coherent cell are pinned last
    # rather than landing at an arbitrary rank.
    def _rank(x):
        d = x["delta_gc_peak_at_coh_run"]
        return float("inf") if (d is None or d != d) else -abs(d)

    for s in sorted(summaries, key=_rank):
        msc = ("n/a" if s.get("mining_score") is None else
               f"{s['mining_score']:+.3f} ({s['mining_tstat']:.1f})")
        body += (
            f"| `{s['source']}` | {s['arm']} | {s['mode']} | {msc} | "
            f"{_f(s['gc_baseline'])} | "
            f"{_f(s['delta_gc_peak_at_coh_run'], 3, sign=True)} | "
            f"{_m(s['peak_run_magnitude'])} | "
            f"{_ci(s['peak_run_ci95'])} | "
            f"{_f(s['mean_abs_delta_gc_over_coh'], 3)} | "
            f"{_f(s['delta_gc_peak_at_coh_sonnet'], 3, sign=True)} | "
            f"{_m(s['peak_sonnet_magnitude'])} | "
            f"{_f(s['delta_gc_peak_any'], 3, sign=True)} | "
            f"{_m(s['peak_any_magnitude'])} | "
            f"{_f(s['gc_min'])} | "
            f"{_f(s['event_rate_baseline'])} → "
            f"{_f(s['event_rate_baseline'] + s['delta_event_rate_at_peak_run'])} | "
            f"{s['n_coherent_cells_run']}/{s['n_coherent_cells_sonnet']} of "
            f"{s['n_cells']} |\n")
    return head + body


def load_selectivity(summary_path: Path | None) -> dict[str, dict]:
    """-> {arm: {"score": float, "tstat": float}} from a mining_summary.json.

    Carried into the results table because the arms do not enter the sweep on
    equal footing: the mined meandiff scores span ~0.18-0.81, so an arm whose
    steering vector moves gc less may simply have had a weaker backtracking
    feature to steer with, independent of the dictionary's temporal structure.
    """
    if not summary_path or not summary_path.exists():
        return {}
    obj = json.loads(summary_path.read_text())
    return {a["arm"]: {"score": a["top_score"], "tstat": a["top_tstat"]}
            for a in obj.get("arms", [])}


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--wave-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--mining-summary", type=Path, default=None,
                   help="mining_summary.json, for the selectivity columns")
    a = p.parse_args(argv)

    data = load_wave(a.wave_dir)
    sel = load_selectivity(a.mining_summary)
    summaries = [summarise(tag, d["rows"], d["meta"]) for tag, d in data.items()]
    for s in summaries:
        s["mining_score"] = sel.get(s["arm"], {}).get("score")
        s["mining_tstat"] = sel.get(s["arm"], {}).get("tstat")
    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "summary.json").write_text(json.dumps(summaries, indent=1))
    (a.out_dir / "table.md").write_text(table_md(summaries))
    (a.out_dir / "curves.json").write_text(json.dumps(
        {s["source"]: s["curve"] for s in summaries}, indent=1))
    print(table_md(summaries))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
