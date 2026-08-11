"""Gate 2, re-run under the published coherence filter.

The unfiltered gate judges every unsteered generation, including the ones that
degenerate into repeat loops. The paper's hill-climb metric never scores those:
metrics.cell_metric drops any cell whose generations fail
`metrics._coh_ok` (max consecutive same-word run <= 2) before taking the peak.
This applies that same filter to the gate's baseline rows and re-means
`genuine_count` over the survivors.

    uv run --no-sync python experiments/backtracking_steering_dsm/coh_check.py \
        --gates <gates.json> --out <coh_check.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.ward_backtracking_txc.metrics import _coh_ok, _max_repeat_run

BAND = (0.45, 0.85)
PUBLISHED_BASELINE = 0.6557377049180327      # 40 / 61, c7_headline seed42


def analyse(gates: dict, coh_threshold: int = 2) -> dict:
    rows = gates["rows"]
    scored = []
    for r in rows:
        txt = r.get("text", "")
        scored.append({
            "prompt_id": r["prompt_id"], "category": r.get("category"),
            "genuine_count": int(r.get("genuine_count", -1)),
            "max_repeat_run": _max_repeat_run(txt),
            "coh_ok": _coh_ok(txt, coh_threshold),
            "n_words": r.get("n_words"), "n_chars": r.get("n_chars"),
        })

    valid = [s for s in scored if s["genuine_count"] >= 0]
    surv = [s for s in valid if s["coh_ok"]]
    mean = lambda xs: (sum(xs) / len(xs)) if xs else float("nan")

    unfiltered = mean([s["genuine_count"] for s in valid])
    filtered = mean([s["genuine_count"] for s in surv])
    return {
        "coh_threshold": coh_threshold,
        "n_rows": len(rows),
        "n_valid_judgements": len(valid),
        "n_unparsed": len(scored) - len(valid),
        "n_coherent": len(surv),
        "n_dropped": len(valid) - len(surv),
        "gc_mean_unfiltered": unfiltered,
        "gc_mean_coherence_filtered": filtered,
        "per_prompt_event_rate_unfiltered":
            mean([1.0 if s["genuine_count"] >= 1 else 0.0 for s in valid]),
        "per_prompt_event_rate_filtered":
            mean([1.0 if s["genuine_count"] >= 1 else 0.0 for s in surv]),
        "published_baseline_61prompt": PUBLISHED_BASELINE,
        "band": list(BAND),
        "in_band_unfiltered": BAND[0] <= unfiltered <= BAND[1],
        "in_band_filtered": BAND[0] <= filtered <= BAND[1],
        "dropped_rows": [s for s in valid if not s["coh_ok"]],
        "rows": scored,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--gates", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--coh-threshold", type=int, default=2)
    a = p.parse_args(argv)

    res = analyse(json.loads(a.gates.read_text()), a.coh_threshold)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1))
    print(f"n_valid={res['n_valid_judgements']} coherent={res['n_coherent']} "
          f"dropped={res['n_dropped']}")
    print(f"gc unfiltered = {res['gc_mean_unfiltered']:.4f}  "
          f"(in band: {res['in_band_unfiltered']})")
    print(f"gc filtered   = {res['gc_mean_coherence_filtered']:.4f}  "
          f"(in band: {res['in_band_filtered']})")
    print(f"per-prompt event rate: unfiltered "
          f"{res['per_prompt_event_rate_unfiltered']:.4f}, filtered "
          f"{res['per_prompt_event_rate_filtered']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
