"""Realised-geometry gate for a GENERATED `sycgen` corpus (generation
card `sycgen/GENERATION_CARD.md` — mac-d executor).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.sycgen_realised_gate \
        --stream <labels/elicit_sycgen_v1.npz>

PRECOUNT_CARD § 2 requires it: the plan's WildChat length priors were a
PLANNING prior, so the § 4 bands must be re-measured on the realised
token layout before any screen. Age face only — `sycgen_rate` is
DEMOTED (§ 7.1 reach kill) and is not resurrected here. $0, CPU.

Instruments and band constants are IMPORTED from the frozen premeasure
builder (`build_sycgen_premeasure`), never re-typed; this file adds no
new thresholds. It can KILL; it cannot clear — the per-token baseline
at the screen stage remains the binding next gate (PRECOUNT § 5).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import msdose_r1_lib as mr1
from . import novelty_lib as nl  # noqa: F401  (via _triage_geom)
from . import wave3_lib as w3
from .build_sycgen_premeasure import (BAND_DOCMEAN_MAX, BAND_EV_PER_CONV_MIN,
                                      BAND_EVENTS_MIN, BAND_POSITION_MAX,
                                      BAND_QUAL_MIN, BAND_USABLE_MIN,
                                      _triage_geom)
from .build_wave3_trio import MIN_POS, _floor_aucs, _spear, _terciles
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    z = np.load(a.stream)
    doc_off = z["doc_off"].astype(np.int64)
    first = z["event_first"].astype(np.int8)
    mask = z["event_mask"].astype(np.int8)
    elig_base = z["probe_eligible"].astype(bool)
    split = z["doc_split"]
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    train_rows = split[doc_of] == 0
    test_rows = split[doc_of] == 1
    elig = elig_base & (mask == 0) & (pos_of >= MIN_POS)

    n_ev = [int(first[doc_off[d]:doc_off[d + 1]].sum())
            for d in range(n_docs)]
    census = {
        "n_convs": n_docs, "n_tokens": int(first.size),
        "events_total": int(sum(n_ev)),
        "events_per_conv_mean": float(np.mean(n_ev)),
        "events_per_conv_min": int(min(n_ev)),
        "events_per_conv_max": int(max(n_ev)),
        "eligible_tokens": int(elig.sum()),
    }
    print(f"[realised] {census['events_total']} events, "
          f"{census['events_per_conv_mean']:.2f}/conv, "
          f"{census['n_tokens']:,} tokens", flush=True)

    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    floors = {
        "censored_age": {T: np.concatenate(
            [w3.sage_floor(first[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS},
    }
    bins, edges = _terciles(age, train_rows, elig)
    st = {"eligible_rows": int((elig & np.isfinite(age)).sum()),
          "tercile_edges": edges,
          "face_position_spearman": _spear(age, pos_of.astype(float), elig)}
    st.update(_triage_geom("sycgen_age", age, bins, pos_of, doc_of,
                           doc_off, test_rows))
    st["floors"] = _floor_aucs("sycgen_age", bins, test_rows, floors)
    st["strata_census"] = mr1.strata_census(age, pos_of, elig)
    c = st["strata_census"]

    # in-window fraction table (PRECOUNT § 7.1's clock exhibit, realised)
    raw = age[elig & np.isfinite(age)]
    st["in_window_frac"] = {int(T): float((raw < T).mean())
                            for T in FLOOR_TS}

    bands = {
        "doc_mean_le_0.88": {"value": st["doc_mean_only_auc"],
                             "pass": st["doc_mean_only_auc"]
                             <= BAND_DOCMEAN_MAX},
        "position_le_0.95": {"value": st["position_auc"],
                             "pass": st["position_auc"]
                             <= BAND_POSITION_MAX},
        "qualifying_ge_8": {"value": c["n_qualifying"],
                            "pass": c["n_qualifying"] >= BAND_QUAL_MIN},
        "usable_ge_250k": {"value": c["usable_tokens"],
                           "pass": c["usable_tokens"] >= BAND_USABLE_MIN},
        "ev_per_conv_ge_1.5": {"value": census["events_per_conv_mean"],
                               "pass": census["events_per_conv_mean"]
                               >= BAND_EV_PER_CONV_MIN},
        "events_ge_300": {"value": census["events_total"],
                          "pass": census["events_total"]
                          >= BAND_EVENTS_MIN},
    }
    bands = {k: {"value": float(v["value"]), "pass": bool(v["pass"])}
             for k, v in bands.items()}
    all_pass = all(v["pass"] for v in bands.values())
    print("  bands: " + ", ".join(
        f"{k}={'PASS' if v['pass'] else 'FAIL'}" for k, v in bands.items()),
        flush=True)
    print(f"  REALISED GEOMETRY: {'ALL PASS' if all_pass else 'KILL'} "
          "(clears nothing — per-token baseline at screen is next)",
          flush=True)

    out = {"stream": a.stream, "census": census, "sycgen_age": st,
           "bands": bands, "all_pass": bool(all_pass),
           "note": ("age face only: sycgen_rate DEMOTED per "
                    "PRECOUNT_CARD §7.1, not resurrected")}
    out_path = Path(a.out) if a.out else HERE / "sycgen_realised_gate.json"
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
