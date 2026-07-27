"""$0 pre-measure for the ``reask_hr`` PRIMARY variant (gate census
~19:05 entry: apology-only events 57.5 % ⇒ hard-refusal-gated
variant pre-registers primary). NO new constants: the event is a
committed-logic conjunction — ``wave3_lib.reask_events`` ∧ the
deflected assistant turn fires ≥ 1 ``census_reask_gate.
HARD_REFUSAL`` substring. Own artifact (the ratified
``wave3_trio_stats.json`` record is not touched).

Run: .venv/bin/python -m experiments.explorations.task_hunt.labels.build_reask_hr
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import pull_refmark2k as pull
from . import wave3_lib as w3
from .build_wave3_trio import (MIN_POS, _floor_aucs, _spear, _terciles,
                               _triage)
from .census_reask_gate import gate_hits
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent
TOKS = ("gpt2", "gemma2", "llama31")
N_REPS = 500


def reask_hr_events(msgs) -> np.ndarray:
    ev = w3.reask_events(msgs)
    for i in np.flatnonzero(ev):
        hard, _ = gate_hits(msgs[i - 1][1])
        if not hard:
            ev[i] = 0
    return ev


def main():
    convs, _ = pull.load()
    counts = np.array([int(reask_hr_events(m).sum()) for m in convs])
    stats: dict = {
        "event_census": {
            "events_total": int(counts.sum()),
            "frac_convs_ge1": float((counts >= 1).mean()),
            "frac_convs_ge2": float((counts >= 2).mean()),
            "events_per_conv_max": int(counts.max()),
        },
        "per_tokenizer": {},
    }
    print(json.dumps(stats["event_census"], indent=1), flush=True)

    for key in TOKS:
        z = np.load(HERE / f"refmark2k_wildchat_{key}.npz")
        zp = np.load(HERE / f"wave3_refmark2k_{key}.npz")
        ids, doc_off = z["token_ids"], z["doc_off"]
        turn_idx, is_assist = z["turn_idx"], z["is_assistant"]
        boundary, rlam, split = z["is_boundary"], z["rlam"], z["doc_split"]
        n_docs = len(doc_off) - 1
        doc_of = np.repeat(np.arange(n_docs, dtype=np.int32),
                           np.diff(doc_off))
        pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                                 for n in np.diff(doc_off)])
        train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

        first = np.zeros(len(ids), np.int8)
        mask = np.zeros(len(ids), np.int8)
        for d in range(n_docs):
            lo, hi = doc_off[d], doc_off[d + 1]
            ev = reask_hr_events(convs[d])
            first[lo:hi] = w3.event_first_token_flags(turn_idx[lo:hi], ev)
            mask[lo:hi] = w3.event_token_flags(turn_idx[lo:hi], ev)
        age = np.concatenate(
            [w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
             for d in range(n_docs)])
        elig = ((mask == 0) & (boundary == 0) & (is_assist == 1)
                & (pos_of >= MIN_POS))
        bins, edges = _terciles(age, train_rows, elig)
        st = {"eligible_rows": int((elig & np.isfinite(age)).sum()),
              "tercile_edges": edges}
        print(f"[{key}] reask_hr: {st['eligible_rows']:,} eligible rows",
              flush=True)
        st.update(_triage("reask_hr_age", age, bins, ids, pos_of, doc_of,
                          doc_off, train_rows, test_rows, N_REPS))
        cage = {T: np.concatenate(
            [w3.sage_floor(first[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}
        cnt = {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}
        st["floors"] = _floor_aucs("reask_hr_age", bins, test_rows,
                                   {"censored_age": cage,
                                    "in_window_event_tokens": cnt})
        elig_any = (boundary == 0) & (is_assist == 1) & (pos_of >= MIN_POS)
        st["anti_dup_spearman"] = {
            "hr_age_vs_pooled_reask_age": _spear(age, zp["reask_age"],
                                                 elig_any),
            "hr_age_vs_refmark_rlam": _spear(age, rlam, elig_any),
        }
        for k2, v in st["anti_dup_spearman"].items():
            print(f"[{key}] anti-dup {k2}: rho={v['rho']:.3f} "
                  f"(n={v['n_rows']:,})", flush=True)
        np.savez_compressed(HERE / f"wave3_reask_hr_{key}.npz",
                            reask_hr_age=age, reask_hr_event_first=first,
                            reask_hr_event_mask=mask)
        st["artifact"] = f"wave3_reask_hr_{key}.npz"
        stats["per_tokenizer"][key] = st

    p = HERE / "reask_hr_premeasure.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
