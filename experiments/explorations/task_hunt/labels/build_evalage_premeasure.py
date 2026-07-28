"""``evalage`` label-side PRE-MEASURE — the card § 6 bands.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_evalage_premeasure

$0, CPU. Consumes the generated stream (gpt2 ids) and re-tokenizes
nothing: the harness wrote token_ids with the gpt2 tokenizer, so the
gpt2 leg is exact and the other two legs are reported as NOT RUN rather
than faked. Instruments are the trio's, imported unchanged.

The decisive band here is **unigram**: `retryesc` died at 0.689-0.716
because task vocabulary predicted the label. The corpus-level control
passed (cv 0.1346), but that is a property of the CORPUS; this measures
whether it holds at the FACE.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import msdose_r1_lib as mr1
from . import evalage_lib as ev
from . import elicit_lib as el
from . import wave3_lib as w3
from .build_wave3_trio import N_REPS, SEED, _floor_aucs, _spear, _terciles, _triage
from .gen4c_lib import section_age
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent
BANDS = {"unigram_le_0.60": 0.60, "doc_mean_le_0.88": 0.88,
         "position_le_0.95": 0.95}
QUAL_MIN, USABLE_MIN, EVENTS_MIN = 8, 250_000, 300


def main():
    z = np.load(HERE / "elicit_evalage_v1.npz")
    ids, doc_off = z["token_ids"], z["doc_off"]
    first, mask, elig_f = z["event_first"], z["event_mask"], z["probe_eligible"]
    split = z["doc_split"]
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1
    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d+1]])
                          for d in range(n_docs)])
    raw = np.concatenate([section_age(first[doc_off[d]:doc_off[d+1]])
                          for d in range(n_docs)])
    elig = (mask == 0) & (elig_f == 1) & (pos_of >= ev.MIN_POS) & np.isfinite(age)
    n_ev = int(first.sum())
    print(f"[evalage] {n_docs} docs, {ids.size:,} tok, {n_ev} events, "
          f"{int(elig.sum()):,} eligible rows", flush=True)
    cz = el.claim_zone(raw, elig, FLOOR_TS)
    print("  claim zone:", {k: round(v*100, 2) for k, v in
                            cz["frac_in_window"].items()}, flush=True)
    bins, edges = _terciles(age, train_rows, elig)
    st = {"n_docs": n_docs, "n_tokens": int(ids.size), "events": n_ev,
          "eligible_rows": int(elig.sum()), "tercile_edges": edges,
          "claim_zone": cz,
          "face_position_spearman": _spear(age, pos_of.astype(float), elig)}
    st.update(_triage("evalage_age", age, bins, ids, pos_of, doc_of, doc_off,
                      train_rows, test_rows, N_REPS))
    floors = {"censored_age": {T: np.concatenate(
        [w3.sage_floor(first[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d+1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}}
    st["floors"] = _floor_aucs("evalage_age", bins, test_rows, floors)
    st["strata_census"] = mr1.strata_census(age, pos_of, elig)
    c = st["strata_census"]
    print(f"  census: {c['n_qualifying']}/{c['n_strata_any']} strata, "
          f"{c['usable_tokens']:,} usable", flush=True)
    b = {"unigram_le_0.60": (st["unigram_auc"], st["unigram_auc"] <= 0.60),
         "doc_mean_le_0.88": (st["doc_mean_only_auc"],
                              st["doc_mean_only_auc"] <= 0.88),
         "position_le_0.95": (st["position_auc"], st["position_auc"] <= 0.95),
         "qualifying_ge_8": (c["n_qualifying"], c["n_qualifying"] >= QUAL_MIN),
         "usable_ge_250k": (c["usable_tokens"], c["usable_tokens"] >= USABLE_MIN),
         "events_ge_300": (n_ev, n_ev >= EVENTS_MIN)}
    st["bands"] = {k: {"value": float(v), "pass": bool(p)}
                   for k, (v, p) in b.items()}
    st["all_pass"] = all(x["pass"] for x in st["bands"].values())
    print("  bands: " + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                                  for k, v in st["bands"].items()), flush=True)
    st["tokenizer_legs"] = {"gpt2": "exact (stream written with gpt2 ids)",
                            "gemma2": "NOT RUN — needs re-tokenized stream",
                            "llama31": "NOT RUN — needs re-tokenized stream"}
    p = HERE / "evalage_premeasure.json"
    p.write_text(json.dumps(st, indent=1))
    print(f"-> {p} | all_pass={st['all_pass']}", flush=True)


if __name__ == "__main__":
    main()
