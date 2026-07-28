"""``evalage`` label-side PRE-MEASURE — the card § 6 bands, ALL THREE
TOKENIZER LEGS.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_evalage_premeasure

$0, CPU. Consumes the per-tokenizer grids built by
`evalage/screen_grids.py` (mac-d's transplant), which recover the corpus
text from the gpt2-ids stream under a hard round-trip receipt and
re-tokenize per model.

**History, stated rather than quietly overwritten.** The first version of
this script read the raw stream and reported gemma2/llama31 as `NOT RUN`
because the harness never persisted text (my defect; the fix is owed
separately). That gap is now closed by the grids, so the script loops.
The gpt2 leg is ASSERTED to reproduce the previously published band
values exactly — the rewrite is not allowed to move the number the LOG
already quotes.

The decisive band is **unigram**: `retryesc` died at 0.689-0.716 because
task vocabulary predicted the label. The corpus-level control passed
(cv 0.1346), but that is a property of the CORPUS; this measures whether
it holds at the FACE — and now, whether it holds under three tokenizers
rather than the one the corpus happened to be written in.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from . import msdose_r1_lib as mr1
from . import evalage_lib as ev
from . import elicit_lib as el
from . import wave3_lib as w3
from .build_wave3_trio import N_REPS, _floor_aucs, _spear, _terciles, _triage
from .gen4c_lib import section_age
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent
GRIDS = HERE.parent / "evalage" / "grids"
LEGS = ("gpt2", "gemma2", "llama31")
BANDS = {"unigram_le_0.60": 0.60, "doc_mean_le_0.88": 0.88,
         "position_le_0.95": 0.95}
QUAL_MIN, USABLE_MIN, EVENTS_MIN = 8, 250_000, 300

PUBLISHED = HERE / "evalage_premeasure.json"   # the gpt2-only artifact


def gpt2_published() -> dict:
    """The gpt2 band values this script published at `ad21f651d`, read
    from the committed artifact rather than transcribed — the 3-leg
    rewrite must reproduce them or it has changed the evidence rather
    than extended it. (Read, not hardcoded: my first cut hand-typed
    these and the mismatch was the typo, not the pipeline.)"""
    d = json.loads(PUBLISHED.read_text())
    return {k: float(v["value"]) for k, v in d["bands"].items()}


def leg_bands(tag: str) -> dict:
    z = np.load(GRIDS / f"elicit_evalage_screen_{tag}.npz")
    ids, doc_off = z["token_ids"], z["doc_off"].astype(np.int64)
    first, mask, elig_f = z["event_first"], z["event_mask"], z["is_assistant"]
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
    print(f"[evalage/{tag}] {n_docs} docs, {ids.size:,} tok, {n_ev} events, "
          f"{int(elig.sum()):,} eligible rows", flush=True)
    cz = el.claim_zone(raw, elig, FLOOR_TS)
    print("  claim zone:", {k: round(v*100, 2) for k, v in
                            cz["frac_in_window"].items()}, flush=True)
    bins, edges = _terciles(age, train_rows, elig)
    st = {"tokenizer_leg": tag, "n_docs": n_docs, "n_tokens": int(ids.size),
          "events": n_ev, "eligible_rows": int(elig.sum()),
          "tercile_edges": edges, "claim_zone": cz,
          "face_position_spearman": _spear(age, pos_of.astype(float), elig)}
    st.update(_triage(f"evalage_age[{tag}]", age, bins, ids, pos_of, doc_of,
                      doc_off, train_rows, test_rows, N_REPS))
    floors = {"censored_age": {T: np.concatenate(
        [w3.sage_floor(first[doc_off[d]:doc_off[d+1]], T)
         for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d+1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}}
    st["floors"] = _floor_aucs(f"evalage_age[{tag}]", bins, test_rows, floors)
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
    return st


def main():
    legs = {tag: leg_bands(tag) for tag in LEGS}

    got = {k: v["value"] for k, v in legs["gpt2"]["bands"].items()}
    for k, want in gpt2_published().items():
        assert abs(got[k] - want) < 1e-9, \
            f"gpt2 leg moved: {k} {got[k]} != published {want}"
    print("[evalage] gpt2 leg reproduces the published bands exactly "
          "(6/6) — the 3-leg rewrite extends the evidence, does not "
          "change it", flush=True)

    out = {"legs": legs,
           "grids_receipt": json.loads(
               (GRIDS / "grids_receipt.json").read_text()),
           "all_legs_pass": all(v["all_pass"] for v in legs.values()),
           "worst_by_band": {
               k: {"value": max(v["bands"][k]["value"] for v in legs.values())
                   if k in ("unigram_le_0.60", "doc_mean_le_0.88",
                            "position_le_0.95")
                   else min(v["bands"][k]["value"] for v in legs.values()),
                   "leg": None}
               for k in legs["gpt2"]["bands"]}}
    for k, rec in out["worst_by_band"].items():
        rec["leg"] = next(t for t, v in legs.items()
                          if v["bands"][k]["value"] == rec["value"])
        rec["pass"] = legs[rec["leg"]]["bands"][k]["pass"]
    p = HERE / "evalage_premeasure_3leg.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p} | all_legs_pass={out['all_legs_pass']}", flush=True)
    print("  worst leg per band: " + ", ".join(
        f"{k}={r['value']:.4f}({r['leg']},{'PASS' if r['pass'] else 'FAIL'})"
        for k, r in out["worst_by_band"].items()), flush=True)


if __name__ == "__main__":
    main()
