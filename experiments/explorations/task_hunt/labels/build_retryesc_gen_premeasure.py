"""``retryesc_gen`` label-side PRE-MEASURE — GENERATION_CARD § 5 bands.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_retryesc_gen_premeasure [--tag retryesc_gen_pilot]

$0, CPU, seconds. Reads the stream `run_elicit.py --scaffold
retryesc_gen` writes and scores the seven pre-registered bands. Shares
every helper with the `evalage` premeasure -- same `_triage`, same
`_floor_aucs`, same `strata_census` -- so the numbers are comparable
across candidates by construction rather than by claim.

THE DECISIVE BAND IS `unigram_auc <= 0.60`. The organic `retryesc` died
there at 0.689-0.716 because task difficulty GENUINELY drove failure
rate, so ordinary task nouns predicted event age, and masking could not
fix it. This scaffold designs that out structurally (schedule drawn
before the task, global strategy pool), and the plan-time check already
passes at cv 0.0567 -- but that is a property of the SCHEDULE. This
script is the first test of whether it holds in the model's actual
PROSE, which is the one thing the $0 dry run could not reach.

BAND 4 IS TWO-SIDED. `floor_excess` must land in [+0.15, +0.25]:
too sparse and the window has nothing to beat the anchor token with;
too dense and the floor -- computed from ground truth -- outruns any
activation-based arm (3 of 5 record cells above +0.25 lose to their own
floor). `claim_zone.frac_in_window["T64"]` IS that quantity (K = 0.96
on real data, card § 2.2a), so it is measured here, not projected.

PILOT vs FULL. The pilot runs the gpt2 leg only -- enough to decide
whether to buy the full corpus. The card's "all three tokenizers" rule
binds the FULL run, which needs the per-tokenizer grids
(`evalage/screen_grids.py` pattern). `--legs` is wired for that.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import elicit_lib as el
from . import msdose_r1_lib as mr1
from . import retryesc_gen_lib as rg
from . import wave3_lib as w3
from .build_wave3_trio import N_REPS, _floor_aucs, _spear, _terciles, _triage
from .gen4c_lib import section_age
from .hunt3_lib import FLOOR_TS

HERE = Path(__file__).resolve().parent
OUT_DIR = HERE.parent / "retryesc_gen" / "results"

# Card § 5, absolute only (the `msdose_r1` lesson). Bands 1-3 and 5-7
# are inherited unchanged from the organic `retryesc` card so the
# rebuild is judged by the same ruler that killed it.
UNIGRAM_MAX, DOC_MEAN_MAX, POSITION_MAX = 0.60, 0.88, 0.95
QUAL_MIN, USABLE_MIN, EVENTS_MIN = 8, 250_000, 300
MIN_POS = 64          # OFF_MIN + 1; a T=64 window needs anchor >= T-1

# Pilot-scale relief on MASS bands only. Rationale, stated so it cannot
# be mistaken for moving a bar: a ~20-doc pilot cannot physically carry
# 250k usable tokens or 300 events, so scoring it against the full-run
# mass bars would fail it for being small rather than for being wrong.
# The DISCRIMINATING bands (unigram, doc-mean, position, floor_excess)
# are NOT relaxed -- they are scale-free and bind identically here.
PILOT_QUAL_MIN, PILOT_USABLE_MIN, PILOT_EVENTS_MIN = 4, 20_000, 60


GRIDS = HERE.parent / "retryesc_gen" / "grids"


def _stream_path(tag: str) -> Path:
    """Per-tokenizer grids first (the 3-leg rule), else the raw stream."""
    g = GRIDS / f"elicit_retryesc_gen_v1_screen_{tag}.npz"
    return g if g.exists() else HERE / f"elicit_{tag}.npz"


def leg_bands(tag: str, pilot: bool) -> dict:
    z = np.load(_stream_path(tag))
    ids, doc_off = z["token_ids"], z["doc_off"].astype(np.int64)
    first, mask = z["event_first"], z["event_mask"]
    elig_f = z["probe_eligible"]
    split = z["doc_split"]
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    # PRIMARY face (card § 3.1): log2(1 + tokens since last
    # repeat-failure). Same shape as sage_face, so the same helper --
    # the face family is `sycgen_age`'s, which is the program's gold.
    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    raw = np.concatenate([section_age(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    elig = ((mask == 0) & (elig_f == 1) & (pos_of >= MIN_POS)
            & np.isfinite(age))
    n_ev = int(first.sum())
    print(f"[retryesc_gen/{tag}] {n_docs} docs, {ids.size:,} tok, "
          f"{n_ev} events, {int(elig.sum()):,} eligible rows", flush=True)

    cz = el.claim_zone(raw, elig, FLOOR_TS)
    f64 = float(cz["frac_in_window"]["T64"])
    print("  claim zone: "
          + str({k: round(v * 100, 2) for k, v in
                 cz["frac_in_window"].items()}), flush=True)

    bins, edges = _terciles(age, train_rows, elig)
    st = {"tokenizer_leg": tag.split("_")[-1], "n_docs": n_docs,
          "n_tokens": int(ids.size), "events": n_ev,
          "eligible_rows": int(elig.sum()), "tercile_edges": edges,
          "claim_zone": cz, "floor_excess_T64": f64,
          "face_position_spearman": _spear(age, pos_of.astype(float), elig)}
    st.update(_triage(f"retryesc_age[{tag}]", age, bins, ids, pos_of, doc_of,
                      doc_off, train_rows, test_rows, N_REPS))

    floors = {
        "censored_age": {T: np.concatenate(
            [w3.sage_floor(first[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS}}
    st["floors"] = _floor_aucs(f"retryesc_age[{tag}]", bins, test_rows, floors)
    st["strata_census"] = mr1.strata_census(age, pos_of, elig)
    c = st["strata_census"]
    print(f"  census: {c['n_qualifying']}/{c['n_strata_any']} strata, "
          f"{c['usable_tokens']:,} usable", flush=True)

    q, u, e = ((PILOT_QUAL_MIN, PILOT_USABLE_MIN, PILOT_EVENTS_MIN) if pilot
               else (QUAL_MIN, USABLE_MIN, EVENTS_MIN))
    lo, hi = rg.FLOOR_EXCESS_BAND
    b = {
        # discriminating, scale-free — NEVER relaxed for the pilot
        "unigram_le_0.60": (st["unigram_auc"],
                            st["unigram_auc"] <= UNIGRAM_MAX),
        "doc_mean_le_0.88": (st["doc_mean_only_auc"],
                             st["doc_mean_only_auc"] <= DOC_MEAN_MAX),
        "position_le_0.95": (st["position_auc"],
                             st["position_auc"] <= POSITION_MAX),
        "floor_excess_in_band": (f64, lo <= f64 <= hi),
        # mass bands — pilot-scaled, and labelled as such in the output
        f"qualifying_ge_{q}": (c["n_qualifying"], c["n_qualifying"] >= q),
        f"usable_ge_{u}": (c["usable_tokens"], c["usable_tokens"] >= u),
        f"events_ge_{e}": (n_ev, n_ev >= e),
    }
    st["bands"] = {k: {"value": float(v), "pass": bool(p)}
                   for k, (v, p) in b.items()}
    st["mass_bands_pilot_scaled"] = bool(pilot)
    st["all_pass"] = all(x["pass"] for x in st["bands"].values())
    print("  bands: " + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                                  for k, v in st["bands"].items()), flush=True)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="retryesc_gen_pilot")
    ap.add_argument("--legs", nargs="*", default=None,
                    help="full-run 3-leg mode; needs per-tokenizer grids")
    ap.add_argument("--full", action="store_true",
                    help="score against FULL-run mass bands, not pilot ones")
    a = ap.parse_args()

    legs = a.legs or [a.tag]
    out = {"card": "retryesc_gen/GENERATION_CARD.md § 5",
           "pilot": not a.full, "legs": {}}
    for tag in legs:
        out["legs"][tag] = leg_bands(tag, pilot=not a.full)

    worst = {}
    for k in out["legs"][legs[0]]["bands"]:
        vals = [out["legs"][t]["bands"][k] for t in legs]
        worst[k] = {"pass_all_legs": all(v["pass"] for v in vals),
                    "values": [v["value"] for v in vals]}
    out["worst_per_band"] = worst
    out["all_pass"] = all(v["pass_all_legs"] for v in worst.values())

    OUT_DIR.mkdir(exist_ok=True)
    # Name by TAG, not by mode. A dry-run smoke test must never land in a
    # file called `pilot_premeasure.json` — stub prose makes its
    # vocabulary bands meaningless, and a reader should not have to know
    # that to avoid quoting them.
    stub = "dryrun" in legs[0]
    out["EVIDENCE"] = ("STUB PROSE — plumbing validation only. Vocabulary "
                       "bands are MEANINGLESS here; every assistant turn "
                       "is near-identical filler." if stub else
                       "generated prose")
    p = OUT_DIR / f"premeasure_{legs[0]}.json"
    p.write_text(json.dumps(out, indent=1))
    if stub:
        print("\n⚠ STUB PROSE — PLUMBING CHECK ONLY, NOT A VERDICT.\n"
              "  Every assistant turn is near-identical filler, so "
              "unigram ~0.5 is an artefact of the stub, not evidence of\n"
              "  vocabulary cleanliness. Only the STRUCTURAL bands "
              "(floor_excess, position, strata) mean anything here.")
    else:
        print(f"\n{'ALL BANDS PASS' if out['all_pass'] else 'BAND FAILURE'}"
              f" — {'buy the full corpus' if out['all_pass'] else 'NO-GO'}")
    if not a.full:
        print("Mass bands were PILOT-SCALED (a 20-doc pilot cannot carry "
              "250k tokens); unigram / doc-mean / position / floor_excess "
              "were NOT relaxed.")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
