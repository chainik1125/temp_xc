"""``sycgen`` GEOMETRY pre-measure (card
``sycgen/PRECOUNT_CARD.md``; scaffold frozen in ``sycgen_lib``).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_sycgen_premeasure

**$0, and it runs BEFORE any generation.** The faces depend only on
where the challenge turns fall, not on what anyone says, so the
scaffold's geometry can be falsified without spending a token of
elicitation budget. That is the whole point: `msdose` died on
geometry AFTER its corpus was built, and `sycgen`'s licensed generator
mode would cost real money to build.

**This pre-measure can KILL but cannot CLEAR.** Content-dependent
traps are invisible to it — unigram leakage, and (the `emoinst`
lesson) whether post-challenge assistant text is per-token readable,
which is the way this candidate is most likely to die. Those need the
generated corpus, and the per-token baseline runs FIRST on it.

Instruments are the trio's, imported unchanged (terciles, bootstrap,
floors, Spearman) plus the § B strata census from ``msdose_r1_lib``.
The unigram triage is absent by necessity — there are no token ids.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from . import boot_lib as bo
from . import msdose_r1_lib as mr1
from . import novelty_lib as nl
from . import punctint_lib as pl
from . import sycgen_lib as sg
from . import wave3_lib as w3
from .build_wave3_trio import (HALF_LIFE_M, MIN_POS, N_REPS, SEED,
                               _floor_aucs, _spear, _terciles, SUPPORT_M)
from .hunt3_lib import FLOOR_TS
from .lib import doc_split

HERE = Path(__file__).resolve().parent
FROZEN_FILES = ("experiments/explorations/task_hunt/labels/sycgen_lib.py",
                "experiments/explorations/task_hunt/labels/"
                "build_sycgen_premeasure.py")

# ── pre-registered bands (card § 4; calibrated to in-repo precedent) ──
BAND_DOCMEAN_MAX = 0.88        # sycpress died at 0.995; survivor ~0.82-0.83
BAND_POSITION_MAX = 0.95       # surviving reask_hr band 0.925-0.946
BAND_QUAL_MIN = 8              # qualifying position strata
BAND_USABLE_MIN = 250_000      # position-matched usable tokens
BAND_EV_PER_CONV_MIN = 1.5     # mean challenges per conversation
BAND_EVENTS_MIN = 300          # total events (MIN_ROWS-class bar)


def _freeze_receipt() -> dict:
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=HERE,
                          capture_output=True, text=True,
                          check=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--"]
                           + list(FROZEN_FILES), cwd=HERE,
                           capture_output=True, text=True,
                           check=True).stdout.strip()
    assert not dirty, f"frozen logic dirty at run time:\n{dirty}"
    return {"head": head, "frozen_files_clean": True,
            "frozen_files": list(FROZEN_FILES)}


def _triage_geom(name, vals, bins, pos_of, doc_of, doc_off,
                 test_rows) -> dict:
    """Trio triage minus the unigram leg (no token ids exist yet)."""
    n_docs = len(doc_off) - 1
    docmean = np.full(n_docs, np.nan)
    for d in range(n_docs):
        seg = vals[doc_off[d]:doc_off[d + 1]]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            docmean[d] = seg.mean()
    scores = {"position_auc": pos_of.astype(float),
              "doc_mean_only_auc": docmean[doc_of]}
    rmask = test_rows & (bins >= 0)
    out = {}
    for s, sc in scores.items():
        out[s] = nl.tercile_auc(sc, bins, rmask)
        b = bo.bootstrap_tercile_auc(sc, bins, rmask, doc_of,
                                     n_reps=N_REPS, seed=SEED)
        out[s + "_ci"] = [b["ci_lo"], b["ci_hi"]]
        print(f"    {name}.{s}: {out[s]:.4f} "
              f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}]", flush=True)
    return out


def main():
    t0 = time.time()
    receipt = _freeze_receipt()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ch_len = len(tok(sg.SYCGEN_CHALLENGE_TEXT,
                     add_special_tokens=False)["input_ids"])
    print(f"[sycgen] challenge template = {ch_len} gpt2 tokens", flush=True)

    rng = np.random.default_rng(sg.SYCGEN_SEED)
    convs = sg.sycgen_plan(rng, ch_len)
    turn_l, asst_l, first_l, mask_l, ev_msg_l, off = [], [], [], [], [], [0]
    for msgs in convs:
        t, a, f, m = sg.layout_arrays(msgs)
        turn_l.append(t); asst_l.append(a); first_l.append(f); mask_l.append(m)
        ev_msg_l.append(np.array([mm["challenge"] for mm in msgs],
                                 dtype=np.int8))
        off.append(off[-1] + len(t))
    doc_off = np.array(off, dtype=np.int64)
    n_docs = len(convs)
    asst = np.concatenate(asst_l)
    first = np.concatenate(first_l)
    mask = np.concatenate(mask_l)
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    split = doc_split(n_docs, seed=SEED)
    train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1

    n_ev = [int(e.sum()) for e in ev_msg_l]
    n_msgs = sum(len(m) for m in convs)
    census = {
        "n_convs": n_docs, "n_tokens": int(len(asst)),
        "n_messages": n_msgs,
        "challenge_len_tokens": int(ch_len),
        "events_total": int(sum(n_ev)),
        "events_per_conv_mean": float(np.mean(n_ev)),
        "events_per_conv_min": int(min(n_ev)),
        "events_per_conv_max": int(max(n_ev)),
        "frac_convs_ge2": float(np.mean(np.array(n_ev) >= 2)),
        "tokens_per_message_mean": float(len(asst) / n_msgs),
    }
    print(f"[sycgen] {census['events_total']} events, "
          f"{census['events_per_conv_mean']:.2f}/conv "
          f"(min {census['events_per_conv_min']}, "
          f"max {census['events_per_conv_max']}); "
          f"{census['tokens_per_message_mean']:.1f} tok/msg", flush=True)

    # faces: T2 age (the clock argument's workhorse) + T1 rate
    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])
    rate = np.zeros(len(asst), dtype=np.float32)
    for d in range(n_docs):
        lo, hi = doc_off[d], doc_off[d + 1]
        lam = pl.sentence_lambda(ev_msg_l[d], half_life=HALF_LIFE_M,
                                 support=SUPPORT_M)
        rate[lo:hi] = pl.token_labels_from_sentences(lam,
                                                     np.concatenate(
                                                         [turn_l[d]]))
    stats: dict = {
        "card": "sycgen/PRECOUNT_CARD.md (WAVE3_SECOND_SOURCE §A "
                "re-entry; dispatch 47040da59)",
        "freeze_receipt": receipt,
        "geometry_only": ("faces depend on challenge POSITIONS only; "
                          "content traps (unigram leakage, per-token "
                          "readability of post-challenge text) are "
                          "NOT measurable here — they need the "
                          "generated corpus, per-token baseline first"),
        "census": census, "faces": {},
    }
    floors = {
        "censored_age": {T: np.concatenate(
            [w3.sage_floor(first[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS},
        "in_window_event_tokens": {T: np.concatenate(
            [w3.dose_window_count(mask[doc_off[d]:doc_off[d + 1]], T)
             for d in range(n_docs)]) for T in FLOOR_TS},
    }
    elig = (mask == 0) & (asst == 1) & (pos_of >= MIN_POS)
    for fname, vals in (("sycgen_age", age), ("sycgen_rate", rate)):
        bins, edges = _terciles(vals, train_rows, elig)
        st = {"eligible_rows": int((elig & np.isfinite(vals)).sum()),
              "tercile_edges": edges,
              "face_position_spearman": _spear(vals,
                                               pos_of.astype(float), elig)}
        print(f"  {fname}: {st['eligible_rows']:,} eligible rows, "
              f"rho(face,pos)="
              f"{st['face_position_spearman']['rho']:.3f}", flush=True)
        st.update(_triage_geom(fname, vals, bins, pos_of, doc_of, doc_off,
                               test_rows))
        st["floors"] = _floor_aucs(fname, bins, test_rows, floors)
        st["strata_census"] = mr1.strata_census(vals, pos_of, elig)
        c = st["strata_census"]
        print(f"    {fname}.census: {c['n_qualifying']}/"
              f"{c['n_strata_any']} strata, {c['usable_tokens']:,} usable",
              flush=True)
        stats["faces"][fname] = st

    bands = {}
    for fname, st in stats["faces"].items():
        c = st["strata_census"]
        b = {
            "doc_mean_le_0.88": {"value": st["doc_mean_only_auc"],
                                 "pass": st["doc_mean_only_auc"]
                                 <= BAND_DOCMEAN_MAX},
            "position_le_0.95": {"value": st["position_auc"],
                                 "pass": st["position_auc"]
                                 <= BAND_POSITION_MAX},
            "qualifying_ge_8": {"value": c["n_qualifying"],
                                "pass": c["n_qualifying"] >= BAND_QUAL_MIN},
            "usable_ge_250k": {"value": c["usable_tokens"],
                               "pass": c["usable_tokens"]
                               >= BAND_USABLE_MIN},
        }
        b = {k: {"value": float(v["value"]), "pass": bool(v["pass"])}
             for k, v in b.items()}
        b["all_pass"] = all(v["pass"] for v in b.values()
                            if isinstance(v, dict))
        bands[fname] = b
        print(f"  {fname} bands: "
              + ", ".join(f"{k}={'PASS' if v['pass'] else 'FAIL'}"
                          for k, v in b.items() if isinstance(v, dict)),
              flush=True)
    bands["event_mass"] = {
        "per_conv_ge_1.5": {"value": census["events_per_conv_mean"],
                            "pass": bool(census["events_per_conv_mean"]
                                         >= BAND_EV_PER_CONV_MIN)},
        "total_ge_300": {"value": census["events_total"],
                         "pass": bool(census["events_total"]
                                      >= BAND_EVENTS_MIN)},
    }
    stats["bands"] = bands
    survivors = [f for f, b in bands.items()
                 if f != "event_mass" and b["all_pass"]]
    mass_ok = all(v["pass"] for v in bands["event_mass"].values())
    stats["verdict_input"] = {
        "surviving_faces": survivors, "event_mass_ok": mass_ok,
        "kill_rule": "no surviving face, or event mass short => the "
                     "scaffold dies before any generation is bought",
        "killed": bool(not survivors or not mass_ok),
    }
    p = HERE / "sycgen_premeasure.json"
    p.write_text(json.dumps(stats, indent=1))
    print(f"-> {p} in {time.time() - t0:.0f}s; surviving faces: "
          f"{survivors or 'NONE'}", flush=True)


if __name__ == "__main__":
    main()
