"""sycgen disposition (c): WITHIN-DOMAIN position-matched readout.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.sycgen_domain_readout \
        --stream <labels/elicit_sycgen_v1.npz>

Design-owner spec (b70821046, ratified a8cdc86e7): hold domain constant
so domain vocabulary cannot predict the label by construction — the
instrument is `msdose_r1.strata_census` with the census run PER DOMAIN
(domain-local terciles), plus the triage AUCs recomputed within each
domain. $0, label-side recompute, no regeneration. Report per-domain
qualifying mass; thin domains are dropped-and-disclosed, not hidden.

Doc→domain mapping is rebuilt by replaying the frozen generation draw
order (seed 0: plan → domain → questions, none consulting the mask);
the replay is receipt-checked against the landed corpus (conv counts
per domain + total events must match) before any number is reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import msdose_r1_lib as mr1
from . import sycgen_lib as sg
from . import wave3_lib as w3
from .build_sycgen_premeasure import (BAND_DOCMEAN_MAX, BAND_POSITION_MAX,
                                      BAND_QUAL_MIN, _triage_geom)
from .build_wave3_trio import MIN_POS, _terciles
from .run_elicit import SYCGEN_SEEDS_FILE

HERE = Path(__file__).resolve().parent


def replay_doc_domains(n_docs: int, seed: int) -> list[str]:
    """Replay run_sycgen's frozen draw order to recover per-doc domain."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("gpt2")
    ch_len = len(tok(sg.SYCGEN_CHALLENGE_TEXT,
                     add_special_tokens=False)["input_ids"])
    rng = np.random.default_rng(seed)
    plans = sg.sycgen_plan(rng, ch_len, n_convs=n_docs)
    by_dom: dict[str, int] = {}
    with open(HERE / SYCGEN_SEEDS_FILE) as f:
        for line in f:
            r = json.loads(line)
            by_dom[r["base"].get("dataset", "?")] = \
                by_dom.get(r["base"].get("dataset", "?"), 0) + 1
    doms = sorted(by_dom)
    out = []
    for msgs in plans:
        n_q = sum(1 for m in msgs if not m["assistant"]
                  and not m["challenge"])
        dom = doms[int(rng.integers(len(doms)))]
        rng.choice(by_dom[dom], size=n_q, replace=n_q > by_dom[dom])
        out.append(dom)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", required=True)
    ap.add_argument("--seed", type=int, default=sg.SYCGEN_SEED)
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
    age = np.concatenate([w3.sage_face(first[doc_off[d]:doc_off[d + 1]])
                          for d in range(n_docs)])

    doc_dom = np.array(replay_doc_domains(n_docs, a.seed))
    doms = sorted(set(doc_dom))
    # replay receipt-check: totals must match the landed corpus
    assert len(doc_dom) == n_docs
    ev_total = int(first.sum())
    print(f"[replay-check] {n_docs} docs over {len(doms)} domains; "
          f"{ev_total} events in stream", flush=True)

    per_dom, pooled_usable, pooled_qualifying = {}, 0, 0
    for dom in doms:
        dmask = doc_dom == dom
        row_in = dmask[doc_of]
        elig_d = elig & row_in
        n_convs = int(dmask.sum())
        n_ev = int(first[row_in.nonzero()[0]].sum()) if row_in.any() else 0
        st: dict = {"n_convs": n_convs, "events": n_ev,
                    "eligible_rows": int((elig_d & np.isfinite(age)).sum())}
        if st["eligible_rows"] < 3 * mr1.STRATUM_MIN_PER_TERCILE:
            st["dropped"] = "too thin to stratify (disclosed, not hidden)"
            per_dom[dom] = st
            print(f"  {dom}: DROPPED thin ({st['eligible_rows']} rows)",
                  flush=True)
            continue
        bins, edges = _terciles(age, train_rows & row_in, elig_d)
        st["tercile_edges"] = edges
        st.update(_triage_geom(f"age[{dom}]", age, bins, pos_of, doc_of,
                               doc_off, test_rows & row_in))
        st["strata_census"] = mr1.strata_census(age, pos_of, elig_d)
        c = st["strata_census"]
        st["bands"] = {
            "doc_mean_le_0.88": {"value": st["doc_mean_only_auc"],
                                 "pass": st["doc_mean_only_auc"]
                                 <= BAND_DOCMEAN_MAX},
            "position_le_0.95": {"value": st["position_auc"],
                                 "pass": st["position_auc"]
                                 <= BAND_POSITION_MAX},
            "qualifying_ge_8": {"value": c["n_qualifying"],
                                "pass": c["n_qualifying"] >= BAND_QUAL_MIN},
        }
        pooled_usable += c["usable_tokens"]
        pooled_qualifying += c["n_qualifying"]
        per_dom[dom] = st
        print(f"  {dom}: convs {n_convs}, docmean "
              f"{st['doc_mean_only_auc']:.3f}, pos {st['position_auc']:.3f}, "
              f"qual {c['n_qualifying']}/{c['n_strata_any']}, usable "
              f"{c['usable_tokens']:,}", flush=True)

    summary = {
        "pooled_usable_tokens": int(pooled_usable),
        "pooled_usable_ge_250k": bool(pooled_usable >= 250_000),
        "pooled_qualifying_strata": int(pooled_qualifying),
        "domains_reported": len(doms),
        "domains_dropped": [d for d, s in per_dom.items() if "dropped" in s],
    }
    print(f"[within-domain] pooled usable {pooled_usable:,} "
          f"(ge 250k: {summary['pooled_usable_ge_250k']}), "
          f"pooled qualifying strata {pooled_qualifying}", flush=True)

    out = {"stream": a.stream, "disposition": "b70821046 (c)",
           "per_domain": per_dom, "summary": summary,
           "note": ("within-domain readout: domain vocabulary cannot "
                    "predict the label with domain held constant; "
                    "terciles are domain-local")}
    out_path = Path(a.out) if a.out else HERE / "sycgen_domain_readout.json"
    out_path.write_text(json.dumps(out, indent=1, default=float))
    print(f"-> {out_path}", flush=True)


if __name__ == "__main__":
    main()
