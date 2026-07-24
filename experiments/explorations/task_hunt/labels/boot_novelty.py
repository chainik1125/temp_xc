"""Doc-level bootstrap CIs for the committed novelty triage (corpus-
scaleup item 3) — label-side only, no new corpus, no new labels.

    .venv/bin/python -m experiments.explorations.task_hunt.labels.boot_novelty

The novelty family screened NEGATIVE, so nothing here is a verdict: its
400-doc triage numbers feed the same threshold-pinning dataset as items
1 and 2, and they are the family with the LOWEST document-level
component measured so far (runpod-e: 22 % between-doc variance,
doc-mean-only AUC 0.792) — which makes them the informative low end of
that distribution.

Recomputed from the committed ``novelty_fineweb_<tok>.npz`` with the
shipped row definitions verbatim (test-doc rows, ``pos >= SUPPORT``,
both the detrended primary ``nov_bin`` and the disclosed
position-confounded ``nov_raw_bin``), plus two things the shipped stats
predate: the **manifest-row view** (the operative convention adopted
later) and the **doc_mean_only_auc** disclosure statistic. Point
estimates on the shipped row set reproduce ``novelty_stats.json``
exactly — asserted here, so a divergence would surface as a failure
rather than as a quietly different number.

Writes ``novelty_bootstrap.json``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from . import boot_lib as bo
from . import novelty_lib as nl

HERE = Path(__file__).resolve().parent
SEED = 0
TOKENIZERS = ("gpt2", "gemma2", "llama31")
FACES = {"resid": ("nov_resid", "nov_bin", "man_nov"),
         "raw": ("nov_rate", "nov_raw_bin", "man_novraw")}


def main():
    shipped = json.loads((HERE / "novelty_stats.json").read_text())
    out = {"source": "novelty_fineweb_<tok>.npz (400 docs, unchanged)",
           "note": "novelty screened NEGATIVE — these are threshold-"
                   "dataset numbers, not a verdict",
           "bootstrap": {"unit": "document (cluster)", "n_reps": bo.N_REPS,
                         "ci_pct": list(bo.CI_PCT), "seed": SEED},
           "per_tokenizer": {}}
    for key in TOKENIZERS:
        z = np.load(HERE / f"novelty_fineweb_{key}.npz")
        doc_off = z["doc_off"]
        n_docs = len(doc_off) - 1
        doc_of = np.repeat(np.arange(n_docs, dtype=np.int32),
                           np.diff(doc_off))
        pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                                 for n in np.diff(doc_off)])
        split = z["doc_split"]
        train_rows, test_rows = split[doc_of] == 0, split[doc_of] == 1
        elig = pos_of >= nl.SUPPORT
        per_face = {}
        for face, (vkey, bkey, mkey) in FACES.items():
            vals, terc = z[vkey], z[bkey]
            unigram = nl.type_mean_scores(z["token_ids"], vals,
                                          train_rows & elig)
            docmean = np.full(n_docs, np.nan)
            for d in range(n_docs):
                seg = vals[doc_off[d]: doc_off[d + 1]]
                seg = seg[np.isfinite(seg)]
                if seg.size:
                    docmean[d] = seg.mean()
            man_rows = np.zeros(len(pos_of), dtype=bool)
            man_rows[doc_off[:-1][z[f"{mkey}_doc"]] + z[f"{mkey}_pos"]] = True
            scores = {"unigram_auc": unigram,
                      "position_auc": pos_of.astype(float),
                      "doc_mean_only_auc": docmean[doc_of]}
            row_sets = {"triage_all_eligible_rows": test_rows & elig,
                        "triage_manifest_rows": man_rows & test_rows}
            face_out = {}
            for rname, rmask in row_sets.items():
                face_out[rname] = {}
                for s, sc in scores.items():
                    t0 = time.time()
                    b = bo.bootstrap_tercile_auc(sc, terc, rmask, doc_of,
                                                 n_reps=bo.N_REPS, seed=SEED)
                    face_out[rname][s] = b
                    print(f"[{key}/{face}] {rname}.{s}: {b['point']:.4f} "
                          f"[{b['ci_lo']:.4f}, {b['ci_hi']:.4f}] "
                          f"({b['n_rows']:,} rows, {time.time() - t0:.0f}s)",
                          flush=True)
            # the shipped point estimates must reproduce exactly
            ship = shipped["per_tokenizer"][key]["triage"][face]
            for s in ("unigram_auc", "position_auc"):
                got = face_out["triage_all_eligible_rows"][s]["point"]
                assert abs(got - ship[s]) < 1e-9, (
                    f"{key}/{face}/{s}: {got} != shipped {ship[s]}")
            per_face[face] = face_out
        out["per_tokenizer"][key] = per_face
    p = HERE / "novelty_bootstrap.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
