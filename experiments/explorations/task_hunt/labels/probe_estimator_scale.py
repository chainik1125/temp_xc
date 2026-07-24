"""Why did the unigram triage bar RISE at 10x corpus? (corpus-scaleup,
item 1 follow-up — label-side only, no new corpus, no new labels.)

    .venv/bin/python -m experiments.explorations.task_hunt.labels.probe_estimator_scale

The scaled punctint build reports current-token type-mean AUCs of
0.558-0.583 on manifest rows where the shipped 400-doc build reported
0.517-0.534. Two readings, with opposite consequences:

- **estimator noise** — the triage score is a TRAIN-SET mean per token
  type (``novelty_lib.type_mean_scores``). With 320 training documents
  most types are seen a handful of times, so the score is mostly noise
  and the measured AUC is attenuated toward 0.5. Then the shipped
  numbers were UNDERSTATEMENTS of a route that was there all along, and
  every 400-doc triage number in the hunt is a lower bound;
- **corpus composition** — the extra 3,600 documents genuinely carry a
  stronger token-identity/label association. Then the shipped bundle's
  own numbers stand and only the scaled corpus is affected.

They are separable, because the scaled corpus CONTAINS the pinned one:
hold the evaluation rows fixed (the scaled build's test manifest rows)
and vary only how many training documents the estimator may use. If the
curve rises with training documents and lands near the shipped value at
the shipped training size, it is estimator noise.

Ladder: 40 / 80 / 160 / 320 (= the shipped build's train size) / 640 /
1280 / 2560 / all 3200 train documents, 3 seeded draws each (the full
set once). Writes ``probe_estimator_scale.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import lib
from . import novelty_lib as nl

HERE = Path(__file__).resolve().parent
LADDER = (40, 80, 160, 320, 640, 1280, 2560)
N_DRAWS = 3
SEED = 0
FACES = ("list", "q")
TOKENIZERS = ("gpt2", "gemma2", "llama31")


def face_rows(z, face):
    """Rebuild the exact triage row sets from the shipped arrays."""
    doc_off = z["doc_off"]
    n_docs = len(doc_off) - 1
    doc_of = np.repeat(np.arange(n_docs, dtype=np.int32), np.diff(doc_off))
    pos_of = np.concatenate([np.arange(n, dtype=np.int32)
                             for n in np.diff(doc_off)])
    bins = z[f"{face}_bin"]
    evt = z[f"is_{face}"]
    masked = np.where(evt == 1, -1, bins).astype(np.int8)
    split = z["doc_split"]
    elig = (masked >= 0) & (pos_of >= lib.MIN_MANIFEST_POS)
    man = np.zeros(len(pos_of), dtype=bool)
    man[doc_off[:-1][z[f"man_{face}_doc"]] + z[f"man_{face}_pos"]] = True
    return {"doc_of": doc_of, "masked": masked, "split": split,
            "train": split[doc_of] == 0, "test": split[doc_of] == 1,
            "elig": elig, "man": man, "lam": z[f"lam_{face}"]}


def main():
    out = {"ladder": list(LADDER), "n_draws": N_DRAWS, "seed": SEED,
           "note": "evaluation rows are FIXED (scaled build's test "
                   "manifest rows and test eligible rows); only the "
                   "number of TRAIN documents feeding the type-mean "
                   "estimator varies",
           "per_tokenizer": {}}
    for key in TOKENIZERS:
        z = np.load(HERE / f"punctint4k_fineweb_{key}.npz")
        per_face = {}
        for face in FACES:
            r = face_rows(z, face)
            train_docs = np.flatnonzero(r["split"] == 0)
            eval_sets = {"manifest": r["man"] & r["test"],
                         "all_eligible": r["test"] & r["elig"]}
            curve = {}
            for n in list(LADDER) + [len(train_docs)]:
                if n > len(train_docs):
                    continue
                vals = {k: [] for k in eval_sets}
                draws = 1 if n == len(train_docs) else N_DRAWS
                for d in range(draws):
                    rng = np.random.default_rng(SEED + d)
                    sub = (train_docs if n == len(train_docs)
                           else rng.choice(train_docs, size=n,
                                           replace=False))
                    keep = np.zeros(len(r["split"]), dtype=bool)
                    keep[sub] = True
                    tmask = keep[r["doc_of"]] & r["elig"]
                    sc = nl.type_mean_scores(z["token_ids"], r["lam"], tmask)
                    for k, m in eval_sets.items():
                        vals[k].append(nl.tercile_auc(sc, r["masked"], m))
                curve[str(n)] = {k: {"mean": float(np.mean(v)),
                                     "min": float(np.min(v)),
                                     "max": float(np.max(v))}
                                 for k, v in vals.items()}
                print(f"[{key}/{face}] train_docs={n:5d} "
                      f"manifest={curve[str(n)]['manifest']['mean']:.4f} "
                      f"all={curve[str(n)]['all_eligible']['mean']:.4f}",
                      flush=True)
            per_face[face] = {"n_train_docs_total": int(len(train_docs)),
                              "curve": curve}
        out["per_tokenizer"][key] = per_face
    p = HERE / "probe_estimator_scale.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"-> {p}")


if __name__ == "__main__":
    main()
