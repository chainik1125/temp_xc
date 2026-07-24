"""Did the label logic really stay frozen? (corpus-scaleup verification)

    .venv/bin/python -m experiments.explorations.task_hunt.labels.verify_prefix_labels

The campaign's central discipline claim is that the scaled builds reuse
the committed label libraries UNCHANGED. That claim is checkable rather
than merely asserted, because the 4,000-doc corpus is a token-for-token
superset of the pinned 400: the shipped bundle and the scaled bundle
must agree EXACTLY on every per-token label of the shared prefix.

Compared per tokenizer, ``punctint_fineweb_<tok>.npz`` (shipped, 400
docs) against ``punctint4k_fineweb_<tok>.npz`` restricted to the first
400 documents:

- must be **bit-identical**: ``token_ids``, ``doc_off``, ``sent_idx``,
  ``in_span``, and both faces' ``lam_*`` (NaN-aware) and ``is_*`` — none
  of these depends on anything outside a document;
- **expected to differ, and quantified rather than waved away**: the
  3-class ``*_bin`` labels and ``doc_split``. Bins come from
  ``zero_split``/tercile edges estimated on the TRAIN rows of whichever
  corpus is being built, and the split itself is drawn for n_docs, so a
  10x corpus legitimately re-estimates both. The report gives the edge
  pair, the fraction of shared-prefix rows whose class changes, and the
  confusion between the two labelings.

Writes ``verify_prefix_labels.json``. A failure here would mean the
scaled bundle is not what it says it is; it is loud on purpose.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOKENIZERS = ("gpt2", "gemma2", "llama31")
FACES = ("list", "q")
EXACT_KEYS = ("token_ids", "sent_idx", "in_span")


def nan_equal(a, b) -> bool:
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        return False
    both_nan = np.isnan(a) & np.isnan(b)
    return bool(np.all(both_nan | (a == b)))


def main():
    out = {"note": "shipped 400-doc bundle vs the scaled bundle restricted "
                   "to the shared prefix; per-token labels must be "
                   "identical, class labels may legitimately differ "
                   "(edges + split are re-estimated per corpus)",
           "per_tokenizer": {}}
    all_ok = True
    for key in TOKENIZERS:
        a = np.load(HERE / f"punctint_fineweb_{key}.npz")
        b = np.load(HERE / f"punctint4k_fineweb_{key}.npz")
        n_docs = len(a["doc_off"]) - 1
        n_tok = int(a["doc_off"][-1])
        rec = {"n_docs_prefix": n_docs, "n_tokens_prefix": n_tok,
               "identical": {}, "reclassified": {}}
        rec["identical"]["doc_off"] = bool(
            np.array_equal(a["doc_off"], b["doc_off"][: n_docs + 1]))
        for k in EXACT_KEYS:
            rec["identical"][k] = bool(np.array_equal(a[k], b[k][:n_tok]))
        for f in FACES:
            rec["identical"][f"lam_{f}"] = nan_equal(a[f"lam_{f}"],
                                                     b[f"lam_{f}"][:n_tok])
            rec["identical"][f"is_{f}"] = bool(
                np.array_equal(a[f"is_{f}"], b[f"is_{f}"][:n_tok]))
            ba, bb = a[f"{f}_bin"], b[f"{f}_bin"][:n_tok]
            lab = (ba >= 0) & (bb >= 0)
            conf = [[int(((ba == i) & (bb == j) & lab).sum())
                     for j in range(3)] for i in range(3)]
            rec["reclassified"][f] = {
                "labeled_rows_both": int(lab.sum()),
                "class_changed_frac": float((ba[lab] != bb[lab]).mean()),
                "confusion_shipped_rows_x_scaled_cols": conf}
        ok = all(rec["identical"].values())
        all_ok &= ok
        rec["all_per_token_labels_identical"] = ok
        out["per_tokenizer"][key] = rec
        print(f"[{key}] per-token labels identical on the shared "
              f"{n_tok:,}-token prefix: {ok}; class relabeled "
              + ", ".join(f"{f}={rec['reclassified'][f]['class_changed_frac']:.4f}"
                          for f in FACES), flush=True)
    out["all_tokenizers_pass"] = bool(all_ok)
    p = HERE / "verify_prefix_labels.json"
    p.write_text(json.dumps(out, indent=1))
    print(f"ALL PASS: {all_ok}\n-> {p}")


if __name__ == "__main__":
    main()
