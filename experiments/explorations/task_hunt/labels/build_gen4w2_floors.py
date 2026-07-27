"""Wave-2 floor bundles — per-T visible floors for the gen-4 wave-2
screens (HUNT4W2 card), recomputed from the COMMITTED gen4c streams
(`gen4c_<corpus>_<tok>.npz`: token ids + boundary masks — no
re-tokenization) and committed fp16 (the hunt4 bundle convention;
mac-c's scout stated the committed-weight deviation, this restores
the screen-side contract under MY freeze).

    .venv/bin/python -m experiments.explorations.task_hunt.labels.build_gen4w2_floors

Floors (scout § 4 instruments):
- wikitext103: `floor_rate_T{T}` (window-novelty kernel rate — shared
  by tret_wt and tretd_wt, the hunt4 § 3 disclosure) +
  `sage_floor_T{T}` (censored age).
- pycode: `floor_rate_T{T}` (tret_py). drev is killed label-side
  (wave-2 card § 1) — its floor is not bundled.

Artifacts: ``gen4w2_floors_<corpus>_<tok>.npz``. Determinism check:
floor values must round-trip the scout's § 4 AUC lines — verified by
the card's freeze notes, not recomputed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import gen4c_lib as g4
from . import hunt3_lib as h3

HERE = Path(__file__).resolve().parent

CORPS = {
    "wikitext103": ("gpt2", "gemma2", "llama31"),
    "pycode": ("gpt2", "gemma2", "llama31"),
}


def build(corpus: str, tok: str):
    z = np.load(HERE / f"gen4c_{corpus}_{tok}.npz")
    ids, off = z["token_ids"], z["doc_off"]
    n = len(ids)
    names = ["floor_rate"] + (["sage_floor"] if corpus == "wikitext103"
                              else [])
    fl = {f"{name}_T{T}": np.full(n, np.nan, dtype=np.float32)
          for T in h3.FLOOR_TS for name in names}
    mark = z["is_boundary"] if corpus == "wikitext103" else None
    for d in range(len(off) - 1):
        s, e = off[d], off[d + 1]
        last_occ = h3.last_occurrence(ids[s:e])
        for T in h3.FLOOR_TS:
            fl[f"floor_rate_T{T}"][s:e] = h3.floor_rate(last_occ, T)
            if mark is not None:
                fl[f"sage_floor_T{T}"][s:e] = g4.sage_floor(mark[s:e], T)
    out = HERE / f"gen4w2_floors_{corpus}_{tok}.npz"
    np.savez_compressed(out, **{k: v.astype(np.float16)
                                for k, v in fl.items()})
    print(f"[{corpus}/{tok}] -> {out.name} ({len(fl)} arrays, {n} tok)")


def main():
    import sys
    only = set(sys.argv[1:])           # e.g. `... llama31` builds one tag
    for corpus, toks in CORPS.items():
        for tok in toks:
            if only and tok not in only:
                continue
            build(corpus, tok)


if __name__ == "__main__":
    main()
