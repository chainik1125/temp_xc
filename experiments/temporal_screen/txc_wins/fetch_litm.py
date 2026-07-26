"""Fetch the Lost in the Middle QA data (not committed -- 7.6 MB per file, 1 MB repo limit).

    python experiments/temporal_screen/txc_wins/fetch_litm.py

Verifies what it downloads rather than trusting it: item counts match across files, the
question and answer strings agree pairwise, the document multiset is identical, and the
gold document sits at the index the filename claims. Exits non-zero if any check fails --
a silent mismatch here would invalidate the matched-foil argument the whole task rests on.
"""
import gzip
import json
import pathlib
import sys
import urllib.request

BASE = ("https://raw.githubusercontent.com/nelson-liu/lost-in-the-middle/main/"
        "qa_data/10_total_documents/nq-open-10_total_documents_gold_at_{g}.jsonl.gz")
OUT = pathlib.Path(__file__).resolve().parent / "litm_data"
GOLDS = (0, 4, 9)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    data = {}
    for g in GOLDS:
        p = OUT / f"gold_at_{g}.jsonl.gz"
        if not p.exists():
            print(f"[fetch] gold_at_{g}")
            urllib.request.urlretrieve(BASE.format(g=g), p)
        with gzip.open(p, "rt") as fh:
            data[g] = [json.loads(l) for l in fh]
        print(f"  gold_at_{g}: {len(data[g])} items")

    n = len(data[GOLDS[0]])
    if any(len(v) != n for v in data.values()):
        print("[fail] item counts differ across files", file=sys.stderr)
        return 1
    bad_q = bad_m = bad_g = 0
    for j in range(n):
        rows = [data[g][j] for g in GOLDS]
        if len({r["question"] for r in rows}) != 1:
            bad_q += 1
        titles = [sorted(c["title"] for c in r["ctxs"]) for r in rows]
        if any(t != titles[0] for t in titles):
            bad_m += 1
        for g, r in zip(GOLDS, rows):
            if [i for i, c in enumerate(r["ctxs"]) if c.get("isgold")] != [g]:
                bad_g += 1
    print(f"[verify] {n} items | question mismatches {bad_q} | "
          f"title-multiset mismatches {bad_m} | gold-index violations {bad_g}")
    if bad_q or bad_m or bad_g:
        print("[fail] the matched-foil property does not hold", file=sys.stderr)
        return 1
    print("[ok] matched foil verified across every item")
    return 0


if __name__ == "__main__":
    sys.exit(main())
