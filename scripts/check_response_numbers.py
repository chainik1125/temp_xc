"""Do the numbers quoted in the reviewer response still match the source data?

Han's standing instruction (2026-07-29): keep `REBUTTAL_HANDOFF.md` and the
reviewer response updated as results land. The failure mode that creates is
obvious in hindsight and invisible in practice — the source data is
regenerated, the prose is not, and a stale number sits in a document nobody
re-reads because it was correct when written.

So the quoted table is re-derived from `frontier.json` and compared, rather
than trusted. Numbers in the response were generated programmatically in the
first place; this keeps them that way.

It checks the sycgen table only: the TXC row, the pooled and stacked rows
(interpolated to the TXC's measured L0 per window, the same rule
`gen_sycgen_budget_table.py` applies), and the TXC L0 row.

    .venv/bin/python scripts/check_response_numbers.py

Exit 0 = every quoted figure reproduces at the printed precision.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "experiments/explorations/task_hunt/sycgen/results/frontier.json"
RESPONSE_DIR = ROOT / "docs/dmitry/reviewer_responses"
DOC = RESPONSE_DIR / "reviewer_responses_1.md"
TS = (2, 4, 8, 16)


CONFLICT = ("<<<<<<<", "=======", ">>>>>>>")


def refuse_if_conflicted(*paths) -> None:
    """A content check cannot express 'this file is in a merge conflict'.

    ⚑ 2026-07-29: a commit certified this proposal with "verified
    programmatically … 0 mismatches" while the file carried live conflict
    markers INSIDE the block a human pastes to a reviewer. The row regexes
    passed because the merge left BOTH variants side by side, so each pattern
    still matched one of them. The check was correct and the file was corrupt.

    Structural integrity is a PRECONDITION of any content check, never an
    implication of one. (mac-d `a55c2109e`.)
    """
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if line.startswith(CONFLICT):
                raise SystemExit(
                    f"REFUSING TO CERTIFY {p.name}: conflict marker at line "
                    f"{i} — {line[:40]!r}. Resolve the merge first; a content "
                    f"check cannot see this and will pass regardless.")


def agg(rows, arm, T, k):
    rs = [r for r in rows if r["arm"] == arm and r["T"] == T
          and r.get("k_tok") == k]
    if not rs:
        return None
    return {"r": mean(x["recovery"] for x in rs),
            "l0": mean(x["realized_l0_per_window"] for x in rs)}


def expected() -> dict:
    rows = json.loads(SRC.read_text())
    ks = sorted({r["k_tok"] for r in rows if r.get("k_tok") is not None})
    out = {}
    for T in TS:
        txc_r = mean(r["recovery"] for r in rows
                     if r["arm"] == "txc" and r["T"] == T)
        txc_l0 = mean(r["realized_l0_per_window"] for r in rows
                      if r["arm"] == "txc" and r["T"] == T)
        row = {"txc": txc_r, "l0": txc_l0}
        for arm in ("pooled", "stacked"):
            pts = [c for c in (agg(rows, arm, T, k) for k in ks) if c]
            lo = [c for c in pts if c["l0"] <= txc_l0 + 1e-9]
            hi = [c for c in pts if c["l0"] > txc_l0 + 1e-9]
            if lo and hi:
                a = max(lo, key=lambda c: c["r"])
                b = min(hi, key=lambda c: c["l0"])
                f = (txc_l0 - a["l0"]) / (b["l0"] - a["l0"])
                row[arm] = a["r"] + f * (b["r"] - a["r"])
            elif hi:
                row[arm] = min(hi, key=lambda c: c["l0"])["r"]
            else:
                row[arm] = max(pts, key=lambda c: c["r"])["r"]
        out[T] = row
    return out


def quoted() -> dict:
    """Parse the sycgen table.

    Handles BOTH shapes: the markdown pipe table (house style since Dmitry's
    2026-07-29 conversion — pipe tables render everywhere, which is what the
    LaTeX arrays kept failing to do) and the older LaTeX array, so this keeps
    working on either branch mid-migration.
    """
    body = DOC.read_text()
    got: dict = {}
    rows = {"Pooled SAE": "pooled", "Stacked SAE": "stacked", "TXC": "txc",
            "TXC L0 per window": "l0"}

    # --- markdown pipe table ------------------------------------------
    for label, key in rows.items():
        pat = re.compile(r"^\|\s*\*{0,2}" + re.escape(label) +
                         r"\*{0,2}\s*\|(.+)$", re.M)
        m = pat.search(body)
        if not m:
            continue
        vals = re.findall(r"(\d+\.\d+)", m.group(1))
        for T, v in zip(TS, vals):
            got.setdefault(T, {})[key] = float(v)

    if got:
        return got

    # --- legacy LaTeX array -------------------------------------------
    for name, key in (("Pooled SAE", "pooled"), ("Stacked SAE", "stacked"),
                      ("TXC", "txc")):
        pat = re.compile(r"\\text\{" + name + r"\}((?:\s*&\s*(?:\\mathbf\{)?"
                         r"-?[\d.]+\}?(?:\^\{\*\})?){4})\s*\\\\")
        m = pat.search(body)
        if not m:
            continue
        for T, v in zip(TS, re.findall(r"(\d\.\d+)", m.group(1))):
            got.setdefault(T, {})[key] = float(v)
    m = re.search(r"\\text\{TXC \}\s*L_0/\\text\{window\}((?:\s*&\s*[\d.]+){4})",
                  body)
    if m:
        for T, v in zip(TS, re.findall(r"(\d+\.\d+)", m.group(1))):
            got.setdefault(T, {})["l0"] = float(v)
    return got


def main() -> int:
    # Structural integrity is checked over EVERY reviewer-bound doc, not just
    # the one this script parses. mac-d 12:5x: the file that actually shipped
    # with markers inside a paste-ready fence was
    # PROPOSED_sycgen_excerpt_reviewer1.md, and NEITHER checker opens it by
    # default — the guard existed and did not cover the file that motivated
    # it. Same shape as the positive control that was aimed at a file the
    # checker never reads. A content check is file-specific; "is this file in
    # a merge conflict" is not.
    refuse_if_conflicted(DOC, SRC, *sorted(RESPONSE_DIR.glob("*.md")))
    if not SRC.exists():
        print(f"no source at {SRC}")
        return 0
    exp, got = expected(), quoted()
    if not got:
        print("could not parse the quoted table — has its shape changed?")
        return 1
    bad = 0
    print(f"{'T':>3}  {'field':<9} {'quoted':>8} {'source':>8}  ok")
    for T in TS:
        for f in ("txc", "pooled", "stacked", "l0"):
            if f not in got.get(T, {}):
                continue
            q, e = got[T][f], exp[T][f]
            # compare at the precision actually printed
            dec = len(str(q).split(".")[1])
            ok = round(e, dec) == q
            bad += 0 if ok else 1
            print(f"{T:>3}  {f:<9} {q:>8} {round(e, dec):>8}  "
                  f"{'ok' if ok else 'MISMATCH'}")
    print(f"\n{bad} mismatch(es). The response quotes numbers that must track "
          f"{SRC.name}; regenerate the prose when the source moves.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
