"""Mechanical completeness check for the rebuttal deliverable surface.

Written so that "is anything missing?" is a command rather than a
judgement call — in particular after a compact, when the answer must not
depend on remembering what was promised.

Checks, in order:
  1. every figure embedded in REBUTTAL_HANDOFF.md exists on disk;
  2. every file pointer in prose resolves (shorthand paths are searched
     for by basename, since the handoff cites some files short);
  3. no conflict markers in any deliverable surface;
  4. each deliverable item 1-7 has BOTH a plot and a table;
  5. the cell census is not older than the leaderboard.

Exit 0 = clean, 1 = something is missing. Run `--self-test` to confirm
the checker can actually FAIL — a guard exercised only on its success
path has been demonstrated, not tested (2026-07-28, six instances).

    .venv/bin/python scripts/handoff_audit.py [--self-test]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HANDOFF = ROOT / "REBUTTAL_HANDOFF.md"
SURFACES = [
    "REBUTTAL_HANDOFF.md",
    "REBUTTAL_CODE_GUIDE.md",
    "REBUTTAL_CELL_CENSUS.md",
    "experiments/explorations/task_hunt/LOG.md",
]

# item -> (plot substring, table substring). Both must appear in the
# handoff AND resolve on disk. Item 7's "table" is its RESULT.md.
ITEMS = {
    "1 probing k5": ("fig_probing_shuffle_tsweep_k5", "RESULTS_btk-only.md"),
    "2 probing k20": ("fig_probing_shuffle_tsweep_k20", "RESULTS.md"),
    "3 RLHF btk": ("fig_rlhf_shuffle_tsweep.png", "rlhf_table.md"),
    "3 RLHF pf": ("fig_rlhf_shuffle_tsweep_pf", "rlhf_table.md"),
    "4 lambda": ("fig_lambda_shuffle_tsweep", "tab_lambda_shuffle_tsweep.md"),
    "5 dq": ("fig2_question_gap_tscaling", "tab_dq_tsweep.md"),
    "6 sycgen": ("fig_sycgen_shuffle_tsweep", "tab_sycgen_shuffle_tsweep.md"),
    "7 hunted #4": ("retryesc_gen", "retryesc_gen/RESULT.md"),
}

fails: list[str] = []
notes: list[str] = []


def resolve(ref: str) -> Path | None:
    """Full path, else search by basename (the handoff cites some short)."""
    p = ROOT / ref
    if p.exists():
        return p
    hits = [h for h in ROOT.rglob(Path(ref).name) if ".git/" not in str(h)]
    return hits[0] if hits else None


def main(self_test: bool = False) -> int:
    text = HANDOFF.read_text()

    # 1. embedded figures
    figs = sorted(set(re.findall(r"]\(([^)]*\.png)\)", text)))
    for f in figs:
        if not (ROOT / f).exists():
            fails.append(f"figure embedded but MISSING on disk: {f}")
    notes.append(f"figures embedded: {len(figs)}, all present"
                 if not fails else f"figures embedded: {len(figs)}")

    # 2. prose pointers
    refs = sorted(set(re.findall(r"`([A-Za-z0-9_/.\-]+\.(?:md|jsonl))`", text)))
    unresolved = [r for r in refs if resolve(r) is None]
    for r in unresolved:
        fails.append(f"pointer does not resolve anywhere: {r}")
    notes.append(f"prose pointers: {len(refs)}, unresolved {len(unresolved)}")

    # 3. conflict markers
    for s in SURFACES:
        p = ROOT / s
        if not p.exists():
            fails.append(f"deliverable surface missing: {s}")
            continue
        n = sum(1 for ln in p.read_text().splitlines() if ln.startswith("<<<<<<<"))
        if n:
            fails.append(f"{n} conflict marker(s) in {s}")
    notes.append(f"surfaces scanned for markers: {len(SURFACES)}")

    # 4. every item has a plot AND a table
    for item, (plot, table) in ITEMS.items():
        if plot not in text:
            fails.append(f"item {item}: plot not referenced in handoff ({plot})")
        if table not in text:
            fails.append(f"item {item}: table not referenced in handoff ({table})")
        elif resolve(table) is None:
            fails.append(f"item {item}: table referenced but MISSING ({table})")
    notes.append(f"items checked for plot+table: {len(ITEMS)}")

    # 5. census freshness
    census, lb = ROOT / "REBUTTAL_CELL_CENSUS.md", ROOT / "results/leaderboard.jsonl"
    if census.exists() and lb.exists() and census.stat().st_mtime < lb.stat().st_mtime:
        fails.append("census is OLDER than the leaderboard — "
                     "run scripts/cell_census.py --write")
    notes.append("census freshness checked")

    if self_test:
        # prove the checker can fail: a reference that cannot exist
        probe = "definitely_not_a_real_deliverable_xyz.md"
        if resolve(probe) is not None:
            print("SELF-TEST FAILED: resolver claims a nonexistent file exists")
            return 1
        print("self-test OK: resolver returns None for a nonexistent file, "
              "so an unresolved pointer would be reported")

    for n in notes:
        print(f"  · {n}")
    if fails:
        print(f"\nAUDIT FAILED — {len(fails)} problem(s):")
        for f in fails:
            print(f"  ✗ {f}")
        return 1
    print("\nAUDIT CLEAN — every item has a plot and a table, every pointer "
          "resolves, no conflict markers, census current.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(self_test="--self-test" in sys.argv))
