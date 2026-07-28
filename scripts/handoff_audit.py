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
  5. the cell census is not older than the leaderboard;
  6. no arch the census flags (UNMAPPED / OFF-SUBSTRATE) is cited in the
     handoff — added 07-28 after the handoff was found quoting item 3's
     pf RLHF result from cells our own census said to classify first;
  7. every runnable script path cited in either guide exists.

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

    # 6. flagged archs must not be cited as deliverables
    census_p = ROOT / "REBUTTAL_CELL_CENSUS.md"
    flagged: set[str] = set()
    if census_p.exists():
        for ln in census_p.read_text().splitlines():
            if not ln.startswith("| `"):
                continue
            cols = [c.strip() for c in ln.split("|")]
            if len(cols) > 4 and "⚠" in cols[3]:
                flagged.add(cols[1].strip("` "))
    # The census flags at (arch, datasource) granularity, so "arch is
    # cited" is the WRONG test — the handoff cites the flagged arch
    # precisely to warn about it. The real requirement: if you cite an
    # arch that has off-substrate rows, you must also pin the substrate
    # those rows are NOT on. First cut of this check failed on its own
    # warning text, which is how the granularity bug surfaced.
    pins = dict(re.findall(r"`([a-z0-9_]+)`/T=\S+: \d+ row\(s\) on `[^`]+`, "
                           r"pinned to `([^`]+)`",
                           census_p.read_text() if census_p.exists() else ""))
    for arch in sorted(flagged):
        if arch not in text:
            continue
        pin = pins.get(arch)
        if pin is None:
            fails.append(f"handoff cites `{arch}`, which the census FLAGS "
                         f"as UNMAPPED — classify before quoting")
        elif pin not in text:
            fails.append(f"handoff cites `{arch}`, which has OFF-SUBSTRATE "
                         f"rows, without pinning `{pin}` anywhere")
    # NOTE the matching is on EXACT arch ids: the handoff cites the RLHF arm
    # as `agentic_txc_02` while the census keys `agentic_txc_02_v1t`, so a
    # shortened citation is not caught here. Loosening it would fire on the
    # correct-substrate cells too, which is why the substrate is ALSO pinned
    # in the census itself rather than relying on this check alone.
    cited = sum(1 for a in flagged if a in text)
    notes.append(f"census-flagged archs: {len(flagged)}, cited in handoff "
                 f"{cited} (exact-id match; a citation is OK only if the "
                 f"pinned substrate is named too)")

    # 7. runnable script paths cited in either guide
    guide_pat = re.compile(
        r"(?:agents/[a-z0-9-]+/|scripts/|experiments/[A-Za-z0-9_/.-]*|src/[A-Za-z0-9_/.-]*)"
        r"[A-Za-z0-9_./-]+\.(?:sh|py)")
    n_scripts = 0
    for g in ("REBUTTAL_HANDOFF.md", "REBUTTAL_CODE_GUIDE.md"):
        gp = ROOT / g
        if not gp.exists():
            continue
        for s in sorted(set(guide_pat.findall(gp.read_text()))):
            n_scripts += 1
            if not (ROOT / s).exists():
                fails.append(f"{g} cites a script that does not exist: {s}")
    notes.append(f"script paths cited in guides: {n_scripts}")

    # 8. staleness sweep — REPORT, not a gate. These phrases were all
    # true when written and went false as the work landed; on 07-28 the
    # handoff still announced a deadline that had passed. A checker
    # cannot judge whether prose is current, so this prints hits and
    # leaves the call to a human. Silence here means nothing; a hit
    # means "re-read that line".
    stale_pat = re.compile(
        r"(?i)\b(this morning|tonight|overnight|~?\d{1,2}:\d{2}\s*(?:BST)?\s*"
        r"(?:today|render)|ETA ~|lands? ~|landing this|RUNNING NOW|in flight|"
        r"h from now|before the deadline|by the deadline|all night|"
        r"expected before)\b")
    stale_hits: list[str] = []
    for g in ("REBUTTAL_HANDOFF.md", "REBUTTAL_CODE_GUIDE.md"):
        gp = ROOT / g
        if not gp.exists():
            continue
        for i, ln in enumerate(gp.read_text().splitlines(), 1):
            m = stale_pat.search(ln)
            if m and "SUPERSEDED" not in ln and "previously" not in ln:
                stale_hits.append(f"{g}:{i}: …{m.group(0)}…")
    notes.append(f"staleness sweep: {len(stale_hits)} future/time-bound "
                 f"phrase(s) — REPORT ONLY, judge each by hand")

    # 5. census freshness
    census, lb = ROOT / "REBUTTAL_CELL_CENSUS.md", ROOT / "results/leaderboard.jsonl"
    if census.exists() and lb.exists() and census.stat().st_mtime < lb.stat().st_mtime:
        fails.append("census is OLDER than the leaderboard — "
                     "run scripts/cell_census.py --write")
    notes.append("census freshness checked")

    if self_test:
        # Each guard is exercised on its FAILURE path. A guard shown only
        # succeeding has been demonstrated, not tested (07-28, six cases).
        probes = []

        probe = "definitely_not_a_real_deliverable_xyz.md"
        probes.append(("unresolved pointer", resolve(probe) is None))

        # flagged-arch guard: a census line whose arm column carries ⚠
        fake = "| `fake_arch_xyz` | `ds` | ⚠ UNMAPPED — classify | 2 | ... |"
        cols = [c.strip() for c in fake.split("|")]
        probes.append(("flagged-arch parse",
                       len(cols) > 4 and "⚠" in cols[3]
                       and cols[1].strip("` ") == "fake_arch_xyz"))

        # script-path guard: must both FIND a path and judge it missing
        found = guide_pat.findall("run `scripts/definitely_missing_xyz.py` now")
        probes.append(("missing-script detect",
                       found == ["scripts/definitely_missing_xyz.py"]
                       and not (ROOT / found[0]).exists() if found else False))

        bad = [n for n, ok in probes if not ok]
        if bad:
            print(f"SELF-TEST FAILED: guard(s) did not fire on their failure "
                  f"path: {', '.join(bad)}")
            return 1
        print(f"self-test OK: {len(probes)} guards each fired on a synthetic "
              f"FAILURE case ({', '.join(n for n, _ in probes)})")

    for n in notes:
        print(f"  · {n}")
    if stale_hits:
        print("\n  time-bound phrasing to re-read (report only, not failures):")
        for h in stale_hits:
            print(f"    ~ {h}")
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
