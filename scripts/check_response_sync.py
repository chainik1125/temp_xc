"""Are the reviewer-response copies on `arxiv` and `dmitry-txcwins-10h` in step?

The same document now lives on two branches (Han, 2026-07-29: keep both
updated as results land). Two copies of a live document is precisely the
hazard that produced four separate contradictions tonight — a claim
corrected on one surface and left standing on another. So this is checked
by command rather than remembered.

It compares the committed copies on the two remote refs. It does NOT push:
writing to another person's branch stays a deliberate act.

    .venv/bin/python scripts/check_response_sync.py [--fetch]

Exit 0 = in step (or a difference that is only the intentional in-file
marker text), 1 = real drift.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ["docs/dmitry/reviewer_responses/reviewer_responses_1.md"]
REFS = ("origin/arxiv", "origin/dmitry-txcwins-10h")

# The marker comment intentionally differs between branches (one says the
# other branch is also carrying it). Normalise it away before comparing so
# the check reports CONTENT drift, not bookkeeping.
MARKER = re.compile(r"<!--\s*=+\s*ADDED.*?=+\s*-->", re.S)


def show(ref: str, path: str) -> str | None:
    r = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main(argv: list[str]) -> int:
    if "--fetch" in argv:
        subprocess.run(["git", "fetch", "-q", "origin"], cwd=ROOT)
    bad = 0
    for path in FILES:
        a, b = (show(REFS[0], path), show(REFS[1], path))
        if a is None or b is None:
            missing = REFS[0] if a is None else REFS[1]
            print(f"  x {path}: not present on {missing}")
            bad += 1
            continue
        na, nb = MARKER.sub("<MARKER>", a), MARKER.sub("<MARKER>", b)
        if na == nb:
            print(f"  ok {path}: identical on both refs "
                  f"({len(a.splitlines())} lines)")
            continue
        bad += 1
        la, lb = na.splitlines(), nb.splitlines()
        only_a = [x for x in la if x not in set(lb)]
        only_b = [x for x in lb if x not in set(la)]
        print(f"  x {path}: DRIFT between {REFS[0]} and {REFS[1]}")
        print(f"      only on {REFS[0]}: {len(only_a)} line(s)")
        for x in only_a[:4]:
            print(f"        + {x[:88]}")
        print(f"      only on {REFS[1]}: {len(only_b)} line(s)")
        for x in only_b[:4]:
            print(f"        - {x[:88]}")
    print(f"\n{len(FILES)} file(s) compared across {REFS[0]} and {REFS[1]}: "
          f"{bad} drifted")
    if bad:
        print("Reconcile before quoting either copy. Pushing to another "
              "person's branch stays a deliberate act — this script will "
              "not do it for you.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
