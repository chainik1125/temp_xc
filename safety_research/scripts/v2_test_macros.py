"""
Smoke test: STEERING_REPORT.md should not contain unresolved macros.

The report builder substitutes M['key'] inline; missing keys yield literal "?".
This test fails if any ?-placeholder remains in non-prose contexts (i.e. the
report should fully resolve once all macros land).

We also list which (method, arm, ds, K) combos are missing from
paper_macros.json — useful to confirm the v2 steering run completed.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path("/home/cs29824/andre/temp_xc/safety_research")
REPORT = ROOT / "STEERING_REPORT.md"
MACROS = ROOT / "results" / "andre_steering_v2" / "paper_macros.json"


def main() -> int:
    text = REPORT.read_text()
    # Strip fenced code blocks, then look for **?**, leakage **?**, fired ?/200.
    body = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    placeholders = re.findall(r"\*\*\?\*\*|fired \?", body)
    macros = json.load(open(MACROS))
    print(f"macros defined: {len(macros)}")
    print(f"unresolved placeholders in report: {len(placeholders)}")
    if placeholders:
        # Show context: 80 chars before & after each placeholder
        for m in re.finditer(r"\*\*\?\*\*|fired \?", body):
            i = m.start()
            ctx = body[max(0, i-60):i+30].replace("\n", " ")
            print(f"  ...{ctx}...")
        return 1
    print("OK — every macro reference in the report resolved to a value.")

    # Coverage report: which (arm, ds) combos have FSGA K=20 macros?
    expected_arms = ("sae", "tsae", "txc")
    expected_ds = ("test_in", "test_ood", "test_mi")
    print("\nFSGA K=20 macro coverage:")
    print(f"  {'arm':<6s} | {'ds':<10s} | dh? | leak? | cFSGA?")
    for arm in expected_arms:
        for ds in expected_ds:
            dh = f"fsgaDh_{arm}_{ds}"
            leak = f"fsgaLeak_{arm}_{ds}"
            cf = f"cfsgaLeak_{arm}_{ds}"
            print(f"  {arm:<6s} | {ds:<10s} | "
                  f"{'Y' if dh in macros else '.':<3s} | "
                  f"{'Y' if leak in macros else '.':<5s} | "
                  f"{'Y' if cf in macros else '.'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
