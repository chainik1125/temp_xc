"""Is a retracted claim still LIVE anywhere, or only quoted inside corrections?

Tonight (2026-07-29) produced five retraction sweeps and the same difficulty
each time: `grep -c` counts a struck-through claim, a claim quoted in order to
refute it, and a live assertion identically. The workaround was
`grep -v "THEY DO NOT"` and friends — excluding correction text by matching
its prose, which breaks the moment someone words a correction differently.

mac-c's improvement (`a41959bb2`) is the right one and this makes it reusable:
**parse the markup and classify each occurrence structurally.** An occurrence
inside `~~...~~`, inside a blockquote, or on a line carrying a correction
marker is *quoted*, not asserted. Only the rest are live.

Two rules this enforces, both earned the hard way:

  1. **The control must fire.** A sweep is run against a git ref where the
     claim still exists; if the pattern does not match there, the sweep is
     broken and a clean result means nothing.
  2. **The control must be evaluated on a state you did NOT just fix.**
     Grepping after your own edit returns zero because you removed the
     pattern, not because the surfaces are clean.

    .venv/bin/python scripts/claim_sweep.py "phrase" [--control-ref origin/arxiv] [paths...]

Exit 0 = no live occurrences (and the control fired), 1 = live ones remain
or the control was silent.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATHS = [
    "REBUTTAL_HANDOFF.md", "REBUTTAL_CODE_GUIDE.md", "REBUTTAL_CELL_CENSUS.md",
    "figs_writeup", "briefings", "experiments/explorations/task_hunt",
]
# Broadened after its first real run: the narrow version missed "Restated",
# "moved from X to Y", "went from", "Cost us", and bare arrows — four
# legitimate correction lines flagged as live. A sweep that cries wolf gets
# ignored, which is the same defect as one that never fires.
CORRECTION = re.compile(
    r"SUPERSEDED|WITHDRAWN|withdrawn|RETRACT|retracted|CORRECTED|corrected|"
    r"restated|amended|WRONG|FALSE|previously (?:said|read)|is NOT|"
    r"THEY DO NOT|no longer|moved from|went from|changed from|cost us|"
    r"used to (?:say|read)|→|->|~~",
    re.I)

# The LOG is append-only history: it quotes every retracted claim by design and
# is correctly historical (hub ruling, 00:23). Excluded by default; pass it
# explicitly to sweep it.
SKIP = ("experiments/explorations/task_hunt/LOG.md",)
STRUCK = re.compile(r"~~.+?~~", re.S)


def classify(text: str, phrase: str) -> tuple[list, list]:
    """(live, quoted) line numbers for each occurrence of phrase."""
    struck_spans = [(m.start(), m.end()) for m in STRUCK.finditer(text)]
    live, quoted = [], []
    for m in re.finditer(re.escape(phrase), text, re.I):
        i = m.start()
        ln = text[:i].count("\n") + 1
        line = text.splitlines()[ln - 1] if ln <= len(text.splitlines()) else ""
        in_struck = any(a <= i < b for a, b in struck_spans)
        in_quote = line.lstrip().startswith(">")
        # A correction marker usually sits on an ADJACENT line, not the one
        # carrying the phrase — the correction reads "X" / was wrong because…
        # Checking only the phrase's own line produced 2 false positives out
        # of 3 on its first run against the hub's own withdrawn rulings.
        lines_all = text.splitlines()
        window = "\n".join(lines_all[max(0, ln - 3): ln + 2])
        near_corr = bool(CORRECTION.search(window))
        (quoted if (in_struck or in_quote or near_corr) else live).append(ln)
    return live, quoted


def files(paths: list[str]):
    for p in paths:
        q = ROOT / p
        if q.is_file():
            yield q
        elif q.is_dir():
            yield from (f for f in q.rglob("*.md")
                        if ".git" not in str(f)
                        and not any(str(f).endswith(k) for k in SKIP))


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[-3])
        return 1
    phrase = argv[0]
    ref = "origin/arxiv"
    if "--control-ref" in argv:
        ref = argv[argv.index("--control-ref") + 1]
        argv = [a for a in argv if a != ref and a != "--control-ref"]
    paths = argv[1:] or DEFAULT_PATHS

    total_live = total_quoted = 0
    for f in files(paths):
        live, quoted = classify(f.read_text(), phrase)
        if live or quoted:
            rel = f.relative_to(ROOT)
            if live:
                print(f"  ✗ LIVE   {rel}: lines {live}")
            if quoted:
                print(f"  · quoted {rel}: lines {quoted} (struck / blockquote / "
                      f"correction line)")
        total_live += len(live)
        total_quoted += len(quoted)

    # --- the control: the phrase must be findable on a ref that predates the fix
    ctrl = subprocess.run(["git", "grep", "-c", "-i", phrase, ref],
                          cwd=ROOT, capture_output=True, text=True)
    hits = sum(int(l.rsplit(":", 1)[1]) for l in ctrl.stdout.splitlines()
               if ":" in l) if ctrl.returncode == 0 else 0
    print(f"\nlive {total_live}   quoted {total_quoted}   "
          f"control on {ref}: {hits} hit(s)")
    # mac-c's method note (ef7b43400), made a guard rather than a rule: the
    # swept phrase must be specific enough to be WRONG. Sweeping 'load-bearing'
    # returned 15 live hits, all ordinary uses of a common phrase. A key that
    # matches everywhere cannot separate a retracted claim from the language
    # it happens to share. Choose the key from the RETRACTED SENTENCE, not
    # from its memorable phrase.
    if hits > 30 or len(phrase.split()) < 3:
        print(f"\n** KEY MAY BE TOO GENERIC: {hits} control hits, "
              f"{len(phrase.split())} word(s). A phrase that appears "
              f"everywhere cannot isolate one retracted claim — sweep a "
              f"distinctive clause FROM THE RETRACTED SENTENCE instead.")
    if hits == 0:
        print(f"** CONTROL SILENT on {ref} — the pattern is not detectable "
              f"there, so a clean result proves nothing. Pick a ref that "
              f"PREDATES the fix.")
        return 1
    if total_live:
        print("** LIVE assertions found — READ each one before concluding.")
        print("   A live hit means the phrase is asserted, NOT that it is")
        print("   wrong: the same words can be correct in another context")
        print("   (e.g. '...sit at chance' is true where chance really is")
        print("   0.5 and false on a 3-class task with a 0.33 null). The")
        print("   tool separates quoted from asserted; only you can judge")
        print("   whether an assertion is true.")
        return 1
    print("clean: every occurrence is struck, quoted, or on a correction line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
