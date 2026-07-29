"""Structural check for LaTeX in the reviewer-response markdown.

These documents are read on GitHub, whose math rendering is unforgiving in
ways a LaTeX compiler is not. Every failure below has actually shipped and
been reported by a human rather than caught here, which is why the file
exists.

Checks:
  1. INLINE MATH SPANNING A NEWLINE. `$...$` does not cross a line break.
     A wrapped span leaves the opening `$` unclosed and the next `$` opens
     a new one, so the braces cascade and GitHub reports "Extra close brace
     or missing open brace" — pointing at a table that is itself fine.
     This one shipped twice. My first KaTeX checker could not see it: its
     regex was `[^$\\n]+`, which excludes newlines *by construction*, so
     the check was structurally incapable of failing on the live bug.
  2. Odd number of `$` in a paragraph — delimiters misalign from there on.
  3. Unbalanced braces inside a math span.
  4. `\\text{}` nested inside `\\mathrm{}` (use `\\!-\\!` for hyphens).
  5. A superscript or subscript with no base (`$^{*}$`), invalid LaTeX.
  6. Escaping consistency: a `$$` block must be all single-backslash or
     all double. The working sections of these docs are single (they are
     rendered); the OpenReview paste sections are double (markdown eats
     one backslash before the math engine). Mixing them breaks silently.

Exit 0 clean, 1 on any finding. `--self-test` proves each check can fail.

    .venv/bin/python scripts/check_response_math.py [file ...] [--self-test]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESPONSE_DIR = ROOT / "docs/dmitry/reviewer_responses"
DEFAULT = RESPONSE_DIR / "reviewer_responses_1.md"

COMMENT = re.compile(r"<!--[\s\S]*?-->")
DISPLAY = re.compile(r"\$\$([\s\S]*?)\$\$")
# NOTE: `[^$]` deliberately ALLOWS newlines — finding the wrapped spans is
# the whole point. Excluding them is the bug this script exists to catch.
INLINE = re.compile(r"(?<!\$)\$([^$]{1,400}?)\$(?!\$)")


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


def inline_spans_newline(body: str) -> list[str]:
    return [m.group(0).replace("\n", " <NEWLINE> ")
            for m in INLINE.finditer(body) if "\n" in m.group(1)]


def odd_dollars(body: str) -> list[str]:
    return [p[:70].replace("\n", " ") for p in body.split("\n\n")
            if p.count("$") % 2]


def unbalanced_braces(tex: str) -> bool:
    d = 0
    for ch in tex:
        if ch == "{":
            d += 1
        elif ch == "}":
            d -= 1
            if d < 0:
                return True
    return d != 0


def text_in_mathrm(tex: str) -> bool:
    return bool(re.search(r"\\{1,2}mathrm\{[^{}]*\\{1,2}text\{", tex))


def baseless_script(body: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"(?<![\w})\]])\$\s*[\^_]\{", body)]


def mixed_escaping(tex: str) -> bool:
    single = len(re.findall(r"(?<!\\)\\(?!\\)[a-zA-Z]", tex))
    double = len(re.findall(r"\\\\[a-zA-Z]", tex))
    return single > 0 and double > 0


def check(path: Path) -> list[str]:
    raw = path.read_text()
    body = COMMENT.sub("", raw)
    out: list[str] = []
    warn: list[str] = []
    for t in inline_spans_newline(DISPLAY.sub("", body)):
        out.append(f"{path.name}: inline math spans a NEWLINE (GitHub breaks "
                   f"here): {t[:90]}")
    for p in odd_dollars(DISPLAY.sub("", body)):
        out.append(f"{path.name}: paragraph has an ODD number of '$': {p}")
    # NOTE: reported, NOT failed. `$^{\\dagger}$` is strictly invalid LaTeX
    # (no base) but BOTH KaTeX and MathJax render it, and it appears in the
    # document's pre-existing text, which demonstrably renders. Failing on it
    # would block on someone else's working line — a check that fires on
    # valid-in-practice input is as useless as one that never fires.
    for t in baseless_script(body):
        warn.append(f"{path.name}: superscript with no base (renders, but not "
                    f"strictly valid): {t}")
    for m in DISPLAY.finditer(body):
        tex = m.group(1)
        head = tex.strip().replace("\n", " ")[:60]
        if unbalanced_braces(tex):
            out.append(f"{path.name}: unbalanced braces in $$ block: {head}")
        if text_in_mathrm(tex):
            out.append(f"{path.name}: \\text{{}} nested in \\mathrm{{}}: {head}")
        if mixed_escaping(tex):
            out.append(f"{path.name}: MIXED single/double backslash in one "
                       f"block: {head}")
    globals()['_WARN'] = warn
    return out


def self_test() -> int:
    """Every check, on an input that must trip it. A guard shown only
    passing has been demonstrated, not tested."""
    probes = [
        ("inline spans newline", bool(inline_spans_newline("a $T\\cdot\nd$ b"))
         and not inline_spans_newline("a $T\\cdot d$ b")),
        ("odd dollars", bool(odd_dollars("has $one only"))
         and not odd_dollars("has $one$ pair")),
        ("unbalanced braces", unbalanced_braces(r"\mathbf{x")
         and not unbalanced_braces(r"\mathbf{x}")),
        ("text in mathrm", text_in_mathrm(r"\mathrm{per\text{-}tok}")
         and not text_in_mathrm(r"\mathrm{per\!-\!tok}")),
        ("baseless script", bool(baseless_script("$^{*}$ note"))
         and not baseless_script("$x^{*}$ note")),
        ("mixed escaping", mixed_escaping(r"\begin{array} \\mathrm{x}")
         and not mixed_escaping(r"\begin{array} \mathrm{x}")),
    ]
    bad = [n for n, ok in probes if not ok]
    if bad:
        print(f"SELF-TEST FAILED: {', '.join(bad)}")
        return 1
    print(f"self-test OK: {len(probes)} checks each fired on a failure case "
          f"and stayed quiet on a passing one")
    return 0


def main(argv: list[str]) -> int:
    if "--self-test" in argv:
        if self_test():
            return 1
        argv = [a for a in argv if a != "--self-test"]
    paths = [Path(a) for a in argv] or [DEFAULT]
    # Widened for the same reason as check_response_numbers.py (mac-d 12:5x):
    # the doc that shipped with markers is not in either default path set, so
    # the structural guard is run over the whole reviewer-bound directory
    # regardless of which file's CONTENT is being checked.
    refuse_if_conflicted(*paths, *sorted(RESPONSE_DIR.glob("*.md")))
    problems: list[str] = []
    for p in paths:
        if p.exists():
            problems += check(p)
    for x in problems:
        print(f"  x {x}")
    for w in globals().get("_WARN", []):
        print(f"  ~ {w}")
    print(f"\n{len(paths)} file(s) checked, {len(problems)} problem(s), "
          f"{len(globals().get('_WARN', []))} warning(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
