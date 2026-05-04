"""Lint: every \\cite{key} in the paper resolves to an entry in refs.bib.

Also reports the reverse direction (entries in refs.bib that no \\cite
uses), which usually indicates dead references to delete.

Runs offline. No network, no dependencies beyond the standard library.

Usage:
    uv run python scripts/verify_cites.py
    uv run python scripts/verify_cites.py --tex paper/somepaper.tex

Exit codes:
    0  OK
    1  usage error
    2  unresolved \\cite keys (hard failure)
    3  unused entries in refs.bib (soft failure; can be waived with --ignore-unused)
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BIB = PROJECT_ROOT / "paper" / "refs.bib"

# natbib commands: \cite, \citep, \citet, \citealp, \citealt, \citeauthor, \citeyear, ...
CITE_RE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])?(?:\[[^\]]*\])?\{([^}]+)\}")
ENTRY_HEAD_RE = re.compile(r"^\s*@\w+\s*\{\s*([^,\s}]+)", re.MULTILINE)


def _find_default_tex() -> Path:
    """Return the sole .tex file in paper/ that contains \\documentclass.

    If there are zero or multiple candidates, fall back to paper/main.tex.
    The caller will get a clear "file not found" error, and the user can
    pass --tex explicitly. This avoids hardcoding a project-specific
    filename while still "just working" for the single-paper case.
    """
    paper_dir = PROJECT_ROOT / "paper"
    if paper_dir.is_dir():
        candidates = []
        for tex in sorted(paper_dir.glob("*.tex")):
            try:
                head = tex.read_text(encoding="utf-8", errors="ignore")[:4096]
            except OSError:
                continue
            if r"\documentclass" in head:
                candidates.append(tex)
        if len(candidates) == 1:
            return candidates[0]
    return paper_dir / "main.tex"


def extract_cite_keys(tex_path: Path) -> set[str]:
    if not tex_path.exists():
        sys.exit(f"ERROR: {tex_path} not found")
    text = tex_path.read_text(encoding="utf-8")
    # Strip LaTeX comments (lines starting with % or %-prefixed tail).
    text = re.sub(r"(?<!\\)%.*", "", text)
    keys: set[str] = set()
    for m in CITE_RE.finditer(text):
        for k in m.group(1).split(","):
            k = k.strip()
            if k:
                keys.add(k)
    return keys


def extract_bib_keys(bib_path: Path) -> set[str]:
    if not bib_path.exists():
        sys.exit(
            f"ERROR: {bib_path} not found. Run `uv run python scripts/build_bib.py` first."
        )
    text = bib_path.read_text(encoding="utf-8")
    return {m.group(1) for m in ENTRY_HEAD_RE.finditer(text)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, default=_find_default_tex())
    parser.add_argument("--bib", type=Path, default=DEFAULT_BIB)
    parser.add_argument("--ignore-unused", action="store_true")
    args = parser.parse_args()

    cited = extract_cite_keys(args.tex)
    defined = extract_bib_keys(args.bib)

    missing = sorted(cited - defined)
    unused = sorted(defined - cited)

    if missing:
        print(f"FAIL: {len(missing)} \\cite key(s) not in {args.bib.name}:", file=sys.stderr)
        for k in missing:
            print(f"  - {k}", file=sys.stderr)
        return 2

    if unused:
        level = "WARN" if args.ignore_unused else "FAIL"
        print(f"{level}: {len(unused)} entry/entries in {args.bib.name} never cited:", file=sys.stderr)
        for k in unused:
            print(f"  - {k}", file=sys.stderr)
        if not args.ignore_unused:
            return 3

    print(f"OK: {len(cited)} \\cite key(s), all resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
