"""Render paper/refs.bib into a markdown reference list (one entry per line).

Output (stdout):
  - **arditi2024refusal** Andy Arditi, ..., Neel Nanda. *Refusal in Language Models...* arXiv:2406.11717 (2024). [link](https://arxiv.org/abs/2406.11717)

This file is GENERATED — never hand-edit. Re-run after build_bib.py.

Why this exists: STEERING_REPORT.md is markdown, not LaTeX, so we can't use
\\bibliography{}. But the bibliography-from-ids rule still applies: we never
hand-type author/title/year metadata into prose. Instead we render the bib
into markdown and inject a fenced section into the report.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
BIB = ROOT / "paper" / "refs.bib"


ENTRY_RE = re.compile(
    r"@(?P<type>\w+)\s*\{\s*(?P<key>[^,\s]+)\s*,\s*(?P<body>.*?)\n\}",
    re.DOTALL,
)
FIELD_RE = re.compile(r"(\w+)\s*=\s*\{(.+?)\}\s*,?\s*\n", re.DOTALL)


def parse_bib(text: str) -> list[dict]:
    out: list[dict] = []
    for m in ENTRY_RE.finditer(text):
        body = m.group("body")
        fields = {f.lower(): v for f, v in FIELD_RE.findall(body)}
        # one trailing field with no comma → re-grep
        for f, v in re.findall(r"(\w+)\s*=\s*\{(.+?)\}\s*\n", body, re.DOTALL):
            fields.setdefault(f.lower(), v)
        out.append({"key": m.group("key"), "type": m.group("type"), **fields})
    return out


def shorten_authors(authors: str) -> str:
    """'A and B and C and D' -> 'A, B, ..., D' (et al if >2 authors).
    Authors are 'First Last' style separated by ' and '."""
    parts = [p.strip() for p in authors.split(" and ") if p.strip()]
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{parts[0]} et al."


def render_entry(e: dict) -> str:
    key = e["key"]
    authors = shorten_authors(e.get("author", ""))
    title = e.get("title", "").replace("\n", " ").strip()
    title = re.sub(r"\s+", " ", title)
    year = e.get("year", "")
    url = e.get("url") or e.get("howpublished", "")
    url = url.strip("\\url{} ").strip()
    arxiv = e.get("eprint", "")
    if arxiv:
        venue = f"arXiv:{arxiv}"
    elif "transformer-circuits.pub" in url:
        venue = "Transformer Circuits Thread"
    else:
        venue = ""
    suffix = []
    if venue:
        suffix.append(venue)
    if year:
        suffix.append(f"({year})")
    suffix_s = " ".join(suffix)
    link = f" [link]({url})" if url else ""
    return f"- **{key}** — {authors}. *{title}.* {suffix_s}.{link}"


def main() -> None:
    if not BIB.exists():
        sys.exit(f"ERROR: {BIB} not found; run build_bib.py first")
    entries = parse_bib(BIB.read_text())
    if not entries:
        sys.exit("ERROR: no entries parsed from refs.bib")
    print(f"<!-- BEGIN: AUTO-GENERATED FROM paper/refs.bib via "
          f"scripts/bib/render_md_refs.py — DO NOT EDIT. -->")
    for e in entries:
        print(render_entry(e))
    print(f"<!-- END: AUTO-GENERATED -->")


if __name__ == "__main__":
    main()
