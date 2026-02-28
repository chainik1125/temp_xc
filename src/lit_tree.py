"""Generate a literature review hierarchy tree in ASCII or Obsidian markdown.

Parses docs/han/literature_review/{sae,temporal_learning,theory_foundation}.md
and outputs a tree organised as:

    category / ## section / paper (authors — title (year))

Usage:
    python src/lit_tree.py          # ASCII tree to stdout
    python src/lit_tree.py --md     # Obsidian markdown list to stdout
    python src/lit_tree.py --write  # write both tree.txt and tree.md
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

CATEGORIES = ["sae", "temporal_learning", "theory_foundation"]
LIT_DIR = Path(__file__).resolve().parent.parent / "docs" / "han" / "literature_review"
REF_PATH = Path(__file__).resolve().parent.parent / "docs" / "shared" / "references.md"


@dataclass
class Paper:
    authors: str
    title: str
    year: str
    url: str
    ref_key: str = ""


@dataclass
class Section:
    name: str
    papers: list[Paper] = field(default_factory=list)


def build_url_to_refkey() -> dict[str, str]:
    """Parse references.md to build a URL -> citation_key mapping."""
    mapping: dict[str, str] = {}
    if not REF_PATH.exists():
        return mapping
    current_key: str | None = None
    for line in REF_PATH.read_text().splitlines():
        m = re.match(r"^### (\S+)", line)
        if m:
            current_key = m.group(1)
            continue
        if current_key:
            for m_url in re.finditer(r"\]\(([^)]+)\)", line):
                mapping[m_url.group(1)] = current_key
    return mapping


def _extract_year(raw: str) -> str:
    """Pull the last (YYYY) from a citation string."""
    m = re.findall(r"\((?:[^)]*?)(\d{4})\)", raw)
    return m[-1] if m else ""


def _split_citation(raw: str) -> tuple[str, str]:
    """Split 'Authors. Title. Venue (Year)' into (authors, title)."""
    parts = re.split(r"(?<=\w{2})\. (?=[A-Z])", raw, maxsplit=1)
    if len(parts) == 2:
        authors, rest = parts
    else:
        return raw, ""
    title = rest.split(". ", 1)[0]
    return authors, title


def parse_file(path: Path, url_to_key: dict[str, str]) -> list[Section]:
    """Return list of Sections with Papers for one markdown file."""
    sections: list[Section] = []
    current: Section | None = None

    for line in path.read_text().splitlines():
        m_sec = re.match(r"^## (.+)", line)
        if m_sec:
            current = Section(name=m_sec.group(1).strip())
            sections.append(current)
            continue

        m_paper = re.match(r"^#### \[(.+?)\]\((.+?)\)", line)
        if m_paper and current is not None:
            raw, url = m_paper.group(1), m_paper.group(2)
            authors, title = _split_citation(raw)
            year = _extract_year(raw)
            ref_key = url_to_key.get(url, "")
            current.papers.append(Paper(authors, title, year, url, ref_key))

    return sections


def _paper_label(p: Paper) -> str:
    """Plain-text label: Authors — Title (Year)."""
    year_suffix = f" ({p.year})" if p.year else ""
    return f"{p.authors} — {p.title}{year_suffix}"


def build_tree(data: dict[str, list[Section]]) -> str:
    """Render as an ASCII tree."""
    lines: list[str] = ["literature_review/"]
    cats = list(data.keys())

    for ci, cat in enumerate(cats):
        last_cat = ci == len(cats) - 1
        cp = "└── " if last_cat else "├── "
        cc = "    " if last_cat else "│   "
        lines.append(f"{cp}{cat}")

        sections = data[cat]
        for si, sec in enumerate(sections):
            last_sec = si == len(sections) - 1
            sp = f"{cc}└── " if last_sec else f"{cc}├── "
            sc = f"{cc}    " if last_sec else f"{cc}│   "
            lines.append(f"{sp}{sec.name}")

            for pi, paper in enumerate(sec.papers):
                last_p = pi == len(sec.papers) - 1
                pp = f"{sc}└── " if last_p else f"{sc}├── "
                lines.append(f"{pp}{_paper_label(paper)}")

    return "\n".join(lines)


def build_md(data: dict[str, list[Section]]) -> str:
    """Render as a nested Obsidian markdown list with wikilinks."""
    lines: list[str] = []

    for cat, sections in data.items():
        lines.append(f"- **{cat}**")
        for sec in sections:
            lines.append(f"  - {sec.name}")
            for p in sec.papers:
                label = _paper_label(p)
                if p.ref_key:
                    lines.append(f"    - [[references#{p.ref_key}|{label}]]")
                else:
                    lines.append(f"    - {label}")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate literature review tree.")
    parser.add_argument("--md", action="store_true", help="Output Obsidian markdown list instead of ASCII tree")
    parser.add_argument("--write", action="store_true", help="Write both tree.txt and tree.md to the literature_review directory")
    args = parser.parse_args()

    url_to_key = build_url_to_refkey()

    data: dict[str, list[Section]] = {}
    for cat in CATEGORIES:
        path = LIT_DIR / f"{cat}.md"
        if not path.exists():
            print(f"Warning: {path} not found, skipping.")
            continue
        data[cat] = parse_file(path, url_to_key)

    if args.write:
        txt = build_tree(data)
        md = build_md(data)
        (LIT_DIR / "tree.txt").write_text(txt + "\n")
        (LIT_DIR / "tree.md").write_text(md + "\n")
        print(f"Wrote {LIT_DIR / 'tree.txt'} and {LIT_DIR / 'tree.md'}")
    elif args.md:
        print(build_md(data))
    else:
        print(build_tree(data))


if __name__ == "__main__":
    main()
