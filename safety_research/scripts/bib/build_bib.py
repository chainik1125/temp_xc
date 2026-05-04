"""Regenerate paper/refs.bib from paper/refs_ids.toml.

Reads the list of citation identifiers in paper/refs_ids.toml and resolves
each via an authoritative source:

    acl:<id>      -> https://aclanthology.org/<id>.bib
    doi:<id>      -> https://doi.org/<id> with Accept: application/x-bibtex
    arxiv:<id>    -> https://export.arxiv.org/api/query?id_list=<id> (XML)
    manual        -> the literal BibTeX block in the TOML entry

Precedence per entry: acl > doi > arxiv > manual.

Fail-loud philosophy: any network error, missing field, or unresolved
identifier aborts the build. Partial .bib files are never written — the
output is a tempfile renamed at the end, so an aborted run leaves the
previous refs.bib intact.

Usage:
    uv run python scripts/build_bib.py [--offline-manual-only]

Flags:
    --offline-manual-only
        Only emit the `manual = true` entries, skip anything requiring
        network. Useful for smoke-testing the parser.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
try:
    import tomllib
except ImportError:  # python < 3.11
    import tomli as tomllib  # type: ignore
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests


# layout: safety_research/scripts/bib/build_bib.py + safety_research/paper/...
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TOML_PATH = PROJECT_ROOT / "paper" / "refs_ids.toml"
BIB_PATH = PROJECT_ROOT / "paper" / "refs.bib"

REQUEST_TIMEOUT = 20  # seconds
USER_AGENT = "build-bib/1.0 (set your email here)"  # arXiv / Crossref appreciate a contact
INTER_REQUEST_DELAY = 4.5  # arxiv asks for >=3s between requests; be safe
MAX_RETRIES = 5
BACKOFF_BASE = 30.0  # seconds; arXiv rate-limit can persist for ~30-60s


@dataclass
class CiteSpec:
    key: str
    claim: str | None
    arxiv: str | None = None
    doi: str | None = None
    acl: str | None = None
    manual: bool = False
    entry: str | None = None
    skip_authors: list[str] | None = None

    def source(self) -> str:
        if self.acl:
            return f"acl:{self.acl}"
        if self.doi:
            return f"doi:{self.doi}"
        if self.arxiv:
            return f"arxiv:{self.arxiv}"
        if self.manual:
            return "manual"
        return "<none>"


def load_specs() -> list[CiteSpec]:
    if not TOML_PATH.exists():
        sys.exit(f"ERROR: {TOML_PATH} not found")
    with TOML_PATH.open("rb") as fh:
        data = tomllib.load(fh)
    cites = data.get("cite", [])
    if not cites:
        sys.exit(f"ERROR: {TOML_PATH} has no [[cite]] entries")
    specs: list[CiteSpec] = []
    seen_keys: set[str] = set()
    for i, entry in enumerate(cites):
        key = entry.get("key")
        if not key:
            sys.exit(f"ERROR: entry #{i} in {TOML_PATH.name} has no key")
        if key in seen_keys:
            sys.exit(f"ERROR: duplicate key {key!r} in {TOML_PATH.name}")
        seen_keys.add(key)
        specs.append(
            CiteSpec(
                key=key,
                claim=entry.get("claim"),
                arxiv=entry.get("arxiv"),
                doi=entry.get("doi"),
                acl=entry.get("acl"),
                manual=bool(entry.get("manual", False)),
                entry=entry.get("entry"),
                skip_authors=entry.get("skip_authors"),
            )
        )
        has_identifier = any([specs[-1].arxiv, specs[-1].doi, specs[-1].acl, specs[-1].manual])
        if not has_identifier:
            sys.exit(f"ERROR: key {key!r} has no identifier (arxiv/doi/acl) and is not manual")
        if specs[-1].manual and not specs[-1].entry:
            sys.exit(f"ERROR: key {key!r} is manual but has no `entry` block")
    return specs


# ---------------------------------------------------------------------------
# Resolvers
# ---------------------------------------------------------------------------


def _get(url: str, headers: dict[str, str] | None = None) -> str:
    merged = {"User-Agent": USER_AGENT}
    if headers:
        merged.update(headers)
    for attempt in range(MAX_RETRIES):
        resp = requests.get(url, headers=merged, timeout=REQUEST_TIMEOUT,
                            allow_redirects=True)
        if resp.ok:
            return resp.text
        # arxiv often returns 429 — exponential backoff and retry
        if resp.status_code in (429, 503, 502, 504):
            sleep = BACKOFF_BASE * (2 ** attempt)
            print(f"    HTTP {resp.status_code}; backing off {sleep:.0f}s "
                  f"(attempt {attempt + 1}/{MAX_RETRIES})")
            time.sleep(sleep)
            continue
        raise RuntimeError(f"GET {url} -> HTTP {resp.status_code}: "
                           f"{resp.text[:200]}")
    raise RuntimeError(f"GET {url} -> exhausted {MAX_RETRIES} retries")


def resolve_acl(spec: CiteSpec) -> str:
    assert spec.acl
    url = f"https://aclanthology.org/{spec.acl}.bib"
    raw = _get(url)
    # ACL Anthology sometimes uses its own citation key. Rewrite to our key.
    bib = _rekey_entry(raw, spec.key)
    return _annotate(bib, spec)


def resolve_doi(spec: CiteSpec) -> str:
    assert spec.doi
    url = f"https://doi.org/{urllib.parse.quote(spec.doi, safe='/')}"
    raw = _get(url, headers={"Accept": "application/x-bibtex"})
    bib = _rekey_entry(raw, spec.key)
    return _annotate(bib, spec)


ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def resolve_arxiv(spec: CiteSpec) -> str:
    assert spec.arxiv
    url = f"https://export.arxiv.org/api/query?id_list={urllib.parse.quote(spec.arxiv)}"
    raw = _get(url)
    root = ET.fromstring(raw)
    entries = root.findall("atom:entry", ARXIV_NS)
    if not entries:
        raise RuntimeError(f"arXiv returned no entries for {spec.arxiv}")
    entry = entries[0]
    title_el = entry.find("atom:title", ARXIV_NS)
    published_el = entry.find("atom:published", ARXIV_NS)
    doi_el = entry.find("arxiv:doi", ARXIV_NS)
    category_el = entry.find("arxiv:primary_category", ARXIV_NS)
    id_el = entry.find("atom:id", ARXIV_NS)
    if title_el is None or title_el.text is None:
        raise RuntimeError(f"arXiv entry for {spec.arxiv} has no title")
    if published_el is None or published_el.text is None:
        raise RuntimeError(f"arXiv entry for {spec.arxiv} has no published date")
    skip = set(spec.skip_authors or [])
    authors = []
    for author in entry.findall("atom:author", ARXIV_NS):
        name_el = author.find("atom:name", ARXIV_NS)
        if name_el is None or not name_el.text:
            raise RuntimeError(f"arXiv entry for {spec.arxiv} has an author without a name")
        name = name_el.text.strip()
        if not re.search(r"[A-Za-z]", name):
            # Artefacts from "corporate author" formatting (arXiv returns the
            # bare ':' between a team name and the author list as its own <author>).
            continue
        if name in skip:
            continue
        if " " not in name:
            # Single-token names (e.g. "Qwen") are corporate/team authors.
            # Brace them so BibTeX treats them as one atomic name and does
            # not try to parse a `First Last` split.
            name = f"{{{name}}}"
        authors.append(name)
    if not authors:
        raise RuntimeError(f"arXiv entry for {spec.arxiv} has no authors")

    title = " ".join(title_el.text.split())  # collapse whitespace
    year = published_el.text[:4]
    abs_url = id_el.text.strip() if id_el is not None and id_el.text else f"https://arxiv.org/abs/{spec.arxiv}"
    primary_class = category_el.get("term") if category_el is not None else None
    doi = doi_el.text.strip() if doi_el is not None and doi_el.text else None

    fields = [
        ("author", " and ".join(authors)),
        ("title", title),
        ("year", year),
        ("eprint", spec.arxiv),
        ("archivePrefix", "arXiv"),
    ]
    if primary_class:
        fields.append(("primaryClass", primary_class))
    if doi:
        fields.append(("doi", doi))
    fields.append(("url", abs_url))

    body = ",\n  ".join(f"{k:<14} = {{{v}}}" for k, v in fields)
    bib = f"@misc{{{spec.key},\n  {body}\n}}\n"
    return _annotate(bib, spec)


def resolve_manual(spec: CiteSpec) -> str:
    assert spec.entry
    bib = spec.entry.strip()
    bib = _rekey_entry(bib, spec.key)
    if not bib.endswith("\n"):
        bib += "\n"
    return _annotate(bib, spec)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


_ENTRY_HEAD_RE = re.compile(r"(@\w+\s*\{)\s*[^,\s]+\s*,", re.MULTILINE)


def _rekey_entry(bibtex: str, key: str) -> str:
    """Replace the BibTeX citation key with our canonical one.

    ACL and Crossref emit their own keys (e.g. `tevet-berant-2021-evaluating`);
    we standardise on the keys used in the .tex file.
    """
    m = _ENTRY_HEAD_RE.search(bibtex)
    if not m:
        raise RuntimeError(f"could not locate @type{{KEY,...}} head in:\n{bibtex[:200]}")
    return _ENTRY_HEAD_RE.sub(lambda _: f"{m.group(1)}{key},", bibtex, count=1)


def _annotate(bibtex: str, spec: CiteSpec) -> str:
    """Prepend a comment naming the authoritative source."""
    lines = [f"% source: {spec.source()}"]
    if spec.claim:
        for claim_line in spec.claim.splitlines():
            lines.append(f"% claim : {claim_line}")
    banner = "\n".join(lines) + "\n"
    return banner + bibtex.strip() + "\n"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


RESOLVERS = [
    ("acl", resolve_acl),
    ("doi", resolve_doi),
    ("arxiv", resolve_arxiv),
]


def resolve(spec: CiteSpec, offline_manual_only: bool) -> str:
    if spec.manual:
        return resolve_manual(spec)
    if offline_manual_only:
        raise RuntimeError(f"{spec.key}: --offline-manual-only set but entry needs network")
    for attr, fn in RESOLVERS:
        if getattr(spec, attr):
            return fn(spec)
    raise RuntimeError(f"{spec.key}: no resolver matched (should be unreachable)")


HEADER = """\
% paper/refs.bib -- GENERATED FILE, DO NOT EDIT BY HAND.
%
% Regenerate with:  uv run python scripts/build_bib.py
% Source of truth:  paper/refs_ids.toml
% See also:         paper/CITATIONS.md
%
% Each entry is preceded by a `% source:` line naming the authoritative
% identifier (arXiv ID, DOI, or ACL Anthology ID) and a `% claim:` line
% naming the specific claim our paper makes against this reference. If
% you find an entry whose author list, title, or year looks wrong, DO
% NOT edit refs.bib -- edit refs_ids.toml and re-run the script.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline-manual-only", action="store_true")
    args = parser.parse_args()

    specs = load_specs()
    print(f"Building {BIB_PATH.relative_to(PROJECT_ROOT)} from {len(specs)} entries...")

    blocks: list[str] = [HEADER]
    for i, spec in enumerate(specs):
        src = spec.source()
        print(f"  [{i+1:2d}/{len(specs)}] {spec.key:<30s} <- {src}")
        try:
            block = resolve(spec, offline_manual_only=args.offline_manual_only)
        except Exception as e:
            print(f"ERROR resolving {spec.key} ({src}): {e}", file=sys.stderr)
            return 2
        blocks.append(block)
        if not spec.manual:
            time.sleep(INTER_REQUEST_DELAY)

    output = "\n".join(b.rstrip() + "\n" for b in blocks)

    # Write atomically.
    BIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".refs.bib.", dir=BIB_PATH.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(output)
        os.replace(tmp_name, BIB_PATH)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise

    print(f"Wrote {BIB_PATH.relative_to(PROJECT_ROOT)} ({len(output):,} bytes).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
