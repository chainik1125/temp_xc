#!/usr/bin/env python3
"""Fill in the Phase 1 reproduction results section of phase1_reproduce.md.

Reads:
  - frontier_grid.json   (from post_phase1_analyze.py — our reproduced numbers)
  - nura_v1_baseline.json (snapshotted from fra_proj/frontier_*.json)

Writes (append/replace the `## Reproduction results` section in):
  - docs/dmitry/c6_em/2026-05-07_em_repl/phase1_reproduce.md

Idempotent: re-running with newer JSON updates the section in place.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


SECTION_HEADER = "## Reproduction results"


def fmt(x):
    if x is None:
        return "—"
    if isinstance(x, float):
        if x != x:  # NaN
            return "NaN"
        return f"{x:.2f}"
    return str(x)


def build_results_section(ours: dict, nura: dict) -> str:
    """Return a markdown chunk to splice into phase1_reproduce.md."""
    lines = [
        SECTION_HEADER,
        "",
        "Auto-generated from `phase1_summary/frontier_grid.json` and `nura_v1_baseline.json`.",
        "",
        "| EM model | Condition | Nura v1 Δ | Ours v2 Δ | gap | Ours peak | Nura peak | n@coh≥70 |",
        "|----------|-----------|----------:|----------:|----:|----------:|----------:|---------:|",
    ]
    for em in ("medical", "finance", "sports"):
        nkey = f"{em}_v1_k1"
        nura_em = nura.get(nkey, {})
        our_em = ours.get(em, {}) or {}
        for method in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
            n = nura_em.get(method, {}) or {}
            o = our_em.get(method, {}) or {}
            nd = n.get("delta_align_coh70")
            od = o.get("delta_align_coh70")
            try:
                gap = od - nd if (isinstance(nd, (int, float)) and isinstance(od, (int, float))
                                  and nd == nd and od == od) else None
            except TypeError:
                gap = None
            lines.append(
                f"| {em} | {method} | {fmt(nd)} | {fmt(od)} | "
                f"{('+' + fmt(gap)) if gap is not None and gap >= 0 else fmt(gap)} | "
                f"{fmt(o.get('peak_alignment'))} | {fmt(n.get('peak_alignment'))} | "
                f"{fmt(o.get('n_points_coh70'))} |"
            )
    lines.append("")

    # Gate
    medical_qkov = (ours.get("medical", {}) or {}).get("qk_to_ov", {}) or {}
    nura_medical = (nura.get("medical_v1_k1", {}) or {}).get("qk_to_ov", {}) or {}
    od = medical_qkov.get("delta_align_coh70")
    nd = nura_medical.get("delta_align_coh70")
    if isinstance(od, (int, float)) and isinstance(nd, (int, float)) and od == od and nd == nd:
        gap = abs(od - nd)
        verdict = "PASS" if gap <= 5.0 else "FAIL"
        lines.append(f"### Phase 1 gate: medical QK→OV reproduces Nura v1 within ±5")
        lines.append("")
        lines.append(
            f"- ours = `{od:.2f}`, Nura v1 = `{nd:.2f}`, |gap| = `{gap:.2f}` → **{verdict}**"
        )
        lines.append("")
        if verdict == "PASS":
            lines.append("Auto-launch path: `bash scripts/launch_phase3_saes.sh GO=1`.")
        else:
            lines.append(
                "Auto-fill path: see `phase1_diagnostic.md` (root cause analysis)."
            )
    else:
        lines.append("### Phase 1 gate")
        lines.append("")
        lines.append("Insufficient data to evaluate gate (missing medical QK→OV result).")
    lines.append("")
    return "\n".join(lines)


def splice(doc: str, new_section: str) -> str:
    """Append-or-replace the SECTION_HEADER block at the end of `doc`."""
    pattern = re.compile(rf"^{re.escape(SECTION_HEADER)}.*?(?=^## |\Z)", re.S | re.M)
    if pattern.search(doc):
        return pattern.sub(new_section + "\n", doc)
    sep = "" if doc.endswith("\n") else "\n"
    return doc + sep + "\n" + new_section + "\n"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frontier-grid-json", required=True,
                   help="Output of post_phase1_analyze.py (e.g. phase1_summary/frontier_grid.json)")
    p.add_argument("--nura-baseline-json", required=True,
                   help="Snapshot of Nura's v1 baselines (fra_proj/nura_v1_baseline.json)")
    p.add_argument("--doc", required=True,
                   help="Path to phase1_reproduce.md to update in-place")
    args = p.parse_args()

    ours = json.loads(Path(args.frontier_grid_json).read_text())
    nura = json.loads(Path(args.nura_baseline_json).read_text())
    doc_path = Path(args.doc)
    text = doc_path.read_text()

    section = build_results_section(ours, nura)
    new_text = splice(text, section)

    if new_text == text:
        print("[fill] no changes to phase1_reproduce.md")
        return 0

    doc_path.write_text(new_text)
    print(f"[fill] updated {doc_path} ({len(section)} chars in results section)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
