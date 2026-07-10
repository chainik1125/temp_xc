"""Render the program-level B×A report from the canonical leaderboard.

    .venv/bin/python -m experiments.explorations.synthetic.render_report

Reads ``results/leaderboard.jsonl``, builds the two fairness-convention matrices
(rows = ``(bench, latent-axis)``, cols = architecture) at the canonical operating
point declared in ``registry.OP``, and fills the ``<!-- AUTO:* -->`` blocks of
``REPORT.md`` plus ``results/program_stats.json``. Idempotent; no hand-typed
numbers. Holes (missing grid cells) render as ``—``; a loose realized-L0 match is
suffixed ``*``. Run the uniform re-grid (see the RunPod briefing) to fill holes.
"""

from __future__ import annotations

from pathlib import Path

from explorations.synthetic import report
from explorations.synthetic.record import populate

from . import registry as reg


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    leaderboard = root / reg.LEADERBOARD_REL
    report_md = root / reg.REPORT_REL
    stats_path = root / reg.STATS_REL

    cells = report.load_program_rows(leaderboard, reg.BENCHES)
    groups = report.group_cells(cells, primary_ds_only=True, benches=reg.BENCHES)

    # The per-token matched recovery matrix, dual-capacity {F, F//2} cells.
    mtx_md, mtx_stats = report.build_matrix(
        groups, reg.BENCHES, reg.ARCHS, reg.capacities, op=reg.OP)

    # Companion panels (capability gate): reconstruction NMSE + content recovery,
    # per benchmark (A×B), from the same matched groups.
    panels = {c.metric: report.build_panel(groups, reg.BENCHES, reg.ARCHS,
                                            reg.capacities, c.metric, op=reg.OP)
              for c in reg.COMPANIONS}
    cov_md = report.coverage(groups, reg.BENCHES, reg.ARCHS, op=reg.OP)

    caps = ", ".join(report._cap_labels(reg.OP))
    cap_lines = "\n".join(
        f"- **{b.name}**: d_sae ∈ {{{', '.join(str(c) for c in reg.capacities(b))}}} "
        f"(F={b.F}" + (f"; {b.F_note}" if b.F_note else "") + ")"
        for b in reg.BENCHES)
    op_md = (
        f"Per-token matched: window **T = {reg.OP.T_can}** (token archs T=1), "
        f"matched to **B\\* = {reg.OP.B_star:g}** atoms/token (nearest realized "
        f"`l0_per_token`; loose match >{reg.OP.l0_tol:g} marked `*`). Each cell is "
        f"normalized recovery `mean` over seeds, `[chance=0, oracle=1]`, shown at "
        f"**{caps}** (`boundary / deep-scarce`):\n\n{cap_lines}")

    blocks = {"operating_point": op_md, "matrix_pertoken": mtx_md, "coverage": cov_md}
    blocks.update({f"panel_{m}": md for m, (md, _st) in panels.items()})
    populate(report_md, blocks)

    report.write_stats(stats_path, {
        "source": reg.LEADERBOARD_REL,
        "n_cells": len(cells), "n_groups": len(groups),
        "operating_point": {"T_can": reg.OP.T_can, "B_star": reg.OP.B_star,
                            "capacity_fracs": list(reg.OP.capacity_fracs),
                            "l0_tol": reg.OP.l0_tol},
        "benches": [b.name for b in reg.BENCHES],
        "archs": [a.name for a in reg.ARCHS],
    }, {"recovery": mtx_stats,
        "panels": {m: st for m, (_md, st) in panels.items()}}, root)

    n_filled = sum(1 for v in mtx_stats.values()
                   if v and any(c is not None for c in v.values()))
    print(f"[report] {len(cells)} cells → {len(groups)} groups; "
          f"per-token matrix rows filled {n_filled}/{len(mtx_stats)}")
    print(f"[report] -> {report_md.relative_to(root)}")


if __name__ == "__main__":
    main()
