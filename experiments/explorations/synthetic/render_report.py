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

    pp_md, pp_stats = report.build_matrix(groups, reg.BENCHES, reg.ARCHS,
                                          convention=report.PER_POSITION, op=reg.OP)
    pw_md, pw_stats = report.build_matrix(groups, reg.BENCHES, reg.ARCHS,
                                          convention=report.PER_WINDOW, op=reg.OP)
    cov_md = report.coverage(groups, reg.BENCHES, reg.ARCHS, op=reg.OP)

    op_md = (f"Canonical cell: **d_sae = F** (per bench), window **T = {reg.OP.T_can}** "
             f"(token archs T=1), matched to **B\\* = {reg.OP.B_star:g}** atoms "
             f"(nearest realized L0; loose match >{reg.OP.l0_tol:g} marked `*`). "
             f"Cells are normalized recovery `mean` over seeds, `[chance=0, oracle=1]`.")

    populate(report_md, {
        "operating_point": op_md,
        "matrix_per_position": pp_md,
        "matrix_per_window": pw_md,
        "coverage": cov_md,
    })

    report.write_stats(stats_path, {
        "source": reg.LEADERBOARD_REL,
        "n_cells": len(cells), "n_groups": len(groups),
        "operating_point": {"T_can": reg.OP.T_can, "B_star": reg.OP.B_star,
                            "l0_tol": reg.OP.l0_tol},
        "benches": [b.name for b in reg.BENCHES],
        "archs": [a.name for a in reg.ARCHS],
    }, {report.PER_POSITION: pp_stats, report.PER_WINDOW: pw_stats}, root)

    n_filled = sum(1 for v in pp_stats.values() if v is not None)
    print(f"[report] {len(cells)} cells → {len(groups)} groups; "
          f"per-position matrix filled {n_filled}/{len(pp_stats)} cells")
    print(f"[report] -> {report_md.relative_to(root)}")


if __name__ == "__main__":
    main()
