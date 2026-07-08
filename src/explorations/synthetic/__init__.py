"""Shared scaffolding for the synthetic-benchmark program.

The single-source *record pipeline* (leaderboard → figures + ``*_stats.json`` +
auto-filled ``<!-- AUTO:* -->`` blocks in ``bench_record.md``) was copy-pasted
per bench (``backtracking`` / ``changepoint`` / ``frequency``). It lives here
once, so each bench's ``run_grid.py`` / ``render_figs.py`` is a thin, config-only
driver that declares its arch list, capacity grid, datasources, and its
bench-specific figure/table specs, then calls these shared functions.

Three modules, split by concern:

- :mod:`explorations.synthetic.record` — read/filter/aggregate the canonical
  leaderboard, format cells, fill the named ``AUTO`` blocks, write the stats JSON.
- :mod:`explorations.synthetic.figs` — shared matplotlib style + save helpers +
  the frontier/curve plot primitives the benches share.
- :mod:`explorations.synthetic.grid` — enumerate cells and drive the canonical
  runner (``temp_bench.core.runner.run_experiment``) in a parallel pool.

The output contract is unchanged: these functions reproduce every published
``AUTO`` block and ``*_stats.json`` byte-for-byte from the unchanged leaderboard.
"""

from explorations.synthetic import figs, grid, record

__all__ = ["record", "figs", "grid"]
