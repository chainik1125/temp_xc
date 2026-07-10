"""Program-level B×A report: cross-bench per-token matrix builder over realized L0.

The per-bench pipeline (``record.py``) renders one benchmark's frontier. This
module renders the **whole program** as ONE matrix — rows = ``(bench, latent
-axis)``, cols = architecture — at a single canonical operating point. Every
number comes from ``results/leaderboard.jsonl``; nothing is hand-typed.

The comparability problem and its resolution (see the registry docstring):

- **Cross-bench metric.** Each recovery metric is pre-normalized to
  ``[chance = 0, oracle = 1]`` by its evaluator, so a cell is one comparable
  scalar. Dual-latent benches give one row per axis.
- **Cross-arch budget.** Architectures are matched on **per-token sparsity** —
  the *realized* ``l0_per_token`` (measured on the shared eval windows, never the
  nominal ``k_pos`` knob — they diverge). This is the controlled comparison:
  equal total atoms per span, so only decode structure varies. (``l0_per_window``
  is also recorded, purely as a diagnostic — we do NOT match on it.)

Canonical cell: each capacity in ``{F, F//2}`` (per bench), window ``T = T_can``
(token archs are ``T=1``), the grid group whose mean realized ``l0_per_token``
sits nearest ``B*``, aggregated ``mean ± std`` over seeds. The full ``d_sae`` /
``T`` frontiers stay in each bench's own section — this is the summary.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

# Reuse the per-bench pipeline's formatters so the reports read identically.
from .record import fmt, fmt_pm  # noqa: F401  (fmt_pm exported for callers)


def load_program_rows(leaderboard: Path, benches, *, trained_only: bool = True) -> list[dict]:
    """Flatten the leaderboard to the program's cells (all registered benches).

    Each cell: ``{bench, ds, arch, T, d_sae, k_pos, seed, l0_t, l0_w, m}`` where
    ``m`` is the metric dict, ``l0_t`` is the realized ``l0_per_token`` matching
    key (``nan`` for pre-increment rows — such cells can't be matched and drop
    out of the matrix). One pass over the file; a row is claimed by the first
    bench whose ``(datasources, protocol)`` it matches.
    """
    by_ds = {}
    for b in benches:
        for ds in b.datasources:
            by_ds[ds] = b
    cells = []
    for line in Path(leaderboard).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        b = by_ds.get(r.get("datasource"))
        if b is None or r.get("evaluator_protocol_version") != b.protocol:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke"):
            continue
        n_steps = int(r.get("training_cfg", {}).get("n_steps", 0))
        if trained_only and n_steps <= 0:
            continue
        ov = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        m = r["metrics"]
        cells.append({
            "bench": b.name, "ds": r["datasource"], "arch": r["arch"],
            "T": int(ov.get("T", 1)), "d_sae": int(ov.get("d_sae")),
            "k_pos": int(ec.get("k_pos", ov.get("k_pos", 1))), "seed": int(r["seed"]),
            "l0_t": float(m.get("l0_per_token", float("nan"))),
            "l0_w": float(m.get("l0_per_window", float("nan"))),
            "kind": "trained" if n_steps > 0 else "untrained", "m": m,
        })
    return cells


def group_cells(cells: list[dict], *, primary_ds_only: bool = True, benches=None) -> dict:
    """Group seeds sharing a knob → per-group realized L0 + per-metric stats.

    Key: ``(bench, arch, T, d_sae, k_pos)``. Value:
    ``{l0_t, l0_w (group means), n_seeds, metrics: {metric: (mean, std, n)}}``.
    With ``primary_ds_only`` (default) only each bench's primary datasource
    (``datasources[0]``) feeds the headline matrix; nulls/controls are excluded.
    """
    primary = None
    if primary_ds_only and benches is not None:
        primary = {b.name: b.datasources[0] for b in benches}
    buck = defaultdict(lambda: {"l0_t": [], "l0_w": [], "m": defaultdict(list)})
    for c in cells:
        if primary is not None and c["ds"] != primary.get(c["bench"]):
            continue
        key = (c["bench"], c["arch"], c["T"], c["d_sae"], c["k_pos"])
        g = buck[key]
        g["l0_t"].append(c["l0_t"])
        g["l0_w"].append(c["l0_w"])
        for k, v in c["m"].items():
            if v is not None and np.isfinite(v):
                g["m"][k].append(float(v))
    def _nanmean(xs):  # avoid the all-NaN RuntimeWarning (pre-increment rows)
        fin = [x for x in xs if np.isfinite(x)]
        return float(np.mean(fin)) if fin else float("nan")

    out = {}
    for key, g in buck.items():
        out[key] = {
            "l0_t": _nanmean(g["l0_t"]),
            "l0_w": _nanmean(g["l0_w"]),
            "n_seeds": len(g["l0_t"]),
            "metrics": {k: (float(np.mean(v)), float(np.std(v)), len(v))
                        for k, v in g["m"].items()},
        }
    return out


def matched_group(groups: dict, bench: str, arch, *, d_sae: int, T_can: int,
                  B_star: float):
    """The canonical group for one (bench, arch, ``d_sae``) cell, or ``None``.

    Restricts to the given ``d_sae`` and the arch's window (``1`` for token archs,
    ``T_can`` for windowed), then picks the group whose mean realized
    ``l0_per_token`` is nearest ``B*`` (per-token matching — the controlled
    comparison). Returns ``(key, group, deviation)`` where ``deviation =
    |realized l0_per_token − B*|`` (so the caller can flag a loose match), or
    ``None`` when no such group exists (rendered as ``—``).
    """
    T = 1 if not arch.windowed else T_can
    best = None
    for key, g in groups.items():
        (bn, an, Tk, d, _kp) = key
        if bn != bench or an != arch.name or Tk != T or d != d_sae:
            continue
        l0 = g["l0_t"]
        if not np.isfinite(l0):
            continue
        dev = abs(l0 - B_star)
        if best is None or dev < best[2]:
            best = (key, g, dev)
    return best


def _cap_labels(op) -> list[str]:
    fr = {1.0: "F", 0.5: "F/2", 0.25: "F/4", 2.0: "2F"}
    return [fr.get(f, f"{f:g}·F") for f in op.capacity_fracs]


def build_matrix(groups: dict, benches, archs, capacities_fn, *, op) -> tuple[str, dict]:
    """The per-token matched matrix (markdown) + stats — dual-capacity cells.

    Rows = ``(bench, latent-axis)``. A cell shows the axis metric's ``mean`` at
    each canonical capacity ``{F, F//2}`` (``capacities_fn(bench)``), joined by
    `` / `` (boundary / deep-scarce), each suffixed ``*`` on a loose realized-L0
    match. Missing → ``—``. Returns ``(markdown, stats)``; stats nests per
    capacity under ``bench/axis/arch``.
    """
    hdr = ("| bench · latent (DC/AC) | " + " | ".join(a.label for a in archs) + " |\n"
           "|---|" + "|".join("---" for _ in archs) + "|\n")
    stats: dict = {}
    body = ""
    for b in benches:
        caps = capacities_fn(b)
        for ax in b.axes:
            label = f"**{b.name}** · {ax.label} ({ax.kind})"
            cells_md = []
            for a in archs:
                parts, cell_stats = [], {}
                for d_sae in caps:
                    mg = matched_group(groups, b.name, a, d_sae=d_sae,
                                       T_can=op.T_can, B_star=op.B_star)
                    if mg is None:
                        parts.append("—")
                        cell_stats[d_sae] = None
                        continue
                    key, g, dev = mg
                    trip = g["metrics"].get(ax.metric, (float("nan"), float("nan"), 0))
                    loose = "*" if dev > op.l0_tol else ""
                    parts.append((fmt(trip) + loose) if trip[2] else "—")
                    cell_stats[d_sae] = {
                        "value": trip[0], "std": trip[1], "n_seeds": trip[2],
                        "k_pos": key[4], "T": key[2], "d_sae": key[3],
                        "realized_l0_token": g["l0_t"], "realized_l0_window": g["l0_w"],
                        "l0_deviation": dev, "loose": bool(dev > op.l0_tol),
                    }
                cells_md.append(" / ".join(parts))
                stats[f"{b.name}/{ax.key}/{a.name}"] = cell_stats
            body += f"| {label} | " + " | ".join(cells_md) + " |\n"
    return hdr + body.rstrip(), stats


def coverage(groups: dict, benches, archs, *, op) -> str:
    """A one-line-per-(bench) coverage note: which arch×T×d_sae cells exist.

    Renders the holes explicitly so a partial grid never reads as complete."""
    lines = []
    have = defaultdict(set)
    for (bn, an, Tk, d, kp) in groups:
        have[bn].add((an, Tk, d))
    for b in benches:
        n = len(have.get(b.name, ()))
        archs_seen = sorted({an for (an, _T, _d) in have.get(b.name, ())})
        lines.append(f"- **{b.name}** (F={b.F}): {n} (arch,T,d_sae) groups · "
                     f"archs: {', '.join(archs_seen) or '—'}")
    return "\n".join(lines)


def write_stats(stats_path: Path, base: dict, matrix: dict, root: Path) -> None:
    """Dump ``{**base, matrix: {bench/axis/arch: {d_sae: cell}}}`` as JSON."""
    stats_path = Path(stats_path)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps({**base, "matrix": matrix}, indent=2))
    print(f"[program-stats] -> {stats_path.relative_to(root)}")
