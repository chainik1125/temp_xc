"""Single-source record pipeline: leaderboard → aggregate → AUTO blocks + stats.

The canonical leaderboard (``results/leaderboard.jsonl``) is the one
code-version-stamped source. A bench renderer:

1. :func:`load_rows` — filter the leaderboard to the bench's rows (datasource(s)
   + protocol + optional ``n_steps`` set), projecting each to a flat cell.
2. :func:`aggregate` — group cells by a key tuple → ``{metric: (mean, std, n)}``
   over seeds.
3. build its ``AUTO`` block strings (headline + tables — :func:`frontier_table`
   covers the common arch×capacity table) and :func:`populate` them into
   ``bench_record.md``.
4. :func:`write_stats` — dump the machine-readable aggregate.

Nothing is hand-typed; re-running rebuilds every number from the leaderboard.
The functions are byte-for-byte faithful to the per-bench copies they replace
(the refactor's acceptance gate is zero numeric drift).
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_rows(leaderboard: Path, datasources, protocol: str, *,
              n_steps_keep=None, deprecated_archs=frozenset(),
              with_ds: bool = False) -> list[dict]:
    """Filtered, flattened leaderboard cells for one bench.

    - ``datasources``: a datasource name or an iterable of names to keep.
    - ``protocol``: required ``evaluator_protocol_version`` (drops stale rows).
    - ``n_steps_keep``: if given, a set/tuple of allowed ``training_cfg.n_steps``
      (e.g. ``{0, 30000}`` drops smoke-length runs); ``None`` keeps any.
    - ``deprecated_archs``: arch names to skip.
    - ``with_ds``: also record ``ds`` (the datasource) on each cell + as a key
      field (for multi-datasource benches).

    Each cell: ``{ds?, arch, T, d_sae, k_pos, seed, kind, m}`` where
    ``kind = "trained" if n_steps>0 else "untrained"``.
    """
    ds_set = {datasources} if isinstance(datasources, str) else set(datasources)
    rows = []
    for line in Path(leaderboard).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("datasource") not in ds_set:
            continue
        if r.get("evaluator_protocol_version") != protocol:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke"):
            continue
        if r.get("arch") in deprecated_archs:
            continue
        n_steps = int(r.get("training_cfg", {}).get("n_steps", 0))
        if n_steps_keep is not None and n_steps not in n_steps_keep:
            continue
        ov = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        cell = {
            "arch": r["arch"], "T": int(ov.get("T", 1)), "d_sae": int(ov.get("d_sae")),
            "k_pos": int(ec.get("k_pos", ov.get("k_pos", 1))), "seed": int(r["seed"]),
            "kind": "trained" if n_steps > 0 else "untrained", "m": r["metrics"],
        }
        if with_ds:
            cell = {"ds": r["datasource"], **cell}
        rows.append(cell)
    return rows


def aggregate(rows: list[dict], key_fields) -> dict:
    """Group cells by ``tuple(cell[f] for f in key_fields)`` → per-metric stats.

    Returns ``{key: {metric: (mean, std, n)}}`` over the finite values, in
    leaderboard-encounter order (so the serialized stats JSON is stable).
    """
    buck = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = tuple(r[f] for f in key_fields)
        for m, v in r["m"].items():
            if v is not None and np.isfinite(v):
                buck[key][m].append(float(v))
    return {k: {m: (float(np.mean(vs)), float(np.std(vs)), len(vs)) for m, vs in d.items()}
            for k, d in buck.items()}


def get(agg: dict, key: tuple, metric: str, default=(float("nan"), float("nan"), 0)):
    """``(mean, std, n)`` for ``agg[key][metric]``, or ``default`` if absent."""
    c = agg.get(key)
    return c[metric] if c and metric in c else default


def fmt(t, dec: int = 3) -> str:
    """Format a ``(mean, std, n)`` triple as the mean, or ``—`` if ``n==0``."""
    m, s, n = t
    return "—" if not n else f"{m:.{dec}f}"


def fmt_pm(t) -> str:
    """Format a ``(mean, std, n)`` triple as ``mean ±std``, or ``—`` if ``n==0``."""
    m, s, n = t
    return "—" if not n else f"{m:.3f} ±{s:.3f}"


def frontier_table(arch_t, d_saes, value_fn, label_fn, *, bold_pred=None,
                   dec: int = 3) -> str:
    """Markdown table: one row per ``(arch, T)``, one column per ``d_sae``.

    ``value_fn(arch, T, d) -> (mean, std, n)``; cells are formatted with
    :func:`fmt`. ``bold_pred(arch, T)`` (default: none) bolds the row label.
    """
    h = ("| arch / T | " + " | ".join(f"d={d}" for d in d_saes) + " |\n"
         "|---|" + "|".join("---" for _ in d_saes) + "|\n")
    for arch, T in arch_t:
        cells = " | ".join(fmt(value_fn(arch, T, d), dec) for d in d_saes)
        name = label_fn(arch, T)
        if bold_pred is not None and bold_pred(arch, T):
            name = f"**{name}**"
        h += f"| {name} | {cells} |\n"
    return h.rstrip()


def populate(record_path: Path, blocks: dict) -> None:
    """Fill each ``<!-- BEGIN AUTO:tag --> … <!-- END AUTO:tag -->`` block.

    Idempotent: only the region between the markers is replaced; hand-written
    prose is untouched. Warns (does not fail) on a missing marker.
    """
    record_path = Path(record_path)
    if not record_path.exists():
        print(f"[warn] {record_path} missing — populate skipped")
        return
    txt = record_path.read_text()
    filled = 0
    for tag, content in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN AUTO:{tag} -->).*?(<!-- END AUTO:{tag} -->)", re.DOTALL)
        if not pat.search(txt):
            print(f"[warn] AUTO:{tag} marker not found in {record_path.name}")
            continue
        txt = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", txt)
        filled += 1
    record_path.write_text(txt)
    print(f"[record] populated {filled}/{len(blocks)} AUTO block(s) in {record_path.name}")


def flatten_agg(agg: dict, key_fn) -> dict:
    """``{key_fn(key_tuple): (mean,std,n) dict}`` — the serializable ``agg`` map."""
    return {key_fn(k): v for k, v in agg.items()}


def write_stats(stats_path: Path, base: dict, agg: dict, key_fn, root: Path) -> None:
    """Write ``{**base, "agg": flatten_agg(agg, key_fn)}`` as indented JSON.

    ``base`` carries the bench's scalar context (source, n_cells, ceilings, …) in
    its own insertion order; ``agg`` is appended last, exactly as the per-bench
    renderers serialized it.
    """
    stats_path = Path(stats_path)
    data = {**base, "agg": flatten_agg(agg, key_fn)}
    stats_path.write_text(json.dumps(data, indent=2))
    print(f"[stats] -> {stats_path.relative_to(root)}")
