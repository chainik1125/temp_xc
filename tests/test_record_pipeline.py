"""Unit tests for the shared synthetic-bench record pipeline
(:mod:`explorations.synthetic`).

The end-to-end acceptance test (every bench's AUTO blocks + ``*_stats.json``
reproduce byte-for-byte from the unchanged leaderboard) is run by hand at refactor
time. These pin the library functions in isolation so a future edit that would
have moved a published number breaks a test.
"""

from __future__ import annotations

import json
import math

from explorations.synthetic import design, figs, grid, record


def test_fmt_and_fmt_pm():
    assert record.fmt((0.1234, 0.01, 3)) == "0.123"
    assert record.fmt((0.1234, 0.01, 3), dec=2) == "0.12"
    assert record.fmt((float("nan"), float("nan"), 0)) == "—"
    assert record.fmt_pm((0.5, 0.02, 3)) == "0.500 ±0.020"
    assert record.fmt_pm((0.0, 0.0, 0)) == "—"


def test_load_rows_filters(tmp_path):
    lb = tmp_path / "leaderboard.jsonl"
    def row(ds, proto, smoke, n_steps, arch="a", seed=1, T=2, d=20, kpos=1, m=None):
        return json.dumps({
            "datasource": ds, "evaluator_protocol_version": proto, "arch": arch,
            "seed": seed, "metrics": m or {"x": 1.0},
            "eval_cfg": {"smoke": smoke, "k_pos": kpos},
            "training_cfg": {"n_steps": n_steps,
                             "arch_hparams_override": {"T": T, "d_sae": d}},
        })
    lb.write_text("\n".join([
        row("keep", "1.2.0", False, 100),       # kept
        row("other", "1.2.0", False, 100),      # wrong datasource
        row("keep", "1.1.0", False, 100),       # wrong protocol
        row("keep", "1.2.0", True, 100),        # smoke
        row("keep", "1.2.0", False, 5),         # dropped by n_steps_keep
        row("keep", "1.2.0", False, 0),         # untrained (kept)
    ]) + "\n")
    rows = record.load_rows(lb, "keep", "1.2.0", n_steps_keep={0, 100})
    assert len(rows) == 2
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["trained", "untrained"]
    assert all("ds" not in r for r in rows)
    # with_ds records the datasource + honours a datasource list
    rows2 = record.load_rows(lb, ["keep", "other"], "1.2.0", with_ds=True)
    assert {r["ds"] for r in rows2} == {"keep", "other"}


def test_aggregate_and_get():
    rows = [
        {"kind": "trained", "k_pos": 1, "arch": "a", "T": 2, "d_sae": 20, "m": {"x": 0.0}},
        {"kind": "trained", "k_pos": 1, "arch": "a", "T": 2, "d_sae": 20, "m": {"x": 1.0}},
        {"kind": "trained", "k_pos": 1, "arch": "b", "T": 4, "d_sae": 20, "m": {"x": 0.5}},
    ]
    agg = record.aggregate(rows, ("kind", "k_pos", "arch", "T", "d_sae"))
    mean, std, n = record.get(agg, ("trained", 1, "a", 2, 20), "x")
    assert (mean, n) == (0.5, 2) and abs(std - 0.5) < 1e-9
    # missing key / metric → default
    assert record.get(agg, ("trained", 1, "z", 2, 20), "x")[2] == 0
    assert math.isnan(record.get(agg, ("trained", 1, "a", 2, 20), "nope")[0])


def test_frontier_table():
    def value_fn(arch, T, d):
        return (0.5, 0.0, 1) if arch == "win" else (float("nan"), 0.0, 0)
    out = record.frontier_table(
        [("tok", 1), ("win", 2)], [8, 20], value_fn, lambda a, T: f"{a}-{T}",
        bold_pred=lambda a, T: a == "win")
    lines = out.splitlines()
    assert lines[0] == "| arch / T | d=8 | d=20 |"
    assert lines[1] == "|---|---|---|"
    assert lines[2] == "| tok-1 | — | — |"
    assert lines[3] == "| **win-2** | 0.500 | 0.500 |"


def test_populate_idempotent(tmp_path):
    rec = tmp_path / "bench_record.md"
    rec.write_text("intro\n<!-- BEGIN AUTO:tab -->\nOLD\n<!-- END AUTO:tab -->\nprose\n")
    record.populate(rec, {"tab": "NEW"})
    body = rec.read_text()
    assert "<!-- BEGIN AUTO:tab -->\nNEW\n<!-- END AUTO:tab -->" in body
    assert "OLD" not in body and "intro" in body and "prose" in body
    record.populate(rec, {"tab": "NEW"})  # idempotent
    assert rec.read_text() == body


def test_write_stats(tmp_path):
    agg = {("trained", 1, "a", 2, 20): {"x": (0.5, 0.1, 2)}}
    out = tmp_path / "s.json"
    record.write_stats(out, {"source": "lb", "n_cells": 3}, agg,
                       lambda k: f"{k[0]}|kpos{k[1]}|{k[2]}|T{k[3]}|d{k[4]}", tmp_path)
    data = json.loads(out.read_text())
    assert list(data) == ["source", "n_cells", "agg"]     # base order preserved, agg last
    assert data["agg"] == {"trained|kpos1|a|T2|d20": {"x": [0.5, 0.1, 2]}}


def test_grid_batch_size():
    assert grid.batch_size(1) == 1024
    assert grid.batch_size(2) == 512
    assert grid.batch_size(16) == 64


def test_save_fig_writes_variants(tmp_path):
    plt = figs.use_agg_style()
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    figs.save_fig(fig, tmp_path, "demo", plt)
    for ext in ("pdf", "png", "thumb.png"):
        assert (tmp_path / f"demo.{ext}").exists()


def test_design_uniform_cells():
    # capacities {F//2, F, 2F}; per-family dict constraint
    assert design.capacities(20) == [10, 20, 40]
    assert design.min_d_sae("pre", 4, 8) == 32 and design.min_d_sae("token", 4, 8) == 4
    assert design.min_d_sae("post", 8, 4) == 8 and design.min_d_sae("spectral", 2, 4) == 8
    # arch_t_list: 2 token (T=1) + 4 window x 3 T = 14
    assert len(design.arch_t_list()) == 14
    cells = design.uniform_cells("ds", 20, 30000, seeds=(1,))
    fam = dict(design.FAIR_BACKBONE)
    # every trained cell satisfies its family's dict constraint
    for c in cells:
        if c["kind"] == "trained":
            assert c["d_sae"] >= design.min_d_sae(fam[c["arch"]], c["k_pos"], c["T"])
    # untrained control: one per (arch,T) at the F anchor, k_pos=1
    unt = [c for c in cells if c["kind"] == "untrained"]
    assert len(unt) == 14 and all(c["d_sae"] == 20 and c["k_pos"] == 1 for c in unt)
    # d_saes override + untrained=False (the null / memo / band extras use this)
    only = design.uniform_cells("ds", 101, 6000, seeds=(1,), d_saes=[101],
                                k_pos_sweep=(1,), untrained=False)
    assert only and all(c["d_sae"] == 101 and c["kind"] == "trained" for c in only)
