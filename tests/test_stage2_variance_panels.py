"""Pre-flight of `support_stats/stage2_variance.py` against the Stage-2
panel row shapes (panel-support-audit item 1).

The oprate (runpod-d) and fineweb (runpod-e) panels emit a row population
the λ̂ panel never did: every row carries the `lambda_probe_v2` eval_cfg
flag AND both probe column sets (paired layout), and TXC-post runs at
per-T nominal k_pos = 8·T from the start. The fineweb replication cells
(gpt2 / llama31) additionally exist at only two window-T values. These
tests construct those populations as synthetic fixtures and pin the
harness's behaviour on each: select the right population, never mix
v1/v2 columns, never emit a trend from two points, degrade with stated
reasons, and keep the committed λ̂ receipts byte-identical.
"""

import json
import math
from pathlib import Path

import pytest

from experiments.explorations.task_hunt.support_stats import (
    stage2_variance as sv)

ROOT = Path(__file__).resolve().parents[1]

PANEL = ("batchtopk_sae", "tsae", "stacked_batchtopk",
         "txc_batchtopk_pre", "txc_batchtopk_post")
REPLICATION = ("batchtopk_sae", "tsae", "txc_batchtopk_pre")
TOKEN_ARCHS = {"batchtopk_sae", "tsae"}
SEEDS = (1, 2, 42)
V2_OFFSET = 0.5
BASE = {"batchtopk_sae": 0.10, "tsae": 0.15, "stacked_batchtopk": 0.20,
        "txc_batchtopk_pre": 0.25, "txc_batchtopk_post": 0.22}


def v1_val(arch, T, seed, kind):
    """Deterministic planted v1 metric; rises in T, varies by seed."""
    s = {1: 0, 2: 1, 42: 2}.get(seed, seed)
    v = BASE[arch] + 0.03 * math.log2(T) + 0.004 * s + 0.002 * s * s
    if kind == "untrained":
        v -= 0.08
    return round(v, 6)


def lb_row(ds, arch, T, k_pos, seed, kind, flagged=True):
    v1 = v1_val(arch, T, seed, kind)
    metrics = {"lambda_recovery": v1, "lambda_r2": v1,
               "lambda_chance": 0.01, "l0_per_token": 8.0}
    eval_cfg = {"smoke": False, "k_pos": k_pos, "eval_window_L": 32}
    if flagged:
        eval_cfg.update({"lambda_probe_v2": True, "lambda_v2_probe": "ridge",
                         "lambda_v2_n_windows": 8192,
                         "lambda_v2_split": "trace"})
        metrics.update({"lambda_recovery_v2": v1 + V2_OFFSET,
                        "lambda_r2_v2": v1 + V2_OFFSET,
                        "lambda_chance_v2": 0.02})
    return {"datasource": ds, "arch": arch, "seed": seed,
            "training_cfg": {
                "n_steps": 0 if kind == "untrained" else 8000,
                "arch_hparams_override": {"k_pos": k_pos, "d_sae": 1152,
                                          "T": T}},
            "eval_cfg": eval_cfg, "metrics": metrics}


def cc_row(lb):
    ov = lb["training_cfg"]["arch_hparams_override"]
    return {"ds": lb["datasource"], "arch": lb["arch"], "T": ov["T"],
            "d_sae": ov["d_sae"], "k_pos": ov["k_pos"], "seed": lb["seed"],
            "n_steps": lb["training_cfg"]["n_steps"],
            "kind": ("untrained" if lb["training_cfg"]["n_steps"] == 0
                     else "trained"),
            "metrics": lb["metrics"], "ok": True}


def build_panel(ds, archs=PANEL, window_ts=(2, 4, 8, 16), flagged=True,
                post_times_t=True):
    """The row population a Stage-2 panel emits (leaderboard + results
    JSON), per `qrate_fineweb/run_stage2.py`: token archs at T=1, window
    archs on the T ladder, post at k_pos = 8·T (both kinds)."""
    rows = []
    for arch in archs:
        for T in ((1,) if arch in TOKEN_ARCHS else window_ts):
            k = 8 * T if (arch == "txc_batchtopk_post" and post_times_t) else 8
            for seed in SEEDS:
                for kind in ("trained", "untrained"):
                    rows.append(lb_row(ds, arch, T, k, seed, kind, flagged))
    return rows


def write_fixture(tmp_path, lb_rows, cc_rows, tag=""):
    lb = tmp_path / f"leaderboard{tag}.jsonl"
    lb.write_text("\n".join(json.dumps(r) for r in lb_rows) + "\n")
    cc = tmp_path / f"stage2{tag}.json"
    cc.write_text(json.dumps(cc_rows))
    return lb, cc


def run(tmp_path, lb, cc, ds, *extra):
    out_dir = tmp_path / "out"
    sv.main(["--ds", ds, "--leaderboard", str(lb),
             "--crosscheck-json", str(cc), "--out-dir", str(out_dir),
             "--out-prefix", "t", *extra])
    return (json.loads((out_dir / "t.json").read_text()),
            (out_dir / "t.md").read_text())


# ---------------------------------------------------------------- paired
# layout (oprate / fineweb primary): 84 cells, post at 8·T, both columns

def test_paired_panel_v1_selects_all_arms(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    out, md = run(tmp_path, lb, cc, ds, "--probe", "v1",
                  "--post-k-rule", "times-T")
    assert out["source"]["leaderboard_rows"] == 84
    assert out["source"]["row_layout"] == "paired"
    # post arm present at every ladder point, not silently dropped
    for T in (2, 4, 8, 16):
        assert f"txc_batchtopk_post/T{T}" in out["per_seed"]["trained"]
    # v1 columns, never v2 (the planted offset would show)
    got = out["per_seed"]["trained"]["txc_batchtopk_pre/T8"]["42"]
    assert got == pytest.approx(v1_val("txc_batchtopk_pre", 8, 42, "trained"))
    got_post = out["per_seed"]["untrained"]["txc_batchtopk_post/T16"]["1"]
    assert got_post == pytest.approx(
        v1_val("txc_batchtopk_post", 16, 1, "untrained"))
    # full ladder -> both frozen trends + secondary present
    assert "txc_pre_trained_2to8" in out["trend"]
    assert "txc_pre_trained_2to16_secondary" in out["trend"]
    assert "T = 8" in out["power"]["recommendation"]["criterion"]
    assert "row layout paired" in md


def test_paired_panel_v2_columns_not_mixed(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    out, _ = run(tmp_path, lb, cc, ds, "--probe", "v2",
                 "--post-k-rule", "times-T")
    assert out["metric"] == "lambda_recovery_v2"
    got = out["per_seed"]["trained"]["txc_batchtopk_pre/T8"]["42"]
    assert got == pytest.approx(
        v1_val("txc_batchtopk_pre", 8, 42, "trained") + V2_OFFSET)


def test_paired_panel_tolerates_seed_topup_rows(tmp_path):
    # extra trained-only seeds (the λ̂ top-up pattern) must be excluded
    # from the receipts and the exact cross-check, not abort them
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    extra = [lb_row(ds, "txc_batchtopk_pre", T, 8, seed, "trained")
             for T in (4, 8) for seed in (3, 4, 5)]
    for r in extra:
        r["metrics"]["lambda_recovery"] = 0.999   # never reported
    lb, cc = write_fixture(tmp_path, rows + extra,
                           [cc_row(r) for r in rows])
    out, _ = run(tmp_path, lb, cc, ds, "--probe", "v1",
                 "--post-k-rule", "times-T")
    assert out["source"]["leaderboard_rows"] == 84


def test_paired_panel_wrong_post_rule_aborts_with_hint(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    with pytest.raises(SystemExit, match="times-T"):
        run(tmp_path, lb, cc, ds, "--probe", "v1")


def test_paired_panel_v1_forced_split_layout_is_loud(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    with pytest.raises(SystemExit, match="0 leaderboard rows"):
        run(tmp_path, lb, cc, ds, "--probe", "v1",
            "--post-k-rule", "times-T", "--row-layout", "split")


# ------------------------------------------------- replication cells
# (gpt2 / llama31): TXC-pre at two T values only + token references

def test_replication_two_T_degrades_honestly(tmp_path):
    ds = "fineweb_punctint_q_gpt2_l7"
    rows = build_panel(ds, archs=REPLICATION, window_ts=(8, 16))
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    out, md = run(tmp_path, lb, cc, ds, "--probe", "v1",
                  "--post-k-rule", "times-T")
    assert out["source"]["leaderboard_rows"] == 24
    # cells reported, trend skipped with the reason stated
    assert "txc_batchtopk_pre/T16" in out["per_seed"]["trained"]
    assert list(out["trend"]) == ["txc_pre_trend"]
    assert "no within-seed permutation resolution" \
        in out["trend"]["txc_pre_trend"]["skipped"]
    assert "txc_pre_trend" in md
    # paired diffs exist at exactly the two T values
    assert sorted(out["paired"]["txc_pre_minus_tsae"]["by_T"]) == \
        ["T16", "T8"]
    # power keys on what exists: T8 present, T4 never fabricated
    assert list(out["power"]["txc_pre_minus_tsae"]) == ["T8"]
    rec = out["power"]["recommendation"]
    assert "T = 8" in rec["criterion"]
    assert "T4_not_cheaply_boundable" not in rec


def test_replication_without_T8_keys_on_largest_T(tmp_path):
    ds = "fineweb_punctint_q_llama31_l14"
    rows = build_panel(ds, archs=REPLICATION, window_ts=(4, 16))
    lb, cc = write_fixture(tmp_path, rows, [cc_row(r) for r in rows])
    out, _ = run(tmp_path, lb, cc, ds, "--probe", "v1",
                 "--post-k-rule", "times-T")
    rec = out["power"]["recommendation"]
    assert "T = 16" in rec["criterion"]
    assert "largest available window T = 16" in rec["headline_T_note"]


# ------------------------------------------------------- split layout
# (the λ̂ panel semantics): v1 unflagged + v2 re-runs + 8·T amendment
# rows coexist; defaults must keep selecting the v1 panel exactly

def test_split_layout_populations_coexist(tmp_path):
    ds = "ward_mini_lambda"
    v1_rows = build_panel(ds, window_ts=(2, 4, 8), flagged=False,
                          post_times_t=False)
    amend = [lb_row(ds, "txc_batchtopk_post", T, 8 * T, seed, kind,
                    flagged=False)
             for T in (2, 4, 8) for seed in SEEDS
             for kind in ("trained", "untrained")]
    v2_rows = build_panel(ds, window_ts=(2, 4, 8), flagged=True,
                          post_times_t=False)
    lb, cc = write_fixture(tmp_path, v1_rows + amend + v2_rows,
                           [cc_row(r) for r in v1_rows])
    out, _ = run(tmp_path, lb, cc, ds, "--probe", "v1")
    assert out["source"]["row_layout"] == "split"
    assert out["source"]["leaderboard_rows"] == 66   # 11 cells × 6
    got = out["per_seed"]["trained"]["txc_batchtopk_pre/T4"]["2"]
    assert got == pytest.approx(v1_val("txc_batchtopk_pre", 4, 2, "trained"))
    # v2 re-run population selected by --probe v2, cross-checked v2-side
    _, cc2 = write_fixture(tmp_path, [], [cc_row(r) for r in v2_rows],
                           tag="_v2")
    out2, _ = run(tmp_path, lb, cc2, ds, "--probe", "v2")
    got2 = out2["per_seed"]["trained"]["txc_batchtopk_pre/T4"]["2"]
    assert got2 == pytest.approx(
        v1_val("txc_batchtopk_pre", 4, 2, "trained") + V2_OFFSET)


# ------------------------------------------------------------- guards

def test_duplicate_cell_aborts(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows + [rows[0]],
                           [cc_row(r) for r in rows])
    with pytest.raises(SystemExit, match="duplicate leaderboard cell"):
        run(tmp_path, lb, cc, ds, "--probe", "v1",
            "--post-k-rule", "times-T")


def test_incomplete_panel_is_a_clear_error(tmp_path):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    lb, cc = write_fixture(tmp_path, rows[:-1],
                           [cc_row(r) for r in rows[:-1]])
    with pytest.raises(SystemExit, match="incomplete panel population"):
        run(tmp_path, lb, cc, ds, "--probe", "v1",
            "--post-k-rule", "times-T")


def test_failed_crosscheck_rows_are_skipped_loudly(tmp_path, capsys):
    ds = "ward_oprate_case_l13"
    rows = build_panel(ds)
    cc_rows = [cc_row(r) for r in rows]
    cc_rows.append({"ds": ds, "arch": "tsae", "T": 1, "seed": 7,
                    "kind": "trained", "ok": False,
                    "error": "OOM"})       # no metrics, no leaderboard row
    lb, cc = write_fixture(tmp_path, rows, cc_rows)
    out, _ = run(tmp_path, lb, cc, ds, "--probe", "v1",
                 "--post-k-rule", "times-T")
    assert out["source"]["crosscheck_failed_rows_skipped"] == 1
    assert "skipping 1 failed" in capsys.readouterr().out


# ------------------------------------------------- committed receipts

def test_legacy_default_reproduces_committed_receipts(tmp_path):
    """The default invocation must reproduce the committed v1 receipts
    from today's leaderboard (which now includes the seed-top-up rows
    the --seeds filter exists for). Floats compare at rel 1e-12, not
    byte-identity: x86<->ARM reduction order drifts the last ulp of
    three r_between_arms values (~2e-16 rel; mac-local review
    2026-07-24) while every statistic of record is identical. Any
    real behavior change is orders of magnitude above this tolerance."""
    import json as _json
    import math as _math

    sv.main(["--out-dir", str(tmp_path)])
    src = ROOT / "experiments" / "explorations" / "task_hunt" / "support_stats"

    def _same(a, b, path=""):
        if isinstance(a, dict):
            assert isinstance(b, dict) and set(a) == set(b), path
            for k in a:
                _same(a[k], b[k], f"{path}.{k}")
        elif isinstance(a, list):
            assert isinstance(b, list) and len(a) == len(b), path
            for i, (u, v) in enumerate(zip(a, b)):
                _same(u, v, f"{path}[{i}]")
        elif isinstance(a, float) or isinstance(b, float):
            assert _math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-15), \
                f"{path}: {a!r} != {b!r}"
        else:
            assert a == b, f"{path}: {a!r} != {b!r}"

    _same(_json.loads((tmp_path / "stage2_variance.json").read_text()),
          _json.loads((src / "stage2_variance.json").read_text()))
    # The .md rounds for display, so byte-identity is platform-safe there.
    assert ((tmp_path / "stage2_variance.md").read_bytes()
            == (src / "stage2_variance.md").read_bytes())
