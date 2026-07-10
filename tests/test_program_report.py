"""Program-level B×A report — matrix builder + realized-L0 matching logic.

Validates that the two fairness conventions pull the *correct grid cell* for each
architecture, keyed on realized L0 (not the nominal knob), and that holes render
as ``—``. No training: fabricates a ``groups`` dict shaped like
``report.group_cells`` output, exercising ``matched_group`` / ``build_matrix``.
"""

from __future__ import annotations

from explorations.synthetic import report
from experiments.explorations.synthetic import registry as reg


def _g(l0t, l0w, **metrics):
    """A fabricated group: realized L0 means + per-metric (mean, std, n) triples."""
    return {"l0_t": l0t, "l0_w": l0w, "n_seeds": 3,
            "metrics": {k: (v, 0.02, 3) for k, v in metrics.items()}}


def _changepoint_groups(F=20, T=4):
    """Realized-L0 behaviour mirrors the measured archs (see the L0 increment):
    token l0_t=l0_w=k_pos; pre l0_t≈k_pos, l0_w≈k_pos·T; post l0_w≈k_pos,
    l0_t≈k_pos/T."""
    return {
        ("changepoint", "batchtopk_sae", 1, F, 2): _g(2, 2, mode_recovery=0.95, tss_recovery=0.02),
        ("changepoint", "batchtopk_sae", 1, F, 4): _g(4, 4, mode_recovery=0.97, tss_recovery=0.02),
        ("changepoint", "txc_batchtopk_pre", T, F, 1): _g(1, 4, mode_recovery=0.66, tss_recovery=0.03),
        ("changepoint", "txc_batchtopk_pre", T, F, 4): _g(4, 16, mode_recovery=0.67, tss_recovery=0.04),
        ("changepoint", "txc_batchtopk_post", T, F, 4): _g(1, 4, mode_recovery=0.60, tss_recovery=0.66),
        ("changepoint", "txc_batchtopk_post", T, F, 16): _g(4, 16, mode_recovery=0.66, tss_recovery=0.52),
    }


def _archs():
    keep = ("batchtopk_sae", "txc_batchtopk_pre", "txc_batchtopk_post")
    return [a for a in reg.ARCHS if a.name in keep]


def _bench():
    return [b for b in reg.BENCHES if b.name == "changepoint"]


def test_per_position_matches_l0_per_token():
    """Per-position holds l0_per_token = B*: token→k_pos=4, pre→k_pos=4, post→k_pos=16."""
    groups = _changepoint_groups()
    _, st = report.build_matrix(groups, _bench(), _archs(),
                               convention=report.PER_POSITION, op=reg.OP)
    assert st["changepoint/mode/batchtopk_sae"]["k_pos"] == 4
    assert st["changepoint/mode/txc_batchtopk_pre"]["k_pos"] == 4      # l0_t=4 (k_win=16 grows)
    assert st["changepoint/mode/txc_batchtopk_post"]["k_pos"] == 16    # l0_t=k_pos/T=4
    for cell in ("batchtopk_sae", "txc_batchtopk_pre", "txc_batchtopk_post"):
        assert abs(st[f"changepoint/mode/{cell}"]["realized_l0_token"] - reg.OP.B_star) < 1e-6


def test_per_window_matches_l0_per_window():
    """Per-window holds l0_per_window = B*: window archs starved to 1/token."""
    groups = _changepoint_groups()
    _, st = report.build_matrix(groups, _bench(), _archs(),
                               convention=report.PER_WINDOW, op=reg.OP)
    assert st["changepoint/mode/batchtopk_sae"]["k_pos"] == 4          # token: T=1, unchanged
    assert st["changepoint/mode/txc_batchtopk_pre"]["k_pos"] == 1      # l0_w=4 → 1/token
    assert st["changepoint/mode/txc_batchtopk_post"]["k_pos"] == 4     # l0_w=4 → 1/token
    for cell in ("txc_batchtopk_pre", "txc_batchtopk_post"):
        s = st[f"changepoint/mode/{cell}"]
        assert abs(s["realized_l0_window"] - reg.OP.B_star) < 1e-6
        assert abs(s["realized_l0_token"] - reg.OP.B_star / reg.OP.T_can) < 1e-6


def test_conventions_pull_different_cells():
    """The whole point: a window arch's per-position and per-window cells differ."""
    groups = _changepoint_groups()
    _, pp = report.build_matrix(groups, _bench(), _archs(),
                               convention=report.PER_POSITION, op=reg.OP)
    _, pw = report.build_matrix(groups, _bench(), _archs(),
                               convention=report.PER_WINDOW, op=reg.OP)
    # post-squash: per-position pulls k_pos=16, per-window pulls k_pos=4 → different tss.
    assert pp["changepoint/tss/txc_batchtopk_post"]["k_pos"] == 16
    assert pw["changepoint/tss/txc_batchtopk_post"]["k_pos"] == 4
    assert pp["changepoint/tss/txc_batchtopk_post"]["value"] != pw["changepoint/tss/txc_batchtopk_post"]["value"]
    # token arch (T=1) is the stable reference: same cell in both.
    assert pp["changepoint/mode/batchtopk_sae"]["k_pos"] == pw["changepoint/mode/batchtopk_sae"]["k_pos"]


def test_missing_cell_renders_dash():
    """An arch with no group at (F, T_can) is a hole, not a crash."""
    groups = _changepoint_groups()
    md, st = report.build_matrix(groups, _bench(),
                                 [a for a in reg.ARCHS if a.name == "spectral_txc"],
                                 convention=report.PER_POSITION, op=reg.OP)
    assert st["changepoint/mode/spectral_txc"] is None
    assert "—" in md


def test_nan_l0_rows_excluded():
    """Pre-increment rows (realized L0 = nan) can't be matched → dropped."""
    groups = {("changepoint", "batchtopk_sae", 1, 20, 4):
              _g(float("nan"), float("nan"), mode_recovery=0.9)}
    mg = report.matched_group(groups, "changepoint",
                              _archs()[0], F=20, T_can=reg.OP.T_can,
                              convention=report.PER_POSITION, B_star=reg.OP.B_star)
    assert mg is None
