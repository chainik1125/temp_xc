"""Program-level B×A report — matrix builder + realized-L0 matching logic.

Validates that the per-token matrix pulls the *correct grid cell* for each
architecture (keyed on realized L0, not the nominal knob), that it renders a
dual-capacity cell at ``{F, F//2}``, and that holes render as ``—``. No training:
fabricates a ``groups`` dict shaped like ``report.group_cells`` output.
"""

from __future__ import annotations

from explorations.synthetic import report
from experiments.explorations.synthetic import registry as reg


def _g(l0t, l0w, **metrics):
    """A fabricated group: realized L0 means + per-metric (mean, std, n) triples."""
    return {"l0_t": l0t, "l0_w": l0w, "n_seeds": 3,
            "metrics": {k: (v, 0.02, 3) for k, v in metrics.items()}}


def _cp_groups():
    """changepoint groups at both capacities {F=20, F//2=10}, T=4 window.
    Realized L0 mirrors the measured archs: token l0_t=l0_w=k_pos; pre l0_t≈k_pos,
    l0_w≈k_pos·T; post l0_w≈k_pos, l0_t≈k_pos/T. B*=2 (reg.OP)."""
    g = {}
    T = 4
    for d in (20, 10):
        g[("changepoint", "batchtopk_sae", 1, d, 2)] = _g(2, 2, mode_recovery=0.90, tss_recovery=0.02)
        g[("changepoint", "batchtopk_sae", 1, d, 4)] = _g(4, 4, mode_recovery=0.95, tss_recovery=0.02)
        g[("changepoint", "txc_batchtopk_pre", T, d, 2)] = _g(2, 8, mode_recovery=0.66, tss_recovery=0.04)
        g[("changepoint", "txc_batchtopk_pre", T, d, 4)] = _g(4, 16, mode_recovery=0.67, tss_recovery=0.05)
        g[("changepoint", "txc_batchtopk_post", T, d, 8)] = _g(2, 8, mode_recovery=0.60, tss_recovery=0.62)
        g[("changepoint", "txc_batchtopk_post", T, d, 4)] = _g(1, 4, mode_recovery=0.55, tss_recovery=0.50)
    return g


def _archs():
    keep = ("batchtopk_sae", "txc_batchtopk_pre", "txc_batchtopk_post")
    return [a for a in reg.ARCHS if a.name in keep]


def _bench():
    return [b for b in reg.BENCHES if b.name == "changepoint"]


def test_pertoken_matches_l0_per_token():
    """Per-token holds l0_per_token = B*=2: token→k_pos=2, pre→k_pos=2, post→k_pos=8."""
    _, st = report.build_matrix(_cp_groups(), _bench(), _archs(), reg.capacities,
                                op=reg.OP)
    for d in (20, 10):   # both capacity slices resolve
        assert st["changepoint/mode/batchtopk_sae"][d]["k_pos"] == 2
        assert st["changepoint/mode/txc_batchtopk_pre"][d]["k_pos"] == 2   # l0_t=2
        assert st["changepoint/mode/txc_batchtopk_post"][d]["k_pos"] == 8  # l0_t=k_pos/T=2
        for cell in ("batchtopk_sae", "txc_batchtopk_pre", "txc_batchtopk_post"):
            assert abs(st[f"changepoint/mode/{cell}"][d]["realized_l0_token"] - reg.OP.B_star) < 1e-6


def test_dual_capacity_cell():
    """A cell shows both {F, F//2} values joined by ' / ', and stats carry both."""
    md, st = report.build_matrix(_cp_groups(), _bench(), _archs(), reg.capacities,
                                 op=reg.OP)
    assert set(st["changepoint/mode/batchtopk_sae"]) == {20, 10}
    # the markdown cell for a resolved arch has the two-capacity separator
    assert " / " in md


def test_missing_capacity_renders_dash():
    """An arch with no group at a capacity → that slot is None → '—' in the cell."""
    md, st = report.build_matrix(_cp_groups(), _bench(),
                                 [a for a in reg.ARCHS if a.name == "spectral_txc"],
                                 reg.capacities, op=reg.OP)
    cell = st["changepoint/mode/spectral_txc"]
    assert cell[20] is None and cell[10] is None
    assert "— / —" in md


def test_matched_group_keys_on_l0_per_token():
    """matched_group picks the group whose realized l0_per_token is nearest B*."""
    groups = _cp_groups()
    mg = report.matched_group(groups, "changepoint", _archs()[2],  # post
                              d_sae=20, T_can=reg.OP.T_can, B_star=reg.OP.B_star)
    assert mg[0][4] == 8    # k_pos=8 → l0_t=2 (post fires k_pos/T per token)
    assert abs(mg[1]["l0_t"] - reg.OP.B_star) < 1e-6


def test_nan_l0_rows_excluded():
    """Pre-increment rows (realized L0 = nan) can't be matched → dropped."""
    groups = {("changepoint", "batchtopk_sae", 1, 20, 2):
              _g(float("nan"), float("nan"), mode_recovery=0.9)}
    mg = report.matched_group(groups, "changepoint", _archs()[0], d_sae=20,
                              T_can=reg.OP.T_can, B_star=reg.OP.B_star)
    assert mg is None
