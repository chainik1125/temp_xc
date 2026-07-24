"""Probe-truth figure (CARD_PROBE_TRUTH.md § 8 deliverable). Committed pre-run.

`figs/probe_truth.{png,pdf}` — six panels off `results/probe_truth.json`:

  Row 1 (the briefing's ask: reported recovery vs T for BOTH probes with the
  TRUE recoverable level marked, per arm) — line C (capacity ladder, the core
  arm) at its dense anchor, line P (matched post, k = 8·T), and lines M + S.
  Row 2 (the axis the question actually lives on, card § 1.1) — the Stage-1
  constructed codes where truth is EXACT, `full` and `token` arms, reported
  recovery vs p/n with the exact truth line; and the trained ladder's signed
  deviation from truth vs p/n for both probes.

Run:  .venv/bin/python -m \
        experiments.explorations.task_hunt.support_synthetic.render_probe_truth
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RES, FIGS = HERE / "results", HERE / "figs"
DPI_FLOOR = 0.41
C_V1, C_V2, C_TR = "#1f77b4", "#d62728", "#000000"


def _line(cells, *, arch, k_pos, d_sae, trained=True, k_is_8T=False):
    out = []
    for c in cells:
        if c["arch"] != arch or c["d_sae"] != d_sae:
            continue
        if (c["n_steps"] > 0) != trained:
            continue
        want = 8 * c["T"] if k_is_8T else k_pos
        if c["k_pos"] != want:
            continue
        out.append(c)
    return sorted(out, key=lambda c: c["T"])


def _panel_vs_T(ax, groups, title, *, floor=True):
    """Reported recovery vs T, both probes, with the TRUE level marked."""
    any_pt = False
    for (cells, label, ls) in groups:
        if not cells:
            continue
        any_pt = True
        Ts = [c["T"] for c in cells]
        ax.plot(Ts, [c["v1_mean"] for c in cells], ls, color=C_V1, marker="o",
                ms=5, label=f"v1 (OLS, nw 1024){label}")
        ax.plot(Ts, [c["v2_mean"] for c in cells], ls, color=C_V2, marker="s",
                ms=5, label=f"v2 (ridge, nw 8192){label}")
        tr = [(c["T"], c["anchor_mean"], c.get("anchor_licensed"))
              for c in cells if "anchor_mean" in c]
        if tr:
            lic = [(t, v) for t, v, ok in tr if ok]
            unl = [(t, v) for t, v, ok in tr if not ok]
            if lic:
                ax.plot([t for t, _ in lic], [v for _, v in lic], "-",
                        color=C_TR, lw=2.2, alpha=.75, marker="_", ms=14,
                        label=f"TRUE (anchor, licensed){label}")
            if unl:
                ax.plot([t for t, _ in unl], [v for _, v in unl], "x",
                        color=C_TR, ms=8, mew=2,
                        label=f"anchor NOT licensed{label}")
    if floor:
        ax.axhline(DPI_FLOOR, color="grey", ls=":", lw=1)
        ax.text(0.02, DPI_FLOOR + .012, "per-token DPI floor 0.41", fontsize=6,
                color="grey", transform=ax.get_yaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set_xlabel("window T")
    ax.set_ylabel("λ recovery (held-out r)")
    ax.set_title(title, fontsize=8)
    ax.grid(alpha=.25)
    if any_pt:
        ax.legend(fontsize=5.5, loc="lower left")


def _panel_calib(ax, calib, arm, T=16):
    """Stage 1: reported vs p/n against the EXACT truth, both code densities.

    Density is drawn because it turned out to matter more than p/n for v1:
    at the same p/n = 1.0 the sag is −0.07 at the card's top-8 construction
    and −0.45 at the real panel's ~6% density.
    """
    def g(c, nw, key):
        return [q for q in c["grid"] if q["n_windows"] == nw][0][key]

    drew = False
    for dens, ls, tag in (("k8", "-", "top-8 code"), ("p6", "--", "6% dense")):
        rows = [c for c in calib if c["arm"] == arm and c["T"] == T
                and c["density"] == dens]
        by_p = {}
        for c in rows:
            by_p.setdefault(c["p"], []).append(c)
        ps = sorted(by_p)
        if not ps:
            continue
        def m(p, f, _bp=by_p):
            return float(np.mean([f(c) for c in _bp[p]]))
        pn = [m(p, lambda c: g(c, 1024, "p_over_n")) for p in ps]
        if not drew:
            ax.plot(pn, [m(p, lambda c: g(c, 1024, "truth")) for p in ps], "-",
                    color=C_TR, lw=2.2, alpha=.75,
                    label="TRUE (exact, by construction)")
            drew = True
        ax.plot(pn, [m(p, lambda c: c["v1"]) for p in ps], ls, marker="o",
                color=C_V1, ms=5, label=f"v1 (OLS, nw 1024) · {tag}")
        ax.plot(pn, [m(p, lambda c: c["v2"]) for p in ps], ls, marker="s",
                color=C_V2, ms=5, label=f"v2 (ridge, nw 8192) · {tag}")
    if not drew:
        ax.set_title(f"Stage 1 — {arm} (no data)", fontsize=8)
        return
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.text(1.05, .04, "p = n", fontsize=6, color="grey",
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log")
    ax.set_xlabel("p / n  (v1's budget: n = 1024·(32/T))")
    ax.set_ylabel("λ recovery (held-out r)")
    ax.set_title(f"Stage 1 constructed code — {arm} arm (truth known exactly)",
                 fontsize=8)
    ax.grid(alpha=.25)
    ax.legend(fontsize=5.5, loc="lower left")


def _panel_delta(ax, cells):
    """Trained ladder: signed deviation from truth vs p/n, both probes."""
    q = [c for c in cells if c.get("anchor_licensed") and "d1_mean" in c]
    if not q:
        ax.set_title("trained ladder — no licensed cells", fontsize=8)
        return
    tr = [c for c in q if c["n_steps"] > 0]
    un = [c for c in q if c["n_steps"] == 0]
    for sub, mk, lab in ((tr, "o", "trained"), (un, "^", "untrained")):
        if not sub:
            continue
        x = [c["p_over_n"] for c in sub]
        ax.scatter(x, [c["d1_mean"] for c in sub], c=C_V1, marker=mk, s=26,
                   label=f"v1 − truth ({lab})")
        ax.scatter(x, [c["d2_mean"] for c in sub], c=C_V2, marker=mk, s=26,
                   label=f"v2 − truth ({lab})")
    ax.axhline(0, color="k", lw=1)
    ax.axhline(-0.05, color="grey", ls=":", lw=1)
    ax.axhline(0.05, color="grey", ls=":", lw=1)
    ax.axvline(1.0, color="grey", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("p / n  (v1's budget)")
    ax.set_ylabel("reported − TRUE")
    ax.set_title("trained mirror ladder — deviation from truth (licensed cells)",
                 fontsize=8)
    ax.grid(alpha=.25)
    ax.legend(fontsize=5.5, loc="lower left")


def main():
    FIGS.mkdir(exist_ok=True)
    d = json.loads((RES / "probe_truth.json").read_text())
    cells, calib = d["cells"], d["calibration"]

    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.6))
    _panel_vs_T(
        axes[0][0],
        [(_line(cells, arch="txc_batchtopk_pre", k_pos=8, d_sae=2048), "", "--")],
        "line C — TXC-pre, k_pos 8, d_sae 2048 (p/n = 1.00 at T16;\n"
        "matched to the real panel in p/n AND window density)")
    _panel_vs_T(
        axes[0][1],
        [(_line(cells, arch="txc_batchtopk_post", k_pos=0, d_sae=2048,
                k_is_8T=True), "", "--")],
        "line P — matched TXC-post, nominal k = 8·T\n"
        "(runpod-d's code-rate convention)")
    _panel_vs_T(
        axes[0][2],
        [(_line(cells, arch="txc_batchtopk_pre", k_pos=1, d_sae=20), " · M", "--"),
         (_line(cells, arch="stacked_batchtopk", k_pos=8, d_sae=512), " · S", ":")],
        "line M — mirror-canonical control (d_sae 20, p/n ≤ 0.010)\n"
        "line S — Stacked, p = T·d_sae (p/n = 4.0 at T16)")
    _panel_calib(axes[1][0], calib, "full")
    _panel_calib(axes[1][1], calib, "token")
    _panel_delta(axes[1][2], cells)

    fig.suptitle(
        f"Which λ readout reports TRUE recovery?  mirror probe-truth campaign "
        f"— branch_evidence: {d['branch_evidence']}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"probe_truth.{ext}", dpi=170)
    print(f"-> {FIGS/'probe_truth.png'}")


if __name__ == "__main__":
    main()
