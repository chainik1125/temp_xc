"""STORY.md assembly — the isolation figure + every scripted number it cites.

    .venv/bin/python -m experiments.explorations.synthetic.story_figs

Companion to ``STORY.md`` (the distilled TXC-vs-T-SAE-vs-per-token story).
Everything here is re-derived from the canonical leaderboard through the SAME
matched-group machinery as ``render_report.py`` (``explorations.synthetic
.report``): per-token matching to B* = 2 realized atoms/token, d_sae = F
(boundary capacity), 3-seed mean with min–max seed whiskers. Nothing is
hand-typed; ``results/story_stats.json`` records every plotted value plus the
exact per-arch parameter counts (instantiated from the registered arch classes,
not transcribed formulas).

Outputs:
- ``figs/story_isolation.{png,pdf}`` — the 4-panel regime-exemplar figure
  (backtracking / frequency / phasepair / recipe residual), bars per arch at
  the bench's canonical verdict slice (T labeled per panel; the recipe
  residual's verdict cell is T=2 — its record's canonical window — while the
  tone benches sit at the program operating point T=4).
- ``results/story_stats.json`` — plotted values (per-seed), the matched knobs
  (k_pos, realized L0), and the § 6 parameter-count table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic import report

from . import registry as reg

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# (bench, metric, T_slice, regime tag, panel title). d_sae = F everywhere.
# T_slice is the bench's canonical verdict window: the program operating point
# T=4 for the three T-swept exemplars; the recipe residual's verdict cell is
# T=2 (its bench_record's canonical window — the equality latent is adjacency-
# local and dilutes at larger T; the T=4 value is in REPORT.md).
PANELS = (
    ("backtracking", "lambda_recovery", 4, "regime 2 — linear-in-window",
     "backtracking · λ intensity"),
    ("frequency", "velocity_recovery", 4, "regime 3 — power (band-aligned)",
     "frequency · tone velocity"),
    ("phasepair", "sign_recovery", 4, "regime 3 — phase",
     "phasepair · rotation sign"),
    ("recipe_instruction_phase_runs", "equality_residual_recovery", 2,
     "regime 3 — equality (grounded)", "recipe · equality residual"),
)

ARCH_COLORS = {
    "batchtopk_sae": "#9e9e9e",
    "tsae": "#616161",
    "stacked_batchtopk": "#64b5f6",
    "txc_batchtopk_pre": "#ff9800",
    "txc_batchtopk_post": "#d32f2f",
    "spectral_txc": "#7b1fa2",
}
ARCH_SHORT = {
    "batchtopk_sae": "Per-token\nSAE",
    "tsae": "T-SAE",
    "stacked_batchtopk": "Stacked",
    "txc_batchtopk_pre": "TXC-pre",
    "txc_batchtopk_post": "TXC-post",
    "spectral_txc": "Spectral\nTXC",
}


def _panel_stats(cells, groups):
    """The matched group + per-seed values for every (panel, arch)."""
    by_bench_f = {b.name: b.F for b in reg.BENCHES}
    out = {}
    for bench, metric, T_slice, regime, title in PANELS:
        F = by_bench_f[bench]
        entry = {"metric": metric, "T": T_slice, "d_sae": F, "regime": regime,
                 "title": title, "archs": {}}
        for a in reg.ARCHS:
            mg = report.matched_group(groups, bench, a, d_sae=F,
                                      T_can=T_slice, B_star=reg.OP.B_star)
            if mg is None:
                entry["archs"][a.name] = None
                continue
            key, g, dev = mg
            (_bn, _an, Tk, d, kp) = key
            seed_vals = sorted(
                (c["seed"], c["m"].get(metric))
                for c in cells
                if (c["bench"], c["arch"], c["T"], c["d_sae"], c["k_pos"])
                == key and c["m"].get(metric) is not None)
            vals = [v for (_s, v) in seed_vals]
            entry["archs"][a.name] = {
                "mean": float(np.mean(vals)), "min": float(np.min(vals)),
                "max": float(np.max(vals)), "n_seeds": len(vals),
                "seeds": {str(s): v for (s, v) in seed_vals},
                "k_pos": kp, "T": Tk, "d_sae": d,
                "realized_l0_token": g["l0_t"],
                "l0_deviation": dev, "loose": bool(dev > reg.OP.l0_tol),
            }
        out[bench] = entry
    return out


def _render_figure(panel_stats, figs_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(PANELS), figsize=(16, 4.6))
    for ax, (bench, _metric, T_slice, regime, title) in zip(axes, PANELS):
        entry = panel_stats[bench]
        xs = np.arange(len(reg.ARCHS))
        for i, a in enumerate(reg.ARCHS):
            st = entry["archs"][a.name]
            if st is None:
                continue
            err = np.array([[st["mean"] - st["min"]], [st["max"] - st["mean"]]])
            ax.bar(i, st["mean"], yerr=err, capsize=3,
                   color=ARCH_COLORS[a.name], width=0.72,
                   error_kw={"lw": 1.1, "ecolor": "black"})
        ax.axhline(0.0, color="black", lw=0.8)
        ax.axhline(1.0, color="black", lw=0.8, ls=":")
        ax.text(0.02, 1.005, "oracle", transform=ax.get_yaxis_transform(),
                fontsize=7, va="bottom", color="dimgray")
        ax.set_xticks(xs)
        ax.set_xticklabels([ARCH_SHORT[a.name] for a in reg.ARCHS], fontsize=7.5)
        ax.set_title(f"{title}\n{regime}   (T={T_slice}, d_sae=F="
                     f"{entry['d_sae']})", fontsize=9)
        lo = min(-0.05, min(st["min"] for st in entry["archs"].values() if st) - 0.08)
        ax.set_ylim(lo, 1.1)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("normalized recovery  [chance=0, oracle=1]")
    fig.suptitle(
        "Where architectures separate — one panel per regime exemplar "
        "(3-seed mean, min–max whiskers; per-token-matched realized L0 ≈ B*=2; "
        "recipe residual is normalized over [additive ceiling, exact])",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    figs_dir.mkdir(exist_ok=True)
    outs = []
    for ext in ("png", "pdf"):
        p = figs_dir / f"story_isolation.{ext}"
        fig.savefig(p, dpi=170)
        outs.append(str(p.relative_to(ROOT)))
    plt.close(fig)
    return outs


def _param_counts(d_in: int = 128, d_sae: int = 101, k_pos: int = 2):
    """Exact parameter counts from the registered arch classes (not formulas).

    Token archs at T=1; window archs at each T ∈ {2, 4, 8}. The canonical
    point (d_in=128, d_sae=101, k_pos=2) is the frequency-substrate boundary
    cell; the T-scaling, not the absolute count, is the story.
    """
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.archs.spectral_txc import SpectralTXCBatchTopK
    from temp_bench.archs.stacked_batchtopk import StackedBatchTopK
    from temp_bench.archs.tsae import TSAEPaper
    from temp_bench.archs.txc_batchtopk import TXCBatchTopKPre, TXCBatchTopKPost

    builders = {
        "batchtopk_sae": lambda T: BatchTopKSAE(
            d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1),
        "tsae": lambda T: TSAEPaper(d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=1),
        "stacked_batchtopk": lambda T: StackedBatchTopK(
            d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T),
        "txc_batchtopk_pre": lambda T: TXCBatchTopKPre(
            d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T),
        "txc_batchtopk_post": lambda T: TXCBatchTopKPost(
            d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T),
        "spectral_txc": lambda T: SpectralTXCBatchTopK(
            d_in=d_in, d_sae=d_sae, k_pos=k_pos, T=T),
    }
    out = {"point": {"d_in": d_in, "d_sae": d_sae, "k_pos": k_pos}, "counts": {}}
    for name, build in builders.items():
        windowed = name not in ("batchtopk_sae", "tsae")
        ts = (2, 4, 8) if windowed else (1,)
        out["counts"][name] = {
            str(T): int(sum(p.numel() for p in build(T).parameters()))
            for T in ts}
    return out


def main() -> None:
    leaderboard = ROOT / reg.LEADERBOARD_REL
    cells = report.load_program_rows(leaderboard, reg.BENCHES)
    primary = {b.name: b.datasources[0] for b in reg.BENCHES}
    cells = [c for c in cells if c["ds"] == primary[c["bench"]]]
    groups = report.group_cells(cells, primary_ds_only=True, benches=reg.BENCHES)

    panel_stats = _panel_stats(cells, groups)
    print(f"{'panel':<30s}{'arch':<20s}{'mean':>7s} {'min':>7s} {'max':>7s} "
          f"{'k_pos':>6s} {'l0/tok':>7s}")
    for bench, entry in panel_stats.items():
        for a in reg.ARCHS:
            st = entry["archs"][a.name]
            if st is None:
                print(f"{bench:<30s}{a.name:<20s}      —")
                continue
            print(f"{bench:<30s}{a.name:<20s}{st['mean']:>7.3f} {st['min']:>7.3f} "
                  f"{st['max']:>7.3f} {st['k_pos']:>6d} "
                  f"{st['realized_l0_token']:>7.2f}{'*' if st['loose'] else ''}")

    figs = _render_figure(panel_stats, HERE / "figs")
    print(f"[story] figures: {', '.join(figs)}")

    params = _param_counts()
    for name, by_t in params["counts"].items():
        print(f"[params] {name:<20s} " +
              "  ".join(f"T={t}: {n:,}" for t, n in by_t.items()))

    stats_path = HERE / "results" / "story_stats.json"
    stats_path.write_text(json.dumps({
        "source": reg.LEADERBOARD_REL,
        "operating_point": {"B_star": reg.OP.B_star, "l0_tol": reg.OP.l0_tol,
                            "d_sae": "F (boundary)",
                            "T": {b: e["T"] for b, e in panel_stats.items()}},
        "panels": panel_stats,
        "param_counts": params,
    }, indent=1))
    print(f"[story] -> {stats_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
