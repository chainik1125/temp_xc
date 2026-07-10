"""Render the AC-only signed-motion frontier figure.

    python -m experiments.explorations.synthetic.signed_motion.render_figs

Reads results/leaderboard.jsonl (signed_motion, protocol 1.2.0, 10K grid) and
writes a 3-panel frontier (s_temp, eAUC, NMSE vs d_sae, one line per (arch, T)).
F = 19 and 2F = 38 are marked; the scarce / memorization-free regime (d_sae < 2F)
is shaded. Also writes a low-res .thumb.png.

The story the figure tells: in the scarce regime no architecture recovers the
order-sensitive sign (s_temp ≈ 0); the crosscoder's recovery only appears at
the over-complete d_sae = 2F reference, where the per-tile probe is confounded
by tabulation.

This is the outlier of the synthetic benches: figure-ONLY (single combined
`bench.md`), on the iter_leaderboard path, with NO record-pipeline. It therefore
does not use `explorations.synthetic.record` (there are no AUTO tables / stats to
regenerate here).

Figures write to this bench's own `figs/` dir (`Path(__file__).parent / "figs"`),
per the per-bench convention — fixed 2026-07-10 (was a CWD-relative path).

NOTE (remaining pre-existing quirk, left as-is): `bench.md` carries a
`<!-- AUTO:* -->` block that this renderer never populates (it has no `populate`
step) — the block is effectively hand-maintained.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

DATASOURCE = "toy_signed_motion_M19_d40"
DEPRECATED_ARCHS = {"txc_pro", "tfa", "tfa_pos"}
N_STEPS = 10000
DEFAULT_SYNTH_D_SAE = 20
F = 19            # ground-truth feature count
N_WINDOWS = 2 * F  # = 38, the per-tile probe's memorization boundary

# Colour per (arch, T): crosscoder = blues by T, stacked = purples, token archs solid.
def _style(arch: str, T: int):
    blues = {2: "#9ecae1", 4: "#4292c6", 8: "#08519c"}
    purples = {2: "#dadaeb", 4: "#9e9ac8", 8: "#6a51a3"}
    if arch == "txc_base":
        return blues.get(T, "#08519c"), "o"
    if arch == "stacked_sae":
        return purples.get(T, "#6a51a3"), "v"
    if arch == "topk_sae":
        return "#d62728", "^"
    if arch == "tsae":
        return "#ff7f0e", "D"
    return "#444", "s"


def _override(row):
    return row.training_cfg.arch_hparams_override or {}


def render_ac_frontier() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from temp_bench.core.cache import iter_leaderboard

    # (arch, T, d_sae) -> {seed: {metric: val}}
    cells: dict = defaultdict(lambda: defaultdict(dict))
    for row in iter_leaderboard():
        if row.experiment != "synthetic" or row.datasource != DATASOURCE:
            continue
        if row.evaluator_protocol_version != "1.2.0":
            continue
        if row.eval_cfg.get("smoke", False) or row.arch in DEPRECATED_ARCHS:
            continue
        if getattr(row.training_cfg, "n_steps", None) != N_STEPS:
            continue
        ov = _override(row)
        d = int(ov.get("d_sae") or DEFAULT_SYNTH_D_SAE)
        T = int(ov.get("T") or 1)
        cells[(row.arch, T, d)][int(row.seed)] = row.metrics

    if not cells:
        raise RuntimeError("No signed_motion 1.2.0 rows. Run the sweep first.")

    series = sorted({(a, T) for (a, T, d) in cells},
                    key=lambda x: ({"txc_base": 0, "stacked_sae": 1,
                                    "topk_sae": 2, "tsae": 3}.get(x[0], 9), x[1]))
    n_seeds = len({s for v in cells.values() for s in v})

    def agg(arch, T, d, metric):
        vals = [m[metric] for m in cells.get((arch, T, d), {}).values()
                if metric in m]
        if not vals:
            return np.nan, 0.0
        return float(np.mean(vals)), (float(np.std(vals)) if len(vals) > 1 else 0.0)

    panels = [("s_temp", "s_temp  (sign recovery)", (-0.15, 1.08)),
              ("eauc", "eAUC  (local feature recovery)", (0.0, 1.02)),
              ("nmse", "NMSE  (reconstruction)", (-0.02, 0.7))]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (metric, ylabel, ylim) in zip(axes, panels):
        dsaes_all = sorted({d for (a, T, d) in cells})
        for (arch, T) in series:
            ds = [d for d in dsaes_all if (arch, T, d) in cells]
            if not ds:
                continue
            m = [agg(arch, T, d, metric)[0] for d in ds]
            e = [agg(arch, T, d, metric)[1] for d in ds]
            color, marker = _style(arch, T)
            ax.errorbar(ds, m, yerr=e, color=color, marker=marker,
                        label=f"{arch} T={T}", lw=1.8, capsize=2, markersize=6)
        # scarce / memorization-free region (d_sae < 2F) shaded
        ax.axvspan(min(dsaes_all) - 1, N_WINDOWS, color="green", alpha=0.05)
        ax.axvline(F, color="gray", ls="--", lw=1, alpha=0.7)
        ax.axvline(N_WINDOWS, color="crimson", ls=":", lw=1, alpha=0.7)
        ax.text(F, ylim[1], " F=19", fontsize=8, va="top", color="gray")
        ax.text(N_WINDOWS, ylim[1], " 2F=38\n (probe\n confounded)", fontsize=7,
                va="top", color="crimson")
        if metric == "s_temp":
            ax.axhline(0.0, color="k", ls=":", lw=0.8, alpha=0.4)
        ax.set_xticks(dsaes_all)
        ax.set_ylim(*ylim)
        ax.set_xlabel("d_sae (dictionary size)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7, ncol=2, loc="upper left")

    fig.suptitle(
        "AC signed-motion frontier: in the scarce regime (d_sae ≤ F, shaded) no "
        "architecture recovers the order-sensitive sign;\nthe crosscoder's "
        "recovery appears only at the over-complete 2F reference, where the probe "
        "is confounded by tabulation.",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out_dir = Path(__file__).resolve().parent / "figs"  # this bench's own figs/ (convention)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_ac_signed_motion.pdf"
    png = out_dir / "fig_ac_signed_motion.png"
    thumb = out_dir / "fig_ac_signed_motion.thumb.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(thumb, dpi=50, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_ac] {len(cells)} cells, {n_seeds} seeds → {pdf}, {png}, {thumb}")
    return pdf


if __name__ == "__main__":
    render_ac_frontier()
