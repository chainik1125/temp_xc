"""Render the AC-only signed-motion figure (FrequencyBench § 5).

    python -m experiments.render_ac_figure

Reads ``results/leaderboard.jsonl`` and writes a two-panel figure to
``docs/figs/``:

- **Left (architectural gap):** s_temp vs k_pos at the ample dictionary
  d_sae=64. The window crosscoder (txc_base) recovers the hidden sign at
  oracle level; every per-token SAE sits at chance — they reconstruct
  perfectly and recover the alphabet, yet by the data-processing
  inequality their per-token codes cannot expose the order-sensitive sign.
- **Right (capacity threshold):** s_temp vs d_sae (mean over k_pos, seeds).
  txc_base's sign recovery switches on only once it has enough atoms to
  represent the 2M=38 distinct windows; the SAE family stays flat at chance
  for every dictionary size.

Also writes a low-res ``.thumb.png`` for safe inline inspection.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

DATASOURCE = "toy_signed_motion_M19_d40"
DEPRECATED_ARCHS = {"txc_pro", "tfa", "tfa_pos"}
DEFAULT_SYNTH_D_SAE = 20
HEADLINE_DSAE = 64
N_STEPS = 10000   # canonical grid; excludes 3K smoke / 30K convergence rows

ARCH_COLOR = {
    "txc_base": "#1f77b4", "topk_sae": "#d62728",
    "stacked_sae": "#9467bd", "tsae": "#ff7f0e",
}
ARCH_MARKER = {
    "txc_base": "o", "topk_sae": "^", "stacked_sae": "v", "tsae": "D",
}


def _row_d_sae(row) -> int:
    ovr = row.training_cfg.arch_hparams_override or {}
    return int(ovr.get("d_sae") or DEFAULT_SYNTH_D_SAE)


def _row_k_pos(row):
    kp = row.eval_cfg.get("k_pos")
    if kp is None:
        kp = (row.training_cfg.arch_hparams_override or {}).get("k_pos")
    return None if kp is None else int(kp)


def render_ac_signed_motion() -> Path:
    import matplotlib.pyplot as plt
    import numpy as np

    from temp_bench.core.cache import iter_leaderboard

    # (d_sae, arch, k_pos) -> {seed: s_temp}  (latest row per seed wins)
    cells: dict = defaultdict(dict)
    for row in iter_leaderboard():
        if row.experiment != "synthetic" or row.datasource != DATASOURCE:
            continue
        if row.evaluator_protocol_version != "1.1.0":
            continue
        if row.eval_cfg.get("smoke", False) or row.arch in DEPRECATED_ARCHS:
            continue
        if getattr(row.training_cfg, "n_steps", None) != N_STEPS:
            continue
        s_temp = row.metrics.get("s_temp")
        kp = _row_k_pos(row)
        if s_temp is None or kp is None:
            continue
        cells[(_row_d_sae(row), row.arch, kp)][int(row.seed)] = float(s_temp)

    if not cells:
        raise RuntimeError(
            "No signed_motion rows. Run scripts/run_ac_minisweep.sh first."
        )

    archs = sorted({a for (_d, a, _k) in cells}, key=lambda a: a != "txc_base")
    n_seeds = len({s for v in cells.values() for s in v})

    def agg(dsae, arch, kp):
        vals = list(cells.get((dsae, arch, kp), {}).values())
        return (np.mean(vals), np.std(vals) if len(vals) > 1 else 0.0) if vals else (np.nan, 0.0)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

    # ── Left: s_temp vs k_pos at d_sae = HEADLINE_DSAE ──
    ks = sorted({k for (d, a, k) in cells if d == HEADLINE_DSAE})
    for arch in archs:
        m = [agg(HEADLINE_DSAE, arch, k)[0] for k in ks]
        e = [agg(HEADLINE_DSAE, arch, k)[1] for k in ks]
        if all(np.isnan(v) for v in m):
            continue
        axL.errorbar(ks, m, yerr=e, color=ARCH_COLOR.get(arch, "#444"),
                     marker=ARCH_MARKER.get(arch, "o"), label=arch,
                     linewidth=2, capsize=3, markersize=8)
    axL.axhline(0.0, color="k", ls=":", alpha=0.4, lw=1)
    axL.set_xticks(ks)
    axL.set_ylim(-0.1, 1.08)
    axL.set_xlabel("k_pos (per-token sparsity)")
    axL.set_ylabel("s_temp   (0 = chance, 1 = oracle)")
    axL.set_title(f"Architectural gap  (d_sae = {HEADLINE_DSAE}, ample)")
    axL.grid(True, alpha=0.3)
    axL.legend(loc="center right", fontsize=10)

    # ── Right: s_temp vs d_sae at the sparsest setting (k_pos=1) ──
    # Fixed k_pos (not averaged): the architectural recovery lives in the
    # sparse regime, and averaging over k_pos would let the higher-k
    # sign-entangled cells mask the clean threshold.
    KP_PANEL = 1
    dsaes = sorted({d for (d, a, k) in cells if k == KP_PANEL})
    for arch in archs:
        m = [agg(d, arch, KP_PANEL)[0] for d in dsaes]
        e = [agg(d, arch, KP_PANEL)[1] for d in dsaes]
        if all(np.isnan(v) for v in m):
            continue
        axR.errorbar(dsaes, m, yerr=e, color=ARCH_COLOR.get(arch, "#444"),
                     marker=ARCH_MARKER.get(arch, "o"), label=arch,
                     linewidth=2, capsize=3, markersize=8)
    axR.axvline(38, color="gray", ls="--", alpha=0.6, lw=1)
    axR.text(38, 0.5, "2M=38\ndistinct windows", fontsize=8,
             ha="center", va="center", color="gray", rotation=90)
    axR.axhline(0.0, color="k", ls=":", alpha=0.4, lw=1)
    axR.set_xticks(dsaes)
    axR.set_ylim(-0.12, 1.08)
    axR.set_xlabel("d_sae (dictionary size)")
    axR.set_ylabel(f"s_temp   (at k_pos={KP_PANEL}, sparsest)")
    axR.set_title("Capacity threshold for window-encoder sign recovery")
    axR.grid(True, alpha=0.3)
    axR.legend(loc="center right", fontsize=10)

    fig.suptitle(
        "AC-only signed-motion bench (FrequencyBench § 5): window crosscoders "
        "recover the order-sensitive sign; per-token SAEs cannot (DPI)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir = Path("docs/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / "fig_ac_signed_motion.pdf"
    png = out_dir / "fig_ac_signed_motion.png"
    thumb = out_dir / "fig_ac_signed_motion.thumb.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, dpi=120, bbox_inches="tight")
    fig.savefig(thumb, dpi=45, bbox_inches="tight")
    plt.close(fig)
    print(f"[render_ac] {len(cells)} cells, {n_seeds} seeds → {pdf}, {png}, {thumb}")
    return pdf


if __name__ == "__main__":
    render_ac_signed_motion()
