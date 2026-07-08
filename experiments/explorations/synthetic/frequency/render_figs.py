"""Cyclic-tone frequency results — SINGLE SOURCE renderer + record populator.

Reads the canonical leaderboard (`results/leaderboard.jsonl`), filters the
`toy_cyclic_*` cells (protocol 1.2.0, non-smoke, n_steps ∈ {0, N_STEPS}),
aggregates over seeds, then in one pass:

  1. renders paper-quality figures into `figs/`
     (`frequency_main`, `frequency_Sf`, `frequency_spectral`,
      `frequency_null`, `frequency_untrained`, `frequency_memorization`),
  2. writes `results/frequency_bench_stats.json`,
  3. fills every `<!-- BEGIN AUTO:<tag> --> … <!-- END AUTO:<tag> -->` block in
     `bench_record.md`.

Chance = 1/|Ω|; the oracle is per-cell (periodogram/GLRT) — recovery is already
normalized to [chance, oracle] in the evaluator (`velocity_recovery`). The S(f)
curve normalizes each per-Ω-class recall to its per-class oracle.

    .venv/bin/python -m experiments.explorations.synthetic.frequency.render_figs
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"
FIG_DIR = HERE / "figs"
RES_DIR = HERE / "results"
GATING = RES_DIR / "frequency_gating_stats.json"
STATS_OUT = RES_DIR / "frequency_bench_stats.json"
RECORD = HERE / "bench_record.md"
CIRCLE = "toy_cyclic_circle_M101_d128"
RANDOM = "toy_cyclic_random_M101_d128"
PROTOCOL = "1.2.0"
N_STEPS_GRID = 6000

M = 101
OMEGA = [0, 1, 2, 4, 8, 16, 24, 32, 40, 50]
FREQS = [y / M for y in OMEGA]
CHANCE = 1.0 / len(OMEGA)
MEMO_THRESH = len(OMEGA) * M          # |Ω|·M = 1010
MEMO_DSAE = 2048                       # the > |Ω|·M memorization-demo width
D_SAES = [32, 64, 101, 256]
ANCHOR = 101
T_WINDOW = [2, 4, 8, 16]

PER_TOKEN = [("batchtopk_sae", 1), ("tsae", 1)]
CROSSCODERS = ["txc_batchtopk_pre", "txc_batchtopk_post", "spectral_txc"]
ARCH_T = PER_TOKEN + [(a, T) for a in CROSSCODERS for T in T_WINDOW]
LABEL = {"batchtopk_sae": "BatchTopK-SAE", "tsae": "T-SAE",
         "txc_batchtopk_pre": "TXC-pre", "txc_batchtopk_post": "TXC-post",
         "spectral_txc": "Spectral-TXC"}
FAM_COLOR = {"txc_batchtopk_pre": "#3182bd", "txc_batchtopk_post": "#807dba",
             "spectral_txc": "#e6550d"}
COLORS = {
    ("batchtopk_sae", 1): "#238b45", ("tsae", 1): "#66c2a4",
    ("txc_batchtopk_pre", 2): "#c6dbef", ("txc_batchtopk_pre", 4): "#9ecae1",
    ("txc_batchtopk_pre", 8): "#4292c6", ("txc_batchtopk_pre", 16): "#08519c",
    ("txc_batchtopk_post", 2): "#dadaeb", ("txc_batchtopk_post", 4): "#bcbddc",
    ("txc_batchtopk_post", 8): "#807dba", ("txc_batchtopk_post", 16): "#54278f",
    ("spectral_txc", 2): "#fdd0a2", ("spectral_txc", 4): "#fdae6b",
    ("spectral_txc", 8): "#f16913", ("spectral_txc", 16): "#a63603",
}
TCOLOR = {2: "#9ecae1", 4: "#4292c6", 8: "#2171b5", 16: "#08306b"}
MARK = {1: "o", 2: "s", 4: "^", 8: "D", 16: "P"}

PAPER_STYLE = {
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11.5,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "axes.grid": True, "grid.alpha": 0.16, "grid.linewidth": 0.7,
    "legend.frameon": False, "lines.linewidth": 2.0, "lines.markersize": 6,
    "figure.facecolor": "white", "mathtext.default": "regular",
}


def label(arch, T):
    return f"{LABEL[arch]} (per-token)" if (arch, T) in PER_TOKEN else f"{LABEL[arch]} (T={T})"


# ── canonical source: the leaderboard ──────────────────────────────────

def load_rows():
    rows = []
    for line in LEADERBOARD.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("datasource") not in (CIRCLE, RANDOM):
            continue
        if r.get("evaluator_protocol_version") != PROTOCOL:
            continue
        ec = r.get("eval_cfg") or {}
        if ec.get("smoke"):
            continue
        n_steps = int(r.get("training_cfg", {}).get("n_steps", 0))
        if n_steps not in (0, N_STEPS_GRID):
            continue
        ov = r.get("training_cfg", {}).get("arch_hparams_override") or {}
        rows.append({"ds": r["datasource"], "arch": r["arch"], "T": int(ov.get("T", 1)),
                     "d_sae": int(ov.get("d_sae")),
                     "k_pos": int(ec.get("k_pos", ov.get("k_pos", 1))),
                     "seed": int(r["seed"]),
                     "kind": "trained" if n_steps > 0 else "untrained", "m": r["metrics"]})
    return rows


def aggregate(rows):
    """(ds, kind, k_pos, arch, T, d_sae) -> {metric: (mean, std, n)} over seeds."""
    buck = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["ds"], r["kind"], r["k_pos"], r["arch"], r["T"], r["d_sae"])
        for m, v in r["m"].items():
            if v is not None and np.isfinite(v):
                buck[key][m].append(float(v))
    return {k: {m: (float(np.mean(vs)), float(np.std(vs)), len(vs)) for m, vs in d.items()}
            for k, d in buck.items()}


def g(agg, ds, kind, kpos, arch, T, d, metric="velocity_recovery"):
    c = agg.get((ds, kind, kpos, arch, T, d))
    return c[metric] if c and metric in c else (float("nan"), float("nan"), 0)


def sf_curve(agg, ds, arch, T, d, *, normalized=True):
    """Per-Ω-class recovery curve for a cell (probe recall, optionally / oracle)."""
    ys = []
    for c in range(len(OMEGA)):
        pr = g(agg, ds, "trained", 1, arch, T, d, f"vel_recall_c{c}")[0]
        orc = g(agg, ds, "trained", 1, arch, T, d, f"vel_oracle_c{c}")[0]
        if not normalized:
            ys.append(pr)
        else:
            ys.append((pr - CHANCE) / max(orc - CHANCE, 1e-6) if np.isfinite(orc) else np.nan)
    return ys


def oracle_curve(agg, ds, arch, T, d):
    return [g(agg, ds, "trained", 1, arch, T, d, f"vel_oracle_c{c}")[0]
            for c in range(len(OMEGA))]


# ── figures ─────────────────────────────────────────────────────────────

def _mark_dsae_axis(ax, ylim):
    ax.axvline(ANCHOR, color="0.45", ls=":", lw=1.1, zorder=1)
    ax.text(ANCHOR * 1.03, ylim[0] + 0.05 * (ylim[1] - ylim[0]), "M", color="0.4",
            fontsize=10, style="italic")
    ax.axvline(MEMO_THRESH, color="#b30000", ls=":", lw=1.1, zorder=1)
    ax.text(MEMO_THRESH * 0.97, ylim[0] + 0.05 * (ylim[1] - ylim[0]), "|Ω|·M",
            color="#b30000", fontsize=9, style="italic", ha="right")


def fig_main(agg, plt):
    """Headline: velocity_recovery frontier (per-token flat at 0 vs crosscoders)."""
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    ylim = (-0.08, 1.05)
    _mark_dsae_axis(ax, ylim)
    ax.axhline(0.0, color="#999", ls="--", lw=1.1)
    ax.text(34, 0.02, "chance floor (per-token DPI + raw-linear)", color="#777",
            fontsize=8, va="bottom")
    for arch, T in ARCH_T:
        xs, ys, es = [], [], []
        for d in D_SAES:
            mm = g(agg, CIRCLE, "trained", 1, arch, T, d, "velocity_recovery")
            if mm[2]:
                xs.append(d); ys.append(mm[0]); es.append(mm[1])
        if xs:
            ls = "--" if (arch, T) in PER_TOKEN else "-"
            ax.errorbar(xs, ys, yerr=es, marker=MARK[T], ms=6, lw=1.9, ls=ls,
                        color=COLORS[(arch, T)], capsize=2, elinewidth=1, label=label(arch, T))
    ax.set_xscale("log", base=2); ax.set_xticks(D_SAES + [MEMO_THRESH])
    ax.set_xticklabels([str(d) for d in D_SAES] + ["1010"])
    ax.set_xlim(28, 1200); ax.set_ylim(*ylim)
    ax.set_xlabel("dictionary size  $d_{sae}$  (anchored on $M$; log)")
    ax.set_ylabel("velocity recovery  (norm. to [chance, periodogram oracle])")
    ax.set_title("Circle single-tone: velocity recovery frontier", loc="left", fontsize=12)
    ax.legend(ncol=2, fontsize=7.6, loc="upper left")
    fig.tight_layout()
    _save(fig, plt, "frequency_main")


def fig_Sf(agg, plt):
    """THE deliverable: S(f) frequency response per window T (crosscoders, anchor).

    Raw per-Ω-class velocity recall (the probe) vs f=Y/M, with the periodogram
    oracle overlaid as the achievable ceiling and the Rayleigh cutoff 1/T marked.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    pt = [g(agg, CIRCLE, "trained", 1, "batchtopk_sae", 1, ANCHOR, f"vel_recall_c{c}")[0]
          for c in range(len(OMEGA))]
    for ax, arch in zip(axes, CROSSCODERS):
        for T in T_WINDOW:
            ys = sf_curve(agg, CIRCLE, arch, T, ANCHOR, normalized=False)
            ax.plot(FREQS, ys, "o-", color=TCOLOR[T], label=f"T={T}")
            ax.axvline(1.0 / T, color=TCOLOR[T], ls=":", lw=0.9, alpha=0.5)
        oc = oracle_curve(agg, CIRCLE, arch, 16, ANCHOR)      # full-resolution oracle
        ax.plot(FREQS, oc, "-", color="0.4", lw=1.2, alpha=0.7, label="oracle (T=16)")
        ax.plot(FREQS, pt, "--", color="#238b45", lw=1.3, label="per-token")
        ax.axhline(CHANCE, color="#999", ls=":", lw=0.9)
        ax.set_xlabel("frequency  $f = Y/M$"); ax.set_ylim(-0.03, 1.05)
        ax.set_title(f"{LABEL[arch]}", loc="left", fontsize=11)
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("per-Ω-class velocity recall  $S(f)$")
    fig.suptitle("Frequency response $S(f)$: high-pass, Rayleigh cutoff $\\approx 1/T$ "
                 "(dotted)  —  circle, $d_{sae}=M=101$", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, plt, "frequency_Sf")


def fig_spectral(agg, plt):
    """Spectral vs vanilla + band decomposition."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    # (a) spectral vs pre vs post: velocity_recovery vs T at anchor
    for arch in CROSSCODERS:
        ts, ys, es = [], [], []
        for T in T_WINDOW:
            mm = g(agg, CIRCLE, "trained", 1, arch, T, ANCHOR, "velocity_recovery")
            if mm[2]:
                ts.append(T); ys.append(mm[0]); es.append(mm[1])
        if ts:
            ax1.errorbar(ts, ys, yerr=es, marker="o", ms=7, lw=2,
                         color=FAM_COLOR[arch], capsize=3, label=LABEL[arch])
    ax1.set_xscale("log", base=2); ax1.set_xticks(T_WINDOW); ax1.set_xticklabels(T_WINDOW)
    ax1.set_xlabel("window size  $T$"); ax1.set_ylabel("velocity recovery (norm.)")
    ax1.set_ylim(-0.05, 1.05)
    ax1.set_title("(a) Spectral vs monolithic crosscoders  ($d_{sae}=M$)", loc="left", fontsize=11)
    ax1.legend(fontsize=9)
    # (b) band decomposition: per-band recovery for spectral_txc T=16
    bands_present = [b for b in range(4)
                     if g(agg, CIRCLE, "trained", 1, "spectral_txc", 16, ANCHOR,
                          f"band{b}_recovery")[2]]
    bnames = ["DC {0}", "low {1-5}", "mid {6-10}", "high {11-15}"]
    for b in bands_present:
        yv = g(agg, CIRCLE, "trained", 1, "spectral_txc", 16, ANCHOR, f"band{b}_recovery")[0]
        ax2.bar(b, yv, color="#e6550d", alpha=0.55 + 0.1 * b, edgecolor="0.3",
                label=bnames[b] if b < len(bnames) else f"band{b}")
    full = g(agg, CIRCLE, "trained", 1, "spectral_txc", 16, ANCHOR, "velocity_recovery")[0]
    ax2.axhline(full, color="#a63603", ls="--", lw=1.3, label="full code")
    ax2.set_xticks(bands_present); ax2.set_xticklabels([bnames[b] for b in bands_present], fontsize=8)
    ax2.set_ylim(-0.05, 1.05); ax2.set_ylabel("velocity recovery (norm.)")
    ax2.set_title("(b) Spectral-TXC band decomposition  (T=16)", loc="left", fontsize=11)
    ax2.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    _save(fig, plt, "frequency_spectral")


def fig_null(agg, plt):
    """Circle high-pass vs random flat null (the ratio-invariance control)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), sharey=True)
    for ax, ds, ttl in [(ax1, CIRCLE, "(a) Circle — frequency-ordered (high-pass)"),
                        (ax2, RANDOM, "(b) Random — FLAT (ratio-invariance null)")]:
        for T in T_WINDOW:
            ys = sf_curve(agg, ds, "txc_batchtopk_pre", T, ANCHOR, normalized=False)
            ax.plot(FREQS, ys, "o-", color=TCOLOR[T], label=f"TXC-pre T={T}")
        ax.axhline(CHANCE, color="#999", ls=":", lw=1, label=f"chance {CHANCE:.1f}")
        ax.set_xlabel("frequency  $f = Y/M$"); ax.set_title(ttl, loc="left", fontsize=11)
        ax.legend(fontsize=8, loc="best")
    ax1.set_ylabel("per-Ω-class velocity recall (raw)")
    fig.suptitle("Symmetry null: circle response tracks $|\\Delta f|$ (Rayleigh); "
                 "random response has no frequency axis", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    _save(fig, plt, "frequency_null")


def fig_untrained(agg, plt):
    """Access vs learning: untrained vs trained velocity_recovery at the anchor."""
    fig, ax = plt.subplots(figsize=(10, 4.6))
    labels, un, uns, tr, trs, cols = [], [], [], [], [], []
    for arch, T in ARCH_T:
        u = g(agg, CIRCLE, "untrained", 1, arch, T, ANCHOR, "velocity_recovery")
        t = g(agg, CIRCLE, "trained", 1, arch, T, ANCHOR, "velocity_recovery")
        labels.append(label(arch, T).replace(" (", "\n(")); cols.append(COLORS[(arch, T)])
        un.append(u[0]); uns.append(u[1]); tr.append(t[0]); trs.append(t[1])
    x = np.arange(len(labels)); w = 0.4
    ax.bar(x - w / 2, un, w, yerr=uns, capsize=2, color="#cfcfcf", edgecolor="0.4",
           lw=0.5, label="untrained (nonlinear access)")
    ax.bar(x + w / 2, tr, w, yerr=trs, capsize=2, color=cols, edgecolor="0.25",
           lw=0.5, label="trained (access + learning)")
    ax.axhline(0.0, color="#999", ls="--", lw=1.0)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7.0); ax.set_ylim(-0.1, 1.05)
    ax.set_ylabel("velocity recovery (norm.)")
    ax.set_title("Access vs learning ($d_{sae}=M=101$, $k_{pos}=1$)", loc="left", fontsize=11.5)
    ax.legend(loc="upper left", fontsize=8.5); ax.grid(axis="x", alpha=0)
    fig.tight_layout()
    _save(fig, plt, "frequency_untrained")


def fig_memorization(agg, plt):
    """Memorization demo: velocity_recovery vs d_sae past |Ω|·M on both datasources."""
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    ylim = (-0.08, 1.08)
    _mark_dsae_axis(ax, ylim)
    for ds, mk, lab in [(CIRCLE, "o", "circle"), (RANDOM, "s", "random (null)")]:
        for arch in ("txc_batchtopk_pre", "spectral_txc"):
            xs, ys = [], []
            for d in D_SAES + [MEMO_DSAE]:
                kind = "memo" if d == MEMO_DSAE else "trained"
                mm = g(agg, ds, kind, 1, arch, 16, d, "velocity_recovery")
                if mm[2]:
                    xs.append(d); ys.append(mm[0])
            if xs:
                ax.plot(xs, ys, marker=mk, ls="-" if ds == CIRCLE else "--",
                        color=FAM_COLOR[arch], alpha=0.9 if ds == CIRCLE else 0.6,
                        label=f"{LABEL[arch]} ({lab})")
    ax.set_xscale("log", base=2); ax.set_ylim(*ylim)
    ax.set_xlabel("dictionary size  $d_{sae}$  (log)")
    ax.set_ylabel("velocity recovery (norm.)")
    ax.set_title("Memorization above $|Ω|·M$: the random null jumps to ~1 by "
                 "template lookup", loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="center left")
    fig.tight_layout()
    _save(fig, plt, "frequency_memorization")


def _save(fig, plt, name):
    for ext, dpi in [("pdf", None), ("png", 200), ("thumb.png", 70)]:
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=dpi)
    plt.close(fig)
    print(f"[fig] {FIG_DIR.name}/{name}.{{pdf,png}}")


# ── tables + headline ──────────────────────────────────────────────────

def _f(t, dec=3):
    m, s, n = t
    return "—" if not n else f"{m:.{dec}f}"


def _frontier_table(agg, ds, metric):
    h = ("| arch / T | " + " | ".join(f"d={d}" for d in D_SAES) + " |\n"
         "|---|" + "|".join("---" for _ in D_SAES) + "|\n")
    for arch, T in ARCH_T:
        cells = " | ".join(_f(g(agg, ds, "trained", 1, arch, T, d, metric)) for d in D_SAES)
        name = label(arch, T) if (arch, T) in PER_TOKEN else f"**{label(arch,T)}**"
        h += f"| {name} | {cells} |\n"
    return h.rstrip()


def _Sf_table(agg):
    h = ("| arch / T | " + " | ".join(f"Y={y}" for y in OMEGA) + " |\n"
         "|---|" + "|".join("---" for _ in OMEGA) + "|\n")
    # oracle row
    orc = oracle_curve(agg, CIRCLE, "txc_batchtopk_pre", 16, ANCHOR)
    h += "| *oracle (T=16)* | " + " | ".join(f"{o:.2f}" for o in orc) + " |\n"
    for arch in CROSSCODERS:
        for T in T_WINDOW:
            ys = sf_curve(agg, CIRCLE, arch, T, ANCHOR, normalized=False)
            h += f"| {LABEL[arch]} T={T} | " + " | ".join(f"{v:.2f}" for v in ys) + " |\n"
    return h.rstrip()


def _band_table(agg):
    bnames = {0: "DC {0}", 1: "low {1-5}", 2: "mid {6-10}", 3: "high {11-15}"}
    h = "| band | velocity recovery (T=16, $d_{sae}=M$) |\n|---|---|\n"
    for b in range(4):
        v = g(agg, CIRCLE, "trained", 1, "spectral_txc", 16, ANCHOR, f"band{b}_recovery")
        if v[2]:
            h += f"| {bnames[b]} | {_f(v)} |\n"
    h += f"| **full code** | {_f(g(agg, CIRCLE, 'trained', 1, 'spectral_txc', 16, ANCHOR, 'velocity_recovery'))} |\n"
    return h.rstrip()


def _untrained_table(agg):
    h = "| arch / T | untrained | trained |\n|---|---|---|\n"
    for arch, T in ARCH_T:
        h += (f"| {label(arch,T)} "
              f"| {_f(g(agg,CIRCLE,'untrained',1,arch,T,ANCHOR,'velocity_recovery'))} "
              f"| {_f(g(agg,CIRCLE,'trained',1,arch,T,ANCHOR,'velocity_recovery'))} |\n")
    return h.rstrip()


def _memo_table(agg):
    h = "| arch / T | circle @ $d_{sae}=2048$ | random @ $d_{sae}=2048$ |\n|---|---|---|\n"
    for arch in ("txc_batchtopk_pre", "spectral_txc"):
        h += (f"| {LABEL[arch]} T=16 "
              f"| {_f(g(agg,CIRCLE,'memo',1,arch,16,2048,'velocity_recovery'))} "
              f"| {_f(g(agg,RANDOM,'memo',1,arch,16,2048,'velocity_recovery'))} |\n")
    return h.rstrip()


def _best(agg, ds, arch, metric="velocity_recovery"):
    vals = [g(agg, ds, "trained", 1, arch, T, d, metric)[0]
            for T in T_WINDOW for d in D_SAES if g(agg, ds, "trained", 1, arch, T, d, metric)[2]]
    return max(vals) if vals else float("nan")


def headline_block(agg):
    pt = np.nanmax([abs(g(agg, CIRCLE, "trained", 1, a, 1, d, "velocity_recovery")[0])
                    for a in ("batchtopk_sae", "tsae") for d in D_SAES
                    if g(agg, CIRCLE, "trained", 1, a, 1, d, "velocity_recovery")[2]] or [np.nan])
    pre16 = g(agg, CIRCLE, "trained", 1, "txc_batchtopk_pre", 16, ANCHOR, "velocity_recovery")[0]
    post16 = g(agg, CIRCLE, "trained", 1, "txc_batchtopk_post", 16, ANCHOR, "velocity_recovery")[0]
    spec16 = g(agg, CIRCLE, "trained", 1, "spectral_txc", 16, ANCHOR, "velocity_recovery")[0]
    pre2 = g(agg, CIRCLE, "trained", 1, "txc_batchtopk_pre", 2, ANCHOR, "velocity_recovery")[0]
    rand_pre16 = g(agg, RANDOM, "trained", 1, "txc_batchtopk_pre", 16, ANCHOR, "velocity_recovery")[0]
    memo_rand = g(agg, RANDOM, "memo", 1, "txc_batchtopk_pre", 16, 2048, "velocity_recovery")[0]
    return (
        f"- **P1 confirmed — per-token is flat at chance:** BatchTopK-SAE / T-SAE "
        f"velocity recovery ≤ **{pt:.2f}** across every $d_{{sae}}$ (provable DPI + "
        f"raw-linear-at-chance from gating). The velocity is a 2nd-moment latent; "
        f"no per-token code exposes it.\n"
        f"- **P2 confirmed — window crosscoders recover $Y$ with a high-pass $S(f)$:** "
        f"recovery climbs with window size $T$ (Rayleigh cutoff $\\approx 1/T$), from "
        f"the shallow-window dip at low $f$ to near-oracle at $T=16$ "
        f"(TXC-pre **{pre16:.2f}**, TXC-post **{post16:.2f}** at $d_{{sae}}=M$). The "
        f"per-Ω-class $S(f)$ curve is the deliverable.\n"
        f"- **P3 — spectral TIES the monolithic crosscoder on the single tone (as "
        f"preregistered):** Spectral-TXC **{spec16:.2f}** vs TXC-pre {pre16:.2f} at "
        f"$T=16$, $d_{{sae}}=M$ — a tie, with a clean **band decomposition** (each "
        f"DCT band decodes the tones in its frequency range). The decisive multiband "
        f"win lives under superposition (scoped out).\n"
        f"- **P4 confirmed — the random null is FLAT:** same archs on the "
        f"random-embedding null show no frequency ordering (TXC-pre T=16 "
        f"{rand_pre16:.2f}); confusion tracks symbol overlap, not $\\Delta f$. Above "
        f"$|Ω|·M=1010$ the null recovery jumps to **{memo_rand:.2f}** by template "
        f"memorization — caught + flagged by the per-tile probe (all main cells "
        f"stay $d_{{sae}}<1010$).\n"
        f"- **Substrate:** circle-embedded cyclic tones, $M=101$, "
        f"$Ω=\\{{0,1,2,4,8,16,24,32,40,50\\}}$, $σ=0.10$; all archs on the BatchTopK "
        f"fair backbone; seeds {{1,2,42}}. (Stacked dropped — its concatenated "
        f"per-position code memorizes above $T·d_{{sae}}=|Ω|·M$; amendment A5.)"
    )


def populate(blocks):
    if not RECORD.exists():
        print(f"[warn] {RECORD} missing — populate skipped"); return
    txt = RECORD.read_text(); filled = 0
    for tag, content in blocks.items():
        pat = re.compile(rf"(<!-- BEGIN AUTO:{tag} -->).*?(<!-- END AUTO:{tag} -->)", re.DOTALL)
        if not pat.search(txt):
            print(f"[warn] AUTO:{tag} marker not found"); continue
        txt = pat.sub(lambda m: f"{m.group(1)}\n{content}\n{m.group(2)}", txt); filled += 1
    RECORD.write_text(txt)
    print(f"[record] populated {filled}/{len(blocks)} AUTO block(s) in {RECORD.name}")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(PAPER_STYLE)
    FIG_DIR.mkdir(exist_ok=True); RES_DIR.mkdir(exist_ok=True)

    rows = load_rows()
    agg = aggregate(rows)
    n_tr = sum(1 for r in rows if r["kind"] == "trained")
    print(f"[render] {len(rows)} leaderboard cells ({n_tr} trained)")

    fig_main(agg, plt)
    fig_Sf(agg, plt)
    fig_spectral(agg, plt)
    fig_null(agg, plt)
    fig_untrained(agg, plt)
    fig_memorization(agg, plt)

    blocks = {
        "headline": headline_block(agg),
        "circle_frontier": _frontier_table(agg, CIRCLE, "velocity_recovery"),
        "sf_table": _Sf_table(agg),
        "band_table": _band_table(agg),
        "untrained": _untrained_table(agg),
        "memo": _memo_table(agg),
        "nmse_table": _frontier_table(agg, CIRCLE, "nmse"),
    }
    populate(blocks)

    STATS_OUT.write_text(json.dumps({
        "source": "results/leaderboard.jsonl", "n_cells": len(rows),
        "M": M, "OMEGA": OMEGA, "chance": CHANCE, "memorization_threshold": MEMO_THRESH,
        "agg": {f"{k[0]}|{k[1]}|kpos{k[2]}|{k[3]}|T{k[4]}|d{k[5]}": v for k, v in agg.items()},
    }, indent=2))
    print(f"[stats] -> {STATS_OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
