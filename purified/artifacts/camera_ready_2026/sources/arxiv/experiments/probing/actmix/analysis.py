"""experiments/probing/actmix/analysis.py — ACTMIX P1 table + figs + gates.

Reads the CANONICAL leaderboard only (results/leaderboard.jsonl),
selects this card's rows (experiment=probing, protocol 1.2.0,
eval_cfg.arm, non-smoke, freeze-stamp allowlist), and emits:

1. ``RESULTS.md`` — Dmitry's table per (arm, k_feat): rows = T,
   columns TXC-pre | TXC-pre-shuf | TXC-post | TXC-post-shuf | SAE |
   TSAE (per-token arms are T-invariant bands printed once), mean ±
   seed-σ, realized-l0 per cell, untrained bands, T=1 anchor deltas.
2. ``figs/tsweep_<arm>_k<k>.{png,pdf}`` — WRITEUP-style (Okabe-Ito,
   fig4 conventions): TXC curves ± seed spread, shuffled overlays
   dashed, SAE/TSAE horizontal bands, untrained gray band; right
   panel = ordered−shuffled difference (Aniket's plot convention).
3. Validity-gate report (CARD § 5 G1–G5) printed + embedded in
   RESULTS.md — any FAIL is flagged loudly; nothing is dropped
   silently.

Run: .venv/bin/python -m experiments.probing.actmix.analysis [--arm btk-only]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

# Freeze-stamp allowlist (CARD § 0 + flag 10): every commit on the
# first-parent lineage from the original freeze to HEAD. Rows are only
# ever produced by PIN-asserted launches (launcher checks HEAD == PIN,
# PIN ∈ origin/arxiv ancestry, clean tree), and mid-queue checkpoint
# commits advance HEAD under running cells — so the honest allowlist is
# the lineage itself. A static sha list rotted twice as relaunch PINs
# advanced past it (dropping trained rows silently); never again.
FREEZE_ROOT = "131ea677f4570181c757472a4d48ca7d3903006d"
_FREEZES: set[str] | None = None


def freeze_allowlist() -> set[str]:
    global _FREEZES
    if _FREEZES is None:
        import subprocess
        out = subprocess.run(
            ["git", "rev-list", "--first-parent", f"{FREEZE_ROOT}^..HEAD"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        _FREEZES = set(out.split())
    return _FREEZES

TXC_PRE = "txc_batchtopk_pre_btkonly"
TXC_POST = "txc_batchtopk_post_btkonly"
PAPER_V1T = "paper_txc_base_v1t"
SAE = "batchtopk_sae_btkonly"
TSAE = "tsae_btkonly"
WINDOW_ARCHS = {TXC_PRE, TXC_POST, PAPER_V1T}
TOKEN_ARCHS = {SAE, TSAE}

# Okabe-Ito, matching figs_writeup/fig4 mappings (post/sae/tsae/untrained
# keep their established hues; pre gets reddish-purple).
C = {TXC_POST: "#D55E00", TXC_PRE: "#CC79A7", PAPER_V1T: "#E69F00",
     SAE: "#0072B2", TSAE: "#009E73", "untrained": "#7f7f7f"}
LABEL = {TXC_PRE: "TXC-pre (k=20·T)", TXC_POST: "TXC-post (k=20/win)",
         PAPER_V1T: "paper §5.1 ReLU∘TopK (k=20·T)",
         SAE: "BatchTopK SAE (20/tok)", TSAE: "T-SAE (20/tok)"}
# Fixed plot/table order for window archs (present-only rendering).
WINDOW_ORDER = (TXC_PRE, TXC_POST, PAPER_V1T)
SHORT = {TXC_PRE: "TXC-pre", TXC_POST: "TXC-post", PAPER_V1T: "paper-v1t"}
L0SHORT = {TXC_PRE: "pre", TXC_POST: "post", PAPER_V1T: "v1t"}


def load_rows(arm: str) -> list[dict]:
    rows = []
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            ec = r.get("eval_cfg", {})
            if (r.get("experiment") != "probing"
                    or r.get("evaluator_protocol_version") != "1.2.0"
                    or ec.get("arm") != arm
                    or ec.get("smoke")
                    # instrument rows, never grid cells:
                    or ec.get("positive_control")):
                continue
            # Canonical-hparam rows only: the grid varies T alone. Rows
            # with other overrides (width-match tsae d_sae=18432, thin-
            # pool controls) belong to their own cards and must not pool
            # into these bands.
            ov = (r.get("training_cfg") or {}).get("arch_hparams_override") or {}
            if set(ov) - {"T"}:
                continue
            if r["code_version"]["commit_sha"] not in freeze_allowlist():
                continue
            rows.append(r)
    return rows


def key_of(r: dict):
    hp = r["training_cfg"].get("arch_hparams_override") or {}
    T = int(hp.get("T", 1))
    untrained = r["training_cfg"]["n_steps"] == 0
    return (r["arch"], untrained, T, int(r["seed"]), int(r["eval_cfg"]["k_feat"]))


def agg(vals):
    a = np.asarray(vals, float)
    return a.mean(), (a.std(ddof=1) if len(a) > 1 else 0.0), len(a)


def gate_report(cells: dict, k_feats, Ts) -> list[str]:
    lines = []

    def flag(ok, msg):
        lines.append(("PASS  " if ok else "FAIL  ") + msg)

    for (arch, untrained, T, seed, k), m in sorted(cells.items()):
        # G4 suite integrity
        if m["n_tasks"] != 38:
            flag(False, f"G4 n_tasks={m['n_tasks']} != 38 at {arch}/T{T}/s{seed}/k{k}")
        # G1 realized-l0 (trained cells only; untrained = fallback path, reported not gated)
        if not untrained:
            l0 = m["realized_l0"]
            if arch == PAPER_V1T:
                # Per-SAMPLE TopK: k_win is a hard cap; ReLU-after-TopK
                # zero-picks pull realized l0 strictly below it (E1).
                lo, hi = 0.9 * 20 * T, 20 * T + 0.5
            elif arch == TXC_PRE:
                # BatchTopK guarantees k_win·B per BATCH, so per-window
                # realized l0 legitimately overshoots k_win (measured
                # +5–6% at T6–T16); symmetric 10% band.
                lo, hi = 0.9 * 20 * T, 1.1 * 20 * T
            else:
                # TXC_POST / SAE / TSAE: all batch-level BatchTopK at
                # k=20 per window/token — same ±10% semantics.
                lo, hi = 18.0, 22.0
            if not (lo <= l0 <= hi):
                flag(False, f"G1 l0={l0:.2f} outside [{lo:.1f},{hi:.1f}] at {arch}/T{T}/s{seed}/k{k}")
        # G2 identity for per-token archs
        if arch in TOKEN_ARCHS and "mean_auc_shuf" in m:
            if m["mean_auc_shuf"] != m["mean_auc"]:
                flag(False, f"G2 shuffle-identity violated at {arch}/s{seed}/k{k}")
    if not any(l.startswith("FAIL") for l in lines):
        lines.append("PASS  G1/G2/G4 hold on all present cells")

    # G3 untrained < trained (seed 42 pairs)
    for (arch, untrained, T, seed, k), m in sorted(cells.items()):
        if untrained:
            tw = cells.get((arch, False, T, seed, k))
            if tw and tw["mean_auc"] <= m["mean_auc"]:
                lines.append(f"FAIL  G3 untrained {m['mean_auc']:.3f} >= trained "
                             f"{tw['mean_auc']:.3f} at {arch}/T{T}/s{seed}/k{k}")

    # G5 anchor: TXC@T1 vs SAE (trained, per k)
    for k in k_feats:
        sae_aucs = [m["mean_auc"] for (a, u, T, s, kk), m in cells.items()
                    if a == SAE and not u and kk == k]
        if len(sae_aucs) >= 2:
            sae_m, sae_sd, _ = agg(sae_aucs)
            for txc in WINDOW_ARCHS:
                t1 = [m["mean_auc"] for (a, u, T, s, kk), m in cells.items()
                      if a == txc and not u and T == 1 and kk == k]
                if t1:
                    d = abs(np.mean(t1) - sae_m)
                    ok = d <= 3 * max(sae_sd, 1e-9)
                    lines.append(("PASS  " if ok else "CAVEAT")
                                 + f" G5 anchor |{txc}@T1−SAE|={d:.4f} vs 3σ_SAE={3*sae_sd:.4f} (k={k})")
    return lines


def make_table(cells: dict, arm: str, k: int, Ts) -> list[str]:
    present = [a for a in WINDOW_ORDER
               if any(key[0] == a and not key[1] for key in cells)]
    # btk-only arm has exactly {TXC_PRE, TXC_POST} present, so its table
    # is unchanged by the dynamic column list (protected render).
    hdr = ("| T | " + " | ".join(f"{SHORT[a]} | {SHORT[a]} shuf"
                                 for a in present)
           + " | " + " | ".join(f"l0 {L0SHORT[a]}" for a in present) + " |")
    out = [f"### arm `{arm}`, k_feat = {k}", "", hdr,
           "|---" * (1 + 3 * len(present)) + "|"]

    def cell(arch, T, key, untrained=False):
        vals = [m[key] for (a, u, t, s, kk), m in cells.items()
                if a == arch and u == untrained and t == T and kk == k and key in m]
        if not vals:
            return "—"
        m_, sd, n = agg(vals)
        return f"{m_:.4f} ± {sd:.4f} (n={n})" if n > 1 else f"{m_:.4f} (n=1)"

    for T in Ts:
        vals = [cell(a, T, key) for a in present
                for key in ("mean_auc", "mean_auc_shuf")]
        vals += [cell(a, T, "realized_l0") for a in present]
        out.append("| " + str(T) + " | " + " | ".join(vals) + " |")
    out += ["", "| per-token band | AUC (T-invariant) | realized l0 |", "|---|---|---|"]
    for arch in (SAE, TSAE):
        out.append(f"| {LABEL[arch]} | {cell(arch, 1, 'mean_auc')} | {cell(arch, 1, 'realized_l0')} |")
    out += ["", "untrained twins (seed 42): "
            + "; ".join(f"{LABEL.get(a, a)}@T{T}: {m['mean_auc']:.3f}"
                        for (a, u, T, s, kk), m in sorted(cells.items())
                        if u and kk == k) or "(none yet)", ""]
    return out


def make_fig(cells: dict, arm: str, k: int, Ts, outdir: Path):
    fig, (ax, axd) = plt.subplots(
        1, 2, figsize=(10.4, 4.2), gridspec_kw={"width_ratios": [1.6, 1.0]})
    x = np.log2(Ts)

    for arch in WINDOW_ORDER:
        for key, ls, mk, lbl in (("mean_auc", "-", "o", LABEL[arch]),
                                 ("mean_auc_shuf", "--", "s", f"{LABEL[arch]} shuffled")):
            ys, es, xs = [], [], []
            for i, T in enumerate(Ts):
                vals = [m[key] for (a, u, t, s, kk), m in cells.items()
                        if a == arch and not u and t == T and kk == k and key in m]
                if vals:
                    m_, sd, n = agg(vals)
                    xs.append(x[i]); ys.append(m_); es.append(sd)
            if xs:
                ax.errorbar(xs, ys, yerr=es, color=C[arch], ls=ls, marker=mk,
                            ms=4, lw=1.6, capsize=2, label=lbl)
        # difference panel
        ds, xs = [], []
        for i, T in enumerate(Ts):
            o = [m["mean_auc"] for (a, u, t, s, kk), m in cells.items()
                 if a == arch and not u and t == T and kk == k]
            sh = [m.get("mean_auc_shuf") for (a, u, t, s, kk), m in cells.items()
                  if a == arch and not u and t == T and kk == k]
            sh = [v for v in sh if v is not None]
            if o and sh:
                xs.append(x[i]); ds.append(np.mean(o) - np.mean(sh))
        if xs:
            axd.plot(xs, ds, color=C[arch], marker="o", ms=4, lw=1.6,
                     label=LABEL[arch])

    for arch in (SAE, TSAE):
        vals = [m["mean_auc"] for (a, u, t, s, kk), m in cells.items()
                if a == arch and not u and kk == k]
        if vals:
            m_, sd, _ = agg(vals)
            ax.axhline(m_, color=C[arch], lw=1.4, label=LABEL[arch] + " band")
            ax.axhspan(m_ - sd, m_ + sd, color=C[arch], alpha=0.14, lw=0)

    untr = [m["mean_auc"] for (a, u, t, s, kk), m in cells.items() if u and kk == k]
    if untr:
        ax.axhline(np.mean(untr), color=C["untrained"], lw=1.2, ls=":",
                   label="untrained twins")

    ax.set_xticks(x); ax.set_xticklabels([str(t) for t in Ts])
    ax.set_xlabel("window length T"); ax.set_ylabel("mean ROC AUC (38 tasks)")
    ax.set_title(f"§5.1 sparse probing — {arm} arm, k_feat={k}")
    ax.legend(fontsize=7.0, loc="lower right", framealpha=0.9)
    axd.axhline(0, color="k", lw=0.8)
    axd.set_xticks(x); axd.set_xticklabels([str(t) for t in Ts])
    axd.set_xlabel("window length T"); axd.set_ylabel("ordered − shuffled AUC")
    axd.set_title("order dependence (fixed probe)")
    axd.legend(fontsize=7.0, framealpha=0.9)
    fig.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"tsweep_{arm}_k{k}.{ext}",
                    dpi=200 if ext == "png" else None)
    plt.close(fig)


# The paper's probe suite is 38-task SAEBench+CT; the camera-ready
# FIGURE headline excludes the two CT tasks (paper-history e77574ffd;
# ≈ −0.027 uniform level shift when included). Directive 89fd5c292:
# headline fig = SAEBench-36 CT-excluded, 38-task raw kept as the
# robustness twin. The evaluator's mean_auc is the raw 38 mean (no
# FLIP); FLIP is moot for the 36 headline (CT excluded) and the twin
# stays raw = the main-text convention.
CT_TASKS = ("winogrande_correct_completion", "wsc_coreference")


def _agg_mean(m: dict, prefix: str, exclude=()) -> float | None:
    p = prefix + "__"
    vals = [v for kk, v in m.items()
            if kk.startswith(p) and kk[len(p):] not in exclude]
    return float(np.mean(vals)) if vals else None


def make_writeup_fig(cells: dict, k: int, Ts, tag: str, outdir: Path,
                     pair_style: str = "mono", exclude=(),
                     n_tasks_label: str = "38 tasks", suffix: str = "",
                     identity_xy: tuple = (0.03, 0.62), arch: str = TXC_PRE):
    """Aniket-template shuffle T-sweep (059a66239 P1 deliverable).

    Knob-for-knob twin of runpod-2's RLHF renderer (421f6fa37,
    `actmix_rlhf/render_writeup_fig.py` — first-to-freeze sets the pair
    template): figsize 5.4x3.7, ONE hue for the TXC family with
    linestyle carrying the order condition (shuffled = open markers),
    faint per-seed traces, per-T seed-mean +- sd, log2 x-scale on real T
    values, "T=16 - T=1" annotation top-left, T=1 shuffle==identity
    note, ragged seed coverage auto-disclosed bottom-right. In the
    writeup-pair namespace #D55E00 = "the TXC family" (matches the RLHF
    fig); actmix-internal figs keep the pre/post hue split.
    """
    HUE = "#D55E00"
    # Pair-hue knob mirrored verbatim from the RLHF twin (2200a346d):
    # mono = single pair-hue, linestyle carries the order condition;
    # blueorange = Aniket backtracking-fig sibling styling (shuffled in
    # house blue #0072B2). Meeting decision — LOG 36655341a.
    colors = {"ordered": HUE,
              "shuffled": HUE if pair_style == "mono" else "#0072B2"}
    pts = {}   # (T, seed) -> {"ordered": v, "shuffled": v}
    for (a, u, T, s, kk), m in cells.items():
        if a == arch and not u and kk == k and "mean_auc" in m:
            o = _agg_mean(m, "auc", exclude)
            sh = _agg_mean(m, "auc_shuf", exclude)
            if sh is None and m.get("shuffle_identity"):
                sh = o
            pts[(T, s)] = {"ordered": o, "shuffled": sh}
    if not pts:
        print(f"[analysis] writeup fig skipped: no {SHORT.get(arch, arch)} rows")
        return
    seeds = sorted({s for (_, s) in pts})

    def mean_sd(field):
        Ts_ = sorted({T for (T, _) in pts})
        mu, sd, n = [], [], []
        for T in Ts_:
            vals = [v[field] for (t, _), v in pts.items()
                    if t == T and v[field] is not None]
            n.append(len(vals))
            mu.append(float(np.mean(vals)))
            sd.append(float(np.std(vals, ddof=1)) if len(vals) > 1 else None)
        return Ts_, mu, sd, n

    fig, ax = plt.subplots(figsize=(5.4, 3.7))

    for s in seeds:            # faint per-seed lines, both conditions
        for field, ls in (("ordered", "-"), ("shuffled", "--")):
            ss = sorted((T, v[field]) for (T, sd_), v in pts.items()
                        if sd_ == s and v[field] is not None)
            if len(ss) > 1:
                ax.plot(*zip(*ss), ls, color=colors[field], alpha=0.25,
                        lw=1, zorder=1)

    for field, ls, mk, mfc, label in (
            ("ordered", "-", "o", None, "ordered"),
            ("shuffled", "--", "s", "white", "within-window shuffled")):
        c = colors[field]
        Ts_, mu, sd, n = mean_sd(field)
        ax.plot(Ts_, mu, ls, color=c, lw=2, marker=mk, ms=6,
                mfc=mfc or c, mec=c, label=label, zorder=3)
        for T, m_, s_ in zip(Ts_, mu, sd):
            if s_ is not None:
                ax.errorbar(T, m_, yerr=s_, color=c, capsize=3,
                            lw=1.2, zorder=2)

    Ts_, mu, _, n = mean_sd("ordered")
    if 16 in Ts_ and 1 in Ts_:
        delta = mu[Ts_.index(16)] - mu[Ts_.index(1)]
        ax.annotate(f"T=16 − T=1: {delta:+.3f}", xy=(0.03, 0.95),
                    xycoords="axes fraction", ha="left", va="top",
                    fontsize=9)
    if (1, 42) in pts:
        # xytext lower than the RLHF twin's (0.03, 0.83): this face's
        # curves occupy the top-left, the twin's don't. Position is the
        # one knob that tracks data geometry; all else stays paired.
        ax.annotate("T=1: shuffle ≡ identity",
                    xy=(1, pts[(1, 42)]["ordered"]),
                    xytext=identity_xy, textcoords="axes fraction",
                    fontsize=8, color="#555555",
                    arrowprops=dict(arrowstyle="-", color="#999999", lw=0.8))

    cov = " ".join(f"T{T}:n={c}" for T, c in zip(Ts_, n))
    tag_note = ("INTERIM — remaining seeds in flight" if tag == "interim"
                else "FINAL — seeds {42, 1, 2}")
    ax.annotate(f"{tag_note} · {cov}", xy=(0.99, 0.02),
                xycoords="axes fraction", ha="right", va="bottom",
                fontsize=6.5, color="#777777")

    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts_)
    ax.set_xticklabels([str(T) for T in Ts_])
    ax.minorticks_off()
    ax.set_xlabel("T (window length)")
    ax.set_ylabel(f"mean probing AUC (k = {k}, {n_tasks_label})")
    ax.grid(True, alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=8, loc="lower right",
              bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    outdir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"fig_probing_shuffle_tsweep{suffix}.{ext}",
                    dpi=300 if ext == "png" else None)
    plt.close(fig)
    print(f"[analysis] writeup fig{suffix or ' (headline)'} ({tag}, "
          f"{n_tasks_label}): seeds {seeds}; coverage {cov}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="btk-only")
    ap.add_argument("--Ts", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    ap.add_argument("--writeup", choices=["interim", "final"],
                    help="also render figs_writeup/fig_probing_shuffle_tsweep"
                         " (k=20 headline, Aniket template)")
    ap.add_argument("--pair-style", choices=("mono", "blueorange"),
                    default="mono",
                    help="mono = single pair-hue (linestyle carries the "
                         "order condition); blueorange = Aniket "
                         "backtracking-fig sibling styling (shuffled in "
                         "house blue #0072B2). Meeting decision — see "
                         "LOG 36655341a.")
    args = ap.parse_args()

    rows = load_rows(args.arm)
    # dict insert order = leaderboard append order (flock-serialized), so
    # duplicate cells — e.g. untrained twins re-run across relaunch PINs —
    # resolve to the LATEST row per display key. Nothing else dedupes.
    cells = {}
    for r in rows:
        cells[key_of(r)] = r["metrics"]
    k_feats = sorted({kk for (_, _, _, _, kk) in cells})
    print(f"[analysis] arm={args.arm}: {len(rows)} rows -> {len(cells)} cells, k_feats={k_feats}")

    gates = gate_report(cells, k_feats, args.Ts)
    md = [f"# ACTMIX P1 results — arm `{args.arm}` (auto-generated by analysis.py)",
          "",
          # BINDING panel-convention pin (LOG a9a7de92f): the paper's
          # headline is the 38-task mean; the writeup/rebuttal figs are
          # SAEBench-36 (CT pair excluded). Same ckpts read ~+0.03 higher
          # on the 36-task panel — self-declare the panel so no exhibit
          # can be cross-quoted against the other by accident.
          "**Panel: 38 tasks, raw** (the paper's convention — `mean_auc` "
          "as evaluated, no exclusions). The `figs_writeup/` twins use "
          f"**SAEBench-36** ({len(CT_TASKS)} CT tasks excluded) and read "
          "~0.03 HIGHER on the same checkpoints. Never cross-quote panels.",
          "", "Verdict discipline: PENDING TEAM REVIEW. Gates (CARD § 5):", ""]
    md += [f"- `{g}`" for g in gates]
    md.append("")
    for k in k_feats:
        md += make_table(cells, args.arm, k, args.Ts)
        make_fig(cells, args.arm, k, args.Ts, HERE / "figs")
    # Per-arm results file (a second arm's run must not clobber the
    # first's tables); RESULTS.md stays the btk-only headline alias.
    (HERE / f"RESULTS_{args.arm}.md").write_text("\n".join(md))
    if args.arm == "btk-only":
        (HERE / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(gates))
    print(f"[analysis] wrote {HERE}/RESULTS_{args.arm}.md + figs/")

    if args.writeup and args.arm == "paper-faithful":
        # Paper-faithful writeup twins (hub order 12:34 London, LOG
        # 7a7ee52c8): same pair template, arch = paper_txc_base_v1t,
        # `_pf` namespace so the btk figs are never clobbered.
        ID_XY = {5: (0.03, 0.87)}
        for kf in k_feats:
            make_writeup_fig(cells, kf, args.Ts, args.writeup,
                             ROOT / "figs_writeup",
                             pair_style=args.pair_style,
                             exclude=CT_TASKS, n_tasks_label="SAEBench-36",
                             suffix=f"_pf_k{kf}",
                             identity_xy=ID_XY.get(kf, (0.03, 0.62)),
                             arch=PAPER_V1T)
    elif args.writeup:
        # Per-k plot family (ccddf01be): one SAEBench-36 headline plot
        # per k present in the rows — k grid extensions get their plot
        # automatically on landing. Base name = k20 alias (link
        # stability) + raw-38 robustness twin (89fd5c292).
        # identity-note position tracks each k's data geometry (the
        # k20 curve owns the top-left; k5's U-shape owns the mid-left)
        ID_XY = {5: (0.03, 0.87)}
        for kf in k_feats:
            make_writeup_fig(cells, kf, args.Ts, args.writeup,
                             ROOT / "figs_writeup",
                             pair_style=args.pair_style,
                             exclude=CT_TASKS, n_tasks_label="SAEBench-36",
                             suffix=f"_k{kf}",
                             identity_xy=ID_XY.get(kf, (0.03, 0.62)))
        make_writeup_fig(cells, 20, args.Ts, args.writeup,
                         ROOT / "figs_writeup", pair_style=args.pair_style,
                         exclude=CT_TASKS, n_tasks_label="SAEBench-36")
        make_writeup_fig(cells, 20, args.Ts, args.writeup,
                         ROOT / "figs_writeup", pair_style=args.pair_style,
                         n_tasks_label="38 tasks, raw", suffix="_38task")


if __name__ == "__main__":
    main()
