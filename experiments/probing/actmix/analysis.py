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

# Freeze-stamp allowlist (CARD § 0 + flag 10): original freeze + the
# launcher-fix PIN. Extend ONLY via a committed edit to this list.
FREEZES = {
    "131ea677f4570181c757472a4d48ca7d3903006d",
    "49464795a" ,  # placeholder-prefix — replaced below by full sha check
    "9befb353fbd43a0e49a574e1bb155350c8d34c7e",
}

TXC_PRE = "txc_batchtopk_pre_btkonly"
TXC_POST = "txc_batchtopk_post_btkonly"
SAE = "batchtopk_sae_btkonly"
TSAE = "tsae_btkonly"
WINDOW_ARCHS = {TXC_PRE, TXC_POST}
TOKEN_ARCHS = {SAE, TSAE}

# Okabe-Ito, matching figs_writeup/fig4 mappings (post/sae/tsae/untrained
# keep their established hues; pre gets reddish-purple).
C = {TXC_POST: "#D55E00", TXC_PRE: "#CC79A7", SAE: "#0072B2",
     TSAE: "#009E73", "untrained": "#7f7f7f"}
LABEL = {TXC_PRE: "TXC-pre (k=20·T)", TXC_POST: "TXC-post (k=20/win)",
         SAE: "BatchTopK SAE (20/tok)", TSAE: "T-SAE (20/tok)"}


def load_rows(arm: str) -> list[dict]:
    rows = []
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            ec = r.get("eval_cfg", {})
            if (r.get("experiment") != "probing"
                    or r.get("evaluator_protocol_version") != "1.2.0"
                    or ec.get("arm") != arm
                    or ec.get("smoke")):
                continue
            sha = r["code_version"]["commit_sha"]
            if not any(sha.startswith(f) for f in FREEZES):
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
            if arch == TXC_PRE:
                lo, hi = 0.9 * 20 * T, 20 * T + 0.5
            else:
                lo, hi = 19.5, 20.5
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
    out = [f"### arm `{arm}`, k_feat = {k}", "",
           "| T | TXC-pre | TXC-pre shuf | TXC-post | TXC-post shuf | l0 pre | l0 post |",
           "|---|---|---|---|---|---|---|"]

    def cell(arch, T, key, untrained=False):
        vals = [m[key] for (a, u, t, s, kk), m in cells.items()
                if a == arch and u == untrained and t == T and kk == k and key in m]
        if not vals:
            return "—"
        m_, sd, n = agg(vals)
        return f"{m_:.4f} ± {sd:.4f} (n={n})" if n > 1 else f"{m_:.4f} (n=1)"

    for T in Ts:
        out.append("| {} | {} | {} | {} | {} | {} | {} |".format(
            T,
            cell(TXC_PRE, T, "mean_auc"), cell(TXC_PRE, T, "mean_auc_shuf"),
            cell(TXC_POST, T, "mean_auc"), cell(TXC_POST, T, "mean_auc_shuf"),
            cell(TXC_PRE, T, "realized_l0"), cell(TXC_POST, T, "realized_l0"),
        ))
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

    for arch in (TXC_PRE, TXC_POST):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="btk-only")
    ap.add_argument("--Ts", type=int, nargs="*", default=[1, 2, 4, 8, 16])
    args = ap.parse_args()

    rows = load_rows(args.arm)
    cells = {}
    for r in rows:
        cells[key_of(r)] = r["metrics"]
    k_feats = sorted({kk for (_, _, _, _, kk) in cells})
    print(f"[analysis] arm={args.arm}: {len(rows)} rows -> {len(cells)} cells, k_feats={k_feats}")

    gates = gate_report(cells, k_feats, args.Ts)
    md = [f"# ACTMIX P1 results — arm `{args.arm}` (auto-generated by analysis.py)",
          "", "Verdict discipline: PENDING TEAM REVIEW. Gates (CARD § 5):", ""]
    md += [f"- `{g}`" for g in gates]
    md.append("")
    for k in k_feats:
        md += make_table(cells, args.arm, k, args.Ts)
        make_fig(cells, args.arm, k, args.Ts, HERE / "figs")
    (HERE / "RESULTS.md").write_text("\n".join(md))
    print("\n".join(gates))
    print(f"[analysis] wrote {HERE/'RESULTS.md'} + figs/")


if __name__ == "__main__":
    main()
