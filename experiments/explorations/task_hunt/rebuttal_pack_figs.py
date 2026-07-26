"""rebuttal_pack_figs.py — REBUTTAL_PACK exhibit figures (mac-b, ACTMIX
overnight § 2). Zero-GPU: reads ONLY the canonical leaderboard + committed
screen/receipt JSONs. Conventions match figs 1–4 (Okabe-Ito, paired-t 95%
CI whiskers, log2 T axis, n annotated). Panel-lane recovery and
screen-instrument order receipts are DIFFERENT instruments in different
units — they get separate panels, never a shared axis.

  .venv/bin/python -m experiments.explorations.task_hunt.rebuttal_pack_figs
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE / "figs"
TCRIT = {3: 4.302652729911275, 6: 2.5705818366147395}
# Okabe-Ito (family palette of figs 1-4)
C_TXC, C_SAE, C_TSAE, C_UNTR = "#D55E00", "#0072B2", "#009E73", "#7f7f7f"
C_SHUF, C_FOREIGN = "#CC79A7", "#E69F00"

TT_FREEZES = {"50af78f121d4c4cbe5024c93aeaa5a4753daed11",
              "85c87fd7602fb36dd2e63488b8d33ad3311789e5"}


def _rows(ds, archs, seeds, freezes=None, k_pos=8):
    sel = {}
    with (ROOT / "results" / "leaderboard.jsonl").open() as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("datasource") != ds or r.get("seed") not in seeds:
                continue
            if r["arch"] not in archs:
                continue
            if freezes and r["code_version"]["commit_sha"] not in freezes:
                continue
            hp = r["training_cfg"]["arch_hparams_override"]
            if hp.get("k_pos") != k_pos:
                continue                      # primary arms only
            kind = "trained" if r["training_cfg"]["n_steps"] else "untrained"
            sel.setdefault((r["arch"], kind, hp.get("T", 1)), []).append(
                r["metrics"]["lambda_recovery"])
    return sel


def _stat(vals):
    v = np.array(vals, float)
    n = len(v)
    m = float(v.mean())
    half = (TCRIT[n] * float(v.std(ddof=1)) / math.sqrt(n)) if n > 1 else 0.0
    return m, half, n


def _recovery_panel(ax, sel, txc_arch, Ts, txc_label, title, claim_span=None,
                    claim_text=None, evidence=None):
    x = np.log2(Ts)
    if claim_span:
        ax.axvspan(np.log2(claim_span[0]) - 0.18, np.log2(claim_span[1]) + 0.35,
                   color="#F0E442", alpha=0.25, lw=0, zorder=0)
        if claim_text:
            ax.text(np.log2(math.sqrt(claim_span[0] * claim_span[1])) + 0.25,
                    ax.get_ylim()[1] * 0.0 + claim_text[1], claim_text[0],
                    ha="center", fontsize=7.5, color="#7a6a00")
    for kind, color, ls in (("trained", C_TXC, "-"),
                            ("untrained", C_UNTR, "--")):
        stats = [_stat(sel[(txc_arch, kind, T)]) for T in Ts]
        m, h, ns = zip(*stats)
        ax.errorbar(x, m, yerr=h, color=color, ls=ls, marker="o", ms=4,
                    capsize=3, lw=1.6, zorder=4,
                    label=f"{txc_label} {kind}")
        if kind == "trained":
            for xi, mi, ni in zip(x, m, ns):
                ax.annotate(f"n={ni}", (xi, mi), textcoords="offset points",
                            xytext=(8, 6), ha="left", fontsize=7,
                            color=color)
    for arch, color, lbl in (("batchtopk_sae", C_SAE,
                              "per-token SAE (8 act/token)"),
                             ("tsae", C_TSAE, "T-SAE (8 act/token)")):
        m, h, _ = _stat(sel[(arch, "trained", 1)])
        ax.axhline(m, color=color, lw=1.4, label=lbl)
        ax.axhspan(m - h, m + h, color=color, alpha=0.14, lw=0)
    if evidence is not None:
        ax.plot(x, evidence, color="k", ls=":", marker="x", ms=5, lw=1.2,
                label="visible-cue evidence line")
    ax.set_xticks(x)
    ax.set_xticklabels([str(T) for T in Ts])
    ax.set_xlabel("window length T (tokens)")
    ax.set_ylabel("recovery (v1, canonical leaderboard)")
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9)


def _order_panel(ax, Ts, win, shuf, foreign, tok, floor_per_T, ylabel, title,
                 floor_label):
    """Screen-instrument order receipt: POINT markers (zoom-safe — bars
    from zero would make these small deltas unreadable) win vs shuffled
    (vs foreign null where present) + per-token and visible-floor refs.
    Identity is marker shape + color (never color alone)."""
    x = np.arange(len(Ts), dtype=float)
    w = 0.22
    ax.plot(x - w, win, "o", ms=9, color=C_TXC, label="window probe",
            zorder=4)
    ax.plot(x, shuf, "s", ms=8, color=C_SHUF, mfc="none", mew=2,
            label="window probe, within-window SHUFFLED", zorder=4)
    if foreign is not None:
        ax.plot(x + w, foreign, "^", ms=8, color=C_FOREIGN,
                label="foreign-window null", zorder=4)
    for xi, wv, sv in zip(x, win, shuf):
        ax.plot([xi - w, xi], [wv, sv], color="#999999", lw=0.9, zorder=3)
    ax.axhline(tok, color=C_SAE, lw=1.3, ls="-", label="per-token probe")
    for xi, fl in zip(x, floor_per_T):
        ax.hlines(fl, xi - 1.9 * w, xi + 1.9 * w, color="k", ls=":", lw=1.3)
    ax.plot([], [], color="k", ls=":", lw=1.3, label=floor_label)
    for xi, wv, sv in zip(x, win, shuf):
        ax.annotate(f"sc {wv - sv:+.3f}", (xi - w / 2, max(wv, sv)),
                    textcoords="offset points", xytext=(0, 9), ha="center",
                    fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"T={T}" for T in Ts])
    ax.set_xlim(-0.75, len(Ts) - 0.25)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=9.5)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.legend(fontsize=7, loc="lower right", framealpha=0.9)


def fig_lambda():
    ds = "ward_real_lambda_base_l12"
    sel = _rows(ds, {"txc_batchtopk_pre", "batchtopk_sae", "tsae"},
                seeds={1, 2, 42, 3, 4, 5})
    for key, v in sel.items():
        assert len(v) in (3, 6), (key, len(v))
    scr = json.load(open(HERE / "lambda_intensity/results/lambda_screen.json"))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axL.set_ylim(-0.02, 0.30)
    _recovery_panel(
        axL, sel, "txc_batchtopk_pre", [2, 4, 8, 16],
        "TXC-pre (8 act/token)",
        "λ̂ backtracking intensity (R1-Distill-8B base, L12) — panel lane",
        claim_span=(8, 8), claim_text=("R22 bounded cell\n(T=8, n=6)", 0.265))
    Ts = [8, 16, 32]
    cells = {T: scr["cells"][f"base/hs13/lam_hat/T{T}"] for T in Ts}
    _order_panel(
        axR, Ts,
        win=[cells[T]["mean"]["auc"] for T in Ts],
        shuf=[cells[T]["shuf"]["auc"] for T in Ts],
        foreign=None,
        tok=scr["cells"]["base/hs13/lam_hat/tok"]["linear"]["auc"],
        floor_per_T=[scr["floors"]["lam_hat"]["auc"]] * len(Ts),
        ylabel="probe AUC (screen instrument)",
        title="λ̂ order receipt (screen, R10 class): shuffle cost is small",
        floor_label="visible-evidence floor")
    fig.suptitle("Exhibit A — backtracking intensity: recovery lane (left) "
                 "vs order receipt (right); different instruments, "
                 "different units", fontsize=9, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"rebuttal_lambda_exhibit.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"[rebuttal] wrote {p}")


def fig_ttrend():
    ds = "dial_real_ttrend_gpt2_l7"
    sel = _rows(ds, {"txc_batchtopk_post", "batchtopk_sae", "tsae"},
                seeds={3, 4, 5, 6, 7, 8}, freezes=TT_FREEZES)
    for key, v in sel.items():
        assert len(v) in (3, 6), (key, len(v))
    ev = json.loads((HERE / "diafaces/results/panel_evidence_line_tt.json")
                    .read_text())["per_T"]
    scr = json.load(open(HERE / "diafaces/results/screen_gpt2.json"))["cells"]
    Ts = [2, 4, 8, 16, 32]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.2, 4.2))
    axL.set_ylim(-0.06, 0.36)
    _recovery_panel(
        axL, sel, "txc_batchtopk_post", Ts,
        "TXC-post (8 act/WINDOW)",
        "ttrend turn-length trend (gpt2, L7) — fresh-seed lane {3–8}",
        claim_span=(16, 32), claim_text=("claiming zone\n(KEEP, n = 6)", 0.315),
        evidence=[ev[str(T)]["pearson_r"] for T in Ts])
    sTs = [16, 32]
    _order_panel(
        axR, sTs,
        win=[scr[f"tt/T{T}/win_linear"]["acc_test"] for T in sTs],
        shuf=[scr[f"tt/T{T}/win_shuf_linear"]["acc_test"] for T in sTs],
        foreign=[scr[f"tt/T{T}/win_foreign_linear"]["acc_test"] for T in sTs],
        tok=scr["tt/tok_linear"]["acc_test"],
        floor_per_T=[scr[f"tt/T{T}/visible_evidence_floor"]["acc_test"]
                     for T in sTs],
        ylabel="probe accuracy (screen instrument)",
        title="ttrend order receipt (screen, R26): within-dialogue shuffle",
        floor_label="visible-evidence floor")
    axR.set_ylim(0.30, 0.62)
    axR.legend(fontsize=7, loc="upper left", framealpha=0.9)
    fig.suptitle("Exhibit B — turn-length trend: recovery lane (left) vs "
                 "order receipt (right); different instruments, different "
                 "units", fontsize=9, y=1.005)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        p = OUT / f"rebuttal_ttrend_exhibit.{ext}"
        fig.savefig(p, dpi=200 if ext == "png" else None,
                    bbox_inches="tight")
        print(f"[rebuttal] wrote {p}")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    fig_lambda()
    fig_ttrend()
