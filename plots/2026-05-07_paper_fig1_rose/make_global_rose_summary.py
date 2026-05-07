"""Build the global rose-diagram summary across all six case studies.

Per arch, one normalized score per case study (each axis = the LHS-subfig
headline metric for that section, min-max normalised to [0, 1] across the
five target architectures). Output to ``figs/global_rose_summary.pdf``.

Headline metrics (all extracted from per-component results files committed
on origin/final, except c5 which is also there but with multi-seed std):

  1. Synthetic 1 (Denoising, c2 Setup B)  →  R²_global from
     ``temp_xc_tex/notes/c2_synthetic/data/denoising_probe_results.json``
     (per-arch best across (t_label, k_pos), averaged over seeds).
  2. Synthetic 2 (Coupling, c2 Setup D, n_parents=10)  →  peak gAUC from
     ``temp_xc_tex/notes/c2_synthetic/data/setup_d_leaderboard.jsonl``
     (per-arch peak across (t_label, k_pos), averaged over seeds).
  3. Sparse Probing (c3, Gemma-2-2B-IT)  →  ``mean AUC across log-k_feats``
     (AUC-of-AUC) from ``purified/experiments/c3_probing/results.json``.
  4. Backtracking (c7)  →  peak Δgc from
     ``purified/experiments/c7_backtracking/results.json``.
  5. RLHF (c5)  →  ``mean_peak_success_grade_at_coh_1.75`` from
     ``purified/experiments/c5_steering/results.json``.
  6. Emergent Misalignment (c6, 7B-medical)  →  Δalign at coh≥70 (max−min),
     averaged across the two SAE-training seeds, computed from the same
     pipeline as ``make_c6_em_figures.py`` (Wang stage-4 canonical grid +
     dense ±110…±200 extension).

Usage::

    python scripts/make_global_rose_summary.py

Outputs ``figs/global_rose_summary.{pdf,png}`` plus a sidecar JSON at
``figs/global_rose_summary_data.json`` recording the raw and normalised
per-arch values per axis (for reproducibility).
"""
from __future__ import annotations

import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_DATA = Path("/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc")
REPO_TEX  = Path("/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc_tex")
FIGS_OUT  = REPO_TEX / "figs"

ARCHS = ["topk_sae", "tsae_paper", "tfa", "txc_base", "txc_pro"]
ARCH_LABEL = {
    "topk_sae":   "TopK SAE",
    "tsae_paper": "T-SAE",
    "tfa":        "TFA",
    "txc_base":   "TXC",
    "txc_pro":    "TXC-pro",
}
ARCH_COLOR = {
    "topk_sae":   "#4477AA",
    "tsae_paper": "#229922",
    "tfa":        "#888888",
    "txc_base":   "#EE6677",
    "txc_pro":    "#CC8800",
}


def git_show(branch_path: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(REPO_DATA), "show", branch_path], text=True
    )


# ── 1. Synthetic Denoising (Setup B) — R²_global per arch ─────────

def synth_denoising_r2() -> dict[str, float]:
    p = REPO_TEX / "notes/c2_synthetic/data/denoising_probe_results.json"
    rows = json.loads(p.read_text())
    g: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        a = r.get("arch_name")
        t = r.get("t_label", "default")
        k = r.get("k_pos")
        v = r.get("lp_mean_global_r2")
        if v is None:
            continue
        g[(a, t, k)].append(float(v))
    # per-arch peak: max over (t,k) of seed-mean
    out: dict[str, float] = {}
    for (a, t, k), vs in g.items():
        m = float(np.mean(vs))
        if m > out.get(a, -1e9):
            out[a] = m
    # Map tfa_pos → tfa for the rose-diagram label.
    if "tfa_pos" in out:
        out["tfa"] = out.pop("tfa_pos")
    return out


# ── 2. Synthetic Coupling (Setup D, np=10) — peak gAUC per arch ──

def synth_coupling_gauc() -> dict[str, float]:
    p = REPO_TEX / "notes/c2_synthetic/data/setup_d_leaderboard.jsonl"
    g: dict[tuple, list[float]] = defaultdict(list)
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        if r.get("eval_cfg", {}).get("n_parents") != 10:
            continue
        a = r.get("arch")
        t = r["eval_cfg"].get("t_label", "default")
        k = r["eval_cfg"].get("k_pos")
        v = r.get("metrics", {}).get("gauc")
        if v is None:
            continue
        g[(a, t, k)].append(float(v))
    out: dict[str, float] = {}
    for (a, t, k), vs in g.items():
        m = float(np.mean(vs))
        if m > out.get(a, -1e9):
            out[a] = m
    if "tfa_pos" in out:
        out["tfa"] = out.pop("tfa_pos")
    return out


# ── 3. c3 Sparse Probing — AUC-of-AUC over log-k_feats sweep ─────

def c3_auc_of_auc() -> dict[str, float]:
    raw = json.loads(git_show("origin/final:purified/experiments/c3_probing/results.json"))
    out: dict[str, float] = {}
    for arch, by_k in raw.items():
        items = sorted((int(k), v["mean_auc"]) for k, v in by_k.items())
        ks = np.array([k for k, _ in items], dtype=float)
        aucs = np.array([v for _, v in items])
        x = np.log2(ks)
        out[arch] = float(np.trapz(aucs, x) / (x[-1] - x[0]))
    # The c3 sweep splits TXC-base into T={5, 10, 20}; pick T=5 (canonical) as
    # the headline for "TXC" to match the rest of the paper.
    if "txc_base_T5" in out:
        out["txc_base"] = out["txc_base_T5"]
    return out


# ── 4. c7 Backtracking — peak Δgc per arch ───────────────────────

def c7_peak_dgc() -> dict[str, float]:
    raw = json.loads(git_show("origin/final:purified/experiments/c7_backtracking/results.json"))
    return {a: float(d["peak"]) for a, d in raw.get("delta_gc", {}).items()}


# ── 5. c5 RLHF — peak success grade @ coh1.75 ────────────────────

def c5_peak_grade() -> dict[str, float]:
    raw = json.loads(git_show("origin/final:purified/experiments/c5_steering/results.json"))
    return {a: float(d["mean_peak_success_grade_at_coh_1.75"])
            for a, d in raw.get("by_arch", {}).items()}


# ── 6. c6 EM — Δalign at coh≥70 (max - min) per arch ─────────────

def c6_em_delta_align() -> dict[str, float]:
    """Compute the same Δalign@coh≥70 metric as ``make_c6_em_figures.py`` for
    the four canonical 7B-medical archs (sae_arditi → 'topk_sae' here for the
    rose; T-SAE / TXC-base / TXC-pro / TFA also from the same pipeline).
    Averaged across the two paired SAE seeds."""
    REDTEAM   = REPO_DATA / "local_data/c6_redteam/h100_em_4/sweep_outputs"
    OVERNIGHT = REPO_DATA / "dmitry/pre_purified/c6_em_overnight/sweep_outputs"
    OVERNIGHT_RUNS = REPO_DATA / "dmitry/pre_purified/c6_em_overnight/runs"
    CELLS = [
        ("sae_arditi",  1,  "9b011dfeea88f8af"),
        ("sae_arditi",  42, "c0da3ed8794554a1"),
        ("txc_base",    1,  "2016074933c41e7f"),
        ("txc_base",    42, "88a4ddf6819d8057"),
        ("txc_pro",     1,  "e561456612fe29ff"),
        ("txc_pro",     42, "0689047ce9bce927"),
        ("tsae_paper",  1,  "6f6d047132771676"),
        ("tsae_paper",  42, "819604d52a131b54"),
        ("tfa",         1,  "e3b029548824f240"),
    ]
    CANONICAL_ON_FINAL = {"sae_arditi", "txc_base"}
    COH_FLOOR = 70.0
    results: dict[str, list[float]] = defaultdict(list)
    for arch, seed, tk in CELLS:
        # canonical stage-4
        if arch in CANONICAL_ON_FINAL:
            canon = json.loads(git_show(f"origin/final:purified/results/runs/c6_{tk}/stage4_frontier.json"))
        else:
            canon = json.loads((OVERNIGHT_RUNS / f"c6_{tk}" / "stage4_frontier.json").read_text())
        # extended (try redteam first, then overnight)
        ext = None
        for root in (REDTEAM, OVERNIGHT):
            f = root / f"c6_{tk}" / "wang_full_extended.json"
            if f.exists():
                ext = json.loads(f.read_text())
                break
        # gather (align, coh) rows
        rows = []
        for fin in canon["finalists"]:
            for r in fin["rows"]:
                a = r.get("mean_align"); c = r.get("mean_coh")
                if a is not None and c is not None:
                    rows.append((float(a), float(c)))
        if ext:
            for r in ext.get("rows", []):
                a = r.get("mean_align"); c = r.get("mean_coh")
                if a is not None and c is not None:
                    rows.append((float(a), float(c)))
        eligible = [a for a, c in rows if c >= COH_FLOOR]
        if not eligible:
            continue
        results[arch].append(max(eligible) - min(eligible))
    out = {a: float(np.mean(v)) for a, v in results.items()}
    # Rename sae_arditi → topk_sae for the rose to match the global arch label set.
    if "sae_arditi" in out:
        out["topk_sae"] = out.pop("sae_arditi")
    return out


# ── Assemble the table + normalize ───────────────────────────────

CASE_STUDIES = [
    ("Denoising",       r"$R^2_{\mathrm{global}}$",                            synth_denoising_r2),
    ("Coupling",        r"peak $g$AUC",                                        synth_coupling_gauc),
    ("Sparse Probing",  r"$\overline{\mathrm{AUC}}$ over $\log_2 k$",          c3_auc_of_auc),
    ("Backtracking",    r"peak $\Delta gc$",                                   c7_peak_dgc),
    ("RLHF",            r"peak grade @coh$\geq$1.75",                          c5_peak_grade),
    ("EM",              r"$\Delta$align @coh$\geq$70",                         c6_em_delta_align),
]


def main():
    print("== loading per-axis raw values ==")
    per_axis: list[tuple[str, str, dict[str, float]]] = []
    for label, metric, fn in CASE_STUDIES:
        data = fn()
        per_axis.append((label, metric, data))
        print(f"  {label:18}  {metric}")
        for a in ARCHS:
            v = data.get(a)
            print(f"      {ARCH_LABEL[a]:10}  raw = {v if v is None else f'{v:.4f}'}")

    # Normalise each axis across the 5 archs.
    norm: dict[str, dict[str, float]] = {a: {} for a in ARCHS}
    for label, _metric, data in per_axis:
        vals = [data.get(a) for a in ARCHS]
        present = [v for v in vals if v is not None]
        if not present:
            continue
        mn, mx = min(present), max(present)
        rng = mx - mn if mx > mn else 1.0
        for a in ARCHS:
            v = data.get(a)
            norm[a][label] = (v - mn) / rng if v is not None else None

    print()
    print("== normalised values (min-max per axis) ==")
    print("axis " + "  ".join(f"{ARCH_LABEL[a]:>10}" for a in ARCHS))
    for label, _m, _d in per_axis:
        line = f"{label:18}  "
        for a in ARCHS:
            v = norm[a].get(label)
            line += f"{'-' if v is None else f'{v:.3f}':>10}  "
        print(line)

    # ── Plot ─────────────────────────────────────────────────────
    n_axes = len(per_axis)
    angles = np.linspace(0, 2 * np.pi, n_axes, endpoint=False).tolist()
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=200,
                           subplot_kw=dict(projection="polar"))
    for a in ARCHS:
        vals = []
        for label, _m, _d in per_axis:
            v = norm[a].get(label)
            vals.append(0.0 if v is None else v)
        vals_closed = vals + [vals[0]]
        col = ARCH_COLOR[a]
        ax.plot(angles_closed, vals_closed, "-o", color=col, lw=1.6, ms=4.5,
                label=ARCH_LABEL[a])
        ax.fill(angles_closed, vals_closed, color=col, alpha=0.10)

    # Axis labels: just the task name.
    tick_labels = [label for label, _metric, _d in per_axis]
    ax.set_xticks(angles)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.5", "0.75", "1.0"], fontsize=7, color="#666")
    ax.set_ylim(0, 1.05)
    ax.set_rlabel_position(90)
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.spines["polar"].set_color("#aaaaaa")

    ax.legend(loc="upper right", bbox_to_anchor=(1.32, 1.08), fontsize=9,
              frameon=False)
    fig.tight_layout()

    FIGS_OUT.mkdir(parents=True, exist_ok=True)
    pdf = FIGS_OUT / "global_rose_summary.pdf"
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(pdf.with_suffix(".png"), format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Sidecar data file for reproducibility.
    sidecar = {
        "case_studies": [
            {"axis": label, "metric": metric, "raw": data}
            for label, metric, data in per_axis
        ],
        "archs": ARCHS,
        "normalised": norm,
    }
    (FIGS_OUT / "global_rose_summary_data.json").write_text(
        json.dumps(sidecar, indent=2)
    )
    print()
    print(f"Wrote {pdf.name} (+ .png + .json sidecar)")


if __name__ == "__main__":
    main()
