"""make_plots.py — figures for the FrequencyBench sprint writeup.

Reads result JSONs from --results (synced from the pod) and writes PNGs to
--figdir. Run locally (rendering only).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WONG = {"token_sae": "#000000", "txc": "#0072B2", "dcac": "#56B4E9",
        "multiband": "#D55E00", "conv": "#009E73"}
ARCH_LABEL = {"token_sae": "per-token SAE (stacked codes)",
              "txc": "window TXC (full band)",
              "dcac": "DC/AC-split TXC",
              "multiband": "multiband (spectral) TXC",
              "conv": "conv dictionary (L=3)"}
OMEGAS = [0, 1, 2, 4, 8, 16, 24, 32, 40, 50]
M = 101


def load(results_dir):
    cells, baselines, sidecars = [], [], {}
    for p in glob.glob(os.path.join(results_dir, "*.json")):
        with open(p) as f:
            r = json.load(f)
        if "shufv2" in p:
            sidecars[r["tag"]] = r
            continue
        (baselines if "baselines" in p else cells).append(r)
    # merge corrected shuffle numbers into cells
    for c in cells:
        tag = f"{c['task']}_{c['arch']}_H{c['H']}_s{c['seed']}"
        if tag in sidecars:
            c["code_shuffled_linear"] = sidecars[tag]["shufv2_linear"]
            c["code_reversed_linear"] = sidecars[tag]["reversed_linear"]
    return cells, baselines


def get_baseline(baselines, task, seed):
    for b in baselines:
        if b["task"] == task and b["seed"] == seed:
            return b
    return None


def fig_response(cells, baselines, task, H, figdir, probe="code_linear",
                 fname=None, title_extra=""):
    """Headline: per-frequency oracle-normalized probe score per architecture."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    freqs = np.array(OMEGAS) / M
    for arch, color in WONG.items():
        rows = []
        for c in cells:
            if (c["task"] == task and c["arch"] == arch and c["H"] == H
                    and probe in c):
                b = get_baseline(baselines, task, c["seed"])
                a = np.array(c[probe]["per_class"], dtype=float)
                a_or = np.array(b["oracle_per_class"], dtype=float)
                rows.append((a - 0.1) / np.maximum(a_or - 0.1, 1e-9))
        if not rows:
            continue
        rows = np.stack(rows)
        mean, lo, hi = rows.mean(0), rows.min(0), rows.max(0)
        ax.plot(freqs, mean, "-o", color=color, label=ARCH_LABEL[arch],
                markersize=4.5, linewidth=1.8)
        ax.fill_between(freqs, lo, hi, color=color, alpha=0.15, linewidth=0)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="grey", linestyle=":", linewidth=1)
    ax.text(0.005, 1.02, "oracle (ML tone detector)", color="grey", fontsize=9)
    ax.text(0.13, 0.02, "chance = best possible from any single token",
            color="grey", fontsize=9)
    ax.axvspan(0, 1 / 16, color="grey", alpha=0.08, linewidth=0)
    ax.text(0.0145, 0.55, "slower than the\nwindow resolves\n(f < 1/W)",
            color="grey", fontsize=8, ha="center")
    ax.set_xlabel("frequency of the hidden tone  (cycles per token)")
    ax.set_ylabel("probe score  S(f) = (acc − chance) / (oracle − chance)")
    ax.set_ylim(-0.12, 1.12)
    emb = "circle-embedding" if "circle" in task else "random-embedding"
    ax.set_title(f"Frequency response: linear probe on dictionary codes, "
                 f"{emb} ten-tone task\n(dictionary size H={H}, window W=16, "
                 f"mean and min–max over seeds){title_extra}", fontsize=10)
    ax.legend(fontsize=8.5, loc="center right", framealpha=0.95)
    fig.tight_layout()
    out = fname or f"fig_response_{task}_H{H}_{probe}.png"
    fig.savefig(os.path.join(figdir, out), dpi=170)
    plt.close(fig)
    print("wrote", out)


def fig_conversion(cells, baselines, figdir):
    """ac_sign: who makes the sign linearly decodable?"""
    task = "ac_sign"
    bars, labels, colors = [], [], []
    b_by_seed = [get_baseline(baselines, task, s) for s in (0, 1, 2)]
    b_by_seed = [b for b in b_by_seed if b]

    def addbar(vals, lab, col):
        bars.append((np.mean(vals), np.min(vals), np.max(vals)))
        labels.append(lab)
        colors.append(col)

    addbar([b["raw_token_linear"]["acc_test"] for b in b_by_seed],
           "raw single token\n(provably chance\nfor ANY probe)", "#bbbbbb")
    addbar([b["raw_stacked_linear"]["acc_test"] for b in b_by_seed],
           "raw 16-token window,\nlinear probe", "#bbbbbb")
    addbar([b["raw_stacked_mlp"]["acc_test"] for b in b_by_seed],
           "raw 16-token window,\nMLP probe\n(info is present)", "#888888")
    for arch in ["token_sae", "txc", "dcac", "multiband", "conv"]:
        vals = [c["code_linear"]["acc_test"] for c in cells
                if c["task"] == task and c["arch"] == arch]
        if vals:
            addbar(vals, ARCH_LABEL[arch].replace(" (", "\n(") + "\nlinear probe",
                   WONG[arch])
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    xs = np.arange(len(bars))
    means = [b[0] for b in bars]
    errs = np.array([[b[0] - b[1] for b in bars], [b[2] - b[0] for b in bars]])
    ax.bar(xs, means, yerr=errs, color=colors, capsize=3)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.text(-0.45, 0.515, "chance", color="grey", fontsize=9)
    ax.axhline(1.0, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7.5, rotation=12)
    ax.set_ylabel("direction-of-motion accuracy")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("Task: classify direction of cyclic motion from a 16-token "
                 "window. Temporal dictionaries make it LINEARLY decodable.\n"
                 "(bars = mean, whiskers = min–max over 3 seeds; all "
                 "dictionaries: same atom count and sparsity)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_conversion_acsign.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_conversion_acsign.png")


def fig_confusions(cells, figdir, arch="txc", H=256):
    """Random vs circle embedding confusion matrices (seed-averaged).
    Random panel uses the MLP probe (the linear probe is near-chance there,
    so its confusion is uninformative); circle panel uses the linear probe."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, task, probe, name in [
            (axes[0], "multifreq", "code_mlp", "random embedding (MLP probe)"),
            (axes[1], "multifreq_circle", "code_linear",
             "circle embedding (linear probe)")]:
        mats = [np.array(c[probe]["confusion"], dtype=float) for c in cells
                if c["task"] == task and c["arch"] == arch and c["H"] == H]
        if not mats:
            ax.set_title(f"{name}: no data")
            continue
        mat = sum(mats)
        mat = mat / mat.sum(1, keepdims=True)
        np.fill_diagonal(mat, np.nan)  # show only confusions; diagonal masked
        im = ax.imshow(mat, cmap="Blues", vmin=0,
                       vmax=max(0.05, np.nanmax(mat)))
        ax.set_xticks(range(10)); ax.set_yticks(range(10))
        ax.set_xticklabels(OMEGAS, fontsize=8)
        ax.set_yticklabels(OMEGAS, fontsize=8)
        ax.set_xlabel("predicted velocity y")
        ax.set_ylabel("true velocity y")
        ax.set_title(f"{name}\n{ARCH_LABEL[arch]}, H={H}", fontsize=10)
        # annotate ratio-2 pairs on the random panel
        if task == "multifreq":
            for (a, b) in [(1, 2), (2, 4), (4, 8), (8, 16), (16, 32)]:
                ia, ib = OMEGAS.index(a), OMEGAS.index(b)
                for (i, j) in [(ia, ib), (ib, ia)]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1,
                                 fill=False, edgecolor="#D55E00", lw=1.6))
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Same symbolic process, two embeddings: confusion is "
                 "multiplicative (ratio pairs, orange) vs spectral (band "
                 "diagonal)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_confusion_{arch}_H{H}.png"), dpi=170)
    plt.close(fig)
    print(f"wrote fig_confusion_{arch}_H{H}.png")


def fig_pairs(theory_dir, figdir):
    rows = []
    for p in glob.glob(os.path.join(theory_dir, "pair_*.json")):
        with open(p) as f:
            rows.append(json.load(f))
    if not rows:
        print("no pair results yet")
        return
    pairs = sorted({tuple(r["pair"]) for r in rows},
                   key=lambda ab: (0 if (ab[1] * pow(ab[0], -1, M)) % M == 2
                                   else 1, ab))
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharey=True)
    for ax, emb in [(axes[0], "random"), (axes[1], "circle")]:
        for k, key in enumerate(["txc_linear", "bag_mlp"]):
            means, los, his = [], [], []
            for pr in pairs:
                vals = [r[key] for r in rows
                        if tuple(r["pair"]) == pr and r["embedding"] == emb]
                if not vals:
                    vals = [np.nan]
                means.append(np.mean(vals)); los.append(np.min(vals))
                his.append(np.max(vals))
            xs = np.arange(len(pairs)) + (k - 0.5) * 0.35
            col = "#0072B2" if key == "txc_linear" else "#009E73"
            ax.bar(xs, means, width=0.33, color=col,
                   yerr=[np.array(means) - los, np.array(his) - means],
                   capsize=2, label={"txc_linear": "TXC codes + linear probe",
                                     "bag_mlp": "bag-of-symbols + MLP"}[key])
        ax.axhline(0.5, color="grey", linestyle=":", lw=1)
        ax.set_xticks(np.arange(len(pairs)))
        ax.set_xticklabels([f"{{{a},{b}}}\nr={(b*pow(a,-1,M))%M}"
                            for a, b in pairs], fontsize=8)
        ax.set_title(f"{emb} embedding")
        ax.set_xlabel("velocity pair (two-class task)")
    axes[0].set_ylabel("pair classification accuracy")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Two-class velocity tasks. Random embedding: uniformly hard "
                 "for TXC codes (ratio-invariance) and trivial for "
                 "bag-of-symbols — except the sign pair {3,98}={3,−3}, whose "
                 "windows have identical symbol sets.\nCircle embedding: "
                 "TXC difficulty tracks the frequency gap (only sub-Rayleigh "
                 "{1,2} is hard); bag-of-symbols degrades to arc-extent "
                 "matching.", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_pairs.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_pairs.png")


def fig_branches(cells, figdir, task="multifreq_circle", H=2048):
    """Multiband branch × velocity heatmap."""
    rows = []
    for c in cells:
        if c["task"] == task and c["arch"] == "multiband" and c["H"] == H:
            row = []
            for b in range(4):
                key = f"code_branch{b}_linear"
                if key in c:
                    row.append(c[key]["per_class"])
            if len(row) == 4:
                rows.append(np.array(row, dtype=float))
    if not rows:
        print("no branch data yet")
        return
    mat = np.mean(rows, axis=0)
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(10))
    ax.set_xticklabels([f"{y}\n{y/M:.3f}" for y in OMEGAS], fontsize=8)
    ax.set_yticks(range(4))
    ax.set_yticklabels(["DC branch\n(w=0)", "low branch\n(w=1–5)",
                        "mid branch\n(w=6–10)", "high branch\n(w=11–15)"],
                       fontsize=8)
    ax.set_xlabel("true velocity y (top) / tone frequency cycles-per-token "
                  "(bottom)")
    ax.set_title(f"Per-branch probe accuracy by hidden frequency\n"
                 f"(multiband TXC, {task}, H={H}, linear probes)",
                 fontsize=10)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("linear probe accuracy\n(chance = 0.1)", fontsize=8)
    ax.text(-0.5, -1.05, "branch = atoms whose temporal kernels are "
            "constrained to the stated DCT frequency indices w (W=16)",
            fontsize=8, color="dimgrey")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_branches_{task}_H{H}.png"), dpi=170)
    plt.close(fig)
    print(f"wrote fig_branches_{task}_H{H}.png")


def fig_freqfrac(results_dir, figdir, task="multifreq_circle", H=2048, seed=0):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.6))
    im = None
    for ax, arch in zip(axes, ["txc", "dcac", "multiband", "conv"]):
        p = os.path.join(results_dir, f"freqprof_{task}_{arch}_H{H}_s{seed}.npy")
        if not os.path.exists(p):
            ax.set_title(f"{arch}: missing")
            continue
        fp = np.load(p)  # (H, W)
        order = np.argsort(fp.argmax(1))
        im = ax.imshow(fp[order], aspect="auto", cmap="magma", vmin=0, vmax=1)
        ax.set_title(ARCH_LABEL[arch], fontsize=11)
        ax.set_xlabel("DCT frequency index w (0 = constant-in-time)",
                      fontsize=9)
        ax.set_ylabel("atom, sorted by peak-energy frequency"
                      if arch == "txc" else "")
    fig.suptitle("Share of each atom's temporal-kernel energy at each DCT "
                 "frequency (FreqFrac). Caution: sorting creates an apparent "
                 "diagonal even for noise — see firing-weighted "
                 f"quantification in the text. ({task}, H={H})", fontsize=11)
    if im is not None:
        fig.colorbar(im, ax=axes, fraction=0.015)
    fig.savefig(os.path.join(figdir, f"fig_freqfrac_{task}_H{H}.png"), dpi=170)
    plt.close(fig)
    print(f"wrote fig_freqfrac_{task}_H{H}.png")


def fig_atomspectra(figdir):
    """Spectral concentration (top-2-adjacent DCT energy fraction) of the 32
    busiest vanilla-TXC atoms: random-init vs trained on each embedding.
    Numbers computed on the pod (firing-weighted FreqFrac analysis)."""
    data = {
        "random init\n(analytic reference)": ([0.205], "#bbbbbb"),
        "trained, random\nembedding task": ([0.212, 0.221, 0.225], "#0072B2"),
        "trained, circle\nembedding task": ([0.579, 0.592, 0.557], "#D55E00"),
    }
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    for i, (lab, (vals, col)) in enumerate(data.items()):
        ax.bar(i, np.mean(vals), color=col,
               yerr=[[np.mean(vals) - np.min(vals)],
                     [np.max(vals) - np.mean(vals)]], capsize=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(list(data.keys()), fontsize=8.5)
    ax.set_ylabel("spectral concentration of kernels\n"
                  "(top-2 adjacent DCT bins, 32 busiest atoms)")
    ax.set_title("TXC atoms become tone-like only when the data\n"
                 "has spectral structure (H=256, 3 seeds)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_atomspectra.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_atomspectra.png")


def fig_dissociation(cells, theory_dir, figdir):
    """Scatter: pairwise confusion vs (left) exact max symbol overlap and
    (right) frequency distance — random vs circle embeddings."""
    import scipy.stats as st
    ovp = os.path.join(theory_dir, "overlaps_all.json")
    if not os.path.exists(ovp):
        print("no overlaps_all.json")
        return
    ov = json.load(open(ovp))

    def conf(task, probe):
        mats = [np.array(c[probe]["confusion"], float) for c in cells
                if c["task"] == task and c["H"] == 256
                and c["arch"] in ("txc", "dcac", "multiband") and probe in c]
        C = sum(mats)
        C = C / C.sum(1, keepdims=True)
        return C - np.diag(np.diag(C))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, task, probe, name, col in [
            (axes[0], "multifreq", "code_mlp", "random embedding", "#0072B2"),
            (axes[1], "multifreq_circle", "code_linear", "circle embedding",
             "#D55E00")]:
        C = conf(task, probe)
        if task == "multifreq":
            xs = [ov[f"{OMEGAS[i]}_{OMEGAS[j]}"]["max"]
                  for i in range(1, 10) for j in range(i + 1, 10)]
            ys = [C[i, j] + C[j, i]
                  for i in range(1, 10) for j in range(i + 1, 10)]
            ax.set_xlabel("max possible shared symbols between windows of "
                          "the two classes\n(exact combinatorics; large iff "
                          "velocity ratio is a small fraction)")
        else:
            xs = [abs(OMEGAS[j] - OMEGAS[i]) / M
                  for i in range(10) for j in range(i + 1, 10)]
            ys = [C[i, j] + C[j, i]
                  for i in range(10) for j in range(i + 1, 10)]
            ax.axvline(1 / 16, color="grey", linestyle="--", lw=1)
            ax.text(1 / 16 + 0.005, max(ys) * 0.75,
                    "window resolution limit\n1/W = 1/16 ≈ 0.06\n"
                    "(all confusion mass\nis below it)",
                    color="grey", fontsize=8)
            ax.set_xlabel("frequency distance |f - f'| (cycles/token)")
        rho, p = st.spearmanr(xs, ys)
        ax.scatter(xs, ys, color=col, s=28, alpha=0.8)
        ax.set_title(f"{name}: Spearman ρ = {rho:.2f} (p = {p:.1g})",
                     fontsize=10)
        ax.set_ylabel("confusion between the two classes\n"
                      "(summed off-diagonal rates, probes on codes, H=256)")
    fig.suptitle("What makes two hidden velocities confusable depends on the "
                 "embedding, not the symbolic process", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_dissociation.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_dissociation.png")


def fig_wscan(base_dir, figdir, arch="txc", task="multifreq_circle",
              H=256):
    """Raw per-class accuracy of the dictionary probe vs the ML oracle, one
    panel per window length W. The low-frequency deficit recedes as 1/W."""
    freqs = np.array(OMEGAS) / M
    Ws = [(4, "results_synced_W4"), (8, "results_synced_W8"),
          (16, "results_synced"), (32, "results_synced_W32")]
    fig, axes = plt.subplots(1, 4, figsize=(13.6, 3.6), sharey=True)
    for ax, (W, cdir) in zip(axes, Ws):
        rdir = os.path.join(base_dir, cdir)
        if not os.path.isdir(rdir):
            ax.set_title(f"W={W}: no data")
            continue
        cells, baselines = load(rdir)
        rows, orows = [], []
        for c in cells:
            if c["task"] == task and c["arch"] == arch and c["H"] == H:
                b = get_baseline(baselines, task, c["seed"])
                if b is None or "oracle_per_class" not in b:
                    continue
                rows.append(np.array(c["code_linear"]["per_class"], float))
                orows.append(np.array(b["oracle_per_class"], float))
        if not rows:
            ax.set_title(f"W={W}: no data")
            continue
        rows, orows = np.stack(rows), np.stack(orows)
        ax.plot(freqs, orows.mean(0), "--", color="grey", linewidth=1.6,
                label="ML oracle (periodogram)")
        ax.plot(freqs, rows.mean(0), "-o", color="#0072B2", markersize=4,
                linewidth=1.7, label="window TXC + linear probe")
        ax.fill_between(freqs, rows.min(0), rows.max(0), color="#0072B2",
                        alpha=0.15, linewidth=0)
        ax.axvline(1 / W, color="#D55E00", linestyle=":", linewidth=1.4)
        ax.text(1 / W * 1.15, 0.18, f"1/W = {1/W:.3f}", color="#D55E00",
                fontsize=8, rotation=90)
        ax.axhline(0.1, color="grey", linestyle=":", lw=0.8)
        ax.set_title(f"window length W = {W}", fontsize=10)
        ax.set_xlabel("tone frequency (cycles/token)")
    axes[0].set_ylabel("10-class accuracy\n(per true class)")
    axes[0].text(0.25, 0.13, "chance", color="grey", fontsize=8)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Slow tones (left of the dotted 1/W line) become resolvable "
                 "as the window W grows — for the oracle and the dictionary "
                 "alike", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, f"fig_wscan_{task}_H{H}.png"), dpi=170)
    plt.close(fig)
    print(f"wrote fig_wscan_{task}_H{H}.png")


def fig_capacity(cells, baselines, figdir):
    """Overall linear acc vs dictionary size H, random vs circle panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for ax, task, name in [(axes[0], "multifreq", "random embedding"),
                           (axes[1], "multifreq_circle", "circle embedding")]:
        for arch, color in WONG.items():
            Hs, means, los, his = [], [], [], []
            for H in [64, 256, 2048]:
                vals = [c["code_linear"]["acc_test"] for c in cells
                        if c["task"] == task and c["arch"] == arch
                        and c["H"] == H]
                if vals:
                    Hs.append(H); means.append(np.mean(vals))
                    los.append(np.min(vals)); his.append(np.max(vals))
            if not Hs:
                continue
            ax.plot(Hs, means, "-o", color=color, label=ARCH_LABEL[arch],
                    markersize=5)
            ax.fill_between(Hs, los, his, color=color, alpha=0.13,
                            linewidth=0)
        ax.axvline(1010, color="grey", linestyle="--", lw=1)
        ax.text(1060, 0.5, "1010 = number of distinct\nclean windows\n"
                "(10 velocities × 101 phases):\nmemorization possible\n"
                "to the right", color="grey", fontsize=7.5)
        ax.axhline(0.1, color="grey", linestyle=":", lw=1)
        ax.text(70, 0.115, "chance (10 classes)", color="grey", fontsize=8)
        ax.set_xscale("log")
        ax.set_xlabel("dictionary size H (atoms)")
        ax.set_title(name)
    axes[0].set_ylabel("10-class linear probe accuracy")
    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.95)
    fig.suptitle("Capacity routes: spectral structure (circle) is exploitable "
                 "at small H; the random task is only solvable above the "
                 "memorization threshold", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_capacity.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_capacity.png")


def fig_multilane(ml_dir, figdir):
    """Per-lane linear accuracy under 3-tone superposition, per architecture."""
    rows = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(ml_dir, "multilane_*.json")):
        r = json.load(open(p))
        rows[r["H"]][r["arch"]].append(r)
    if not rows:
        print("no multilane data")
        return
    Hs = sorted(rows)
    archs = ["token_sae", "txc", "dcac", "multiband", "conv"]
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    width = 0.38
    orc = []
    for hi, H in enumerate(Hs):
        for ai, arch in enumerate(archs):
            cs = rows[H].get(arch, [])
            if not cs:
                continue
            vals = [c["lin_mean"] for c in cs]
            orc += [c["oracle_mean"] for c in cs]
            x = ai + (hi - (len(Hs) - 1) / 2) * width
            ax.bar(x, np.mean(vals), width=width * 0.92, color=WONG[arch],
                   alpha=1.0 if hi == 0 else 0.55,
                   yerr=[[np.mean(vals) - np.min(vals)],
                         [np.max(vals) - np.mean(vals)]], capsize=3,
                   label=f"H={H}" if ai == 1 else None)
    ax.axhline(np.mean(orc), color="grey", linestyle=":", lw=1.2)
    ax.text(3.4, np.mean(orc) + 0.012, "per-lane periodogram oracle",
            color="grey", fontsize=8.5)
    ax.axhline(0.1, color="grey", linestyle=":", lw=1)
    ax.text(-0.45, 0.115, "chance", color="grey", fontsize=8.5)
    ax.set_xticks(range(len(archs)))
    ax.set_xticklabels([ARCH_LABEL[a].replace(" (", "\n(") for a in archs],
                       fontsize=8.5)
    ax.set_ylabel("mean per-lane 10-class linear accuracy")
    hleg = " / ".join(f"H={H}" for H in Hs)
    ax.set_title("Three simultaneous hidden tones (superposition): the "
                 "spectral split now beats the vanilla window TXC\n"
                 f"(solid = H={Hs[0]}, translucent = H={Hs[-1] if len(Hs)>1 else Hs[0]}; "
                 "whiskers = min-max over 3 seeds; memorization impossible: "
                 "10^9 distinct windows)", fontsize=9.5)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "fig_multilane.png"), dpi=170)
    plt.close(fig)
    print("wrote fig_multilane.png")


def table_summary(cells, baselines, out_path):
    """Markdown summary table: per task/arch/H: FVU, linear, MLP, S_temp."""
    lines = ["| task | arch | H | FVU | linear acc | MLP acc | shuffled lin | "
             "S_temp(lin) |",
             "|---|---|---|---|---|---|---|---|"]
    grp = defaultdict(list)
    for c in cells:
        grp[(c["task"], c["arch"], c["H"])].append(c)
    for (task, arch, H), cs in sorted(grp.items()):
        b = get_baseline(baselines, task, cs[0]["seed"])
        a_loc = b["a_loc_star"] if b else float("nan")
        a_or = b["oracle_acc"] if b else float("nan")

        def m(key, sub="acc_test"):
            vals = [c[key][sub] for c in cs if key in c]
            return np.mean(vals) if vals else float("nan")

        fvu = np.mean([c["fvu"] for c in cs])
        lin, mlp = m("code_linear"), m("code_mlp")
        sh = m("code_shuffled_linear")
        st = (lin - a_loc) / max(a_or - a_loc, 1e-9)
        lines.append(f"| {task} | {arch} | {H} | {fvu:.3f} | {lin:.3f} | "
                     f"{mlp:.3f} | {sh:.3f} | {st:.2f} |")
    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results_synced")
    ap.add_argument("--theory", default="theory_synced")
    ap.add_argument("--figdir", default="figures")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    cells, baselines = load(args.results)
    print(f"loaded {len(cells)} cells, {len(baselines)} baselines")
    for H in [256, 2048]:
        for task in ["multifreq", "multifreq_circle"]:
            if any(c["task"] == task and c["H"] == H for c in cells):
                fig_response(cells, baselines, task, H, args.figdir)
                fig_response(cells, baselines, task, H, args.figdir,
                             probe="code_mlp")
    if any(c["task"] == "ac_sign" for c in cells):
        fig_conversion(cells, baselines, args.figdir)
    for H in [256, 2048]:
        fig_confusions(cells, args.figdir, arch="txc", H=H)
    fig_branches(cells, args.figdir)
    fig_branches(cells, args.figdir, H=256)
    fig_freqfrac(args.results, args.figdir)
    if os.path.isdir(args.theory):
        fig_pairs(args.theory, args.figdir)
        fig_dissociation(cells, args.theory, args.figdir)
    fig_capacity(cells, baselines, args.figdir)
    fig_atomspectra(args.figdir)
    fig_wscan(os.path.dirname(os.path.abspath(args.results)) or ".",
              args.figdir)
    table_summary(cells, baselines, os.path.join(args.figdir,
                                                 "summary_table.md"))


if __name__ == "__main__":
    main()
