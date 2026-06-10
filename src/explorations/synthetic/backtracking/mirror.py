"""Autoresearch #1 — synthetic mirror for backtracking (stages 4-5).

The measurement (backtracking.py) found strong self-excitation. Per the
verdict criteria we fit a **discrete self-exciting process** — a
logistic-autoregressive / Hawkes-style conditional intensity

    logit P(b_i = 1) = a + c·pos_i + Σ_{l=1..K} w_l · b_{i-l}

on TRAIN traces, then GENERATE sequences from it and check (on HELD-OUT
real traces) that the synthetic reproduces the temporal signature
(ACF, Fano, self-excitation ratio, inter-event CV) — the § 2.5 weak
validation. The fitted lag weights w_l are the self-excitation kernel.

    .venv/bin/python -m explorations.synthetic.backtracking.mirror
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from explorations.synthetic.backtracking.measure import (
    FIG_DIR, HERE, acf, base_rate, fano, inter_event_cv, load_traces, self_excitation,
)

SEED = 0
K = 8                # autoregressive lags (the kernel length)
OUT_JSON = HERE / "results" / "backtracking_mirror_stats.json"


def _design(seqs):
    """Rows of [pos, b_{i-1..i-K}] → label b_i, for i >= K within each trace."""
    X, y = [], []
    for s in seqs:
        L = s.size
        if L <= K:
            continue
        pos = np.arange(L) / L
        for i in range(K, L):
            X.append([pos[i], *[float(s[i - l]) for l in range(1, K + 1)]])
            y.append(int(s[i]))
    return np.array(X), np.array(y)


def _generate(clf, lengths, rng):
    """Sample synthetic traces of the given lengths from the fitted intensity."""
    a = float(clf.intercept_[0]); w = clf.coef_[0]           # [pos, lag1..lagK]
    c_pos, c_lag = w[0], w[1:]
    out = []
    for L in lengths:
        pos = np.arange(L) / L
        b = np.zeros(L, dtype=np.int8)
        for i in range(L):
            hist = sum(c_lag[l] * (b[i - 1 - l] if i - 1 - l >= 0 else 0.0) for l in range(K))
            logit = a + c_pos * pos[i] + hist
            p = 1.0 / (1.0 + np.exp(-logit))
            b[i] = 1 if rng.random() < p else 0
        out.append(b)
    return out


def _sig(seqs):
    se = self_excitation(seqs)
    return {"base": base_rate(seqs), "acf": acf(seqs).tolist(), "fano": fano(seqs),
            "excite_ratio": se["excite_ratio"], "p11": se["p11"], "p01": se["p01"],
            "gap_cv": inter_event_cv(seqs)["cv"]}


def main():
    from sklearn.linear_model import LogisticRegression
    rng = np.random.default_rng(SEED)
    seqs, _cats = load_traces()
    # split traces train/eval
    idx = rng.permutation(len(seqs)); cut = int(0.7 * len(seqs))
    train = [seqs[i] for i in idx[:cut]]; ev = [seqs[i] for i in idx[cut:]]

    Xtr, ytr = _design(train)
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)

    # generate synthetic traces matching the eval-trace lengths
    syn = _generate(clf, [s.size for s in ev], rng)

    real_sig, syn_sig = _sig(ev), _sig(syn)
    kernel = clf.coef_[0][1:].tolist()      # w_1..w_K (self-excitation kernel)

    stats = {
        "K": K, "n_train": len(train), "n_eval": len(ev),
        "intercept": float(clf.intercept_[0]),
        "coef_position": float(clf.coef_[0][0]),
        "kernel_w": kernel,
        "real_eval": real_sig, "synthetic": syn_sig,
        "acf_abs_err_lag1_5": float(np.mean(np.abs(
            np.array(real_sig["acf"][:5]) - np.array(syn_sig["acf"][:5])))),
    }
    OUT_JSON.write_text(json.dumps(stats, indent=2))
    _plot(real_sig, syn_sig, kernel)

    print("\n================ BACKTRACKING MIRROR (discrete self-exciting) ================")
    print(f"fit on {len(train)} train traces, validated on {len(ev)} held-out; K={K} lags")
    print(f"position coef = {stats['coef_position']:+.3f}  (rate rises through trace)")
    print("self-excitation kernel w_l (lag→weight):")
    for l, wv in enumerate(kernel, 1):
        print(f"    lag {l}: {wv:+.3f}")
    print(f"\n{'stat':<16}{'real(held-out)':>16}{'synthetic':>14}")
    for k, lab in [("base", "base rate"), ("excite_ratio", "excite ratio"),
                   ("fano", "Fano"), ("gap_cv", "gap CV")]:
        print(f"{lab:<16}{real_sig[k]:>16.3f}{syn_sig[k]:>14.3f}")
    print(f"{'ACF(1)':<16}{real_sig['acf'][0]:>16.3f}{syn_sig['acf'][0]:>14.3f}")
    print(f"{'ACF(2)':<16}{real_sig['acf'][1]:>16.3f}{syn_sig['acf'][1]:>14.3f}")
    print(f"\nmean |ACF_real - ACF_syn| over lags 1-5: {stats['acf_abs_err_lag1_5']:.3f}")
    return stats


def _plot(real_sig, syn_sig, kernel):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lags = np.arange(1, len(real_sig["acf"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(lags, real_sig["acf"], "o-", color="#1f77b4", label="real (held-out)", lw=2)
    ax[0].plot(lags, syn_sig["acf"], "s--", color="#d62728", label="synthetic mirror", lw=2)
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("lag (sentences)"); ax[0].set_ylabel("event-indicator ACF")
    ax[0].set_title("Mirror reproduces the autocorrelation"); ax[0].legend(fontsize=9); ax[0].grid(True, alpha=0.25)
    kl = np.arange(1, len(kernel) + 1)
    ax[1].bar(kl, kernel, color="#9467bd")
    ax[1].set_xlabel("lag l (sentences)"); ax[1].set_ylabel("kernel weight w_l")
    ax[1].set_title("Fitted self-excitation kernel"); ax[1].grid(True, alpha=0.25)
    fig.suptitle("Backtracking synthetic mirror: discrete self-exciting (logistic-AR) process", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"backtracking_mirror.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] → {FIG_DIR}/backtracking_mirror.*")


if __name__ == "__main__":
    main()
