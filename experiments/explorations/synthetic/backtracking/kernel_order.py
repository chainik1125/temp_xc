"""How many lags does backtracking self-excitation actually need? (K selection)

K (the kernel length) was hardcoded to 8 in the mirror fit, never selected
against the data. This fits the same logistic-AR conditional intensity

    logit P(b_i=1) = a + c*pos_i + Sum_{l=1..K} w_l b_{i-l}

at a range of K on the SAME 70/30 train/eval split as the mirror (SEED=0),
and reports HELD-OUT negative log-likelihood (the generalization criterion)
plus BIC (parsimony). The K that minimizes held-out NLL is the data-backed
kernel length; the effective memory = smallest L capturing >=90% of Sum|w_l|.

This decouples K from any benchmark result (it's a fit to the real labels),
so it is a clean pre-build justification, not metric shopping.

    .venv/bin/python -m experiments.explorations.synthetic.backtracking.kernel_order
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.explorations.synthetic.backtracking.measure import HERE, load_traces

SEED = 0
K_GRID = [0, 1, 2, 3, 4, 5, 6, 8, 10, 12]
OUT_JSON = HERE / "results" / "backtracking_kernel_order_stats.json"


def design_K(seqs, K):
    """Rows of [pos, b_{i-1..i-K}] -> label b_i, for i >= K within each trace."""
    X, y = [], []
    for s in seqs:
        L = s.size
        if L <= K:
            continue
        pos = np.arange(L) / L
        for i in range(K, L):
            X.append([pos[i], *[float(s[i - l]) for l in range(1, K + 1)]])
            y.append(int(s[i]))
    return np.asarray(X, dtype=np.float64), np.asarray(y)


def main():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    rng = np.random.default_rng(SEED)
    seqs, _ = load_traces()
    idx = rng.permutation(len(seqs)); cut = int(0.7 * len(seqs))
    train = [seqs[i] for i in idx[:cut]]; ev = [seqs[i] for i in idx[cut:]]

    rows = []
    for K in K_GRID:
        Xtr, ytr = design_K(train, K)
        Xev, yev = design_K(ev, K)
        clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xtr, ytr)
        # held-out NLL (per-event), the generalization criterion
        p_ev = clf.predict_proba(Xev)[:, 1]
        nll_ev = float(log_loss(yev, p_ev, labels=[0, 1]))
        # train LL + BIC (parsimony). n_params = intercept + pos + K lags
        p_tr = clf.predict_proba(Xtr)[:, 1]
        ll_tr = float(-log_loss(ytr, p_tr, labels=[0, 1]) * len(ytr))
        n_params = 2 + K
        bic = float(-2 * ll_tr + n_params * np.log(len(ytr)))
        w = clf.coef_[0][1:] if K > 0 else np.array([])      # lag weights
        # effective memory: smallest L with cumulative |w| >= 90% of total
        eff = None
        if K > 0 and np.sum(np.abs(w)) > 0:
            cum = np.cumsum(np.abs(w)) / np.sum(np.abs(w))
            eff = int(np.searchsorted(cum, 0.90) + 1)
        rows.append({"K": K, "n_params": n_params, "heldout_nll": nll_ev,
                     "bic": bic, "kernel_w": w.tolist(),
                     "coef_position": float(clf.coef_[0][0]),
                     "eff_memory_90pct": eff})

    best = min(rows, key=lambda r: r["heldout_nll"])
    best_bic = min(rows, key=lambda r: r["bic"])
    out = {"seed": SEED, "n_train": len(train), "n_eval": len(ev),
           "K_grid": K_GRID, "rows": rows,
           "best_K_heldout_nll": best["K"], "best_K_bic": best_bic["K"]}
    OUT_JSON.write_text(json.dumps(out, indent=2))

    print("\n========= BACKTRACKING — KERNEL LENGTH (K) MODEL SELECTION =========")
    print(f"70/30 trace split (SEED={SEED}); {len(train)} train / {len(ev)} eval traces\n")
    print(f"  {'K':>3}{'n_par':>7}{'heldout_NLL':>14}{'BIC':>12}{'eff_mem(90%)':>14}")
    for r in rows:
        star = "  *minNLL" if r["K"] == best["K"] else ""
        star += "  *minBIC" if r["K"] == best_bic["K"] else ""
        em = r["eff_memory_90pct"] if r["eff_memory_90pct"] is not None else "-"
        print(f"  {r['K']:>3}{r['n_params']:>7}{r['heldout_nll']:>14.5f}{r['bic']:>12.1f}{str(em):>14}{star}")
    print(f"\n  best K by held-out NLL: {best['K']}   best K by BIC: {best_bic['K']}")
    # show the kernel at K=8 (the current default) and at the selected K
    for K_show in sorted({8, best["K"], best_bic["K"]}):
        r = next(x for x in rows if x["K"] == K_show)
        if r["kernel_w"]:
            wfmt = ", ".join(f"{v:+.2f}" for v in r["kernel_w"])
            print(f"  kernel at K={K_show}: [{wfmt}]  (pos coef {r['coef_position']:+.2f}, "
                  f"eff mem {r['eff_memory_90pct']})")
    print(f"\n  -> {OUT_JSON}")
    return out


if __name__ == "__main__":
    main()
