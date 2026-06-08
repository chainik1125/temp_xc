"""Autoresearch #1 — backtracking as a temporal property (measurement).

Stages 2-3 of autoresearch/autoresearch_spec.md on the Ward Stage-A labels:
build the per-trace backtracking event stream, measure its temporal
signature, and run the null-model controls (within-trace permutation N1,
inhomogeneous-Poisson N2, homogeneous-Poisson N3) that gate the
"is it temporal?" verdict.

Pure text-label analysis — no model inference. Writes
autoresearch/backtracking/results/backtracking_stats.json + figures to autoresearch/backtracking/figs/.

    .venv/bin/python -m autoresearch.backtracking.measure
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
N_NULL = 200       # replicates per null model
N_BOOT = 500       # bootstrap resamples for the real CI
MAXLAG = 12
FANO_W = 10
POS_BINS = 20
ROOT = Path(__file__).resolve().parents[2]   # purified/
STAGE_A = ROOT / "results" / "c7_backtracking" / "stage_a"
OUT_JSON = ROOT / "autoresearch" / "backtracking" / "results" / "backtracking_stats.json"
FIG_DIR = ROOT / "autoresearch" / "backtracking" / "figs"


# ── load ────────────────────────────────────────────────────────────────

def load_traces():
    SL = json.loads((STAGE_A / "sentence_labels.json").read_text())
    seqs, cats = [], []
    for d in SL:
        b = np.array([1 if s["is_backtracking"] else 0 for s in d["sentences"]], dtype=np.int8)
        if b.size >= 3:
            seqs.append(b)
            cats.append(d["question_id"].rsplit("_", 1)[0])
    return seqs, cats


# ── statistics on a list of binary sequences ─────────────────────────────

def base_rate(seqs):
    tot = sum(s.size for s in seqs); pos = sum(int(s.sum()) for s in seqs)
    return pos / tot

def position_rate(seqs, nbins=POS_BINS):
    num = np.zeros(nbins); den = np.zeros(nbins)
    for s in seqs:
        L = s.size
        idx = np.minimum((np.arange(L) / L * nbins).astype(int), nbins - 1)
        for j in range(nbins):
            m = idx == j
            den[j] += m.sum(); num[j] += s[m].sum()
    return num / np.maximum(den, 1)

def acf(seqs, maxlag=MAXLAG):
    """Pooled within-trace autocorrelation of the event indicator, per lag."""
    out = []
    for k in range(1, maxlag + 1):
        xs, ys = [], []
        for s in seqs:
            if s.size > k:
                xs.append(s[:-k]); ys.append(s[k:])
        x = np.concatenate(xs).astype(float); y = np.concatenate(ys).astype(float)
        sx, sy = x.std(), y.std()
        out.append(float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else 0.0)
    return np.array(out)

def self_excitation(seqs):
    """P(next=1|cur=1), P(next=1|cur=0), ratio P(next=1|cur=1)/base."""
    n11 = n1 = n01 = n0 = 0
    for s in seqs:
        cur, nxt = s[:-1], s[1:]
        n1 += int((cur == 1).sum()); n11 += int(((cur == 1) & (nxt == 1)).sum())
        n0 += int((cur == 0).sum()); n01 += int(((cur == 0) & (nxt == 1)).sum())
    p = base_rate(seqs)
    p11 = n11 / max(n1, 1); p01 = n01 / max(n0, 1)
    return {"p11": p11, "p01": p01, "base": p, "excite_ratio": p11 / max(p, 1e-9)}

def inter_event_cv(seqs):
    gaps = []
    for s in seqs:
        idx = np.flatnonzero(s)
        if idx.size >= 2:
            gaps.extend(np.diff(idx).tolist())
    gaps = np.array(gaps, dtype=float)
    if gaps.size < 2:
        return {"mean": float("nan"), "cv": float("nan"), "n": int(gaps.size)}
    return {"mean": float(gaps.mean()), "cv": float(gaps.std() / gaps.mean()), "n": int(gaps.size)}

def fano(seqs, w=FANO_W):
    counts = []
    for s in seqs:
        L = s.size
        for start in range(0, L - w + 1, w):
            counts.append(int(s[start:start + w].sum()))
    c = np.array(counts, dtype=float)
    return float(c.var() / c.mean()) if c.size and c.mean() > 0 else float("nan")

def markov_order_test(seqs):
    """LL of order-0/1/2 + LR p-values (chi2)."""
    from math import log
    p = base_rate(seqs)
    # order-0
    ll0 = sum(int(s.sum()) * log(max(p, 1e-12)) + int((s == 0).sum()) * log(max(1 - p, 1e-12)) for s in seqs)
    # order-1: counts of (prev,cur)
    def counts(order):
        from collections import defaultdict
        c = defaultdict(lambda: [0, 0])
        for s in seqs:
            for i in range(order, s.size):
                ctx = tuple(int(v) for v in s[i - order:i])
                c[ctx][int(s[i])] += 1
        return c
    def ll(order):
        c = counts(order); tot = 0.0
        for ctx, (n0, n1) in c.items():
            n = n0 + n1
            if n == 0:
                continue
            q = n1 / n
            if 0 < q < 1:
                tot += n1 * log(q) + n0 * log(1 - q)
        return tot
    ll1, ll2 = ll(1), ll(2)
    from scipy.stats import chi2
    lr10 = 2 * (ll1 - ll0); p10 = float(chi2.sf(lr10, df=1))
    lr21 = 2 * (ll2 - ll1); p21 = float(chi2.sf(lr21, df=2))
    return {"ll0": ll0, "ll1": ll1, "ll2": ll2, "lr10": lr10, "p_order1_vs_0": p10,
            "lr21": lr21, "p_order2_vs_1": p21}

def mi_vs_lag(seqs, maxlag=MAXLAG):
    from math import log
    out = []
    for k in range(1, maxlag + 1):
        j = np.zeros((2, 2))
        for s in seqs:
            if s.size > k:
                for a, b in zip(s[:-k], s[k:]):
                    j[int(a), int(b)] += 1
        if j.sum() == 0:
            out.append(0.0); continue
        pj = j / j.sum(); pa = pj.sum(1, keepdims=True); pb = pj.sum(0, keepdims=True)
        mi = 0.0
        for a in range(2):
            for b in range(2):
                if pj[a, b] > 0:
                    mi += pj[a, b] * log(pj[a, b] / (pa[a, 0] * pb[0, b]))
        out.append(float(mi))
    return np.array(out)


# ── null-model generators ─────────────────────────────────────────────────

def null_permute(seqs, rng):
    return [rng.permutation(s) for s in seqs]

def null_homog(seqs, rng):
    p = base_rate(seqs)
    return [(rng.random(s.size) < p).astype(np.int8) for s in seqs]

def null_inhomog(seqs, posrate, rng, nbins=POS_BINS):
    out = []
    for s in seqs:
        L = s.size
        idx = np.minimum((np.arange(L) / L * nbins).astype(int), nbins - 1)
        out.append((rng.random(L) < posrate[idx]).astype(np.int8))
    return out


def headline(seqs):
    se = self_excitation(seqs)
    return {"acf": acf(seqs), "fano": fano(seqs), "p11": se["p11"],
            "excite_ratio": se["excite_ratio"], "gap_cv": inter_event_cv(seqs)["cv"]}

def null_band(seqs, gen, rng, n=N_NULL):
    accs = {"acf": [], "fano": [], "p11": [], "excite_ratio": [], "gap_cv": []}
    for _ in range(n):
        h = headline(gen(seqs, rng))
        for k in accs:
            accs[k].append(h[k])
    res = {}
    for k, v in accs.items():
        a = np.array(v)
        res[k] = {"mean": np.nanmean(a, axis=0).tolist(), "lo": np.nanpercentile(a, 2.5, axis=0).tolist(),
                  "hi": np.nanpercentile(a, 97.5, axis=0).tolist()}
    return res


# ── main ──────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(SEED)
    seqs, cats = load_traces()
    p = base_rate(seqs)
    posrate = position_rate(seqs)
    real = headline(seqs)
    se = self_excitation(seqs); gap = inter_event_cv(seqs); mk = markov_order_test(seqs)
    mi = mi_vs_lag(seqs)

    # bootstrap CI for real headline (resample traces)
    boot = {"acf1": [], "fano": [], "excite_ratio": [], "gap_cv": []}
    n = len(seqs)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n); bs = [seqs[i] for i in idx]
        h = headline(bs)
        boot["acf1"].append(h["acf"][0]); boot["fano"].append(h["fano"])
        boot["excite_ratio"].append(h["excite_ratio"]); boot["gap_cv"].append(h["gap_cv"])
    realci = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in boot.items()}

    # nulls
    nulls = {
        "N1_permute":  null_band(seqs, null_permute, rng),
        "N3_homog":    null_band(seqs, null_homog, rng),
        "N2_inhomog":  null_band(seqs, lambda s, r: null_inhomog(s, posrate, r), rng),
    }

    # label-noise robustness: independent symmetric flips
    def flip(seqs, eps, rng):
        return [np.where(rng.random(s.size) < eps, 1 - s, s).astype(np.int8) for s in seqs]
    noise = {}
    for eps in (0.05, 0.10):
        h = headline(flip(seqs, eps, rng))
        noise[f"eps={eps}"] = {"acf1": float(h["acf"][0]), "excite_ratio": float(h["excite_ratio"])}

    # per-category ACF(1), Fano, mean position-rate slope
    bycat = {}
    for c in sorted(set(cats)):
        cs = [seqs[i] for i in range(len(seqs)) if cats[i] == c]
        bycat[c] = {"n_traces": len(cs), "base": base_rate(cs),
                    "acf1": float(acf(cs, 1)[0]), "fano": fano(cs), "excite_ratio": self_excitation(cs)["excite_ratio"]}

    stats = {
        "n_traces": len(seqs), "base_rate": p,
        "position_rate": posrate.tolist(),
        "position_rate_first_vs_last": [float(posrate[0]), float(posrate[-1])],
        "acf_real": real["acf"].tolist(), "acf_real_lag1_ci": realci["acf1"],
        "fano_real": real["fano"], "fano_real_ci": realci["fano"],
        "self_excitation": se, "excite_ratio_ci": realci["excite_ratio"],
        "inter_event": gap, "gap_cv_ci": realci["gap_cv"],
        "markov": mk, "mi_vs_lag": mi.tolist(),
        "nulls": nulls, "label_noise": noise, "by_category": bycat,
        "params": {"seed": SEED, "n_null": N_NULL, "n_boot": N_BOOT, "maxlag": MAXLAG,
                   "fano_w": FANO_W, "pos_bins": POS_BINS},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))

    _plots(stats)
    _summary(stats)
    return stats


def _plots(stats):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lags = np.arange(1, len(stats["acf_real"]) + 1)

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    # ACF real vs nulls
    ax[0].plot(lags, stats["acf_real"], "o-", color="#1f77b4", label="real", lw=2)
    for name, col in [("N1_permute", "#999"), ("N2_inhomog", "#d62728"), ("N3_homog", "#2ca02c")]:
        m = np.array(stats["nulls"][name]["acf"]["mean"]); lo = np.array(stats["nulls"][name]["acf"]["lo"]); hi = np.array(stats["nulls"][name]["acf"]["hi"])
        ax[0].plot(lags, m, "--", color=col, label=name, lw=1)
        ax[0].fill_between(lags, lo, hi, color=col, alpha=0.15)
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("lag (sentences)"); ax[0].set_ylabel("event-indicator ACF"); ax[0].set_title("Autocorrelation: real vs nulls"); ax[0].legend(fontsize=8)
    # position rate
    pr = stats["position_rate"]; xb = np.linspace(0, 1, len(pr))
    ax[1].plot(xb, pr, "o-", color="#1f77b4"); ax[1].axhline(stats["base_rate"], color="gray", ls=":", label="base rate")
    ax[1].set_xlabel("normalized position in trace"); ax[1].set_ylabel("backtracking rate"); ax[1].set_title("Position trend"); ax[1].legend(fontsize=8)
    # MI vs lag
    ax[2].plot(lags, stats["mi_vs_lag"], "o-", color="#9467bd")
    ax[2].set_xlabel("lag (sentences)"); ax[2].set_ylabel("MI (nats)"); ax[2].set_title("Mutual information vs lag")
    for a in ax:
        a.grid(True, alpha=0.25)
    fig.suptitle(f"Backtracking temporal signature (n={stats['n_traces']} traces, base rate {stats['base_rate']:.3f})", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"backtracking_signature.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[plots] → {FIG_DIR}/backtracking_signature.*")


def _summary(s):
    print("\n================ BACKTRACKING TEMPORAL SIGNATURE ================")
    print(f"traces={s['n_traces']}  base_rate={s['base_rate']:.4f}")
    print(f"position rate first→last bin: {s['position_rate_first_vs_last'][0]:.3f} → {s['position_rate_first_vs_last'][1]:.3f}")
    n = s["nulls"]
    print(f"\nACF(1):  real={s['acf_real'][0]:.3f}  CI{tuple(round(x,3) for x in s['acf_real_lag1_ci'])}")
    print(f"         N1_permute={n['N1_permute']['acf']['mean'][0]:+.3f}  N2_inhomog={n['N2_inhomog']['acf']['mean'][0]:+.3f}  N3_homog={n['N3_homog']['acf']['mean'][0]:+.3f}")
    print(f"Fano(w={FANO_W}): real={s['fano_real']:.3f} CI{tuple(round(x,2) for x in s['fano_real_ci'])} | N1={n['N1_permute']['fano']['mean']:.2f} N2={n['N2_inhomog']['fano']['mean']:.2f} N3={n['N3_homog']['fano']['mean']:.2f}")
    se = s["self_excitation"]
    print(f"self-excite: P(1|1)={se['p11']:.3f}  P(1|0)={se['p01']:.3f}  base={se['base']:.3f}  ratio={se['excite_ratio']:.2f} CI{tuple(round(x,2) for x in s['excite_ratio_ci'])}")
    print(f"             N2 excite_ratio={n['N2_inhomog']['excite_ratio']['mean']:.2f}  (real>>N2 ⇒ self-excitation beyond trend)")
    g = s["inter_event"]
    print(f"inter-event gap: mean={g['mean']:.2f} CV={g['cv']:.3f} CI{tuple(round(x,3) for x in s['gap_cv_ci'])} (CV>1 ⇒ bursty; N3 geometric CV={n['N3_homog']['gap_cv']['mean']:.2f})")
    mk = s["markov"]
    print(f"Markov: LL0={mk['ll0']:.0f} LL1={mk['ll1']:.0f} LL2={mk['ll2']:.0f} | order1>0 p={mk['p_order1_vs_0']:.1e}  order2>1 p={mk['p_order2_vs_1']:.1e}")
    print(f"label-noise robustness: {s['label_noise']}")
    print("\nverdict inputs — compare real to N2 (inhomogeneous Poisson = trend only):")
    print(f"  ACF(1) real {s['acf_real'][0]:.3f} vs N2 {n['N2_inhomog']['acf']['mean'][0]:.3f} (N2 95% hi {n['N2_inhomog']['acf']['hi'][0]:.3f})")


if __name__ == "__main__":
    main()
