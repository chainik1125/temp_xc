"""Appendix-B generating-process menu: fit + generate + held-out validate.

Each mirror is two functions — ``fit_*(train_seqs, ...) -> params`` (a plain
JSON-serializable dict) and ``gen_*(params, lengths, rng) -> seqs`` — plus the
shared ``validate_mirror`` that compares the *matched, preregistered*
statistic on held-out real sequences vs synthetic draws (the § 2.5 weak
validation, exactly what ``backtracking/mirror.py`` did).

Menu (README Appendix B):

    bursty / clustered events   -> logistic-AR (discrete Hawkes)   [binary]
    exponential autocorrelation -> k-state Markov / AR(1)          [cat/scalar]
    heavy-tailed dwell          -> semi-Markov (empirical dwell)   [categorical]
    rhythmic / periodic         -> periodic rate + noise           [binary]

A candidate's card must key its mirror to its measured statistic; bespoke
processes require a written justification (README guardrails).
"""

from __future__ import annotations

import numpy as np

from explorations.synthetic.expansion import signature as sig


# ── logistic-AR (discrete self-exciting / Hawkes-style), binary ────────────

def fit_logistic_ar(train_seqs, K: int = 8, position: bool = True) -> dict:
    """logit P(b_i=1) = a + c·pos_i + Σ_{l=1..K} w_l · b_{i−l} (backtracking's)."""
    from sklearn.linear_model import LogisticRegression

    X, y = [], []
    for s in train_seqs:
        L = s.size
        if L <= K:
            continue
        pos = np.arange(L) / L
        for i in range(K, L):
            row = ([pos[i]] if position else []) + [float(s[i - l]) for l in range(1, K + 1)]
            X.append(row)
            y.append(int(s[i]))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    w = clf.coef_[0]
    return {"process": "logistic_ar", "K": K, "position": position,
            "intercept": float(clf.intercept_[0]),
            "coef_position": float(w[0]) if position else 0.0,
            "kernel_w": w[1:].tolist() if position else w.tolist()}


def gen_logistic_ar(params: dict, lengths, rng) -> list:
    a = params["intercept"]
    c_pos = params["coef_position"]
    w = np.array(params["kernel_w"])
    K = params["K"]
    out = []
    for L in lengths:
        pos = np.arange(L) / L
        b = np.zeros(L, dtype=np.int8)
        for i in range(L):
            hist = sum(w[l] * (b[i - 1 - l] if i - 1 - l >= 0 else 0.0) for l in range(K))
            p = 1.0 / (1.0 + np.exp(-(a + c_pos * pos[i] + hist)))
            b[i] = 1 if rng.random() < p else 0
        out.append(b)
    return out


# ── k-state Markov chain, categorical (or binary) ──────────────────────────

def fit_markov(train_seqs, n_symbols: int | None = None) -> dict:
    pooled = np.concatenate([s for s in train_seqs])
    k = n_symbols or int(pooled.max()) + 1
    P = np.ones((k, k))  # add-one smoothing
    for s in train_seqs:
        np.add.at(P, (s[:-1].astype(int), s[1:].astype(int)), 1)
    P /= P.sum(1, keepdims=True)
    pi = np.bincount(pooled.astype(int), minlength=k).astype(float)
    pi /= pi.sum()
    return {"process": "markov", "n_symbols": k, "P": P.tolist(), "pi": pi.tolist()}


def gen_markov(params: dict, lengths, rng) -> list:
    P = np.array(params["P"])
    pi = np.array(params["pi"])
    k = params["n_symbols"]
    out = []
    for L in lengths:
        s = np.zeros(L, dtype=np.int8)
        s[0] = rng.choice(k, p=pi)
        for i in range(1, L):
            s[i] = rng.choice(k, p=P[s[i - 1]])
        out.append(s)
    return out


# ── semi-Markov (empirical dwell + jump chain), categorical ────────────────

def fit_semi_markov(train_seqs, n_symbols: int | None = None, max_dwell: int = 400) -> dict:
    pooled = np.concatenate([s for s in train_seqs])
    k = n_symbols or int(pooled.max()) + 1
    dwell: list[list[int]] = [[] for _ in range(k)]
    J = np.ones((k, k)) - np.eye(k)  # jump chain, smoothed, no self-jumps
    pi = np.zeros(k)
    for s in train_seqs:
        change = np.flatnonzero(np.diff(s) != 0)
        edges = np.concatenate([[-1], change, [s.size - 1]])
        states = s[np.concatenate([change, [s.size - 1]])].astype(int)
        runs = np.diff(edges).astype(int)
        pi[int(s[0])] += 1
        for j, (st, r) in enumerate(zip(states, runs)):
            dwell[st].append(min(int(r), max_dwell))
            if j + 1 < len(states):
                J[st, states[j + 1]] += 1
    J /= J.sum(1, keepdims=True)
    pi = pi / max(pi.sum(), 1)
    return {"process": "semi_markov", "n_symbols": k, "jump_P": J.tolist(),
            "pi": pi.tolist(),
            "dwell": [d if d else [1] for d in dwell]}


def gen_semi_markov(params: dict, lengths, rng) -> list:
    J = np.array(params["jump_P"])
    pi = np.array(params["pi"])
    k = params["n_symbols"]
    dwell = [np.array(d) for d in params["dwell"]]
    out = []
    for L in lengths:
        s = np.empty(L, dtype=np.int8)
        cur = rng.choice(k, p=pi)
        i = 0
        while i < L:
            r = int(rng.choice(dwell[cur]))
            s[i:i + r] = cur
            i += r
            cur = rng.choice(k, p=J[cur])
        out.append(s)
    return out


# ── AR(1), scalar ──────────────────────────────────────────────────────────

def fit_ar1(train_seqs) -> dict:
    xs, ys = [], []
    for s in train_seqs:
        if s.size > 1:
            xs.append(s[:-1].astype(float))
            ys.append(s[1:].astype(float))
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    mu = float(np.concatenate([s.astype(float) for s in train_seqs]).mean())
    xc, yc = x - mu, y - mu
    rho = float((xc * yc).sum() / max((xc * xc).sum(), 1e-12))
    resid = yc - rho * xc
    return {"process": "ar1", "mu": mu, "rho": rho, "sigma": float(resid.std())}


def gen_ar1(params: dict, lengths, rng) -> list:
    mu, rho, sg = params["mu"], params["rho"], params["sigma"]
    stat_sd = sg / max(np.sqrt(max(1 - rho ** 2, 1e-9)), 1e-9)
    out = []
    for L in lengths:
        x = np.empty(L)
        x[0] = mu + stat_sd * rng.standard_normal()
        for i in range(1, L):
            x[i] = mu + rho * (x[i - 1] - mu) + sg * rng.standard_normal()
        out.append(x)
    return out


# ── periodic rate + noise, binary ──────────────────────────────────────────

def fit_periodic_rate(train_seqs, max_period: int = 64) -> dict:
    """Bernoulli with rate a + b·cos(2πt/P + φ); P from the pooled cyclogram."""
    best = None
    p0 = sig.base_rate(train_seqs)
    for P in range(2, max_period + 1):
        num = np.zeros(P)
        den = np.zeros(P)
        for s in train_seqs:
            ph = np.arange(s.size) % P
            np.add.at(num, ph, s.astype(float))
            np.add.at(den, ph, 1.0)
        prof = num / np.maximum(den, 1)
        power = float(((prof - p0) ** 2).mean())
        if best is None or power > best[0]:
            best = (power, P, prof)
    _, P, prof = best
    t = np.arange(P)
    c = np.cos(2 * np.pi * t / P)
    s_ = np.sin(2 * np.pi * t / P)
    A = np.stack([np.ones(P), c, s_], 1)
    coef, *_ = np.linalg.lstsq(A, prof, rcond=None)
    return {"process": "periodic_rate", "period": int(P), "a": float(coef[0]),
            "b_cos": float(coef[1]), "b_sin": float(coef[2])}


def gen_periodic_rate(params: dict, lengths, rng) -> list:
    P, a, bc, bs = params["period"], params["a"], params["b_cos"], params["b_sin"]
    out = []
    for L in lengths:
        t = np.arange(L)
        rate = np.clip(a + bc * np.cos(2 * np.pi * t / P) + bs * np.sin(2 * np.pi * t / P),
                       1e-4, 1 - 1e-4)
        out.append((rng.random(L) < rate).astype(np.int8))
    return out


# ── registry + held-out validation ─────────────────────────────────────────

MENU = {
    "logistic_ar": (fit_logistic_ar, gen_logistic_ar),
    "markov": (fit_markov, gen_markov),
    "semi_markov": (fit_semi_markov, gen_semi_markov),
    "ar1": (fit_ar1, gen_ar1),
    "periodic_rate": (fit_periodic_rate, gen_periodic_rate),
}


def validate_mirror(real_eval_seqs, syn_seqs, kind: str, *, maxlag: int = 12) -> dict:
    """Weak validation: matched statistics side-by-side on held-out real vs syn."""
    hr = sig.headline(real_eval_seqs, kind, maxlag=maxlag)
    hs = sig.headline(syn_seqs, kind, maxlag=maxlag)
    out = {"real": {}, "synthetic": {}, "abs_err": {}}
    for k in hr:
        vr, vs = np.asarray(hr[k], dtype=float), np.asarray(hs[k], dtype=float)
        if vr.ndim == 0:
            out["real"][k] = float(vr)
            out["synthetic"][k] = float(vs)
            out["abs_err"][k] = float(abs(vr - vs))
        else:
            out["real"][k] = vr.tolist()
            out["synthetic"][k] = vs.tolist()
            out["abs_err"][k + "_lag1_5"] = float(np.mean(np.abs(vr[:5] - vs[:5])))
    return out
