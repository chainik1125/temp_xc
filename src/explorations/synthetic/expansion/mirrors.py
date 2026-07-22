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
    long-memory ACF plateau     -> hierarchical AR(1)              [scalar]
    rhythmic AND clustered      -> periodic + self-exciting hybrid [binary]
    categorical plateau         -> hierarchical categorical        [categorical]

(``hier_ar1`` and ``periodic_hawkes`` are the Cycle-3 extensions the Cycle-2
review mandated: the short-memory menu could not generate the hedging stream's
ACF plateau nor the verification stream's periodic-plus-bursty events.
``hier_categorical`` is the Cycle-4 extension both C3 interaction/equality
aborts point at: real categorical phase streams hold pooled self-match
plateaus and inflated pooled MI(lag) that any single global dwell+jump process
understates — proof-operation MI(2) halved, recipe-instruction ACF(4)
undershot — because each document carries its own phase-propensity profile.)

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

def fit_ar1(train_seqs, position: bool = False) -> dict:
    """AR(1), optionally around a linear position trend.

    ``position=True`` fits ``m(pos) = mu + beta·pos`` (pos ∈ [0,1) within each
    sequence) by pooled least squares, then AR(1) on the de-trended residual —
    matching a card that pins BOTH the lag-1 persistence and the drift trend.
    """
    allx = np.concatenate([s.astype(float) for s in train_seqs])
    if position:
        allp = np.concatenate([np.arange(s.size) / s.size for s in train_seqs])
        A = np.stack([np.ones_like(allp), allp], 1)
        (mu, beta), *_ = np.linalg.lstsq(A, allx, rcond=None)
        mu, beta = float(mu), float(beta)
    else:
        mu, beta = float(allx.mean()), 0.0

    xs, ys = [], []
    for s in train_seqs:
        if s.size > 1:
            pos = np.arange(s.size) / s.size
            r = s.astype(float) - (mu + beta * pos)
            xs.append(r[:-1])
            ys.append(r[1:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rho = float((x * y).sum() / max((x * x).sum(), 1e-12))
    resid = y - rho * x
    return {"process": "ar1", "mu": mu, "beta_position": beta, "rho": rho,
            "sigma": float(resid.std())}


def gen_ar1(params: dict, lengths, rng) -> list:
    mu, rho, sg = params["mu"], params["rho"], params["sigma"]
    beta = params.get("beta_position", 0.0)
    stat_sd = sg / max(np.sqrt(max(1 - rho ** 2, 1e-9)), 1e-9)
    out = []
    for L in lengths:
        pos = np.arange(L) / L
        r = np.empty(L)
        r[0] = stat_sd * rng.standard_normal()
        for i in range(1, L):
            r[i] = rho * r[i - 1] + sg * rng.standard_normal()
        out.append(mu + beta * pos + r)
    return out


# ── hierarchical AR(1): per-sequence latent level + AR(1) within, scalar ───

def fit_hier_ar1(train_seqs, position: bool = True) -> dict:
    """Per-sequence latent level + AR(1) residual (+ optional pooled trend).

    x_i^(j) = mu + beta·pos_i + l_j + r_i,  r_i = rho·r_{i−1} + sigma·ε_i,
    with l_j one latent level per sequence, kept as the EMPIRICAL level list
    (heavy tails preserved, nothing assumed Gaussian). The level variance puts
    a floor under the pooled within-sequence ACF at long lags — the plateau
    that no single-timescale menu process can produce.
    """
    if position:
        allx = np.concatenate([s.astype(float) for s in train_seqs])
        allp = np.concatenate([np.arange(s.size) / s.size for s in train_seqs])
        A = np.stack([np.ones_like(allp), allp], 1)
        (mu, beta), *_ = np.linalg.lstsq(A, allx, rcond=None)
        mu, beta = float(mu), float(beta)
    else:
        mu = float(np.concatenate([s.astype(float) for s in train_seqs]).mean())
        beta = 0.0

    levels, xs, ys = [], [], []
    for s in train_seqs:
        pos = np.arange(s.size) / s.size
        d = s.astype(float) - (mu + beta * pos)
        l_j = float(d.mean())
        levels.append(l_j)
        r = d - l_j
        if s.size > 1:
            xs.append(r[:-1])
            ys.append(r[1:])
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    rho = float((x * y).sum() / max((x * x).sum(), 1e-12))
    resid = y - rho * x
    return {"process": "hier_ar1", "mu": mu, "beta_position": beta, "rho": rho,
            "sigma": float(resid.std()), "levels": levels}


def gen_hier_ar1(params: dict, lengths, rng) -> list:
    mu, rho, sg = params["mu"], params["rho"], params["sigma"]
    beta = params.get("beta_position", 0.0)
    levels = np.array(params["levels"], dtype=float)
    stat_sd = sg / max(np.sqrt(max(1 - rho ** 2, 1e-9)), 1e-9)
    out = []
    for L in lengths:
        pos = np.arange(L) / L
        l_j = float(rng.choice(levels))
        r = np.empty(L)
        r[0] = stat_sd * rng.standard_normal()
        for i in range(1, L):
            r[i] = rho * r[i - 1] + sg * rng.standard_normal()
        out.append(mu + beta * pos + l_j + r)
    return out


# ── hierarchical categorical: per-doc phase propensities + jump chain ──────

def fit_hier_categorical(train_seqs, n_symbols: int | None = None,
                         max_dwell: int = 400, alpha_grid: int = 51) -> dict:
    """The categorical ``hier_ar1`` (Cycle-4 menu extension).

    Layer 1: each document ``j`` carries its own phase-propensity vector
    ``pi_j`` — kept as the EMPIRICAL list of per-doc symbol distributions
    (heavy tails preserved, nothing assumed Dirichlet). Layer 2: within a doc,
    phases persist via per-symbol empirical dwell draws (as ``semi_markov``);
    on a jump out of ``c`` the target mixes the doc's own propensities with a
    global jump chain:

        P(d | c, j) = alpha · pi_j(d | d != c)  +  (1 - alpha) · J(c, d)

    with ``alpha`` fit by MLE over the observed jump events (grid search —
    likelihood-based, so no measured moment is fit directly). Doc-propensity
    heterogeneity puts a floor under the pooled self-match ACF at long lags
    and inflates pooled MI(lag) — the categorical plateau a single global
    dwell+jump process cannot generate (the C3 gate-8 failure mode of both
    interaction/equality candidates).
    """
    pooled = np.concatenate([s for s in train_seqs])
    k = n_symbols or int(pooled.max()) + 1

    dwell: list[list[int]] = [[] for _ in range(k)]
    J = np.ones((k, k)) - np.eye(k)  # jump chain, smoothed, no self-jumps
    props = []
    jumps = []  # (doc_idx, src, dst)
    for j, s in enumerate(train_seqs):
        cnt = np.bincount(s.astype(int), minlength=k).astype(float) + 1.0
        props.append((cnt / cnt.sum()).tolist())
        change = np.flatnonzero(np.diff(s) != 0)
        edges = np.concatenate([[-1], change, [s.size - 1]])
        states = s[np.concatenate([change, [s.size - 1]])].astype(int)
        runs = np.diff(edges).astype(int)
        for i, (st, r) in enumerate(zip(states, runs)):
            dwell[st].append(min(int(r), max_dwell))
            if i + 1 < len(states):
                J[st, states[i + 1]] += 1
                jumps.append((j, int(st), int(states[i + 1])))
    J /= J.sum(1, keepdims=True)

    # MLE for the tilt weight alpha over the recorded jump events.
    P = np.array(props)
    best_alpha, best_ll = 0.0, -np.inf
    for alpha in np.linspace(0.0, 1.0, alpha_grid):
        ll = 0.0
        for j, c, d in jumps:
            tilde = P[j].copy()
            tilde[c] = 0.0
            tot = tilde.sum()
            p_doc = tilde[d] / tot if tot > 0 else 0.0
            p = alpha * p_doc + (1.0 - alpha) * J[c, d]
            ll += np.log(max(p, 1e-12))
        if ll > best_ll:
            best_ll, best_alpha = ll, float(alpha)

    # Self-consistency deconvolution: the generator's within-doc stationary is
    # FLATTER than its propensity vector (the no-self-jump exclusion spreads
    # mass), so storing the observed doc marginal as the propensity would
    # flatten the heterogeneity every fit->generate round and leak the plateau.
    # Solve per doc for u_j whose stationary matches the OBSERVED marginal.
    m_dwell = np.array([np.mean(d) if d else 1.0 for d in dwell])

    def _stationary(u, alpha):
        K = np.empty((k, k))
        for c in range(k):
            tilde = u.copy()
            tilde[c] = 0.0
            tot = tilde.sum()
            row = (alpha * (tilde / tot) if tot > 0 else 0.0) + (1 - alpha) * J[c]
            K[c] = row / row.sum()
        nu = np.full(k, 1.0 / k)
        for _ in range(200):
            nu = nu @ K
        occ = nu * m_dwell
        return occ / occ.sum()

    adj = []
    for pi_obs in P:
        u = pi_obs.copy()
        for _ in range(25):
            pred = _stationary(u, best_alpha)
            u = np.clip(u * pi_obs / np.maximum(pred, 1e-9), 1e-6, None)
            u /= u.sum()
        adj.append(u.tolist())

    return {"process": "hier_categorical", "n_symbols": k,
            "jump_P": J.tolist(), "alpha": best_alpha,
            "doc_props": adj, "doc_marginals": props,
            "dwell": [d if d else [1] for d in dwell]}


def gen_hier_categorical(params: dict, lengths, rng) -> list:
    J = np.array(params["jump_P"])
    k = params["n_symbols"]
    alpha = params["alpha"]
    props = np.array(params["doc_props"])
    dwell = [np.array(d) for d in params["dwell"]]
    out = []
    for L in lengths:
        pi_j = props[rng.integers(len(props))]
        s = np.empty(L, dtype=np.int8)
        cur = int(rng.choice(k, p=pi_j))
        i = 0
        while i < L:
            r = int(rng.choice(dwell[cur]))
            s[i:i + r] = cur
            i += r
            tilde = pi_j.copy()
            tilde[cur] = 0.0
            tot = tilde.sum()
            if tot > 0:
                p = alpha * (tilde / tot) + (1.0 - alpha) * J[cur]
            else:  # doc propensity degenerate on cur — global chain only
                p = J[cur].copy()
            p = np.clip(p, 0.0, None)
            p /= p.sum()
            cur = int(rng.choice(k, p=p))
        out.append(s)
    return out


# ── periodic + self-exciting hybrid (periodic-Hawkes), binary ──────────────

def fit_periodic_hawkes(train_seqs, K: int = 8, max_period: int = 64) -> dict:
    """logit P(b_i=1) = a + b_c·cos(2πi/P) + b_s·sin(2πi/P) + Σ_l w_l·b_{i−l}.

    The period P comes from the pooled cyclogram (as ``fit_periodic_rate``);
    the phase profile and the K-lag excitation kernel are then fit jointly by
    logistic regression — a rhythmic base rate that events also self-excite
    around, for streams that are periodic AND bursty at once.
    """
    from sklearn.linear_model import LogisticRegression

    # Cyclogram power per P, minus the per-bin sampling-noise floor (raw power
    # is biased toward large P: more, noisier bins), then the SMALLEST P within
    # 5% of the max (every multiple of the true period ties in signal power).
    p0 = sig.base_rate(train_seqs)
    powers = {}
    for P in range(2, max_period + 1):
        num = np.zeros(P)
        den = np.zeros(P)
        for s in train_seqs:
            ph = np.arange(s.size) % P
            np.add.at(num, ph, s.astype(float))
            np.add.at(den, ph, 1.0)
        prof = num / np.maximum(den, 1)
        noise = float((prof * (1 - prof) / np.maximum(den, 1)).mean())
        powers[P] = float(((prof - p0) ** 2).mean()) - noise
    pmax = max(powers.values())
    P = min(p for p, v in powers.items() if v >= 0.95 * pmax)

    X, y = [], []
    for s in train_seqs:
        L = s.size
        if L <= K:
            continue
        for i in range(K, L):
            row = [np.cos(2 * np.pi * i / P), np.sin(2 * np.pi * i / P)]
            row += [float(s[i - l]) for l in range(1, K + 1)]
            X.append(row)
            y.append(int(s[i]))
    clf = LogisticRegression(C=1.0, max_iter=2000).fit(np.array(X), np.array(y))
    w = clf.coef_[0]
    return {"process": "periodic_hawkes", "period": int(P), "K": K,
            "intercept": float(clf.intercept_[0]),
            "b_cos": float(w[0]), "b_sin": float(w[1]),
            "kernel_w": w[2:].tolist()}


def gen_periodic_hawkes(params: dict, lengths, rng) -> list:
    a = params["intercept"]
    P, bc, bs = params["period"], params["b_cos"], params["b_sin"]
    w = np.array(params["kernel_w"])
    K = params["K"]
    out = []
    for L in lengths:
        b = np.zeros(L, dtype=np.int8)
        for i in range(L):
            hist = sum(w[l] * (b[i - 1 - l] if i - 1 - l >= 0 else 0.0) for l in range(K))
            logit = (a + bc * np.cos(2 * np.pi * i / P)
                     + bs * np.sin(2 * np.pi * i / P) + hist)
            b[i] = 1 if rng.random() < 1.0 / (1.0 + np.exp(-logit)) else 0
        out.append(b)
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
    "hier_ar1": (fit_hier_ar1, gen_hier_ar1),
    "periodic_hawkes": (fit_periodic_hawkes, gen_periodic_hawkes),
    "hier_categorical": (fit_hier_categorical, gen_hier_categorical),
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
