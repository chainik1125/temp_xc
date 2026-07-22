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
    three-timescale plateau     -> segment-level hier categorical  [categorical]

(``hier_ar1`` and ``periodic_hawkes`` are the Cycle-3 extensions the Cycle-2
review mandated: the short-memory menu could not generate the hedging stream's
ACF plateau nor the verification stream's periodic-plus-bursty events.
``hier_categorical`` is the Cycle-4 extension both C3 interaction/equality
aborts point at: real categorical phase streams hold pooled self-match
plateaus and inflated pooled MI(lag) that any single global dwell+jump process
understates — proof-operation MI(2) halved, recipe-instruction ACF(4)
undershot — because each document carries its own phase-propensity profile.
``seg_hier_categorical`` is the Cycle-5 extension the C4 proof-operation-r2
ABORT points at: reasoning-trace phase streams hold a THIRD, segment-scale
layer between run and doc — the doc-level hierarchy matched the lag-12 floor
but undershot lags 2–8 — so within-doc segment regimes carry locally elevated
propensities over the same run dynamics.)

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


# ── segment-level hierarchical categorical (run/segment/doc), categorical ──

def _tilted_excl(v: np.ndarray, c: int):
    """v restricted to d != c, renormalized; None if degenerate on c."""
    t = v.astype(float).copy()
    t[c] = 0.0
    s = t.sum()
    return t / s if s > 0 else None


def _mix_row(c: int, pi_seg, pi_doc, J, a: float, b: float) -> np.ndarray:
    """Jump law P(· | c): a·pi_seg(·|≠c) + b·pi_doc(·|≠c) + (1−a−b)·J(c,·).

    A degenerate tilt target (all mass on c) falls back to the global chain
    for its weight share, mirroring gen_hier_categorical's degenerate branch.
    """
    ts, td = _tilted_excl(pi_seg, c), _tilted_excl(pi_doc, c)
    p = (1.0 - a - b) * J[c] \
        + a * (ts if ts is not None else J[c]) \
        + b * (td if td is not None else J[c])
    p = np.clip(p, 0.0, None)
    return p / p.sum()


def _segment_dp(sym: np.ndarray, w: np.ndarray, k: int, min_seg: int,
                pen: float) -> list:
    """BIC-penalized weighted-multinomial changepoint DP over one sequence.

    A segment's cost is its weighted multinomial NLL (composition entropy ×
    total weight); each boundary pays ``pen``. Deterministic,
    likelihood-based — no pooled temporal moment enters the objective.
    Returns [(start, end)] with end exclusive (indices into ``sym``).
    """
    from scipy.special import xlogy

    n = int(sym.size)
    if n < 2 * min_seg:
        return [(0, n)]
    C = np.zeros((n + 1, k))
    C[1:] = np.cumsum(np.eye(k)[sym.astype(int)] * w[:, None], axis=0)
    W = np.concatenate([[0.0], np.cumsum(w)])
    best = np.full(n + 1, np.inf)
    prev = np.zeros(n + 1, dtype=int)
    best[0] = -pen                     # the first segment pays no boundary
    for j in range(min_seg, n + 1):
        i_max = j - min_seg + 1
        cnt = C[j] - C[:i_max]                            # (i_max, k)
        m = W[j] - W[:i_max]
        nll = -(xlogy(cnt, cnt).sum(1) - m * np.log(m))
        tot = best[:i_max] + pen + nll
        i_best = int(np.argmin(tot))
        if tot[i_best] < best[j]:
            best[j] = tot[i_best]
            prev[j] = i_best
    segs = []
    j = n
    while j > 0:
        i = int(prev[j])
        segs.append((i, j))
        j = i
    return segs[::-1]


# Segmentation-evidence knobs, calibrated on the committed harness toys (a
# strong three-timescale truth must round-trip; a doc-homogeneous null must
# not gain structure) BEFORE any real-data calibration — see the harness
# tests. Module-level so the calibration protocol is explicit and frozen.
CAP_MULT = 1.5
PEN_MULT = 1.0
PERM_NULL_SEED = 0      # seed for run_permuted_streams (the insertion control)


def run_permuted_streams(seqs, rng=None, max_tries: int = 40) -> list:
    """Within-doc run-order shuffle with NO adjacent same-type repeats — the
    mirror's insertion-control null.

    Shuffles each stream's (type, length) RUN pairs, rejecting/repairing
    arrangements that put two same-type runs adjacent — a plain permutation
    merges those into longer runs and distorts the dwell material itself
    (measured: +58% ACF(4) on a heavy-dwell null's PERMUTED moments, making
    control tolerances meaningless). This null preserves the doc
    composition, every run's length, and the no-self-jump property, while
    destroying all segment-scale clustering. Used by the preregistered
    calibration-level control: the seg mirror fit to these streams must not
    reproduce lag-2–8 structure beyond what they themselves carry. (An
    automatic winner's-curse-safe scaling of segment compositions was
    attempted and abandoned — every variant measured on the harness toys
    either drowned genuine signal [complementary in-block halves:
    hypergeometrically anti-correlated; interleaved splits: confounded by
    dominant/excursion alternation; analytic multinomial floors:
    no-self-jump deflates real variance below them] or failed to cancel
    selection noise [permutation-matched DP split-half: segment-size
    mismatch between real and permuted DP output]. The per-dataset control
    is the honest replacement: it measures the estimator's actual
    hallucination on THIS data.)
    """
    if rng is None:
        rng = np.random.default_rng(PERM_NULL_SEED)
    out = []
    for s in seqs:
        s = np.asarray(s).astype(int)
        change = np.flatnonzero(np.diff(s) != 0)
        starts = np.concatenate([[0], change + 1])
        run_sym = s[starts]
        run_len = np.diff(np.concatenate([starts, [s.size]])).astype(int)
        n = run_sym.size
        idx = np.arange(n)
        for _ in range(max_tries):
            idx = rng.permutation(n)
            if not (run_sym[idx][1:] == run_sym[idx][:-1]).any():
                break
        else:
            # Greedy repair of remaining same-type adjacencies (possible
            # leftovers when one type dominates the doc's runs).
            idx = list(idx)
            for i in range(1, n):
                if run_sym[idx[i]] != run_sym[idx[i - 1]]:
                    continue
                for m in range(n):
                    ok_here = run_sym[idx[m]] != run_sym[idx[i - 1]] and \
                        (i + 1 >= n or run_sym[idx[m]] != run_sym[idx[i + 1]])
                    ok_there = run_sym[idx[i]] != (run_sym[idx[m - 1]] if m > 0 else -1) and \
                        (m + 1 >= n or idx[m + 1] == idx[i] or run_sym[idx[i]] != run_sym[idx[m + 1]])
                    if m != i and ok_here and ok_there:
                        idx[i], idx[m] = idx[m], idx[i]
                        break
            idx = np.array(idx)
        out.append(np.repeat(run_sym[idx], run_len[idx]).astype(np.int8))
    return out


def _segment_stream(s: np.ndarray, k: int, min_seg_runs: int = 4) -> list:
    """Run-aware segmentation: boundaries at RUN edges, composition in POSITIONS.

    Segment-scale structure is about which symbols recur locally — it must
    not be confused with the dwell timescale. A raw position-level
    changepoint mistakes a single long run for a concentrated 'segment'
    (measured: on a doc-homogeneous heavy-dwell null it hallucinated ~7
    segments/doc and over-generated mid-lag ACF by +25%). So candidate
    boundaries live at run edges only (each DP token = one run, ≥
    ``min_seg_runs`` runs per segment), while each run contributes its
    LENGTH-weighted counts — composition contrast is measured in positions,
    where it lives. On an exchangeable run stream BIC then finds no
    boundaries and the mirror degenerates to ``hier_categorical``.
    Returns [(start, end)] position spans, end exclusive.
    """
    s = s.astype(int)
    change = np.flatnonzero(np.diff(s) != 0)
    starts = np.concatenate([[0], change + 1])
    run_sym = s[starts]
    run_len = np.diff(np.concatenate([starts, [s.size]])).astype(float)
    # Capped run weights: composition contrast is measured in positions
    # (where it lives — pure run counts are blind to realistic contrast) but
    # a single run's weight is capped at CAP_MULT × the doc's mean run
    # length, so one heavy-dwell run cannot masquerade as strong composition
    # evidence (uncapped position weights inserted +33% ACF(4) on a
    # doc-homogeneous null; pure run counts missed toy segments entirely —
    # both calibrated on the committed harness toys, never on real data).
    w = np.minimum(run_len, CAP_MULT * run_len.mean())
    pen = PEN_MULT * 0.5 * (k - 1) * np.log(max(s.size, 2))
    rsegs = _segment_dp(run_sym, w, k, min_seg=min_seg_runs, pen=pen)
    ends = np.concatenate([starts[1:], [s.size]])
    return [(int(starts[a]), int(ends[b - 1])) for a, b in rsegs]


def fit_seg_hier_categorical(train_seqs, n_symbols: int | None = None,
                             max_dwell: int = 400, min_seg_runs: int = 4,
                             tilt_grid: int = 13) -> dict:
    """The three-timescale categorical mirror (Cycle-5 menu extension).

    The C4 ``hier_categorical`` ABORT on reasoning traces
    (proof-operation-phase-runs-r2) localized the miss to lags 2–8 while the
    lag-12 doc floor matched: reasoning phase streams carry a **segment**
    scale between the run (~2) and the document — within-doc regimes of
    locally elevated phase propensity. This mirror adds exactly that layer:

    - **run**: per-symbol empirical dwell (as ``semi_markov``);
    - **segment**: each doc is segmented by a RUN-AWARE BIC changepoint DP
      (:func:`_segment_stream` — boundaries found in run space so dwell and
      segment timescales cannot be confused); per-doc EMPIRICAL segment
      tables [(length, propensity)] are stored (the ``hier_ar1`` /
      ``hier_categorical`` empirical-list precedent, one level down);
    - **doc**: the observed per-doc marginal, kept as a secondary tilt target.

    On a jump out of ``c`` the target mixes the ACTIVE SEGMENT's propensity,
    the doc propensity, and the global no-self-jump chain::

        P(d | c, seg, doc) = a·pi_seg(d|≠c) + b·pi_doc(d|≠c) + (1−a−b)·J(c,d)

    with ``(a, b)`` fit jointly by MLE over the observed jump events (simplex
    grid — likelihood-based, so no measured moment is fit directly; the
    segmentation objective is likewise composition likelihood, never a lag
    statistic — the preregistered gate-8 moments remain genuinely
    non-fitted). Segment propensities are then DECONVOLVED by the C4
    self-consistency fixed point (raw segment marginals would flatten every
    fit→generate round exactly as raw doc marginals did), against the
    infinite-segment stationary approximation.
    """
    pooled = np.concatenate([s for s in train_seqs])
    k = n_symbols or int(pooled.max()) + 1

    dwell: list[list[int]] = [[] for _ in range(k)]
    J = np.ones((k, k)) - np.eye(k)
    doc_props, doc_runs = [], []
    for s in train_seqs:
        s = s.astype(int)
        cnt = np.bincount(s, minlength=k).astype(float) + 1.0
        doc_props.append(cnt / cnt.sum())
        change = np.flatnonzero(np.diff(s) != 0)
        starts = np.concatenate([[0], change + 1])
        run_sym = s[starts]
        run_len = np.diff(np.concatenate([starts, [s.size]])).astype(float)
        doc_runs.append((run_sym, run_len))
        for st, r in zip(run_sym, run_len):
            dwell[st].append(min(int(r), max_dwell))
        np.add.at(J, (run_sym[:-1], run_sym[1:]), 1)
    J /= J.sum(1, keepdims=True)

    doc_segs, jumps = [], []
    for j, s in enumerate(train_seqs):
        s = s.astype(int)
        segs = _segment_stream(s, k, min_seg_runs=min_seg_runs)
        seg_list = []
        for (a0, b0) in segs:
            scnt = np.bincount(s[a0:b0], minlength=k).astype(float)
            p_hat = np.clip(scnt / max(scnt.sum(), 1.0), 1e-6, None)
            seg_list.append([int(b0 - a0), p_hat / p_hat.sum()])
        doc_segs.append(seg_list)
        bounds = np.array([b0 for _, b0 in segs])
        change = np.flatnonzero(np.diff(s) != 0)
        for t in change:                       # jump lands at position t+1
            seg_i = int(np.searchsorted(bounds, t + 1, side="right"))
            seg_i = min(seg_i, len(seg_list) - 1)
            jumps.append((j, seg_i, int(s[t]), int(s[t + 1])))

    # Joint MLE for the segment/doc tilt weights over the jump events.
    P1 = np.empty(len(jumps))          # segment-tilt probability of the target
    P2 = np.empty(len(jumps))          # doc-tilt probability
    P3 = np.empty(len(jumps))          # global-chain probability
    for i, (j, seg_i, c, d) in enumerate(jumps):
        ts = _tilted_excl(doc_segs[j][seg_i][1], c)
        td = _tilted_excl(doc_props[j], c)
        P1[i] = (ts[d] if ts is not None else J[c, d])
        P2[i] = (td[d] if td is not None else J[c, d])
        P3[i] = J[c, d]
    grid = np.linspace(0.0, 1.0, tilt_grid)
    best_ab, best_ll = (0.0, 0.0), -np.inf
    for a in grid:
        for b in grid:
            if a + b > 1.0 + 1e-9:
                continue
            ll = np.log(np.maximum(a * P1 + b * P2 + (1 - a - b) * P3,
                                   1e-12)).sum()
            if ll > best_ll:
                best_ll, best_ab = ll, (float(a), float(b))
    a, b = best_ab

    # Segment-level self-consistency deconvolution (the C4 lesson, one level
    # down): solve per segment for u whose within-segment stationary — under
    # the fitted mix with THIS doc's propensity — matches the OBSERVED
    # segment composition. Infinite-length stationary approximation.
    m_dwell = np.array([np.mean(d) if d else 1.0 for d in dwell])

    def _stationary(u, pi_doc):
        K = np.empty((k, k))
        for c in range(k):
            K[c] = _mix_row(c, u, pi_doc, J, a, b)
        nu = np.full(k, 1.0 / k)
        for _ in range(80):
            nu = nu @ K
        occ = nu * m_dwell
        return occ / occ.sum()

    seg_tables = []
    for j, seg_list in enumerate(doc_segs):
        table = []
        for length, pi_obs in seg_list:
            u = pi_obs.copy()
            for _ in range(15):
                pred = _stationary(u, doc_props[j])
                u = np.clip(u * pi_obs / np.maximum(pred, 1e-9), 1e-6, None)
                u /= u.sum()
            table.append([int(length), u.tolist()])
        seg_tables.append(table)

    return {"process": "seg_hier_categorical", "n_symbols": k,
            "jump_P": J.tolist(), "tilt_seg": a, "tilt_doc": b,
            "doc_props": [p.tolist() for p in doc_props],
            "seg_tables": seg_tables,
            "dwell": [d if d else [1] for d in dwell]}


def gen_seg_hier_categorical(params: dict, lengths, rng) -> list:
    J = np.array(params["jump_P"])
    k = params["n_symbols"]
    a, b = float(params["tilt_seg"]), float(params["tilt_doc"])
    doc_props = [np.asarray(p) for p in params["doc_props"]]
    seg_tables = params["seg_tables"]
    dwell = [np.array(d) for d in params["dwell"]]
    out = []
    for L in lengths:
        j = int(rng.integers(len(seg_tables)))
        pi_doc = doc_props[j]
        table = seg_tables[j]
        # Segment schedule: draw (length, propensity) with replacement from
        # THIS doc's empirical table until the sequence is covered.
        seg_of = np.empty(L, dtype=np.int64)
        seg_props = []
        pos = 0
        while pos < L:
            li, pr = table[int(rng.integers(len(table)))]
            seg_of[pos:pos + int(li)] = len(seg_props)
            seg_props.append(np.asarray(pr, dtype=float))
            pos += int(li)
        s = np.empty(L, dtype=np.int8)
        p0 = np.clip(seg_props[0], 0.0, None)
        cur = int(rng.choice(k, p=p0 / p0.sum()))
        t = 0
        while t < L:
            r = int(rng.choice(dwell[cur]))
            s[t:t + r] = cur
            t += r
            if t >= L:
                break
            # Runs straddle segment boundaries; only the JUMP law switches
            # with the active segment (the fit's generative story).
            cur = int(rng.choice(
                k, p=_mix_row(cur, seg_props[seg_of[t]], pi_doc, J, a, b)))
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
    "seg_hier_categorical": (fit_seg_hier_categorical, gen_seg_hier_categorical),
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
