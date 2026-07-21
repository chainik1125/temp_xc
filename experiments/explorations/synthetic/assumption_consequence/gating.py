"""Assumption→consequence bench — § 8 gating due-diligence (BEFORE the grid).

Confirms, on the real generator (briefing gate): (i) each latent's oracle is
reachable by a probe on the noiseless emission, and (ii) the chance floor sits
where the spec says. Also states — before any architecture runs — the one
structural fact of this substrate:

**The mirror is order-1 Markov, so the current state ``s_i`` is a sufficient
statistic for ``s_{i+1}``.** The Bayes ceiling for the AC (next-state) probe is
the one-step conditional (spec § 3), and since the state direction is in the
per-token span, the per-token INFO ceiling *equals* that oracle — there is no
information-theoretic per-token/window separation on this substrate (unlike
backtracking's DPI floor or changepoint's equality-pattern argument). The
frozen § 5 prediction ("per-token blind to the directed dependency") is
therefore expected to be adjudicated by what the *trained scarce code* exposes
linearly at the leading edge, not by an access bound. Recorded here so the
blind grid's verdict is interpreted against the right ceiling.

Gate (preregistered here, before any grid):
- state oracle reachable: per-token multinomial probe on noiseless ``x_t``
  reaches balanced acc ≥ 0.99;
- next-state separable: analytic Bayes-balanced oracle − 1/3 ≥ 0.10, and the
  empirical probe-on-x oracle agrees with the analytic value within 0.02;
- chance floors: shuffled-label probes within 0.02 of 1/3; the realized
  directed edge P(C@t+1 | A@t) within 0.03 of the g7 fit (0.363).

    .venv/bin/python -m experiments.explorations.synthetic.assumption_consequence.gating

Deterministic (SEED = 0); standalone (no framework / runner involvement).
Writes results/assumption_gating_stats.json + a figure.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SEED = 0
N_SEQS_PROBE = 6000
SEQ_LEN = 64
T_GRID = [2, 4, 8]
MAX_PROBE_ROWS = 120_000
GATE_STATE_ORACLE = 0.99      # per-token state balacc on noiseless x_t
GATE_NEXT_SEP = 0.10          # analytic oracle balacc - 1/3
GATE_ORACLE_AGREE = 0.02      # |empirical - analytic| oracle balacc
GATE_CHANCE_TOL = 0.02        # shuffled floors vs 1/3
GATE_FWD_TOL = 0.03           # realized P(C|A) vs the g7 fit

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "assumption_gating_stats.json"
FIG_DIR = HERE / "figs"


def _subsample(rng, *arrays, cap=MAX_PROBE_ROWS):
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = rng.choice(n, size=cap, replace=False)
    return tuple(a[idx] for a in arrays)


def _logistic_balacc(X_tr, y_tr, X_ev, y_ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(X_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(X_ev)))


def _strided_tiles(x, T):
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T * d)


def _leading_edge(labels, T):
    n, Ls = labels.shape[:2]
    n_tiles = Ls // T
    return labels[:, :n_tiles * T].reshape(n, n_tiles, T)[:, :, T - 1].reshape(-1)


def analytic_oracle(P: np.ndarray, pi: np.ndarray) -> dict:
    """Bayes rules of the one-step conditional + their exact scores.

    Balanced: rule_b = argmax_j P(j|i)/pi_j; balacc = mean_j recall_j with
    recall_j = Σ_i pi_i P_ij 1{rule_b(i)=j} / pi_j. Raw: rule_r = argmax_j
    P(j|i); acc = Σ_i pi_i max_j P_ij; chance (modal marginal) = max_j pi_j.
    """
    rule_b = np.argmax(P / pi[None, :], axis=1)
    recall = np.zeros(3)
    for j in range(3):
        num = sum(pi[i] * P[i, j] for i in range(3) if rule_b[i] == j)
        recall[j] = num / pi[j]
    acc_r = float(sum(pi[i] * P[i].max() for i in range(3)))
    return {
        "rule_balanced": rule_b.tolist(),
        "oracle_balacc": float(recall.mean()),
        "recall_by_class": recall.tolist(),
        "oracle_rawacc": acc_r,
        "chance_rawacc_modal_marginal": float(pi.max()),
        "chance_balacc": 1.0 / 3.0,
    }


def main():
    from sklearn.metrics import balanced_accuracy_score

    from temp_bench.data.synthetic import assumption_consequence
    from temp_bench.evals.assumption_recovery import bayes_balanced_rule

    rng = np.random.default_rng(SEED)
    data = assumption_consequence(seq_len=SEQ_LEN, n_seqs=N_SEQS_PROBE, seed=SEED)
    x = data.x.numpy()
    s = data.extra["state_labels"].numpy()
    nx = data.extra["next_state_labels"].numpy()
    P = np.array(data.extra["P"])
    pi = np.array(data.extra["pi"])

    results = {"meta": {
        "N_SEQS_PROBE": N_SEQS_PROBE, "seq_len": SEQ_LEN, "T_grid": T_GRID,
        "gates": {"state_oracle": GATE_STATE_ORACLE, "next_sep": GATE_NEXT_SEP,
                  "oracle_agree": GATE_ORACLE_AGREE, "chance_tol": GATE_CHANCE_TOL,
                  "fwd_tol": GATE_FWD_TOL},
        "P": P.tolist(), "pi": pi.tolist(),
    }}

    # ── mirror sanity: the realized directed edge ─────────────────────────
    cur, nxt = s[:, :-1].reshape(-1), s[:, 1:].reshape(-1)
    fwd = float((nxt[cur == 1] == 2).mean())         # P(C@t+1 | A@t)
    rev = float((cur[nxt == 1] == 2).mean())         # time-reversed
    marg_C = float((s == 2).mean())
    results["mirror"] = {"fwd_rate": fwd, "rev_rate": rev, "asym": fwd - rev,
                         "marginal_C": marg_C, "fwd_fit_g7": float(P[1, 2])}

    # ── analytic oracle of the one-step conditional ───────────────────────
    results["analytic"] = analytic_oracle(P, pi)
    assert results["analytic"]["rule_balanced"] == \
        bayes_balanced_rule(P, pi).tolist()

    # ── per-token probes on the noiseless emission ────────────────────────
    half = N_SEQS_PROBE // 2
    m_val = nx >= 0
    Xtr = x[:half].reshape(-1, x.shape[-1])[m_val[:half].reshape(-1)]
    Xev = x[half:].reshape(-1, x.shape[-1])[m_val[half:].reshape(-1)]
    str_, sev = s[:half][m_val[:half]], s[half:][m_val[half:]]
    ntr, nev = nx[:half][m_val[:half]], nx[half:][m_val[half:]]
    Xtr, str_, ntr = _subsample(rng, Xtr, str_, ntr)
    Xev, sev, nev = _subsample(rng, Xev, sev, nev)

    state_bal = _logistic_balacc(Xtr, str_, Xev, sev)
    next_bal = _logistic_balacc(Xtr, ntr, Xev, nev)
    perm = rng.permutation(len(ntr))
    next_floor = _logistic_balacc(Xtr, ntr[perm], Xev, nev)
    state_floor = _logistic_balacc(Xtr, str_[perm], Xev, sev)
    # empirical oracle: the Bayes-balanced rule on the TRUE current state
    rule = np.array(results["analytic"]["rule_balanced"])
    next_oracle_emp = float(balanced_accuracy_score(nev, rule[sev]))
    results["per_token"] = {
        "state_balacc_on_x": state_bal, "state_shuffled_floor": state_floor,
        "next_balacc_on_x": next_bal, "next_shuffled_floor": next_floor,
        "next_oracle_empirical": next_oracle_emp,
    }

    # ── window raw-linear probes (concatenated raw tile, leading edge) ────
    results["window"] = {}
    for T in T_GRID:
        Xt = _strided_tiles(x[:half], T)
        Xe = _strided_tiles(x[half:], T)
        st_t, st_e = _leading_edge(s[:half], T), _leading_edge(s[half:], T)
        nx_t, nx_e = _leading_edge(nx[:half], T), _leading_edge(nx[half:], T)
        mt, me = nx_t >= 0, nx_e >= 0
        Xt2, st_t2, nx_t2 = _subsample(rng, Xt[mt], st_t[mt], nx_t[mt])
        Xe2, st_e2, nx_e2 = _subsample(rng, Xe[me], st_e[me], nx_e[me])
        results["window"][str(T)] = {
            "state_balacc_raw_linear": _logistic_balacc(Xt2, st_t2, Xe2, st_e2),
            "next_balacc_raw_linear": _logistic_balacc(Xt2, nx_t2, Xe2, nx_e2),
        }

    # ── verdict ───────────────────────────────────────────────────────────
    an = results["analytic"]
    pt = results["per_token"]
    sep = an["oracle_balacc"] - 1.0 / 3.0
    results["verdict"] = {
        "state_oracle_reachable": bool(pt["state_balacc_on_x"] >= GATE_STATE_ORACLE),
        "next_separable": bool(sep >= GATE_NEXT_SEP),
        "next_separation": float(sep),
        "oracle_empirical_agrees": bool(
            abs(pt["next_oracle_empirical"] - an["oracle_balacc"]) <= GATE_ORACLE_AGREE),
        "chance_floors_ok": bool(
            abs(pt["next_shuffled_floor"] - 1 / 3) <= GATE_CHANCE_TOL
            and abs(pt["state_shuffled_floor"] - 1 / 3) <= GATE_CHANCE_TOL),
        "fwd_edge_matches_fit": bool(abs(fwd - P[1, 2]) <= GATE_FWD_TOL),
        "per_token_info_ceiling_equals_oracle": bool(
            pt["next_balacc_on_x"] >= pt["next_oracle_empirical"] - GATE_ORACLE_AGREE),
    }
    results["verdict"]["passes_gate"] = bool(
        results["verdict"]["state_oracle_reachable"]
        and results["verdict"]["next_separable"]
        and results["verdict"]["oracle_empirical_agrees"]
        and results["verdict"]["chance_floors_ok"]
        and results["verdict"]["fwd_edge_matches_fit"])

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    m, an, pt, v = r["mirror"], r["analytic"], r["per_token"], r["verdict"]
    print("\n======= ASSUMPTION→CONSEQUENCE — § 8 GATING DUE-DILIGENCE =======")
    print(f"mirror: fwd P(C|A) = {m['fwd_rate']:.3f} (g7 fit {m['fwd_fit_g7']:.3f}), "
          f"rev = {m['rev_rate']:.3f}, asym = {m['asym']:.3f}, "
          f"marginal C = {m['marginal_C']:.3f}")
    print(f"\n  analytic (order-1 conditional): oracle balacc = {an['oracle_balacc']:.3f} "
          f"(chance 1/3; separation {an['oracle_balacc'] - 1/3:.3f}); "
          f"raw-acc oracle {an['oracle_rawacc']:.3f} vs modal-marginal "
          f"{an['chance_rawacc_modal_marginal']:.3f}")
    print(f"  balanced Bayes rule (from N/A/C): {an['rule_balanced']}  "
          "(state persists)")
    print("\n  per-token probes on noiseless x_t:")
    print(f"    state balacc  {pt['state_balacc_on_x']:.3f}  (oracle 1; shuffled "
          f"{pt['state_shuffled_floor']:.3f})")
    print(f"    next  balacc  {pt['next_balacc_on_x']:.3f}  (empirical oracle "
          f"{pt['next_oracle_empirical']:.3f}; shuffled {pt['next_shuffled_floor']:.3f})")
    print("\n  window raw-linear (concatenated tile, leading edge):")
    print(f"   {'T':>3}{'state':>9}{'next':>9}")
    for T, w in r["window"].items():
        print(f"   {T:>3}{w['state_balacc_raw_linear']:>9.3f}"
              f"{w['next_balacc_raw_linear']:>9.3f}")
    print("\n  NOTE: order-1 mirror ⇒ s_i sufficient ⇒ per-token INFO ceiling "
          f"= oracle ({v['per_token_info_ceiling_equals_oracle']}). No "
          "info-theoretic per-token/window separation on this substrate; the "
          "grid adjudicates what trained scarce codes expose linearly.")
    print(f"\n  VERDICT: state oracle reachable   {v['state_oracle_reachable']}")
    print(f"           next-state separable      {v['next_separable']} "
          f"(sep {v['next_separation']:.3f} ≥ {GATE_NEXT_SEP})")
    print(f"           oracle empirical agrees   {v['oracle_empirical_agrees']}")
    print(f"           chance floors ok          {v['chance_floors_ok']}")
    print(f"           fwd edge matches g7 fit   {v['fwd_edge_matches_fit']}")
    print(f"           ==> passes_gate = {v['passes_gate']}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    an, pt = r["analytic"], r["per_token"]
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    Ts = [1] + [int(t) for t in r["window"]]
    next_vals = [pt["next_balacc_on_x"]] + \
        [r["window"][str(T)]["next_balacc_raw_linear"] for T in Ts[1:]]
    state_vals = [pt["state_balacc_on_x"]] + \
        [r["window"][str(T)]["state_balacc_raw_linear"] for T in Ts[1:]]
    xpos = np.arange(len(Ts))
    ax.bar(xpos - 0.2, state_vals, 0.38, color="#D55E00", alpha=0.85,
           label="state $s_i$ (DC; oracle = 1)")
    ax.bar(xpos + 0.2, next_vals, 0.38, color="#3182bd", alpha=0.85,
           label="next state $s_{i+1}$ (AC-directed)")
    ax.axhline(an["oracle_balacc"], color="#3182bd", ls="--", lw=1.2,
               label=f"order-1 Bayes oracle = {an['oracle_balacc']:.3f}")
    ax.axhline(1 / 3, color="0.4", ls=":", lw=1.2, label="chance (balanced) = 1/3")
    ax.set_xticks(xpos)
    ax.set_xticklabels(["per-token\n(x_t)"] + [f"raw tile\nT={T}" for T in Ts[1:]])
    ax.set_ylabel("held-out balanced accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("assumption_consequence § 8 gating: every raw readout sits at the\n"
                 "order-1 oracle — s_i is sufficient (no per-token/window info gap)",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="center right")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext, dpi in [("pdf", None), ("png", 120)]:
        fig.savefig(FIG_DIR / f"assumption_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/assumption_gating.*")


if __name__ == "__main__":
    main()
