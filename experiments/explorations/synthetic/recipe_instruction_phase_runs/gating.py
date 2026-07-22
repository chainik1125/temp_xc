"""Recipe-instruction bench — § 8 EQUALITY-VARIANT gating STOP-gate (pre-grid).

The C4-review addition to the validity gates, mandatory before any grid. The
bench claims its primary latent ``e_t = [c_t = c_{t-1}]`` is **regime 3**
(order-2 / position-mixing: raw-LINEAR readouts blind, nonlinear/trained
access required). Verify, on the noiseless and noisy substrate:

- **(i) raw-LINEAR access to e_t ≈ chance** — per-token AND window
  concatenation. Scored as the balanced-accuracy access CEILING: the probe's
  decision threshold is optimized on train and scored held-out (a plain
  argmax probe under base rate 0.63 can predict all-1s and sit at balacc 0.5
  while real access exists — a validity gate must not hide behind that).
  If raw-linear reads e_t ≫ chance, the latent is regime 2 (additively
  readable): record it, **STOP before the grid**, report for re-scope.
- **(ii) the latent is PRESENT** — a nonlinear readout (MLP on raw window
  tiles) reaches e_t well above chance (the exact pair rule gives oracle 1
  for T ≥ 2, in-tile). If even nonlinear access fails → non-discriminating
  the other way — STOP, record.
- **DC control**: phase class c_t per-token near oracle (expected and fine).

**Known-at-freeze caveat this gate adjudicates:** unlike changepoint (Π
rebalanced uniform BY DESIGN so m_t carries no boundary information), this
grounded mirror has class-dependent continuation rates (per-symbol dwell
means 4.0/3.0/2.4/1.7/1.5 → P(e_t=1|c_t) spans ≈ 0.33–0.74), so ``c_t``
ALONE — linearly readable per token by design — predicts ``e_t`` above
chance. The decomposition below separates that DC leak (from-c_t line) from
genuine additive cross-position access (pair-additive ceiling) and from the
nonlinear-only residual (exact rule = 1), so the review can re-scope
precisely if (i) fails.

Gate (preregistered here, before any grid):
- (i)  raw-linear e ceiling within 0.05 of 0.5, per-token AND every T;
- (ii) MLP on raw T=2 tiles reaches e balacc ≥ 0.90 (noiseless);
- control: per-token c_t balacc ≥ 0.99 (noiseless);
- mirror sanity: match rate within [0.58, 0.68]; marginal max-dev < 0.08.

    .venv/bin/python -m experiments.explorations.synthetic.recipe_instruction_phase_runs.gating

Deterministic (SEED = 0); standalone (no framework / runner involvement).
Writes results/recipe_gating_stats.json + a figure.
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
SIGMA_NOISY = 0.5
MAX_PROBE_ROWS = 120_000
MAX_MLP_ROWS = 40_000
GATE_RAW_LINEAR_TOL = 0.05    # |e raw-linear balacc ceiling - 0.5|
GATE_MLP_PRESENT = 0.90       # MLP on raw T=2 tiles, noiseless
GATE_CONTROL_ORACLE = 0.99    # per-token c_t balacc on noiseless x_t
GATE_MATCH_BAND = (0.58, 0.68)
GATE_MARGINAL_TOL = 0.08
REAL_MARGINAL = (0.29, 0.51, 0.06, 0.07, 0.07)

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "recipe_gating_stats.json"
FIG_DIR = HERE / "figs"


def _subsample(rng, *arrays, cap=MAX_PROBE_ROWS):
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = rng.choice(n, size=cap, replace=False)
    return tuple(a[idx] for a in arrays)


def _balacc(y_true, y_pred):
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(y_true, y_pred))


def _logistic_balacc(X_tr, y_tr, X_ev, y_ev):
    """Plain argmax logistic probe (the eval convention), held-out balacc."""
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(X_tr, y_tr)
        return float(_balacc(y_ev, clf.predict(X_ev)))


def _score_threshold_ceiling(s_tr, y_tr, s_ev, y_ev):
    """Balanced-accuracy CEILING of a scalar score: best threshold on train,
    scored held-out. The honest access measure for a binary latent."""
    order = np.argsort(s_tr)
    ss, yy = s_tr[order], y_tr[order]
    n1, n0 = max(yy.sum(), 1), max((1 - yy).sum(), 1)
    # balacc of "predict 1 iff s > c" for every cut c between sorted scores
    tp = n1 - np.cumsum(yy)              # positives above the cut
    tn = np.cumsum(1 - yy)               # negatives at/below the cut
    bal = 0.5 * (tp / n1 + tn / n0)
    best = int(np.argmax(bal))
    cut = ss[best]
    pred = (s_ev > cut).astype(np.int64)
    fwd = _balacc(y_ev, pred)
    rev = _balacc(y_ev, 1 - pred)        # allow the inverted rule
    return float(max(fwd, rev))


def _linear_ceiling_balacc(X_tr, y_tr, X_ev, y_ev):
    """Raw-LINEAR access ceiling: logistic score + optimized threshold."""
    from sklearn.linear_model import LogisticRegression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(X_tr, y_tr)
        s_tr = clf.decision_function(X_tr)
        s_ev = clf.decision_function(X_ev)
    return _score_threshold_ceiling(s_tr, y_tr, s_ev, y_ev)


def _mlp_balacc(X_tr, y_tr, X_ev, y_ev, rng):
    """Nonlinear presence check: small MLP on the raw tile, argmax + ceiling."""
    from sklearn.neural_network import MLPClassifier
    X_tr, y_tr = _subsample(rng, X_tr, y_tr, cap=MAX_MLP_ROWS)
    X_ev, y_ev = _subsample(rng, X_ev, y_ev, cap=MAX_MLP_ROWS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=200,
                            random_state=SEED).fit(X_tr, y_tr)
        arg = _balacc(y_ev, clf.predict(X_ev))
        s_tr = clf.predict_proba(X_tr)[:, 1]
        s_ev = clf.predict_proba(X_ev)[:, 1]
    return float(arg), _score_threshold_ceiling(s_tr, y_tr, s_ev, y_ev)


def _strided_tiles(x, T):
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T * d)


def _leading_edge(labels, T):
    n, Ls = labels.shape[:2]
    n_tiles = Ls // T
    return labels[:, :n_tiles * T].reshape(n, n_tiles, T)[:, :, T - 1].reshape(-1)


def analytic_lines(c, e):
    """Access lines computable from the LABELS alone (no emission).

    - from_c_t: balanced-optimal rule on P(e=1|c_t) — the per-token DC-leak
      line (what any code that exposes c_t hands to a linear e-readout).
    - pair_additive: logistic on one-hot(c_{t-1}) ⊕ one-hot(c_t) — the
      ceiling for ANY additive readout with perfect class access.
    - exact_pair: e is deterministic in (c_{t-1}, c_t) → 1.0 (T ≥ 2 in-tile).
    """
    half = c.shape[0] // 2
    ct_tr, ct_ev = c[:half, 1:].reshape(-1), c[half:, 1:].reshape(-1)
    cp_tr, cp_ev = c[:half, :-1].reshape(-1), c[half:, :-1].reshape(-1)
    e_tr, e_ev = (e[:half, 1:].reshape(-1).astype(np.int64),
                  e[half:, 1:].reshape(-1).astype(np.int64))

    cond = {int(k): float(e_tr[ct_tr == k].mean()) for k in range(5)}
    s_tr = np.array([cond[int(k)] for k in ct_tr])
    s_ev = np.array([cond[int(k)] for k in ct_ev])
    from_ct = _score_threshold_ceiling(s_tr, e_tr, s_ev, e_ev)

    def onehot_pair(cp, ct):
        Z = np.zeros((cp.size, 10), dtype=np.float64)
        Z[np.arange(cp.size), cp] = 1.0
        Z[np.arange(ct.size), 5 + ct] = 1.0
        return Z
    pair_add = _linear_ceiling_balacc(onehot_pair(cp_tr, ct_tr), e_tr,
                                      onehot_pair(cp_ev, ct_ev), e_ev)
    return {
        "cond_match_rate_by_class": cond,
        "base_match_rate": float(e_tr.mean()),
        "e_balacc_from_c_t": from_ct,
        "e_balacc_pair_additive": pair_add,
        "e_exact_pair_rule": 1.0,
    }


def probes_on_x(data, rng, tag):
    """Per-token + window raw-linear + MLP probes on the emission."""
    x = data.x.numpy()
    c = data.extra["phase_class_labels"].numpy()
    e = data.extra["equality_labels"].numpy().astype(np.int64)
    half = x.shape[0] // 2

    out = {}
    # per-token (position t sees x_t only; target e_t, t >= 1)
    Xtr = x[:half, 1:].reshape(-1, x.shape[-1])
    Xev = x[half:, 1:].reshape(-1, x.shape[-1])
    ctr, cev = c[:half, 1:].reshape(-1), c[half:, 1:].reshape(-1)
    etr, eev = e[:half, 1:].reshape(-1), e[half:, 1:].reshape(-1)
    Xtr, ctr, etr = _subsample(rng, Xtr, ctr, etr)
    Xev, cev, eev = _subsample(rng, Xev, cev, eev)
    out["per_token"] = {
        "c_balacc_on_x": _logistic_balacc(Xtr, ctr, Xev, cev),
        "e_balacc_probe": _logistic_balacc(Xtr, etr, Xev, eev),
        "e_balacc_ceiling": _linear_ceiling_balacc(Xtr, etr, Xev, eev),
    }

    # window raw-linear (concatenated tile, leading-edge target)
    out["window"] = {}
    for T in T_GRID:
        Xt = _strided_tiles(x[:half], T)
        Xe = _strided_tiles(x[half:], T)
        et, ee = _leading_edge(e[:half], T), _leading_edge(e[half:], T)
        Xt2, et2 = _subsample(rng, Xt, et)
        Xe2, ee2 = _subsample(rng, Xe, ee)
        blk = {
            "e_balacc_probe": _logistic_balacc(Xt2, et2, Xe2, ee2),
            "e_balacc_ceiling": _linear_ceiling_balacc(Xt2, et2, Xe2, ee2),
        }
        if T == 2:   # presence check on the smallest window
            arg, ceil = _mlp_balacc(Xt2, et2, Xe2, ee2, rng)
            blk["e_balacc_mlp"] = arg
            blk["e_balacc_mlp_ceiling"] = ceil
        out["window"][str(T)] = blk
    print(f"  [{tag}] per-token e ceiling "
          f"{out['per_token']['e_balacc_ceiling']:.3f}; window ceilings "
          + ", ".join(f"T={T}: {out['window'][str(T)]['e_balacc_ceiling']:.3f}"
                      for T in T_GRID))
    return out


def main():
    from temp_bench.data.synthetic import recipe_instruction_phase_runs

    rng = np.random.default_rng(SEED)
    results = {"meta": {
        "N_SEQS_PROBE": N_SEQS_PROBE, "seq_len": SEQ_LEN, "T_grid": T_GRID,
        "sigma_noisy": SIGMA_NOISY,
        "gates": {"raw_linear_tol": GATE_RAW_LINEAR_TOL,
                  "mlp_present": GATE_MLP_PRESENT,
                  "control_oracle": GATE_CONTROL_ORACLE,
                  "match_band": GATE_MATCH_BAND,
                  "marginal_tol": GATE_MARGINAL_TOL},
    }}

    data = recipe_instruction_phase_runs(seq_len=SEQ_LEN, n_seqs=N_SEQS_PROBE,
                                         seed=SEED)
    c = data.extra["phase_class_labels"].numpy()
    e = data.extra["equality_labels"].numpy()

    # ── mirror sanity ─────────────────────────────────────────────────────
    marg = (np.bincount(c.ravel(), minlength=5) / c.size)
    results["mirror"] = {
        "marginal": marg.tolist(),
        "marginal_real": list(REAL_MARGINAL),
        "marginal_max_dev": float(np.abs(marg - np.array(REAL_MARGINAL)).max()),
        "match_rate": float(e[:, 1:].mean()),
    }

    # ── analytic access lines (labels only) ───────────────────────────────
    results["analytic"] = analytic_lines(c, e)

    # ── probes on the noiseless + noisy emission ──────────────────────────
    results["noiseless"] = probes_on_x(data, rng, "noiseless")
    data_n = recipe_instruction_phase_runs(seq_len=SEQ_LEN, n_seqs=N_SEQS_PROBE,
                                           sigma=SIGMA_NOISY, seed=SEED)
    results["noisy"] = probes_on_x(data_n, rng, f"noisy sigma={SIGMA_NOISY}")

    # ── verdict ───────────────────────────────────────────────────────────
    nl = results["noiseless"]
    raw_lines = [nl["per_token"]["e_balacc_ceiling"]] + \
        [nl["window"][str(T)]["e_balacc_ceiling"] for T in T_GRID]
    worst_raw = max(abs(v - 0.5) for v in raw_lines)
    mlp = nl["window"]["2"].get("e_balacc_mlp_ceiling", 0.0)
    m = results["mirror"]
    results["verdict"] = {
        "raw_linear_lines": raw_lines,
        "worst_raw_linear_dev": float(worst_raw),
        "i_raw_linear_at_chance": bool(worst_raw <= GATE_RAW_LINEAR_TOL),
        "ii_latent_present_nonlinear": bool(mlp >= GATE_MLP_PRESENT),
        "control_oracle_reachable": bool(
            nl["per_token"]["c_balacc_on_x"] >= GATE_CONTROL_ORACLE),
        "mirror_sane": bool(
            GATE_MATCH_BAND[0] <= m["match_rate"] <= GATE_MATCH_BAND[1]
            and m["marginal_max_dev"] < GATE_MARGINAL_TOL),
    }
    v = results["verdict"]
    v["passes_gate"] = bool(v["i_raw_linear_at_chance"]
                            and v["ii_latent_present_nonlinear"]
                            and v["control_oracle_reachable"]
                            and v["mirror_sane"])
    if not v["i_raw_linear_at_chance"]:
        v["stop_reason"] = (
            "regime-2 leak: raw-LINEAR access to e_t exceeds chance — "
            "decompose via analytic lines (from-c_t vs pair-additive) and "
            "STOP before the grid; the review must re-scope the claim.")
    elif not v["ii_latent_present_nonlinear"]:
        v["stop_reason"] = ("latent not present: even nonlinear access fails "
                            "— non-discriminating; STOP.")

    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    a, nl, v, m = r["analytic"], r["noiseless"], r["verdict"], r["mirror"]
    print("\n===== RECIPE-INSTRUCTION — § 8 EQUALITY-VARIANT STOP-GATE =====")
    print(f"mirror: marginal {[round(x, 3) for x in m['marginal']]} "
          f"(real {m['marginal_real']}; max dev {m['marginal_max_dev']:.3f}); "
          f"match rate {m['match_rate']:.3f}")
    print(f"\n  analytic access lines (labels only; balacc ceilings):")
    print(f"    from c_t alone (DC leak)      {a['e_balacc_from_c_t']:.3f}")
    print(f"    pair-additive (one-hot ⊕)     {a['e_balacc_pair_additive']:.3f}")
    print(f"    exact pair rule (in-tile T≥2)  1.000")
    print(f"    cond match rate by class      "
          f"{ {k: round(x, 3) for k, x in a['cond_match_rate_by_class'].items()} }")
    print("\n  probes on noiseless x (balacc; probe = eval convention, "
          "ceiling = threshold-optimized):")
    pt = nl["per_token"]
    print(f"    per-token: c {pt['c_balacc_on_x']:.3f} | e probe "
          f"{pt['e_balacc_probe']:.3f}, e CEILING {pt['e_balacc_ceiling']:.3f}")
    for T in T_GRID:
        w = nl["window"][str(T)]
        extra = (f", MLP {w['e_balacc_mlp']:.3f} (ceil "
                 f"{w['e_balacc_mlp_ceiling']:.3f})") if "e_balacc_mlp" in w else ""
        print(f"    raw tile T={T}: e probe {w['e_balacc_probe']:.3f}, "
              f"e CEILING {w['e_balacc_ceiling']:.3f}{extra}")
    print(f"\n  VERDICT: (i)  raw-linear at chance    "
          f"{v['i_raw_linear_at_chance']}  (worst dev {v['worst_raw_linear_dev']:.3f} "
          f"vs tol {GATE_RAW_LINEAR_TOL})")
    print(f"           (ii) latent present (MLP)    {v['ii_latent_present_nonlinear']}")
    print(f"           control oracle reachable    {v['control_oracle_reachable']}")
    print(f"           mirror sane                 {v['mirror_sane']}")
    print(f"           ==> passes_gate = {v['passes_gate']}")
    if "stop_reason" in v:
        print(f"           STOP: {v['stop_reason']}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, nl = r["analytic"], r["noiseless"]
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    names = ["per-token\n(x_t)"] + [f"raw tile\nT={T}" for T in T_GRID]
    ceilings = [nl["per_token"]["e_balacc_ceiling"]] + \
        [nl["window"][str(T)]["e_balacc_ceiling"] for T in T_GRID]
    xpos = np.arange(len(names))
    ax.bar(xpos, ceilings, 0.55, color="#3182bd", alpha=0.85,
           label="raw-LINEAR access ceiling for $e_t$")
    if "e_balacc_mlp_ceiling" in nl["window"]["2"]:
        ax.scatter([1], [nl["window"]["2"]["e_balacc_mlp_ceiling"]], s=90,
                   marker="*", color="#D55E00", zorder=5,
                   label="nonlinear (MLP) on raw T=2 tile")
    ax.axhline(a["e_balacc_from_c_t"], color="#7b3294", ls="--", lw=1.4,
               label=f"from $c_t$ alone (DC leak) = {a['e_balacc_from_c_t']:.3f}")
    ax.axhline(a["e_balacc_pair_additive"], color="#008837", ls="-.", lw=1.4,
               label=f"pair-additive ceiling = {a['e_balacc_pair_additive']:.3f}")
    ax.axhline(0.5, color="0.4", ls=":", lw=1.2, label="chance = 0.5")
    ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names)
    ax.set_ylabel("held-out balanced accuracy (threshold-optimized)")
    ax.set_ylim(0.4, 1.05)
    ax.set_title("recipe_instruction § 8 equality-variant gate: raw-linear access "
                 "to $e_t$\nvs the analytic DC-leak / additive / nonlinear lines",
                 fontsize=10.5)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    for ext, dpi in [("pdf", None), ("png", 120)]:
        fig.savefig(FIG_DIR / f"recipe_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/recipe_gating.*")


if __name__ == "__main__":
    main()
