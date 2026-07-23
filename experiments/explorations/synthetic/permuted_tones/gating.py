"""Permuted tones (FB-5) — T1 proof-gate numerics + § 8 STOP-gate.

Runs on the ACTUAL built generator (``permuted_tones``) at the frozen card
parameters (freqbench/cards/FB-5.md § 2). Deterministic (SEED = 0).

**T1 discharges:**
- **P1 numerical:** per-class pooled symbol marginal ≈ uniform (TV), raw
  per-token probe ≈ chance, and the frozen T=1 falsifier (> 0.1 recovery ⇒
  bug — STOP).
- **Ceiling:** the matched-filter oracle curve per T ∈ {2,4,8} (the exact-
  template ML decoder; card expects near-saturation by T=8 — window
  uniqueness verified at freeze).
- **Non-absorption re-verification:** per-class lag-1 trajectory
  autocorrelation on the BUILT generator ≈ 0 ± O(1/√M), vs the tone
  ladder's ±1 span (the card § 1 distinguishing statistic).

**§ 8 discriminability STOP-gate (equality-variant):** raw-linear readouts
(per-token, window-concat) ≈ chance while the oracle reads the latent ≫
chance at T ∈ {4,8}. A-priori bars: per-token ≤ chance+0.02 (P1 is exact);
window-concat linear ≤ chance+0.05 — the +0.05 (not +0.02) is set a priori
citing the FB-4 datum: on an equally mean-dead task (P2 bounds means only),
a large-sample multiclass linear probe reads ≈ +0.03 off 2nd-moment
structure; the gate's discriminating quantity is the probe-vs-oracle GAP.
Oracle bar: ≥ chance+0.3 at T=8 (card § 4; expected ≈ 1.0).

**Envelope reference (NOT a gate):** multinomial-logistic classifier on the
window's circle-plane per-DCT-index energies ONLY (order/cross-channel
phase discarded) per T ∈ {2,4,8} — the card's spectral-envelope reference
curve, against which the grid's spectral-vs-post comparison is read.

    .venv/bin/python -m experiments.explorations.synthetic.permuted_tones.gating

Writes ``results/permuted_gating_stats.json``. Gating script under LOOP.md
T3 strict commit-then-run: committed before first execution; amendments as
their own commits.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEED = 0
M, K = 101, 10
D_IN = 128
SIGMA = 0.10
SEQ_LEN = 64
N_SEQS = 6000
T_GRID = [2, 4, 8]

TOL_TOKEN_FLOOR = 0.02          # chance + tol (P1 exact)
TOL_WINDOW_FLOOR = 0.05         # chance + tol (FB-4 2nd-moment probe datum)
GATE_ORACLE_T8 = 0.30           # oracle − chance at T=8 (card § 4)
FALSIFIER_T1_RECOVERY = 0.10    # frozen card § 6.5

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "permuted_gating_stats.json"


def _dct(T):
    tau = np.arange(T)
    psi = np.zeros((T, T))
    for w in range(T):
        psi[w] = (np.sqrt(1 / T) if w == 0 else
                  np.sqrt(2 / T) * np.cos(np.pi * (tau + 0.5) * w / T))
    return psi


def _tiles(x, labels, T, n_max=30_000, rng=None):
    n, s, d = x.shape
    k = s // T
    tx = x[:, : k * T, :].reshape(n * k, T, d)
    ty = labels[:, : k * T][:, ::T].reshape(n * k)
    if rng is not None and tx.shape[0] > n_max:
        i = rng.choice(tx.shape[0], n_max, replace=False)
        tx, ty = tx[i], ty[i]
    return tx, ty


def _logistic(z_tr, y_tr, z_ev, y_ev):
    import warnings
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=300).fit(z_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(z_ev)))


def main() -> None:
    from temp_bench.data.synthetic import permuted_tones
    from temp_bench.evals.permuted_recovery import _matched_filter_pred

    data = permuted_tones(M=M, K=K, d_in=D_IN, sigma=SIGMA, seq_len=SEQ_LEN,
                          n_seqs=N_SEQS, seed=SEED)
    x = data.x.numpy().astype(np.float64)
    lab = data.extra["schedule_labels"].numpy()
    U = data.emission_features.numpy().astype(np.float64)
    P = data.extra["schedule_table"].numpy()
    R = data.extra["circle_plane"].numpy().astype(np.float64)
    chance = 1.0 / K
    rng = np.random.default_rng(SEED)
    out = {"card": "freqbench/cards/FB-5.md", "seed": SEED,
           "params": {"M": M, "K": K, "d_in": D_IN, "sigma": SIGMA,
                      "n_seqs": N_SEQS}}

    # ── P1: pooled marginal TV per class + non-absorption lag-1 statistic ──
    B = data.extra["offset_labels"].numpy()
    Y0 = lab[:, 0]
    t64 = np.arange(SEQ_LEN)
    tvs, lag1 = [], []
    for k in range(K):
        z = P[k][(B[Y0 == k][:, None] + t64[None, :]) % M].ravel()
        hist = np.bincount(z, minlength=M) / len(z)
        tvs.append(float(0.5 * np.abs(hist - 1.0 / M).sum()))
        ang = 2 * np.pi * P[k] / M
        lag1.append(float(np.cos(ang[np.r_[1:M, 0]] - ang).mean()))
    out["p1_marginal_tv_per_class"] = tvs
    out["nonabsorption_lag1_per_schedule"] = lag1

    # ── floors: per-token probe (+ frozen T=1 falsifier) & window-concat ──
    half = N_SEQS // 2
    tok_tr = x[:half].reshape(-1, D_IN)
    tok_ev = x[half:].reshape(-1, D_IN)
    yt_tr = lab[:half].reshape(-1)
    yt_ev = lab[half:].reshape(-1)
    i_tr = rng.choice(len(tok_tr), 30_000, replace=False)
    i_ev = rng.choice(len(tok_ev), 30_000, replace=False)
    tok_acc = _logistic(tok_tr[i_tr], yt_tr[i_tr], tok_ev[i_ev], yt_ev[i_ev])
    T8 = 8
    wtr, wy_tr = _tiles(x[:half], lab[:half], T8, rng=np.random.default_rng(1))
    wev, wy_ev = _tiles(x[half:], lab[half:], T8, rng=np.random.default_rng(2))
    win_acc = _logistic(wtr.reshape(len(wtr), -1), wy_tr,
                        wev.reshape(len(wev), -1), wy_ev)
    t1_recovery = (tok_acc - chance) / (1 - chance)
    out["floors"] = {"chance": chance, "per_token_balacc": tok_acc,
                     "window_concat_linear_balacc": win_acc,
                     "t1_recovery": t1_recovery}

    # ── ceiling + envelope reference per T ────────────────────────────────
    out["by_T"] = {}
    for T in T_GRID:
        tx, ty = _tiles(x, lab, T, n_max=12_000, rng=np.random.default_rng(3 + T))
        opred = _matched_filter_pred(tx, U, P, M)
        oracle = float((opred == ty).mean())
        # envelope: circle-plane projection → per-DCT-index energies only
        proj = tx @ R                                     # (N, T, 2)
        coef = np.einsum("wt,ntc->nwc", _dct(T), proj)
        env = (coef ** 2).sum(axis=2)                     # (N, T) energies
        n2 = len(env) // 2
        env_acc = _logistic(env[:n2], ty[:n2], env[n2:], ty[n2:])
        out["by_T"][T] = {"oracle": oracle, "envelope_ref": env_acc,
                          "n_tiles": int(len(ty))}

    # ── verdict (a fail is a STOP, never a retune) ────────────────────────
    oracle_t8 = out["by_T"][8]["oracle"]
    out["verdict"] = {
        "p1_marginal_uniform": max(tvs) < 0.10,
        "falsifier_t1_fired": t1_recovery > FALSIFIER_T1_RECOVERY,
        "token_floor_ok": tok_acc - chance <= TOL_TOKEN_FLOOR,
        "window_floor_ok": win_acc - chance <= TOL_WINDOW_FLOOR,
        "oracle_reads_latent": oracle_t8 - chance >= GATE_ORACLE_T8,
        "oracle_t8": oracle_t8,
    }
    v = out["verdict"]
    v["passes_gate"] = bool(v["p1_marginal_uniform"] and v["token_floor_ok"]
                            and v["window_floor_ok"] and v["oracle_reads_latent"]
                            and not v["falsifier_t1_fired"])
    print(json.dumps(out["verdict"], indent=1), flush=True)
    print(json.dumps(out["by_T"], indent=1), flush=True)
    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
