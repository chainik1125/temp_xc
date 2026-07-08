"""Frequency / cyclic-tone bench — § 8 gating due-diligence (compute BEFORE building).

Port of Dmitry's FrequencyBench (``origin/dmitry-spectral-sprint2``,
``docs/dmitry/sprints/2026-06-10_freqbench_sprint/``) onto our fair backbone +
conventions. This standalone analysis (no framework / runner) settles the three
OPEN design decisions in ``frequency/bench_spec.md`` §§ 3/8 and confirms the
ceilings are separated, BEFORE anything is built. Analogue of
``backtracking.gating`` / ``changepoint.gating``.

The substrate (bench_spec § 1–2): a symbol walks a cyclic alphabet ``Z_M`` at a
hidden velocity ``Y`` — ``Q_t = (B + Y·t) mod M``, phase ``B ~ Unif(Z_M)``,
``Y ~ Unif(Ω)`` (the label is the *index* into Ω). Two embeddings:

- **circle (headline):** ``u_a = R·[cos 2πa/M, sin 2πa/M]``, ``R`` a random
  ``d×2`` isometry. Velocity ``Y`` becomes a temporal **tone** at ``f = Y/M``
  cycles/token; the ML decoder is the **periodogram peak-pick** (project onto
  the 2-D circle plane with the true ``R`` → complex tone → correlate against
  ``e^{-2πi(Y/M)t}``, argmax over Ω).
- **random (symmetry null):** ``M`` orthonormal directions in ``d ≥ M``. For
  prime ``M`` and an exchangeable frame the relabel ``a ↦ c·a`` maps ``Y ↦ cY``
  bijectively over ``Z_M^*`` → all nonzero velocities are one orbit → **flat**
  response (the ratio-invariance null). The GLRT template decoder still reads
  ``Y`` off the symbol differences, but with NO frequency ordering.

What this script confirms (the gate):

1. **Ground truth settled** — the circle codebook is the ``M`` circle atoms (a
   2-D object; ``d_sae`` anchors on ``M``, NOT 2); random codebook = ``M``
   orthonormal dirs; memorization threshold = ``|Ω|·M`` distinct clean windows.
   Final ``M / d_in / Ω / σ / seq_len / L / T``.
2. **Per-token velocity ceiling ≈ chance** — single token uniform over symbols
   ⇒ ``I(Y; x_t) = 0`` (provable DPI); empirical logistic probe ≈ ``1/|Ω|``.
3. **Raw-linear WINDOW ceiling ≈ chance** — the velocity lives in the 2nd moment
   (the phase progression), NOT the class-conditional mean (``E[x_t|Y]≈0`` since
   the circle is centred and ``B`` uniform). So a *linear* probe on the raw
   concatenated tile is at chance too — a window *win* on a trained code is the
   nonlinear feature-learning, not linear access (mirrors changepoint / the
   signed-motion additive-score argument). The untrained control checks the
   nonlinear-access residual.
4. **Window oracle = periodogram accuracy** — near-1 for ``f ≳ 1/T`` at the
   chosen ``σ``; the per-Ω-class oracle traces the **Rayleigh** structure
   (``|Δf| < 1/T`` unresolvable). Reported per tile size ``T``.
5. **Random-embedding response FLAT** — the per-Ω-class oracle shows no
   frequency ordering (verifies the ratio theorem; separates "frequency" from
   "symbol identity").
6. **Separation gate** — circle window oracle (best ``T``, resolvable band)
   minus per-token ≥ the bar.

    .venv/bin/python -m experiments.explorations.synthetic.frequency.gating

Deterministic (SEED = 0). Writes frequency/results/frequency_gating_stats.json
+ a figure. Proceed to build ONLY if ``verdict.passes_gate`` is true.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np

SEED = 0

# ── settled-candidate design (confirmed / chosen by this script) ──────────
M = 101                                   # prime alphabet (ratio-invariance null)
OMEGA = [0, 1, 2, 4, 8, 16, 24, 32, 40, 50]   # velocity set (sprint ladder DC→Nyquist)
D_IN = 128                                # d_in ≥ M for the null; unified across modes
SEQ_LEN = 64                              # matches the family (signed_motion/changepoint)
L = 32                                    # common tiled eval window (power of two)
T_GRID = [2, 4, 8, 16]                    # arch/eval tiles (powers of two ≤ L)
SIGMA_SWEEP = [0.05, 0.10, 0.20, 0.25]    # noise sweep → pick σ (settle SNR)
SIGMA = 0.10                              # chosen default (confirmed by the sweep below)

N_SEQS = 8000                             # sequences for the Monte-Carlo estimates
N_PROBE_ROWS = 40_000                     # subsample cap for sklearn probes

# gates
GATE_PER_TOKEN_TOL = 0.03                 # |per-token acc - chance|
GATE_RAW_LINEAR_TOL = 0.05                # |raw-linear window acc - chance|
GATE_ORACLE_RESOLVABLE = 0.85             # circle oracle on the resolvable band (best T)
GATE_SEPARATION = 0.50                    # circle window oracle - per-token (resolvable band)
GATE_NULL_FLAT_RANGE = 0.20               # max spread of random per-Ω-class oracle (flatness)

CHANCE = 1.0 / len(OMEGA)

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "frequency_gating_stats.json"
FIG_DIR = HERE / "figs"


# ── the generative process (mirrors the planned cyclic_tones generator) ──


def circle_embedding(d_in, rng):
    """u_a = R·[cos 2πa/M, sin 2πa/M], R a random d×2 isometry. Returns (U, R)."""
    A = rng.standard_normal((d_in, 2))
    R, _ = np.linalg.qr(A)                            # (d_in, 2), R^T R = I_2
    ang = 2 * np.pi * np.arange(M) / M
    V = np.stack([np.cos(ang), np.sin(ang)], axis=1)  # (M, 2)
    U = V @ R.T                                       # (M, d_in), unit-norm rows
    return U.astype(np.float64), R.astype(np.float64)


def random_embedding(d_in, rng):
    """M orthonormal directions in R^{d_in} (d_in ≥ M). Returns (U, None)."""
    if d_in < M:
        raise ValueError(f"random embedding needs d_in ({d_in}) >= M ({M})")
    A = rng.standard_normal((d_in, M))
    Q, _ = np.linalg.qr(A)
    return Q.T[:M].astype(np.float64), None


def simulate(mode, d_in, sigma, n_seqs, seq_len, rng):
    """Return x (n, seq_len, d_in), vel_class (n,), Y (n,), U (M,d), R or None."""
    lab = rng.integers(0, len(OMEGA), size=n_seqs)           # class index into Ω
    Y = np.asarray(OMEGA, dtype=np.int64)[lab]               # velocity value
    B = rng.integers(0, M, size=n_seqs)                      # phase
    t = np.arange(seq_len)[None, :]
    Q = (B[:, None] + Y[:, None] * t) % M                    # (n, seq_len)
    U, R = (circle_embedding(d_in, rng) if mode == "circle"
            else random_embedding(d_in, rng))
    x = U[Q]                                                 # (n, seq_len, d_in)
    if sigma > 0:
        x = x + sigma * rng.standard_normal(x.shape)
    return x.astype(np.float64), lab.astype(np.int64), Y, U, R


# ── tiling helpers (leading-edge target, as the eval probes it) ──────────


def strided_tiles(x, T):
    """Non-overlapping T-tiles → rows=(seq,tile), cols=T·d_in (raw concat)."""
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T * d)


def tile_windows(x, T):
    """Non-overlapping T-tiles as windows → (n·n_tiles, T, d_in)."""
    n, Ls, d = x.shape
    n_tiles = Ls // T
    return x[:, :n_tiles * T].reshape(n * n_tiles, T, d)


def tile_labels(lab, T, seq_len):
    """Per-sequence label broadcast to each tile (velocity is constant/seq)."""
    n_tiles = seq_len // T
    return np.repeat(lab, n_tiles)


# ── probes ───────────────────────────────────────────────────────────────


def _subsample(rng, *arrays, cap=N_PROBE_ROWS):
    n = arrays[0].shape[0]
    if n <= cap:
        return arrays
    idx = rng.choice(n, size=cap, replace=False)
    return tuple(a[idx] for a in arrays)


def logistic_acc(X_tr, y_tr, X_ev, y_ev):
    """Held-out balanced accuracy of a multinomial-logistic probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=200).fit(X_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(X_ev)))


# ── the ML oracle (periodogram for circle; GLRT template for both) ───────


def periodogram_oracle(x_tiles, R, Y_true_class):
    """Circle ML: project to the 2-D plane, correlate against candidate tones.

    ``x_tiles: (N, T, d_in)`` → per-tile argmax over Ω of |DFT at f=Y/M|.
    Returns (overall_acc, per_class_acc (len |Ω|), pred_class).
    """
    N, T, _d = x_tiles.shape
    proj = x_tiles @ R                                       # (N, T, 2)
    c = proj[..., 0] + 1j * proj[..., 1]                     # (N, T) complex tone
    t = np.arange(T)
    vel = np.asarray(OMEGA, dtype=np.float64)
    basis = np.exp(-2j * np.pi * vel[:, None] * t[None, :] / M)   # (|Ω|, T)
    scores = np.abs(c @ basis.T)                             # (N, |Ω|)
    pred = scores.argmax(axis=1)
    return _acc_by_class(pred, Y_true_class)


def glrt_oracle(x_tiles, U, Y_true_class):
    """General ML/GLRT: argmax over (B,Y) templates of the correlation.

    Works for BOTH embeddings. Precompute symbol scores S_t[a]=⟨x_t,u_a⟩, then
    template(B,Y) score = Σ_t S_t[(B+Y·t) mod M]; argmax over B∈Z_M, Y∈Ω → Y.
    Returns (overall_acc, per_class_acc, pred_class).
    """
    N, T, _d = x_tiles.shape
    Sscore = np.einsum("ntd,md->ntm", x_tiles, U)           # (N, T, M) symbol scores
    t = np.arange(T)
    best_score = np.full(N, -np.inf)
    pred = np.zeros(N, dtype=np.int64)
    for yi, Yv in enumerate(OMEGA):
        idx = (np.arange(M)[:, None] + Yv * t[None, :]) % M      # (M_B, T) positions
        sc = np.zeros((N, M))                                    # score for each phase B
        for tt in range(T):
            sc += Sscore[:, tt, idx[:, tt]]                      # (N, M_B)
        best_B = sc.max(axis=1)                                  # marginalize phase (GLRT)
        upd = best_B > best_score
        best_score = np.where(upd, best_B, best_score)
        pred = np.where(upd, yi, pred)
    return _acc_by_class(pred, Y_true_class)


def _acc_by_class(pred, y_true):
    overall = float((pred == y_true).mean())
    per_class = []
    for c in range(len(OMEGA)):
        m = y_true == c
        per_class.append(float((pred[m] == y_true[m]).mean()) if m.any() else float("nan"))
    return overall, per_class, pred


# ── per-σ analysis block ─────────────────────────────────────────────────


def analyse(mode, d_in, sigma, rng):
    """Full ceiling block for one (mode, d_in, σ)."""
    x, lab, Y, U, R = simulate(mode, d_in, sigma, N_SEQS, SEQ_LEN, rng)
    half = N_SEQS // 2

    # per-token velocity probe (empirical; provable chance)
    Xt_tr = x[:half].reshape(-1, d_in)
    yt_tr = np.repeat(lab[:half], SEQ_LEN)
    Xt_ev = x[half:].reshape(-1, d_in)
    yt_ev = np.repeat(lab[half:], SEQ_LEN)
    Xt_tr, yt_tr = _subsample(rng, Xt_tr, yt_tr)
    Xt_ev, yt_ev = _subsample(rng, Xt_ev, yt_ev)
    per_token = logistic_acc(Xt_tr, yt_tr, Xt_ev, yt_ev)

    blk = {"per_token_acc": per_token, "by_T": {}}
    for T in T_GRID:
        # raw-linear window ceiling (multinomial-logistic on concatenated tile)
        Xw_tr = strided_tiles(x[:half], T)
        yw_tr = tile_labels(lab[:half], T, SEQ_LEN)
        Xw_ev = strided_tiles(x[half:], T)
        yw_ev = tile_labels(lab[half:], T, SEQ_LEN)
        Xw_tr, yw_tr = _subsample(rng, Xw_tr, yw_tr)
        Xw_ev, yw_ev = _subsample(rng, Xw_ev, yw_ev)
        raw_linear = logistic_acc(Xw_tr, yw_tr, Xw_ev, yw_ev)

        # ML oracle on the (held-out half) tiles
        xt = tile_windows(x[half:], T)
        yt = tile_labels(lab[half:], T, SEQ_LEN)
        glrt_overall, glrt_pc, _ = glrt_oracle(xt, U, yt)
        entry = {"raw_linear_acc": raw_linear,
                 "glrt_oracle_acc": glrt_overall,
                 "glrt_oracle_by_class": glrt_pc}
        if mode == "circle":
            po, ppc, _ = periodogram_oracle(xt, R, yt)
            entry["periodogram_oracle_acc"] = po
            entry["periodogram_oracle_by_class"] = ppc
        blk["by_T"][str(T)] = entry
    return blk


def resolvable_band_mean(per_class, T):
    """Mean oracle over the Ω-classes resolvable at tile T (f = Y/M ≳ 1/T)."""
    vals = [pc for Yv, pc in zip(OMEGA, per_class)
            if Yv > 0 and (Yv / M) >= (1.0 / T) and not np.isnan(pc)]
    return float(np.mean(vals)) if vals else float("nan")


def main():
    rng = np.random.default_rng(SEED)
    results = {"meta": {
        "M": M, "OMEGA": OMEGA, "d_in": D_IN, "seq_len": SEQ_LEN, "L": L,
        "T_grid": T_GRID, "sigma_default": SIGMA, "sigma_sweep": SIGMA_SWEEP,
        "chance": CHANCE, "n_seqs": N_SEQS,
        "memorization_threshold": len(OMEGA) * M,   # |Ω|·M distinct clean windows
        "codebook_circle": "M circle atoms (2-D object; d_sae anchors on M, not 2)",
        "codebook_random": "M orthonormal directions (F = M)",
        "gates": {"per_token_tol": GATE_PER_TOKEN_TOL,
                  "raw_linear_tol": GATE_RAW_LINEAR_TOL,
                  "oracle_resolvable": GATE_ORACLE_RESOLVABLE,
                  "separation": GATE_SEPARATION,
                  "null_flat_range": GATE_NULL_FLAT_RANGE},
    }}

    # (A) σ sweep on the circle mode — settle σ (oracle at best T on resolvable band)
    sweep = []
    for s in SIGMA_SWEEP:
        blk = analyse("circle", D_IN, s, np.random.default_rng(SEED + 1))
        Tmax = str(T_GRID[-1])
        pc = blk["by_T"][Tmax]["periodogram_oracle_by_class"]
        sweep.append({
            "sigma": s,
            "per_token_acc": blk["per_token_acc"],
            "oracle_overall_Tmax": blk["by_T"][Tmax]["periodogram_oracle_acc"],
            "oracle_resolvable_Tmax": resolvable_band_mean(pc, T_GRID[-1]),
        })
    results["sigma_sweep_circle"] = sweep

    # (B) headline blocks at the chosen σ
    results["circle"] = analyse("circle", D_IN, SIGMA, np.random.default_rng(SEED + 2))
    results["random"] = analyse("random", D_IN, SIGMA, np.random.default_rng(SEED + 3))

    # confirm periodogram ≈ GLRT for the circle mode (sanity: same ML oracle)
    circ = results["circle"]
    results["circle_oracle_agreement"] = {
        T: abs(circ["by_T"][T]["periodogram_oracle_acc"]
               - circ["by_T"][T]["glrt_oracle_acc"])
        for T in circ["by_T"]}

    # ── verdict ──
    Tmax = str(T_GRID[-1])
    pt_circle = circ["per_token_acc"]
    raw_lin_max = max(circ["by_T"][T]["raw_linear_acc"] for T in circ["by_T"])
    oracle_res = resolvable_band_mean(
        circ["by_T"][Tmax]["periodogram_oracle_by_class"], T_GRID[-1])
    # null flatness: spread of the random per-Ω-class oracle over nonzero Y at T=max
    rand_pc = [pc for Yv, pc in zip(
        OMEGA, results["random"]["by_T"][Tmax]["glrt_oracle_by_class"])
        if Yv > 0 and not np.isnan(pc)]
    null_range = float(max(rand_pc) - min(rand_pc)) if rand_pc else float("nan")

    v = {
        "per_token_at_chance": bool(abs(pt_circle - CHANCE) <= GATE_PER_TOKEN_TOL),
        "raw_linear_window_at_chance": bool(abs(raw_lin_max - CHANCE) <= GATE_RAW_LINEAR_TOL),
        "oracle_resolvable_high": bool(oracle_res >= GATE_ORACLE_RESOLVABLE),
        "separation": float(oracle_res - pt_circle),
        "separation_passes": bool(oracle_res - pt_circle >= GATE_SEPARATION),
        "null_flat_range": null_range,
        "null_is_flat": bool(null_range <= GATE_NULL_FLAT_RANGE),
        "per_token_circle": pt_circle,
        "raw_linear_window_max": raw_lin_max,
        "oracle_resolvable_Tmax": oracle_res,
    }
    v["passes_gate"] = bool(
        v["per_token_at_chance"] and v["raw_linear_window_at_chance"]
        and v["oracle_resolvable_high"] and v["separation_passes"]
        and v["null_is_flat"])
    results["verdict"] = v

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(results, indent=2))
    _print(results)
    _plot(results)
    return results


def _print(r):
    v = r["verdict"]
    print("\n========== FREQUENCY BENCH — § 8 GATING DUE-DILIGENCE ==========")
    print(f"M={M} (prime)  |Ω|={len(OMEGA)}  chance={CHANCE:.3f}  d_in={D_IN}  "
          f"seq_len={SEQ_LEN}  L={L}  T={T_GRID}")
    print(f"memorization threshold |Ω|·M = {len(OMEGA)*M} distinct clean windows")
    print("\n  σ sweep (circle) — settle noise (oracle at T=%d on resolvable band):" % T_GRID[-1])
    print(f"   {'σ':>6}{'per-tok':>9}{'oracle(all)':>13}{'oracle(res)':>13}")
    for s in r["sigma_sweep_circle"]:
        mark = "  <- chosen" if abs(s["sigma"] - SIGMA) < 1e-9 else ""
        print(f"   {s['sigma']:>6.2f}{s['per_token_acc']:>9.3f}"
              f"{s['oracle_overall_Tmax']:>13.3f}{s['oracle_resolvable_Tmax']:>13.3f}{mark}")

    for mode in ("circle", "random"):
        print(f"\n  {mode.upper()} mode  (σ={SIGMA})   per-token velocity acc = "
              f"{r[mode]['per_token_acc']:.3f}  (chance {CHANCE:.3f})")
        print(f"   {'T':>3}{'raw-lin':>9}{'GLRT orc':>10}{'periodo':>9}   per-class oracle (Y=%s)"
              % ",".join(str(y) for y in OMEGA))
        for T in map(str, T_GRID):
            e = r[mode]["by_T"][T]
            po = e.get("periodogram_oracle_acc", float("nan"))
            pc = e.get("periodogram_oracle_by_class", e["glrt_oracle_by_class"])
            pcs = " ".join(f"{p:.2f}" for p in pc)
            print(f"   {T:>3}{e['raw_linear_acc']:>9.3f}{e['glrt_oracle_acc']:>10.3f}"
                  f"{po:>9.3f}   {pcs}")

    print("\n  VERDICT")
    print(f"    per-token velocity at chance        {v['per_token_at_chance']}  ({v['per_token_circle']:.3f})")
    print(f"    raw-linear WINDOW at chance         {v['raw_linear_window_at_chance']}  ({v['raw_linear_window_max']:.3f})")
    print(f"    circle oracle high (resolvable)     {v['oracle_resolvable_high']}  ({v['oracle_resolvable_Tmax']:.3f})")
    print(f"    separation (oracle - per-token)     {v['separation']:.3f}  "
          f"{'>= %.2f PASS' % GATE_SEPARATION if v['separation_passes'] else 'FAIL'}")
    print(f"    random null FLAT (Ω-class spread)   {v['null_is_flat']}  (range {v['null_flat_range']:.3f})")
    print(f"    ==> passes_gate = {v['passes_gate']}")
    print(f"\n  -> {OUT_JSON}")


def _plot(r):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    freqs = [y / M for y in OMEGA]
    colors = {2: "#c7c7c7", 4: "#9edae5", 8: "#1f77b4", 16: "#d62728"}

    # left: circle S(f) oracle per T + per-token + raw-linear
    for T in T_GRID:
        pc = r["circle"]["by_T"][str(T)].get(
            "periodogram_oracle_by_class",
            r["circle"]["by_T"][str(T)]["glrt_oracle_by_class"])
        ax[0].plot(freqs, pc, "o-", color=colors[T], label=f"circle oracle, T={T}")
    ax[0].axhline(r["circle"]["per_token_acc"], color="k", ls="--", lw=1,
                  label="per-token (probe)")
    ax[0].axhline(CHANCE, color="grey", ls=":", lw=1, label=f"chance {CHANCE:.2f}")
    for T in T_GRID:
        ax[0].axvline(1.0 / T, color=colors[T], ls=":", lw=0.8, alpha=0.6)
    ax[0].set_xlabel("temporal frequency  f = Y/M  (cycles/token)")
    ax[0].set_ylabel("oracle velocity accuracy")
    ax[0].set_title("Circle S(f): periodogram oracle per tile T\n(dotted = Rayleigh 1/T)")
    ax[0].legend(fontsize=7); ax[0].grid(True, alpha=0.25); ax[0].set_ylim(-0.03, 1.03)

    # right: random null — flat oracle vs frequency (no ordering)
    for T in T_GRID:
        pc = r["random"]["by_T"][str(T)]["glrt_oracle_by_class"]
        ax[1].plot(freqs, pc, "s-", color=colors[T], label=f"random oracle, T={T}")
    ax[1].axhline(r["random"]["per_token_acc"], color="k", ls="--", lw=1,
                  label="per-token (probe)")
    ax[1].axhline(CHANCE, color="grey", ls=":", lw=1, label=f"chance {CHANCE:.2f}")
    ax[1].set_xlabel("temporal frequency  f = Y/M  (cycles/token)")
    ax[1].set_ylabel("oracle velocity accuracy")
    ax[1].set_title("Random null: FLAT response (no frequency axis)\nratio-invariance theorem")
    ax[1].legend(fontsize=7); ax[1].grid(True, alpha=0.25); ax[1].set_ylim(-0.03, 1.03)

    fig.suptitle("Frequency bench § 8 gating: circle S(f) high-pass vs random flat null; "
                 "per-token & raw-linear window at chance", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    for ext, dpi in [("pdf", None), ("png", 120), ("thumb.png", 55)]:
        fig.savefig(FIG_DIR / f"frequency_gating.{ext}", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {FIG_DIR}/frequency_gating.*")


if __name__ == "__main__":
    main()
