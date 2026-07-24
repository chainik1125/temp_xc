"""Stage 1 — constructed-code calibration with EXACTLY known truth.

`CARD_PROBE_TRUTH.md` § 2.1. **Off the leaderboard** (the `probe_capacity.py`
precedent: nothing here is appended to `results/leaderboard.jsonl`, no
checkpoint is written, no eval protocol is touched).

The question the campaign must answer — *does a probe's reported recovery
track the truth?* — needs the truth. Here it is available by construction:
the trained encoder is replaced by an **analytic** one whose λ-information
content is set, not learned. Everything below the encoder is the committed
code path: `_sample_windows` at v1's seeds (train pool 0 / eval pool 1),
v1's `n // 2` sequence split, `_tile_lambda_examples`' tiling and
leading-edge target. The committed probes themselves
(`_train_lambda_probe`, `_train_lambda_probe_v2`) are *called*, not
re-implemented, for the two headline numbers.

**The construction.** A tile is `(T, d_in)`; the target is λ at its leading
edge (position `T-1`), which the generator sets as
`σ(a + w₁·b_{i-1} + w₂·b_{i-2})` — so the tile positions `T-2` and `T-3`
carry the entire driver. The dictionary is orthonormal and the backtracking
emission is `b·|N(2.5, .75)|·u_bt`, so `x·u_bt` is **exactly** 0 wherever
`b = 0`: thresholding the projection recovers `b` exactly (checked per run
and recorded as `b_exact`).

    signal dims  binary `b` at chosen tile positions — the arm (below).
    noise  dims  fixed random Gaussian map of the tile's CONTENT subspace
                 (`u_bt` projected out) → ReLU → top-k per row. Content
                 directions are drawn independently of `b`, so these columns
                 carry **zero** λ information by construction, while being
                 sparse, non-negative and mutually correlated like a real
                 SAE code.

Because the noise columns are population-independent of λ, the
population-optimal linear predictor over all `p` columns is the one that
uses the signal columns alone. So **truth = the held-out correlation of an
OLS fit on the signal columns**, evaluated on the *same* eval rows the
probe is scored on — an exactly paired comparison whose only error is the
O(S/n) estimation error of a ≤ 2-parameter fit.

Arms (card § 2.1), each with an independently documented truth:

    full   b at tile positions T-2, T-3  → window ceiling ≈ 0.91 (T=2, only
                                           one lag fits) / ≈ 0.99 (T ≥ 4)
    token  b at tile position T-1        → per-token DPI floor ≈ 0.41
    null   none                          → 0

Reproducing those three constants is validity gate **G1**: they come from
the bench's own analysis, independently of anything measured here.

Density note (disclosed addition, not a deviation): the card freezes the
noise readout at top-8 per row. Real dense panel cells sit near
`l0_per_window / d_sae ≈ 6%`, which at p = 4096 is 250× denser. The frozen
top-8 construction is run over the whole ladder as the primary; a 6%
variant is run at the top of the ladder as an ADDITIONAL arm, so the
frozen design is executed as written and the real-panel density is covered
too.

Run (leave cores for the training grid):
    OMP_NUM_THREADS=8 .venv/bin/python -m \
      experiments.explorations.task_hunt.support_synthetic.probe_truth_calib
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from temp_bench.core.config import load_datasource
from temp_bench.data.synthetic import materialise
from temp_bench.evals.lambda_recovery import (
    _tile_lambda_examples,
    _train_lambda_probe,
)
from temp_bench.evals.lambda_recovery_v2 import (
    DEFAULT_ALPHAS,
    _train_lambda_probe_v2,
)
from temp_bench.evals.synthetic_recovery import _sample_windows

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
DS = "toy_backtracking_selfexcite_d64"
L = 32
SEEDS = (1, 2, 42)
BT_EPS = 1e-4                       # x·u_bt is exactly 0 when b = 0

# The ladder (card § 2.1). n_rows = n_windows·(L/T); at T = 16 the four
# n_windows values give n = 2048 / 4096 / 8192 / 16384, so n_windows = 1024
# IS v1's committed setting and 8192 IS v2's.
P_LADDER = (8, 32, 128, 512, 2048, 4096)
P_STACKED_CORNER = 8192             # p/n = 4 at n = 2048, the Stacked regime
N_WINDOWS = (1024, 2048, 4096, 8192)
PROBES = ("ols", "ridge")
T_MAIN = 16
T_GATE = (2, 4, 8, 16)              # G1 sweep (at p = 32, deep in the safe regime)
P_GATE = 32
DENSITY_PS = (512, 2048, 4096)      # the added 6%-density arm


def _anchor_n_windows(p: int, T: int) -> tuple[int, int]:
    """Truth-anchor budget (card § 3): n_rows ≥ 32·p, floor 16384, cap 65536."""
    target = min(max(32 * p, 16384), 65536)
    per_window = L // T
    nw = int(np.ceil(target / per_window))
    return nw, nw * per_window


class ConstructedArch(torch.nn.Module):
    """Analytic encoder with a known λ-information content (module docstring)."""

    def __init__(self, *, T: int, d_in: int, u_bt: torch.Tensor,
                 sig_pos: tuple[int, ...], p_noise: int, k_noise: int,
                 seed: int):
        super().__init__()
        self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.config = SimpleNamespace(T=T)
        self.T, self.d_in = T, d_in
        self.register_buffer("u_bt", u_bt.detach().clone())
        self.sig_pos = tuple(sig_pos)
        self.p_noise, self.k_noise = int(p_noise), int(k_noise)
        g = torch.Generator().manual_seed(int(seed) * 1_000_003 + T * 101 + p_noise)
        # Content rows have ‖·‖ ≈ √(n_c)·mag_content per position → √(3T) over
        # the tile; scaling by 1/√(3T) puts the pre-ReLU units at std ≈ 1, the
        # same order as the binary signal dims (ridge penalises a raw scale).
        w = torch.randn(T * d_in, max(p_noise, 1), generator=g) / np.sqrt(3.0 * T)
        self.register_buffer("W", w)

    @property
    def p(self) -> int:
        return len(self.sig_pos) + self.p_noise

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, d_in) → (B, p): signal dims ⊕ sparse content-only noise dims."""
        if x.dim() == 2:
            x = x.unsqueeze(1)
        proj = x @ self.u_bt                                    # (B, T) = b·mbt
        b = (proj > BT_EPS).to(x.dtype)                         # exact b
        parts = []
        if self.sig_pos:
            parts.append(b[:, list(self.sig_pos)])
        if self.p_noise:
            content = x - proj.unsqueeze(-1) * self.u_bt        # ⊥ u_bt
            h = torch.relu(content.reshape(x.shape[0], -1) @ self.W)
            if 0 < self.k_noise < self.p_noise:
                kth = h.topk(self.k_noise, dim=1).values[:, -1:]
                h = h * (h >= kth)
            parts.append(h)
        return torch.cat(parts, dim=1) if parts else b[:, :0]


def _rows(model, x, lam, *, n_windows: int, seed: int = 0):
    """The committed sampler/tiler on v1's split and seeds → train/eval rows."""
    T = model.config.T
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    wx_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    wl_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    wx_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    wl_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)
    z_tr, t_tr = _tile_lambda_examples(model, wx_tr, wl_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, wx_ev, wl_ev, T)
    return z_tr, t_tr, z_ev, t_ev


def _fit(z_tr, t_tr, z_ev, t_ev, probe: str):
    """One probe fit → held-out Pearson r (the headline metric) + receipts."""
    from sklearn.linear_model import LinearRegression, RidgeCV
    if z_tr.shape[1] == 0:                       # null arm's signal-only fit
        return {"r": 0.0, "r2": 0.0, "alpha": 0.0,
                "n_rows": int(z_tr.shape[0]), "p": 0}
    reg = (LinearRegression() if probe == "ols"
           else RidgeCV(alphas=np.asarray(DEFAULT_ALPHAS, dtype=float)))
    reg.fit(z_tr, t_tr)
    pred = reg.predict(z_ev)
    r = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"r": r, "r2": float(reg.score(z_ev, t_ev)),
            "alpha": float(getattr(reg, "alpha_", 0.0)),
            "n_rows": int(z_tr.shape[0]), "p": int(z_tr.shape[1])}


def _arms(T: int) -> dict[str, tuple[int, ...]]:
    """Signal tile-positions per arm; λ's drivers are at T-2 and T-3."""
    return {"full": tuple(q for q in (T - 2, T - 3) if q >= 0),
            "token": (T - 1,),
            "null": ()}


def _cell(arm, T, p, dens, seed, x, lam, u_bt, d_in, *, do_anchor: bool):
    """One constructed-code cell: v1, v2, the (n_windows × probe) grid, truth."""
    sig = _arms(T)[arm]
    p_noise = max(p - len(sig), 0)
    k_noise = 8 if dens == "k8" else max(1, int(round(0.06 * p_noise)))
    model = ConstructedArch(T=T, d_in=d_in, u_bt=u_bt, sig_pos=sig,
                            p_noise=p_noise, k_noise=k_noise, seed=seed)
    model.eval()
    out = {"arm": arm, "T": T, "p_nominal": p, "p": model.p, "n_sig": len(sig),
           "density": dens, "k_noise": k_noise, "seed": seed, "grid": []}

    # Headline numbers straight from the committed probes.
    v1 = _train_lambda_probe(model, x, lam, L=L)                 # OLS, nw 1024
    v2 = _train_lambda_probe_v2(model, x, lam, L=L, n_windows=8192,
                                probe="ridge", alphas=DEFAULT_ALPHAS,
                                split_mode="half", trace_ids=None)
    out["v1"] = float(v1["lambda_recovery"])
    out["v1_chance"] = float(v1["lambda_chance"])
    out["v2"] = float(v2["lambda_recovery_v2"])
    out["v2_chance"] = float(v2["lambda_chance_v2"])
    out["v2_alpha"] = float(v2["lambda_alpha_v2"])

    for nw in N_WINDOWS:
        z_tr, t_tr, z_ev, t_ev = _rows(model, x, lam, n_windows=nw)
        # truth = OLS on the signal columns alone, on the SAME eval rows.
        truth = _fit(z_tr[:, :len(sig)], t_tr, z_ev[:, :len(sig)], t_ev, "ols")
        row = {"n_windows": nw, "n_rows": int(z_tr.shape[0]),
               "p_over_n": float(z_tr.shape[1]) / max(z_tr.shape[0], 1),
               "truth": float(truth["r"])}
        for probe in PROBES:
            f = _fit(z_tr, t_tr, z_ev, t_ev, probe)
            row[probe] = f["r"]
            row[f"{probe}_r2"] = f["r2"]
            row[f"{probe}_alpha"] = f["alpha"]
        out["grid"].append(row)
        if nw == 1024:
            # Licence (card § 3c): the local path must reproduce the committed
            # v1 probe exactly, else nothing else here may be read.
            out["v1_replication_delta"] = abs(row["ols"] - out["v1"])
        del z_tr, z_ev

    if do_anchor:
        nw_a, n_a = _anchor_n_windows(model.p, T)
        z_tr, t_tr, z_ev, t_ev = _rows(model, x, lam, n_windows=nw_a)
        a = {"n_windows": nw_a, "n_rows": int(z_tr.shape[0]),
             "n_over_p": float(z_tr.shape[0]) / max(model.p, 1),
             "truth": float(_fit(z_tr[:, :len(sig)], t_tr,
                                 z_ev[:, :len(sig)], t_ev, "ols")["r"])}
        for probe in PROBES:
            a[probe] = _fit(z_tr, t_tr, z_ev, t_ev, probe)["r"]
        a["ols_ridge_gap"] = abs(a["ols"] - a["ridge"])
        a["anchor"] = 0.5 * (a["ols"] + a["ridge"])
        a["licensed"] = bool(a["n_over_p"] >= 16.0 and a["ols_ridge_gap"] <= 0.02)
        out["anchor"] = a
        del z_tr, z_ev
    return out


def main():
    RES.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    for seed in SEEDS:
        data = materialise(load_datasource(DS), seed=seed)
        x = data.x
        lam = torch.as_tensor(data.extra["lambda_labels"]).float()
        u_bt = torch.as_tensor(np.asarray(data.emission_features)[0]).float()
        d_in = int(x.shape[-1])
        b_true = torch.as_tensor(data.extra["b_labels"]).float()
        b_hat = ((x.float() @ u_bt) > BT_EPS).float()
        b_exact = bool(torch.equal(b_hat, b_true))
        print(f"[seed {seed}] b extraction exact: {b_exact}", flush=True)

        jobs = []                                    # (arm, T, p, dens, anchor)
        for arm in ("full", "token", "null"):
            for p in P_LADDER:
                jobs.append((arm, T_MAIN, p, "k8", True))
            jobs.append((arm, T_MAIN, P_STACKED_CORNER, "k8", False))
            for p in DENSITY_PS:
                jobs.append((arm, T_MAIN, p, "p6", True))
            for T in T_GATE:                          # G1: the arm truths vs T
                if T != T_MAIN:
                    jobs.append((arm, T, P_GATE, "k8", False))

        for arm, T, p, dens, anch in jobs:
            r = _cell(arm, T, p, dens, seed, x, lam, u_bt, d_in, do_anchor=anch)
            r["b_exact"] = b_exact
            rows.append(r)
            g16 = [q for q in r["grid"] if q["n_windows"] == 8192][0]
            print(f"[{time.time()-t0:6.0f}s] {arm:<5} T{T:<3} p={r['p']:<5} "
                  f"{dens} truth={g16['truth']:+.3f} v1={r['v1']:+.3f} "
                  f"v2={r['v2']:+.3f} p/n={g16['p_over_n']:.3f} "
                  f"rep={r.get('v1_replication_delta', float('nan')):.2e}",
                  flush=True)
            (RES / "probe_truth_calib.json").write_text(json.dumps(rows, indent=1))
    print(f"-> {RES/'probe_truth_calib.json'} ({len(rows)} cells, "
          f"{time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
