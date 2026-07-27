"""λ̂ shuffle overlay — post-hoc on the retrained checkpoints
(SHUFFLE_OVERLAY_CARD.md § 4; instrument = probing 1.2.0's shuffle
control, Aniket's cross-task convention).

For every card § 2 cell: recompute the ORDERED recovery from the
persisted checkpoint via the v1 pipeline (`lambda_recovery`'s own
`_sample_windows` seeds 0/1 + tiling, LinearRegression) and assert it
equals the canonical-runner metric to |Δ| ≤ 1e-6 (identity receipt);
only then score the SAME fixed probe on eval tiles whose T positions
are per-row permuted pre-encode (`shuffle_within_window`, seed 0).
The probe is never refit. T=1 anchors are identity by construction.
Also prints the card § 3 anchor-gate table (frozen constants below).

Run:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.shuffle_overlay
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from explorations.synthetic.grid import batch_size
from explorations.task_hunt.real_lambda import ward_lambda_real
from temp_bench.core.config import (
    compute_data_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.runner import _load_checkpoint
from temp_bench.core.schemas import TrainingConfig
from temp_bench.evals.synthetic_recovery import _sample_windows
from temp_bench.utils.shuffles import shuffle_within_window

from experiments.explorations.task_hunt.lambda_intensity.run_shuffle_overlay_retrain import (
    ARMS,
    RETRAIN_TAG,
)
from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    BUFFER_TOKENS,
    D_SAE,
    DS_DEFAULT,
    EVAL_L,
    K_POS,
    N_STEPS,
    WINDOW_TS,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SHUF_EVAL_SEED = 0            # card § 4, disclosed
IDENTITY_TOL = 5e-4           # card § 4 amendment A1: cross-process GPU
# kernel nondeterminism (no TF32/determinism pins anywhere in the
# framework) drifts encode outputs ~1e-7 relative, amplified through
# the p=2048 OLS probe to ~1e-4 on r (observed 1.28e-4, first tt
# cell). 5e-4 remains 6-60x below every anchor-gate sigma, so the
# receipt still catches any real protocol divergence (wrong seed/
# window/probe moves r by >>1e-3). Amended BEFORE any shuffled
# column was read; original 1e-6 assumed same-process determinism.
SEEDS = (1, 2, 42)

# Card § 3 anchor-gate constants (stage2_summary.json trained block,
# frozen in the card BEFORE any retrain cell ran).
GATE = {
    ("txc_batchtopk_post", 2): (0.1296, 0.0171),
    ("txc_batchtopk_post", 4): (0.1607, 0.0160),
    ("txc_batchtopk_post", 8): (0.1848, 0.0244),
    ("txc_batchtopk_post", 16): (0.2548, 0.0473),
    ("batchtopk_sae", 1): (0.1130, 0.0218),
    ("tsae", 1): (0.1541, 0.0367),
}


def _fit_ordered_and_shuffled(model, x, lam, *, L, T, n_windows=1024, seed=0):
    """v1's `_train_lambda_probe` flow verbatim, plus the shuffled column."""
    from sklearn.linear_model import LinearRegression

    device = next(model.parameters()).device
    model.eval()
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)

    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)

    def tiles_and_targets(win_x, win_l):
        W, L_, d_in = win_x.shape
        n_tiles = L_ // T
        tiles = win_x.to(device, dtype=torch.float32).reshape(W * n_tiles, T, d_in)
        tgt = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1)
        return tiles, tgt.detach().float().cpu().numpy()

    with torch.no_grad():
        tiles_tr, t_tr = tiles_and_targets(win_x_tr, win_l_tr)
        z_tr = model.encode(tiles_tr).reshape(tiles_tr.shape[0], -1).float().cpu().numpy()
        tiles_ev, t_ev = tiles_and_targets(win_x_ev, win_l_ev)
        z_ev = model.encode(tiles_ev).reshape(tiles_ev.shape[0], -1).float().cpu().numpy()
        if T > 1:
            tiles_sh = shuffle_within_window(tiles_ev, T=T, seed=SHUF_EVAL_SEED)
            z_sh = model.encode(tiles_sh).reshape(tiles_sh.shape[0], -1).float().cpu().numpy()
        else:
            z_sh = z_ev                      # length-1 shuffle is the identity

    tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
    if not tr_m.all():
        z_tr, t_tr = z_tr[tr_m], t_tr[tr_m]
    if not ev_m.all():
        z_ev, z_sh, t_ev = z_ev[ev_m], z_sh[ev_m], t_ev[ev_m]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
        pred_sh = reg.predict(z_sh)          # SAME fixed probe, never refit
    r_ord = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    r_sh = float(np.corrcoef(pred_sh, t_ev)[0, 1]) if np.std(pred_sh) > 1e-12 else 0.0
    return r_ord, r_sh


def main():
    retrain = json.loads((RES / "shuffle_overlay_retrain.json").read_text())
    rows = retrain if isinstance(retrain, list) else retrain["rows"]
    by_cell = {(r["arch"], r["T"], r["seed"]): r for r in rows if r.get("ok")}

    ds_spec = load_datasource(DS_DEFAULT)
    data = ward_lambda_real(**ds_spec.params)
    data_key = compute_data_key(ds_spec)
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    lam = lam.float()

    out, means = [], {}
    for arch_name, fam in ARMS:
        ts = (1,) if fam == "token" else WINDOW_TS
        for T in ts:
            per_seed = []
            for seed in SEEDS:
                row = by_cell.get((arch_name, T, seed))
                assert row is not None, f"missing retrain cell {arch_name}/T{T}/s{seed}"
                override = {"k_pos": K_POS[0], "d_sae": D_SAE, "T": T}
                tcfg = TrainingConfig(
                    n_steps=N_STEPS, batch_size=batch_size(T),
                    buffer_tokens=BUFFER_TOKENS,
                    arch_hparams_override=override)
                spec = load_arch(arch_name, section="synthetic")
                spec = spec.model_copy(
                    update={"hparams": {**spec.hparams, **override}})
                tk = compute_train_key(arch=spec, seed=seed, training_cfg=tcfg,
                                       data_key=data_key, section="synthetic")
                model = _load_checkpoint(spec, tk, ds_spec).cuda()
                r_ord, r_sh = _fit_ordered_and_shuffled(
                    model, data.x, lam, L=EVAL_L, T=T)
                canon = row["metrics"]["lambda_recovery"]
                ident = abs(r_ord - canon)
                assert ident <= IDENTITY_TOL, (
                    f"identity receipt FAILED {arch_name}/T{T}/s{seed}: "
                    f"recomputed {r_ord:.8f} vs canonical {canon:.8f}")
                cell = {"arch": arch_name, "T": T, "seed": seed,
                        "canonical_r": canon, "recomputed_r": r_ord,
                        "identity_abs": ident,
                        "shuffle_identity": int(T == 1),
                        "r_shuf": r_sh, "gap": r_ord - r_sh,
                        "shuf_seed": SHUF_EVAL_SEED, "tag": RETRAIN_TAG}
                out.append(cell)
                per_seed.append((r_ord, r_sh))
                del model
                torch.cuda.empty_cache()
                print(f"[overlay] {arch_name}/T{T}/s{seed} "
                      f"ord={r_ord:.4f} shuf={r_sh:.4f} gap={r_ord - r_sh:+.4f}",
                      flush=True)
            o = np.array([p[0] for p in per_seed])
            s = np.array([p[1] for p in per_seed])
            means[f"{arch_name}/T{T}"] = {
                "ordered_mean": float(o.mean()), "ordered_sd": float(o.std(ddof=1)),
                "shuf_mean": float(s.mean()), "shuf_sd": float(s.std(ddof=1)),
                "gap_mean": float((o - s).mean()), "n": len(per_seed)}

    gate = {}
    print("\n[anchor gate] card § 3 — per-cell |mean_retrained - mean_quoted| <= 1σ_quoted")
    for (arch_name, T), (q_mean, q_sd) in GATE.items():
        m = means[f"{arch_name}/T{T}"]["ordered_mean"]
        d = abs(m - q_mean)
        ok = d <= q_sd
        gate[f"{arch_name}/T{T}"] = {
            "quoted_mean": q_mean, "sigma_tol": q_sd,
            "retrained_mean": m, "abs_delta": d, "pass": bool(ok)}
        print(f"  {arch_name}/T{T}: retrained {m:.4f} vs quoted {q_mean:.4f} "
              f"|Δ|={d:.4f} tol={q_sd:.4f} -> {'PASS' if ok else 'FAIL'}")
    all_pass = all(g["pass"] for g in gate.values())
    print(f"[anchor gate] {'ALL PASS — overlay licensed' if all_pass else 'FAIL — STOP per card § 3 (finding, not license)'}")

    payload = {"card": "SHUFFLE_OVERLAY_CARD.md", "status": "PENDING TEAM REVIEW",
               "shuf_eval_seed": SHUF_EVAL_SEED, "identity_tol": IDENTITY_TOL,
               "anchor_gate": gate, "anchor_gate_all_pass": all_pass,
               "summary": means, "cells": out}
    (RES / "shuffle_overlay.json").write_text(json.dumps(payload, indent=1))
    print(f"[overlay] -> {RES / 'shuffle_overlay.json'}")


if __name__ == "__main__":
    main()
