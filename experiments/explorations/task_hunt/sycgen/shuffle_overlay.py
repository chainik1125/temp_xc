"""sycgen shuffle overlay — post-hoc on the retrained checkpoints
(RETRAIN_CARD.md § 3; instrument = probing 1.2.0's shuffle control,
Aniket's cross-task convention, λ̂ overlay transplant).

For every card § 2 trained cell: recompute the ORDERED recovery from
the persisted checkpoint via the v1 pipeline and assert it equals the
canonical-runner metric to |Δ| ≤ 2e-3 (identity receipt — tolerance
inherited from the λ̂ card's A2 with its conditioning analysis); only
then score the SAME fixed probe on eval tiles whose T positions are
per-row permuted pre-encode (`shuffle_within_window`, seed 0). The
probe is never refit. T=1 anchors are identity by construction.

No anchor gate exists here BY DESIGN: there is no quoted panel — this
is the substrate's first training and the T-sweep itself is the
exhibit. The identity receipt certifies the code path; the untrained
twins and shuffle columns are the instruments.

Run:  .venv/bin/python -m experiments.explorations.task_hunt.sycgen.shuffle_overlay
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

from explorations.synthetic.grid import batch_size
from explorations.task_hunt.real_sycgen import sycgen_age_real
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

from experiments.explorations.task_hunt.sycgen.run_retrain import (
    ARMS,
    BUFFER_TOKENS,
    D_SAE,
    DS,
    EVAL_L,
    K_POS,
    N_STEPS,
    RETRAIN_TAG,
    WINDOW_TS,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SHUF_EVAL_SEED = 0
IDENTITY_TOL = 2e-3
SEEDS = (1, 2, 42)


def _fit_ordered_and_shuffled(model, x, lam, *, L, T, n_windows=1024,
                              seed=0):
    from sklearn.linear_model import LinearRegression

    device = next(model.parameters()).device
    model.eval()
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)

    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows,
                                  seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows,
                                  seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows,
                                  seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows,
                                  seed=seed + 1)

    def tiles_and_targets(win_x, win_l):
        W, L_, d_in = win_x.shape
        n_tiles = L_ // T
        tiles = win_x.to(device, dtype=torch.float32).reshape(
            W * n_tiles, T, d_in)
        tgt = win_l.reshape(W, n_tiles, T)[:, :, T - 1].reshape(-1)
        return tiles, tgt.detach().float().cpu().numpy()

    with torch.no_grad():
        tiles_tr, t_tr = tiles_and_targets(win_x_tr, win_l_tr)
        z_tr = model.encode(tiles_tr).reshape(
            tiles_tr.shape[0], -1).float().cpu().numpy()
        tiles_ev, t_ev = tiles_and_targets(win_x_ev, win_l_ev)
        z_ev = model.encode(tiles_ev).reshape(
            tiles_ev.shape[0], -1).float().cpu().numpy()
        if T > 1:
            tiles_sh = shuffle_within_window(tiles_ev, T=T,
                                             seed=SHUF_EVAL_SEED)
            z_sh = model.encode(tiles_sh).reshape(
                tiles_sh.shape[0], -1).float().cpu().numpy()
        else:
            z_sh = z_ev

    tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
    if not tr_m.all():
        z_tr, t_tr = z_tr[tr_m], t_tr[tr_m]
    if not ev_m.all():
        z_ev, z_sh, t_ev = z_ev[ev_m], z_sh[ev_m], t_ev[ev_m]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reg = LinearRegression().fit(z_tr, t_tr)
        pred = reg.predict(z_ev)
        pred_sh = reg.predict(z_sh)
    r_ord = (float(np.corrcoef(pred, t_ev)[0, 1])
             if np.std(pred) > 1e-12 else 0.0)
    r_sh = (float(np.corrcoef(pred_sh, t_ev)[0, 1])
            if np.std(pred_sh) > 1e-12 else 0.0)
    return r_ord, r_sh


def main():
    rows = []
    for shard in (0, 1):
        p = RES / f"retrain_shard{shard}.json"
        blob = json.loads(p.read_text())
        rows += blob if isinstance(blob, list) else blob["rows"]
    by_cell = {(r["arch"], r["T"], r["seed"]): r for r in rows
               if r.get("ok") and r.get("n_steps", N_STEPS) > 0}

    ds_spec = load_datasource(DS)
    data = sycgen_age_real(**ds_spec.params)
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
                assert row is not None, \
                    f"missing retrain cell {arch_name}/T{T}/s{seed}"
                override = {"k_pos": K_POS[0], "d_sae": D_SAE, "T": T}
                tcfg = TrainingConfig(
                    n_steps=N_STEPS, batch_size=batch_size(T),
                    buffer_tokens=BUFFER_TOKENS,
                    arch_hparams_override=override)
                spec = load_arch(arch_name, section="synthetic")
                spec = spec.model_copy(
                    update={"hparams": {**spec.hparams, **override}})
                tk = compute_train_key(arch=spec, seed=seed,
                                      training_cfg=tcfg,
                                      data_key=data_key,
                                      section="synthetic")
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
                      f"ord={r_ord:.4f} shuf={r_sh:.4f} "
                      f"gap={r_ord - r_sh:+.4f}", flush=True)
            o = np.array([p[0] for p in per_seed])
            s = np.array([p[1] for p in per_seed])
            means[f"{arch_name}/T{T}"] = {
                "ordered_mean": float(o.mean()),
                "ordered_sd": float(o.std(ddof=1)),
                "shuf_mean": float(s.mean()),
                "shuf_sd": float(s.std(ddof=1)),
                "gap_mean": float((o - s).mean()), "n": len(per_seed)}

    payload = {"card": "sycgen/RETRAIN_CARD.md",
               "status": "PENDING TEAM REVIEW",
               "shuf_eval_seed": SHUF_EVAL_SEED,
               "identity_tol": IDENTITY_TOL,
               "anchor_gate": "none by design (first training; the "
                              "T-sweep is the exhibit)",
               "summary": means, "cells": out}
    (RES / "sycgen_shuffle_overlay.json").write_text(
        json.dumps(payload, indent=1))
    print(f"[overlay] -> {RES / 'sycgen_shuffle_overlay.json'}")


if __name__ == "__main__":
    main()
