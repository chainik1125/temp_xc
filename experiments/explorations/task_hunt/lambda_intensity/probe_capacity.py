"""Stage-2 diagnostic — is the λ probe capacity-limited at large T?

Pre-registered in `card_stage2_postmatched.md` § 4 reading (c), BEFORE
the matched cells ran, so its sign could not be chosen after the fact.

`lambda_recovery` fits an **unregularized** `LinearRegression` on the
single-tile code (`p = d_sae = 2048` features) using `n = 1024·(32/T)`
rows: 32768 / 16384 / 8192 / 4096 / **2048** at T = 1 / 2 / 4 / 8 / 16.
At T = 16, `n = p` exactly — OLS interpolates. A code with ~8 nonzeros
per row survives that; a code with ~100 nonzeros per row need not. So a
T = 16 drop can be *probe* capacity rather than representation, and the
round-1 T = 16 drops of TXC-pre (0.206 → 0.138) and Stacked
(0.125 → 0.094) — both realizing `l0_per_window ≈ 125` — are already
under that suspicion.

This re-fits the probe on the SAME trained checkpoints with more probe
data and with ridge. It **cannot** change any leaderboard cell: nothing
here is appended to `results/leaderboard.jsonl`, and the eval protocol
is untouched. It can only change what the record is allowed to *claim*
about the T = 16 column.

Faithfulness: the `n_windows = 1024`, `est = ols` row re-implements the
eval exactly (same `_sample_windows` seeds, same `_tile_lambda_examples`
tiling, same split) and MUST reproduce the leaderboard's
`lambda_recovery` to ~1e-6. That replication check is printed first and
is the licence for reading any other row. The private helpers are
imported rather than re-typed precisely so the replication is exact.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.lambda_intensity.probe_capacity [ds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from temp_bench.core.config import (
    checkpoint_dir,
    compute_data_key,
    compute_train_key,
    import_by_path,
    load_arch,
    load_datasource,
)
from temp_bench.core.schemas import TrainingConfig
from temp_bench.data.synthetic import materialise
from temp_bench.evals.lambda_recovery import _tile_lambda_examples
from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

DS_DEFAULT = "ward_real_lambda_base_l12"
D_SAE = 2048
EVAL_L = 32
N_STEPS = 8_000
BUFFER_TOKENS = 524_288
SEEDS = (1, 2, 42)

# (arch, T, nominal k_pos) — the matched post ladder, the round-1 post
# ladder it replaces, and the two dense window archs whose T=16 drops are
# under the same suspicion. Token archs are the n≫p reference.
CELLS = (
    [("txc_batchtopk_post", T, 8 * T) for T in (2, 4, 8, 16)]        # matched
    + [("txc_batchtopk_post", T, 8) for T in (8, 16)]                # round 1
    + [("txc_batchtopk_pre", T, 8) for T in (8, 16)]
    + [("stacked_batchtopk", T, 8) for T in (8, 16)]
    + [("batchtopk_sae", 1, 8), ("tsae", 1, 8)]
)
N_WINDOWS = (1024, 8192)
ESTIMATORS = ("ols", "ridge")


def _load_model(arch: str, T: int, k_pos: int, seed: int, ds: str):
    """Re-instantiate the trained checkpoint for one Stage-2 cell.

    Rebuilds the exact `train_key` `grid.run_cell` produced, so this
    reads the same weights the leaderboard row was scored on.
    """
    arch_spec = load_arch(arch, section="synthetic")
    data_spec = load_datasource(ds)
    override = {"k_pos": k_pos, "d_sae": D_SAE, "T": T}
    arch_spec = arch_spec.model_copy(
        update={"hparams": {**arch_spec.hparams, **override}})
    tcfg = TrainingConfig(
        n_steps=N_STEPS, batch_size=1024 if T == 1 else 1024 // T,
        buffer_tokens=BUFFER_TOKENS, arch_hparams_override=override)
    train_key = compute_train_key(
        arch=arch_spec, seed=seed, training_cfg=tcfg,
        data_key=compute_data_key(data_spec), section="synthetic")
    path = checkpoint_dir(train_key) / "model.safetensors"
    if not path.exists():
        return None, train_key, data_spec
    from safetensors.torch import load_file
    from temp_bench.core.trainer import _infer_d_in
    cls = import_by_path(arch_spec.class_path)
    model = cls(d_in=_infer_d_in(data_spec), **arch_spec.hparams)
    model.load_state_dict(load_file(str(path)))
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model, train_key, data_spec


def _fit(model, x, lam, *, L: int, n_windows: int, est: str, seed: int = 0):
    """The eval's λ probe with `n_windows` and the estimator exposed.

    `n_windows=1024, est='ols'` is byte-equivalent to
    `lambda_recovery._train_lambda_probe`.
    """
    from sklearn.linear_model import LinearRegression, RidgeCV

    T = _check_tileable(model, L)
    model.eval()
    n = x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)

    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_l_tr, _ = _sample_windows(lam3[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_l_ev, _ = _sample_windows(lam3[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, t_tr = _tile_lambda_examples(model, win_x_tr, win_l_tr, T)
    z_ev, t_ev = _tile_lambda_examples(model, win_x_ev, win_l_ev, T)

    reg = (LinearRegression() if est == "ols"
           else RidgeCV(alphas=np.logspace(-2, 4, 13)))
    reg.fit(z_tr, t_tr)
    pred = reg.predict(z_ev)
    corr = float(np.corrcoef(pred, t_ev)[0, 1]) if np.std(pred) > 1e-12 else 0.0
    return {"r": corr, "r2_heldout": float(reg.score(z_ev, t_ev)),
            "r2_train": float(reg.score(z_tr, t_tr)),
            "n_rows": int(z_tr.shape[0]), "p": int(z_tr.shape[1]),
            "nnz_per_row": float((z_tr != 0).sum(axis=1).mean()),
            "n_active_cols": int((z_tr != 0).any(axis=0).sum()),
            "alpha": float(getattr(reg, "alpha_", float("nan")))}


def main():
    ds = sys.argv[1] if len(sys.argv) > 1 else DS_DEFAULT
    data_cache: dict[int, object] = {}
    out = []
    for arch, T, k_pos in CELLS:
        for seed in SEEDS:
            model, train_key, data_spec = _load_model(arch, T, k_pos, seed, ds)
            if model is None:
                print(f"[skip] {arch}/T{T}/k{k_pos}/s{seed}: no checkpoint "
                      f"({train_key})", flush=True)
                continue
            if seed not in data_cache:
                data_cache[seed] = materialise(load_datasource(ds), seed=seed)
            data = data_cache[seed]
            lam = torch.as_tensor(data.extra["lambda_labels"]).float()
            for nw in N_WINDOWS:
                for est in ESTIMATORS:
                    rec = _fit(model, data.x, lam, L=EVAL_L, n_windows=nw, est=est)
                    row = {"arch": arch, "T": T, "k_pos": k_pos, "seed": seed,
                           "n_windows": nw, "est": est, "train_key": train_key,
                           **rec}
                    out.append(row)
                    print(f"{arch:<20} T{T:<3} k{k_pos:<4} s{seed:<3} "
                          f"nw={nw:<5} {est:<5} r={rec['r']:+.4f} "
                          f"r2ev={rec['r2_heldout']:+.3f} "
                          f"r2tr={rec['r2_train']:+.3f} "
                          f"n={rec['n_rows']:<6} nnz={rec['nnz_per_row']:.1f} "
                          f"act={rec['n_active_cols']}", flush=True)
            del model
            torch.cuda.empty_cache()
    RES.mkdir(exist_ok=True)
    (RES / f"probe_capacity_{ds}.json").write_text(json.dumps(out, indent=2))
    print(f"-> {RES}/probe_capacity_{ds}.json  ({len(out)} rows)")


if __name__ == "__main__":
    main()
