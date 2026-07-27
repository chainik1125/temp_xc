"""λ̂ T{6,10} fill — shuffle-overlay columns (T_FILL_CARD.md pre-registration).

The frozen overlay machinery (`shuffle_overlay._fit_ordered_and_shuffled`,
v1 pipeline verbatim) pointed at the fill checkpoints, with the fill's
eval_window_L=30. No anchor gate — these are fresh primary cells; the
identity receipt (|ordered r − canonical row metric| ≤ 2e-3, amended-tol
precedent) must pass per cell before its shuffled column is read.
Receipt failure ⇒ STOP + report (partials written). T10/s2 (the collapsed
cell) runs LAST so a stop there cannot mask the other five.

Run:  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m \
        experiments.explorations.task_hunt.lambda_intensity.t_fill_overlay
"""

from __future__ import annotations

import json
from pathlib import Path

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

from experiments.explorations.task_hunt.lambda_intensity.shuffle_overlay import (
    IDENTITY_TOL,
    SHUF_EVAL_SEED,
    _fit_ordered_and_shuffled,
)
from experiments.explorations.task_hunt.lambda_intensity.run_stage2 import (
    BUFFER_TOKENS,
    D_SAE,
    DS_DEFAULT,
    K_POS,
    N_STEPS,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
EVAL_L_FILL = 30
ARCH = "txc_batchtopk_post"
# T10/s2 last (see module docstring).
CELL_ORDER = [(6, 1), (6, 2), (6, 42), (10, 1), (10, 42), (10, 2)]


def main():
    fill = json.loads(
        (RES / f"stage2_t6t10_{DS_DEFAULT}.json").read_text())
    rows = fill if isinstance(fill, list) else fill["rows"]
    by_cell = {(r["T"], r["seed"]): r for r in rows
               if r.get("ok") and r["kind"] == "trained"}
    assert len(by_cell) == 6, f"expected 6 trained fill rows, got {len(by_cell)}"

    ds_spec = load_datasource(DS_DEFAULT)
    data = ward_lambda_real(**ds_spec.params)
    data_key = compute_data_key(ds_spec)
    lam = data.extra["lambda_labels"]
    if not torch.is_tensor(lam):
        lam = torch.as_tensor(lam)
    lam = lam.float()

    out = []
    out_path = RES / "t6t10_overlay.json"
    try:
        for T, seed in CELL_ORDER:
            row = by_cell[(T, seed)]
            override = {"k_pos": K_POS[0], "d_sae": D_SAE, "T": T}
            tcfg = TrainingConfig(
                n_steps=N_STEPS, batch_size=batch_size(T),
                buffer_tokens=BUFFER_TOKENS,
                arch_hparams_override=override)
            spec = load_arch(ARCH, section="synthetic")
            spec = spec.model_copy(
                update={"hparams": {**spec.hparams, **override}})
            tk = compute_train_key(arch=spec, seed=seed, training_cfg=tcfg,
                                   data_key=data_key, section="synthetic")
            model = _load_checkpoint(spec, tk, ds_spec).cuda()
            r_ord, r_sh = _fit_ordered_and_shuffled(
                model, data.x, lam, L=EVAL_L_FILL, T=T)
            canon = row["metrics"]["lambda_recovery"]
            ident = abs(r_ord - canon)
            assert ident <= IDENTITY_TOL, (
                f"identity receipt FAILED {ARCH}/T{T}/s{seed}: "
                f"recomputed {r_ord:.8f} vs canonical {canon:.8f} "
                f"(|Δ|={ident:.2e} > {IDENTITY_TOL:.0e})")
            cell = {"arch": ARCH, "T": T, "seed": seed,
                    "canonical_r": canon, "recomputed_r": r_ord,
                    "identity_abs": ident, "r_shuf": r_sh,
                    "gap": r_ord - r_sh, "eval_window_L": EVAL_L_FILL,
                    "shuf_seed": SHUF_EVAL_SEED, "tag": "t6t10_fill"}
            out.append(cell)
            print(f"[overlay] T{T}/s{seed}: ord {r_ord:+.4f} "
                  f"(receipt |Δ| {ident:.1e}) shuf {r_sh:+.4f} "
                  f"gap {r_ord - r_sh:+.4f}", flush=True)
            del model
            torch.cuda.empty_cache()
    finally:
        out_path.write_text(json.dumps(out, indent=1))
        print(f"[overlay] wrote {len(out)}/6 cells -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
