"""ttrend shuffle overlay — post-hoc on the retrained checkpoints
(TT_SHUFFLE_OVERLAY_CARD.md § 4; instrument identical to the approved
λ̂ overlay: probing-1.2.0 shuffle control, identity receipt first).

Run:  CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m \
        experiments.explorations.task_hunt.diafaces.tt_shuffle_overlay
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from explorations.synthetic.grid import batch_size
from explorations.task_hunt.real_dialogue import dialogue_face_real
from temp_bench.core.config import (
    compute_data_key,
    compute_train_key,
    load_arch,
    load_datasource,
)
from temp_bench.core.runner import _load_checkpoint
from temp_bench.core.schemas import TrainingConfig

from experiments.explorations.task_hunt.diafaces.run_tt_shuffle_overlay_retrain import (
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
from experiments.explorations.task_hunt.lambda_intensity.shuffle_overlay import (
    IDENTITY_TOL,
    SHUF_EVAL_SEED,
    _fit_ordered_and_shuffled,
)

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEEDS = (1, 2, 42)

# Card § 3 gate constants (quoted v2 payload seed spread, frozen in
# the card BEFORE any retrain cell ran).
GATE = {
    ("txc_batchtopk_post", 2): (0.0363, 0.0058),
    ("txc_batchtopk_post", 4): (0.0501, 0.0087),
    ("txc_batchtopk_post", 8): (0.0709, 0.0291),
    ("txc_batchtopk_post", 16): (0.1421, 0.0099),
    ("txc_batchtopk_post", 32): (0.2968, 0.0127),
    ("batchtopk_sae", 1): (0.0320, 0.0030),
    ("tsae", 1): (0.0408, 0.0040),
}


def main():
    retrain = json.loads((RES / "tt_shuffle_overlay_retrain.json").read_text())
    rows = retrain if isinstance(retrain, list) else retrain["rows"]
    by_cell = {(r["arch"], r["T"], r["seed"]): r for r in rows if r.get("ok")}

    ds_spec = load_datasource(DS)
    data = dialogue_face_real(**ds_spec.params)
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
                out.append({"arch": arch_name, "T": T, "seed": seed,
                            "canonical_r": canon, "recomputed_r": r_ord,
                            "identity_abs": ident,
                            "shuffle_identity": int(T == 1),
                            "r_shuf": r_sh, "gap": r_ord - r_sh,
                            "shuf_seed": SHUF_EVAL_SEED, "tag": RETRAIN_TAG})
                per_seed.append((r_ord, r_sh))
                del model
                torch.cuda.empty_cache()
                print(f"[tt-overlay] {arch_name}/T{T}/s{seed} "
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

    payload = {"card": "TT_SHUFFLE_OVERLAY_CARD.md", "status": "PENDING TEAM REVIEW",
               "shuf_eval_seed": SHUF_EVAL_SEED, "identity_tol": IDENTITY_TOL,
               "anchor_gate": gate, "anchor_gate_all_pass": all_pass,
               "summary": means, "cells": out}
    (RES / "tt_shuffle_overlay.json").write_text(json.dumps(payload, indent=1))
    print(f"[tt-overlay] -> {RES / 'tt_shuffle_overlay.json'}")


if __name__ == "__main__":
    main()
