"""Setup F driver — coupled with additive observation noise.

Same protocol as Setup D's run_hunt.py / run_t_sweep.py but uses the
``coupled_obs_noise_hmm`` generator. Sweeps T-values × k_pos at a
single σ datasource to show denoising as a function of T.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import instantiate_arch, load_arch, load_datasource
from temp_bench.data.toy.coupled_obs_noise import coupled_obs_noise_hmm
from temp_bench.data.toy.coupled import make_batch_iter
from temp_bench.eval.synthetic import (
    feature_recovery,
    global_recovery_gAUC,
)
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str):
    if datasource_name not in _DATA_CACHE:
        spec = load_datasource(datasource_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {
            "K_hidden": int(spec.K_hidden),
            "M_emissions": int(spec.M_emissions),
            "n_parents": int(spec.n_parents),
            "d_in": int(spec.d_in),
            "seq_len": int(spec.seq_len),
            "pi": float(spec.pi),
            "rho": float(spec.rho),
            "obs_noise_sigma": float(spec.obs_noise_sigma),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
            "device": device,
        }
        _DATA_CACHE[datasource_name] = coupled_obs_noise_hmm(**kwargs)
    return _DATA_CACHE[datasource_name]


def make_train_fn(datasource_name: str):
    def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg,
                    act_cache_key, component):
        data = _get_data(datasource_name)
        spec = load_arch(arch_name, component=component)
        if training_cfg.arch_hparams_override:
            merged = {**spec.hparams, **training_cfg.arch_hparams_override}
            spec = spec.model_copy(update={"hparams": merged})
        d_in = data.x.shape[-1]
        model = instantiate_arch(spec, d_in=d_in)
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
        raw_iter = make_batch_iter(data, seed=int(seed))
        result = train_sae(model, raw_iter, training_cfg, device="cuda")
        return result["state_dict"]
    return my_train_fn


def make_eval_fn(datasource_name: str):
    def my_eval_fn(*, model=None, eval_cfg, component):
        arch_name = eval_cfg["_arch_name"]
        state = eval_cfg["_state_dict"]
        overrides = eval_cfg.get("_arch_hparams_override")
        data = _get_data(datasource_name)
        spec = load_arch(arch_name, component=component)
        if overrides:
            merged = {**spec.hparams, **overrides}
            spec = spec.model_copy(update={"hparams": merged})
        d_in = data.x.shape[-1]
        model = instantiate_arch(spec, d_in=d_in).to("cuda").eval()
        model.load_state_dict(state, strict=True)
        decoder = model.decoder_directions().detach().cpu()
        recov = feature_recovery(decoder, data.emission_features.cpu())
        glob = global_recovery_gAUC(decoder, data.hidden_features.cpu())
        metrics = {
            "eauc": float(recov["auc"]),
            "e_mean_max_cos": float(recov["mean_max_cos"]),
            "e_frac_recovered_90": float(recov["frac_recovered_90"]),
            **{k: float(v) for k, v in glob.items()},
        }
        return metrics, "gauc"
    return my_eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--T", required=True, type=int)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--n-steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k-poses", nargs="+", type=int, required=True)
    ap.add_argument("--d-sae", type=int, default=40)
    ap.add_argument("--archs", nargs="+", default=["txc_base"],
                    help="Archs to run (txc_base only by default).")
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    sigma = float(spec.obs_noise_sigma)
    rho = float(spec.rho)
    T_val = int(args.T)
    t_label = f"T={T_val}"

    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)

    for arch_name in args.archs:
        # topk_sae has no T axis; only run topk_sae once if it's first time.
        if arch_name == "topk_sae" and T_val != 5:
            continue  # Skip duplicate runs across T values for topk_sae
        for k_pos in args.k_poses:
            if arch_name in ("txc_base",) and int(k_pos) * T_val > int(args.d_sae):
                print(f"[setupF] SKIP {arch_name} T={T_val} k={k_pos}", flush=True)
                continue
            override: dict[str, Any] = {"k_pos": int(k_pos)}
            if arch_name in ("txc_base",):
                override["T"] = T_val
            cfg = TrainingConfig(
                n_steps=int(args.n_steps),
                batch_size=int(args.batch_size),
                plateau_early_stop=False,
                arch_hparams_override=override,
            )
            eval_cfg = {
                "k_pos": int(k_pos),
                "smoke": False,
                "_arch_hparams_override": override,
                "rho": rho,
                "obs_noise_sigma": sigma,
                "setup": "F",
                "n_steps_train": int(args.n_steps),
                "t_label": t_label if arch_name == "txc_base" else "default",
            }
            print(f"[setupF] {arch_name} T={T_val} k={k_pos} seed={args.seed} "
                  f"σ={sigma} ds={args.datasource}", flush=True)
            runner.run_cell(
                component=COMPONENT,
                arch_name=arch_name,
                seed=int(args.seed),
                datasource_name=args.datasource,
                training_cfg=cfg,
                eval_cfg=eval_cfg,
                eval_protocol_version=EVAL_PROTOCOL_VERSION,
                train_fn=train_fn,
                eval_fn=eval_fn,
            )


if __name__ == "__main__":
    main()
