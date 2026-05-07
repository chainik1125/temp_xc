"""Setup A + C baseline backfill driver.

Setup A (`toy_coupled_K10_M20_d256`): missing tfa_pos.
Setup C (`toy_coupled_K10_M20_d256_rho{00,03,06,09}`): missing all 5
baselines (topk_sae, tfa_pos, tsae_paper, stacked_sae T=2/T=5).

Generator: ``coupled_hmm`` (= clean Setup A, no Bernoulli noise).
Different from Setup D's ``coupled_noisy_hmm`` and from F's
``coupled_obs_noise_hmm``. Per-datasource params come from
``configs/datasources.yaml`` (loaded via ``load_datasource``).

Per-arch knobs:
- topk_sae:    per-token, no T axis.       arch_hparams_override = {k_pos, d_sae=40}
- tfa_pos:     per-token attention.         arch_hparams_override = {k_pos, d_sae=40}
- tsae_paper:  T=2 paper-faithful.         arch_hparams_override = {k_pos, d_sae=40}
                                            train_window_size=2 in TrainingConfig.
- stacked_sae T=2 / T=5: arch_hparams_override = {k_pos, T, d_sae=40}

n_steps=30000 to match the existing Setup A coverage (12 k_pos × 3
seeds at n_steps=30000, per agent_filler's c2 sweep). Agent_synth's
later setups use 8000; Setup A/C are paper-headline so we keep the
deeper training for fair comparison with the existing TXC cells.
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
from temp_bench.data.toy.coupled import coupled_hmm, make_batch_iter
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
        kwargs = {
            "K_hidden": int(spec.K_hidden),
            "M_emissions": int(spec.M_emissions),
            "n_parents": int(spec.n_parents),
            "d_in": int(spec.d_in),
            "seq_len": int(spec.seq_len),
            "pi": float(spec.pi),
            "rho": float(spec.rho),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
        }
        _DATA_CACHE[datasource_name] = coupled_hmm(**kwargs)
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


def _build_override(arch: str, k_pos: int, T: int | None) -> dict[str, Any]:
    base = {"k_pos": int(k_pos), "d_sae": 40}
    if arch in ("topk_sae", "tfa_pos", "tsae_paper"):
        return base
    if arch in ("stacked_sae", "txc_base"):
        if T is None:
            raise SystemExit(f"{arch} requires --T.")
        return {**base, "T": int(T)}
    raise SystemExit(f"Unsupported arch '{arch}'.")


def _t_label(arch: str, T: int | None) -> str:
    if arch in ("stacked_sae", "txc_base") and T is not None:
        return f"T={T}"
    return "default"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True)
    ap.add_argument("--arch", required=True,
                    choices=["topk_sae", "tfa_pos", "tsae_paper", "stacked_sae",
                             "txc_base"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--T", type=int, default=None)
    ap.add_argument("--k-poses", nargs="+", type=int, required=True)
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    rho = float(spec.rho)

    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)
    train_window_size = 2 if args.arch == "tsae_paper" else None

    for k_pos in args.k_poses:
        override = _build_override(args.arch, int(k_pos), args.T)
        cfg = TrainingConfig(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
            train_window_size=train_window_size,
        )
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": False,
            "_arch_hparams_override": override,
            "t_label": _t_label(args.arch, args.T),
            "rho": rho,
        }
        print(f"[ac_baselines] {args.arch:12s} T={args.T} k={k_pos:2d} "
              f"seed={args.seed} ρ={rho} ds={args.datasource}", flush=True)
        runner.run_cell(
            component=COMPONENT,
            arch_name=args.arch,
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
