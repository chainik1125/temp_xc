"""C2 Phase 3 ENGINEER driver — hierarchical features (slow globals × fast locals).

agent_synth mission 2026-05-06T23:00Z. Engineered for the global/local
divide: K_g slow globals × K_l fast locals modulated by globals. Per-
token SAE prefers locals; window-pooling TXC prefers globals.

Reuses ``runner.run_cell`` and the standard C2 eval (eAUC against
``f_l`` = locals; gAUC against ``f_g`` = globals). Headline metric:
gAUC vs k_pos line plot, all 6 archs, TXC family clearly above SAE
family.

Usage (from purified/):
    bash experiments/c2_hierarchical/run.sh

or (single shard for debug):
    .venv/bin/python -m experiments.c2_hierarchical.run \\
        --datasource toy_hierarchical_Kg10_Kl30_d256
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import instantiate_arch, load_arch, load_datasource
from temp_bench.data.toy.hierarchical import hierarchical_features
from temp_bench.data.toy.coupled import make_batch_iter
from temp_bench.eval.synthetic import (
    feature_recovery,
    global_recovery_gAUC,
)
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


# Phase 3 arch list — full TXC + SAE family.
ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),                          # T=5
    ("txc_base",    None),                          # T=5
    ("txc_pro",     {"T_max": 2,  "t_sample": 2}),
    ("txc_pro",     {"T_max": 5,  "t_sample": 2}),
]

# k_pos × T must be ≤ d_sae=40 (per_component_hparams.c2). For T=5 archs
# (txc_base, stacked_sae T=5, txc_pro T=5), k_pos ≤ 8. We cap globally
# at 8 so all archs run on the same k_pos grid (fair comparison).
DEFAULT_K_POSES = (1, 2, 3, 4, 5, 6, 8)
DEFAULT_SEEDS = (1, 2, 42)


_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
    if datasource_name not in _DATA_CACHE:
        spec = load_datasource(datasource_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        kwargs = {
            "K_global": int(spec.K_global),
            "K_local": int(spec.K_local),
            "n_global_parents": int(getattr(spec, "n_global_parents", 1)),
            "d_in": int(spec.d_in),
            "seq_len": int(spec.seq_len),
            "pi_g": float(spec.pi_g),
            "rho_g": float(spec.rho_g),
            "p_l_high": float(getattr(spec, "p_l_high", 0.8)),
            "p_l_low":  float(getattr(spec, "p_l_low",  0.1)),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
            "device": device,
        }
        _DATA_CACHE[datasource_name] = hierarchical_features(**kwargs)
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

        # eAUC = local recovery (against f_l = emission_features).
        # gAUC = global recovery (against f_g = hidden_features).
        decoder = model.decoder_directions().detach().cpu()
        recov = feature_recovery(decoder, data.emission_features.cpu())
        glob = global_recovery_gAUC(decoder, data.hidden_features.cpu())

        metrics: dict[str, float] = {
            "eauc": float(recov["auc"]),
            "e_mean_max_cos": float(recov["mean_max_cos"]),
            "e_frac_recovered_90": float(recov["frac_recovered_90"]),
            **{k: float(v) for k, v in glob.items()},
        }
        return metrics, "gauc"
    return my_eval_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", default="toy_hierarchical_Kg10_Kl30_d256")
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--arch-t-idx", type=int, default=None,
                    help="Run ONLY the entry at this index of ARCH_TS. "
                         "Used to fan out (arch_t, seed) tuples to GPUs.")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--n-steps", type=int, default=20_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    rho_g = float(spec.rho_g)
    K_g = int(spec.K_global)
    K_l = int(spec.K_local)

    arch_filter = set(args.archs) if args.archs else None
    arch_ts = [(a, t) for (a, t) in ARCH_TS if arch_filter is None or a in arch_filter]
    if args.arch_t_idx is not None:
        arch_ts = [ARCH_TS[int(args.arch_t_idx)]]

    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)

    print(
        f"[hierarchical] datasource={args.datasource} K_g={K_g} K_l={K_l} "
        f"ρ_g={rho_g} archs={[a for a,_ in arch_ts]} k_pos={args.k_poses} "
        f"seeds={args.seeds} n_steps={args.n_steps}",
        flush=True,
    )

    for arch_name, t_override in arch_ts:
        for k_pos in args.k_poses:
            override: dict[str, Any] = {"k_pos": int(k_pos)}
            if t_override:
                override.update(t_override)
            cfg = TrainingConfig(
                n_steps=int(args.n_steps),
                batch_size=int(args.batch_size),
                plateau_early_stop=False,
                arch_hparams_override=override,
            )
            for seed in args.seeds:
                eval_cfg = {
                    "k_pos": int(k_pos),
                    "smoke": bool(args.smoke),
                    "_arch_hparams_override": override,
                    "rho_g": float(rho_g),
                    "K_global": K_g,
                    "K_local": K_l,
                    "bench": "hierarchical",
                }
                if t_override:
                    eval_cfg["t_label"] = "T=" + str(
                        t_override.get("T_max") or t_override.get("T")
                    )
                else:
                    eval_cfg["t_label"] = "default"
                print(
                    f"[c2 hier] {arch_name:12} {eval_cfg['t_label']:10} "
                    f"k={k_pos:2d} seed={seed} K_g={K_g} K_l={K_l} "
                    f"ds={args.datasource} steps={cfg.n_steps}",
                    flush=True,
                )
                runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch_name,
                    seed=int(seed),
                    datasource_name=args.datasource,
                    training_cfg=cfg,
                    eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=train_fn,
                    eval_fn=eval_fn,
                )


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
