"""Unified baseline-fill driver for C2 synthetic Setups D, E, F, G (and J).

agent_synth post-compact mission 2026-05-07: each of agent_synth's setups
(D-np5, D-np10, E, F, G) is missing tsae_paper + tfa_pos rows. F + G are
also missing stacked_sae T=2/T=5. This driver runs any single missing-arch
× seed sweep against any datasource by routing on the YAML ``generator``
field.

Generator routing:
  - ``coupled_noisy_hmm``           → Setup D (noisy + overlap)
  - ``hierarchical_features``       → Setup E / J (hierarchical)
  - ``coupled_obs_noise_hmm``       → Setup F (coupled + obs noise)
  - ``hierarchical_obs_noise_features`` → Setup G (hier + obs noise)

Per-arch overrides:
  - tsae_paper   → arch_hparams_override={"d_sae": 40, "k_pos": k},
                   train_window_size=2 (Bhalla/Ye 2025 paper-faithful)
  - tfa_pos      → arch_hparams_override={"d_sae": 40, "k_pos": k}
  - stacked_sae  → no d_sae override (already 40 via per_component_hparams.c2);
                   T_override picks T=2 vs T=5

Usage::

    .venv/bin/python -m experiments.c2_synthetic_coupled.fill_baselines \\
        --datasource toy_coupled_noisy_K10_M20_d256_pB05_np10 \\
        --arch tsae_paper --seed 1 \\
        --k-poses 1 2 3 4 5 6 8

The launcher fans out (datasource, arch, seed) tuples across GPUs.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Callable

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch

from temp_bench import runner
from temp_bench.schemas import TrainingConfig
from temp_bench.config import instantiate_arch, load_arch, load_datasource
from temp_bench.data.toy.coupled import coupled_hmm
from temp_bench.data.toy.coupled_noisy import coupled_noisy_hmm
from temp_bench.data.toy.hierarchical import hierarchical_features
from temp_bench.data.toy.coupled_obs_noise import coupled_obs_noise_hmm
from temp_bench.data.toy.hierarchical_obs_noise import (
    hierarchical_obs_noise_features,
)
from temp_bench.data.toy.heterogeneous_rho import heterogeneous_rho_features
from temp_bench.data.toy.dewdrop import dewdrop_features
from temp_bench.data.toy.chord import chord_features
from temp_bench.data.toy.coupled import make_batch_iter
from temp_bench.eval.synthetic import feature_recovery, global_recovery_gAUC
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"

# k_pos × T ≤ d_sae=40. tsae_paper is per-token (T=1 effective), tfa_pos is
# per-token. stacked_sae has T=2 or 5. We cap at k_pos=8 globally so all
# archs run on the same grid.
DEFAULT_K_POSES = (1, 2, 3, 4, 5, 6, 8)
DEFAULT_SEEDS = (1, 2, 42)


_DATA_CACHE: dict[str, Any] = {}


def _build_data(spec, *, device: str):
    """Dispatch on YAML ``generator`` field to the right toy-data builder."""
    gen = getattr(spec, "generator", "")
    seed = 0
    if gen.endswith(":coupled_hmm"):
        return coupled_hmm(
            K_hidden=int(spec.K_hidden),
            M_emissions=int(spec.M_emissions),
            n_parents=int(spec.n_parents),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi=float(spec.pi),
            rho=float(spec.rho),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":coupled_noisy_hmm"):
        return coupled_noisy_hmm(
            K_hidden=int(spec.K_hidden),
            M_emissions=int(spec.M_emissions),
            n_parents=int(spec.n_parents),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi=float(spec.pi),
            rho=float(spec.rho),
            p_A=float(getattr(spec, "p_A", 0.0)),
            p_B=float(getattr(spec, "p_B", 1.0)),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":hierarchical_features"):
        return hierarchical_features(
            K_global=int(spec.K_global),
            K_local=int(spec.K_local),
            n_global_parents=int(getattr(spec, "n_global_parents", 1)),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi_g=float(spec.pi_g),
            rho_g=float(spec.rho_g),
            p_l_high=float(getattr(spec, "p_l_high", 0.8)),
            p_l_low=float(getattr(spec, "p_l_low", 0.1)),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":coupled_obs_noise_hmm"):
        return coupled_obs_noise_hmm(
            K_hidden=int(spec.K_hidden),
            M_emissions=int(spec.M_emissions),
            n_parents=int(spec.n_parents),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi=float(spec.pi),
            rho=float(spec.rho),
            obs_noise_sigma=float(spec.obs_noise_sigma),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":chord_features"):
        return chord_features(
            K_global=int(spec.K_global),
            K_local=int(spec.K_local),
            n_groups=int(spec.n_groups),
            n_global_parents=int(getattr(spec, "n_global_parents", 1)),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi_g=float(spec.pi_g),
            rho_g=float(spec.rho_g),
            p_l_high=float(getattr(spec, "p_l_high", 0.8)),
            p_l_low=float(getattr(spec, "p_l_low", 0.1)),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":dewdrop_features"):
        return dewdrop_features(
            K_global=int(spec.K_global),
            K_local=int(spec.K_local),
            n_global_parents=int(getattr(spec, "n_global_parents", 1)),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            period=int(spec.period),
            stride=int(getattr(spec, "stride", 1)),
            p_l_high=float(getattr(spec, "p_l_high", 0.8)),
            p_l_low=float(getattr(spec, "p_l_low", 0.1)),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":heterogeneous_rho_features"):
        return heterogeneous_rho_features(
            K_global=int(spec.K_global),
            K_local=int(spec.K_local),
            n_global_parents=int(getattr(spec, "n_global_parents", 1)),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi_g=float(spec.pi_g),
            rho_g_list=list(spec.rho_g_list),
            p_l_high=float(getattr(spec, "p_l_high", 0.8)),
            p_l_low=float(getattr(spec, "p_l_low", 0.1)),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    if gen.endswith(":hierarchical_obs_noise_features"):
        return hierarchical_obs_noise_features(
            K_global=int(spec.K_global),
            K_local=int(spec.K_local),
            n_global_parents=int(getattr(spec, "n_global_parents", 1)),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            pi_g=float(spec.pi_g),
            rho_g=float(spec.rho_g),
            p_l_high=float(getattr(spec, "p_l_high", 0.8)),
            p_l_low=float(getattr(spec, "p_l_low", 0.1)),
            obs_noise_sigma=float(spec.obs_noise_sigma),
            magnitude_dist=getattr(spec, "magnitude_dist", "folded_normal"),
            magnitude_mean=float(getattr(spec, "magnitude_mean", 1.0)),
            magnitude_std=float(getattr(spec, "magnitude_std", 0.15)),
            n_seqs=int(getattr(spec, "n_seqs", 4096)),
            seed=seed,
            device=device,
        )
    raise ValueError(f"Unknown generator: {gen!r} for datasource")


def _get_data(datasource_name: str):
    if datasource_name not in _DATA_CACHE:
        spec = load_datasource(datasource_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _DATA_CACHE[datasource_name] = _build_data(spec, device=device)
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


def _is_valid_cell(arch_name: str, k_pos: int, T: int | None) -> bool:
    """k_pos × T ≤ d_sae=40 is the binding constraint for windowed archs.
    Per-token archs (topk_sae, tsae_paper BatchTopK, tfa_pos) have T=1 and
    only need k_pos ≤ d_sae=40."""
    if T is None or T <= 1:
        return k_pos <= 40
    return k_pos * T <= 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True,
                    help="YAML datasource name (any setup D/E/F/G/J).")
    ap.add_argument("--arch", required=True,
                    choices=["tsae_paper", "tfa_pos", "stacked_sae",
                             "topk_sae", "txc_base", "txc_pro"])
    ap.add_argument("--T", type=int, default=None,
                    help="T override for stacked_sae / txc_base / txc_pro. "
                         "Ignored for tsae_paper / tfa_pos / topk_sae.")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--k-poses", nargs="+", type=int,
                    default=list(DEFAULT_K_POSES))
    ap.add_argument("--n-steps", type=int, default=8000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)

    # Per-arch TrainingConfig + arch_hparams_override.
    arch_train_window: int | None = None
    if args.arch == "tsae_paper":
        arch_train_window = 2  # paper-faithful adjacent-pair training
    t_label = "default" if args.T is None else f"T={args.T}"

    print(
        f"[fill_baselines] arch={args.arch} T={args.T} seed={args.seed} "
        f"ds={args.datasource} k_poses={args.k_poses} n_steps={args.n_steps}",
        flush=True,
    )

    for k_pos in args.k_poses:
        if not _is_valid_cell(args.arch, int(k_pos), args.T):
            print(f"[fill_baselines] SKIP {args.arch} T={args.T} k={k_pos} "
                  f"(k_pos × T > d_sae=40)", flush=True)
            continue
        override: dict[str, Any] = {"k_pos": int(k_pos)}
        if args.arch in ("tsae_paper", "tfa_pos"):
            override["d_sae"] = 40
        if args.T is not None:
            if args.arch == "stacked_sae":
                override["T"] = int(args.T)
            elif args.arch == "txc_base":
                override["T"] = int(args.T)
            elif args.arch == "txc_pro":
                override["T_max"] = int(args.T)
                override["t_sample"] = 2
        cfg_kwargs: dict[str, Any] = dict(
            n_steps=int(args.n_steps),
            batch_size=int(args.batch_size),
            plateau_early_stop=False,
            arch_hparams_override=override,
        )
        if arch_train_window is not None:
            cfg_kwargs["train_window_size"] = arch_train_window
        cfg = TrainingConfig(**cfg_kwargs)

        # Tag txc_base × T cells as tsweep so plot_headline.render_tsweep
        # picks them up. Other archs get hunt_phase="fill" for the line/
        # scatter plots.
        is_tsweep = (args.arch == "txc_base" and args.T is not None)
        eval_cfg = {
            "k_pos": int(k_pos),
            "smoke": bool(args.smoke),
            "_arch_hparams_override": override,
            "t_label": t_label,
            "fill_baselines": True,
            "hunt_phase": "tsweep" if is_tsweep else "fill",
            "tsweep": is_tsweep,
            "bench": "hierarchical" if "hierarchical" in args.datasource else None,
            "datasource": args.datasource,
        }
        print(
            f"[fill_baselines] {args.arch:12s} {t_label:8s} k={k_pos:2d} "
            f"seed={args.seed} ds={args.datasource} n_steps={cfg.n_steps}",
            flush=True,
        )
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
