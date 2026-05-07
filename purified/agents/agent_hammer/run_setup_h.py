"""Setup H driver — ρ-sweep on Setup D's max-overlap regime (pB05_np10).

Disambiguates Effect 1 (sample aggregation, ρ-independent) from Effect 2
(temporal pattern detection, ρ-dependent) on the regime where TXC's gAUC
win is largest. Same disambiguation Setup C runs on Setup A; Setup H
runs the same on Setup D-np10.

Generator: ``coupled_noisy_hmm`` (same as Setup D's ``run_hunt.py``).
Datasources at ρ ∈ {0.0, 0.3, 0.6, 0.9} — the latter already exists as
``toy_coupled_noisy_K10_M20_d256_pB05_np10``; the new three were added
by this same agent in ``configs/datasources.yaml`` 2026-05-07.

Per-arch sweep: handles topk_sae, stacked_sae T=2/T=5, tfa_pos,
tsae_paper, and the txc_base T-sweep (T ∈ {2, 4, 5, 6, 8, 10, 12}).
All d_sae overrides are 40 (toy regime); k_pos × T must be ≤ d_sae.

Usage (one (arch, seed, T) per call; launcher shards across GPUs):

    .venv/bin/python -m agents.agent_hammer.run_setup_h \\
        --datasource toy_coupled_noisy_K10_M20_d256_pB05_np10_rho00 \\
        --arch txc_base --T 5 --seed 1 --k-poses 1 2 3
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
from temp_bench.data.toy.coupled_noisy import coupled_noisy_hmm
from temp_bench.data.toy.coupled import make_batch_iter
from temp_bench.eval.synthetic import (
    feature_recovery,
    global_recovery_gAUC,
)
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


# Process-global data cache (datasource-keyed) — generator is expensive.
_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
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
            "p_A": float(spec.p_A),
            "p_B": float(spec.p_B),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
            "device": device,
        }
        _DATA_CACHE[datasource_name] = coupled_noisy_hmm(**kwargs)
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
    """All baselines need d_sae=40 override (locked YAML has 18432)."""
    base = {"k_pos": int(k_pos), "d_sae": 40}
    if arch == "topk_sae":
        return base   # per-token, no T axis
    if arch == "tfa_pos":
        return base   # per-token attention, no T axis
    if arch == "tsae_paper":
        return base   # train_window_size=2 set in TrainingConfig, not here
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
                    choices=["topk_sae", "stacked_sae", "tfa_pos",
                             "tsae_paper", "txc_base"])
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--T", type=int, default=None,
                    help="Required for stacked_sae and txc_base.")
    ap.add_argument("--k-poses", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--n-steps", type=int, default=8_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    args = ap.parse_args()

    spec = load_datasource(args.datasource)
    rho = float(spec.rho)
    n_parents = int(spec.n_parents)

    train_fn = make_train_fn(args.datasource)
    eval_fn  = make_eval_fn(args.datasource)

    train_window_size = 2 if args.arch == "tsae_paper" else None
    d_sae = 40

    for k_pos in args.k_poses:
        # Skip cells where k_pos * T exceeds d_sae for window-style archs.
        if args.arch in ("stacked_sae", "txc_base") and args.T is not None:
            if int(k_pos) * int(args.T) > d_sae:
                print(f"[setup_h] SKIP {args.arch} T={args.T} k={k_pos} "
                      f"(k_pos × T > d_sae=40)", flush=True)
                continue
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
            "n_parents": n_parents,
            "n_steps_train": int(args.n_steps),
            "setup": "H",
        }
        print(f"[setup_h] {args.arch:12s} T={args.T} k={k_pos:2d} "
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
