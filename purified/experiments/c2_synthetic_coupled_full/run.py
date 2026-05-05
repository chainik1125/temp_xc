"""C2 full-suite driver — same sweep as agent_filler's c2, routed through
``temp_bench.data.toy_full.api`` (the wasteland-faithful port).

Sweep dimensions (identical to agent_filler):
  - arch ∈ {topk_sae, stacked_sae (T=2/5), txc_base (T=5), txc_pro (T=2/5/12)}
  - k_pos ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20}.
  - seed ∈ {1, 2, 42}.

Per-cell metrics:
  - eAUC (vs M=20 emission directions)
  - gAUC (vs K=10 hidden directions) — paper headline
  - mean_max_cos / frac_recovered_{80,90} per axis

Datasource: ``toy_coupled_K10_M20_d256_full`` — same hparams as
agent_filler's ``toy_coupled_K10_M20_d256`` but distinct
``act_cache_key`` (different generator path), so leaderboard rows
from this driver don't collide with agent_filler's runs.

Usage::

    .venv/bin/python -m experiments.c2_synthetic_coupled_full.run \\
        --seeds 1 2 42 --n-steps 30000

Smoke::

    .venv/bin/python -m experiments.c2_synthetic_coupled_full.run \\
        --archs txc_pro --k-poses 5 --seeds 42 --n-steps 200 --smoke
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch

from temp_bench import runner
from temp_bench.config import instantiate_arch, load_arch, load_datasource
from temp_bench.data.toy_full.api import coupled_hmm, make_batch_iter
from temp_bench.schemas import TrainingConfig
from temp_bench.training.sae_trainer import train_sae


@torch.no_grad()
def feature_recovery(decoder_directions: torch.Tensor,
                     true_features: torch.Tensor,
                     *, n_thresholds: int = 50) -> dict[str, float]:
    """Per-feature AUC; inline copy of agent_paper's eval/synthetic.feature_recovery."""
    a = decoder_directions / decoder_directions.norm(dim=1, keepdim=True).clamp_min(1e-12)
    b = true_features / true_features.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sims = (a @ b.t()).abs()
    max_per_true = sims.max(dim=0).values
    thresholds = np.linspace(0, 1, n_thresholds)
    curve = np.array([
        (max_per_true.cpu().numpy() >= t).mean() for t in thresholds
    ])
    return {
        "auc": float(np.trapezoid(curve, thresholds)),
        "mean_max_cos": float(max_per_true.mean()),
        "frac_recovered_90": float((max_per_true >= 0.9).float().mean()),
        "frac_recovered_80": float((max_per_true >= 0.8).float().mean()),
    }


def global_recovery_gAUC(decoder_directions: torch.Tensor,
                         hidden_features: torch.Tensor) -> dict[str, float]:
    """gAUC + companion stats vs K hidden directions."""
    base = feature_recovery(decoder_directions, hidden_features)
    return {
        "gauc": base["auc"],
        "g_mean_max_cos": base["mean_max_cos"],
        "g_frac_recovered_90": base["frac_recovered_90"],
        "g_frac_recovered_80": base["frac_recovered_80"],
    }

COMPONENT = "c2"
DATASOURCE = "toy_coupled_K10_M20_d256_full"
EVAL_PROTOCOL_VERSION = "1.0.0"

ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),
    ("txc_base",    None),
    ("txc_pro",     {"T_max": 2,  "t_sample": 2}),
    ("txc_pro",     {"T_max": 5,  "t_sample": 2}),
    ("txc_pro",     {"T_max": 12, "t_sample": 2}),
]

DEFAULT_K_POSES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20)
DEFAULT_SEEDS = (1, 2, 42)

_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
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
            "delta": float(getattr(spec, "delta", 0.0)),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "coupling_mode": getattr(spec, "coupling_mode", "or_gate"),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
        }
        _DATA_CACHE[datasource_name] = coupled_hmm(**kwargs)
    return _DATA_CACHE[datasource_name]


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg,
                act_cache_key, component):
    data = _get_data(DATASOURCE)
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


def my_eval_fn(*, model=None, eval_cfg, component):
    arch_name = eval_cfg["_arch_name"]
    state = eval_cfg["_state_dict"]
    overrides = eval_cfg.get("_arch_hparams_override")

    data = _get_data(DATASOURCE)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    arch_filter = set(args.archs) if args.archs else None

    for arch_name, t_override in ARCH_TS:
        if arch_filter is not None and arch_name not in arch_filter:
            continue
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
                    "_full_suite": True,
                }
                if t_override:
                    eval_cfg["t_label"] = "T=" + str(
                        t_override.get("T_max") or t_override.get("T")
                    )
                else:
                    eval_cfg["t_label"] = "default"
                t_label = eval_cfg["t_label"]
                print(
                    f"[c2_full] {arch_name:12} {t_label:10} k={k_pos:2d} "
                    f"seed={seed} steps={cfg.n_steps}",
                    flush=True,
                )
                runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch_name,
                    seed=int(seed),
                    datasource_name=DATASOURCE,
                    training_cfg=cfg,
                    eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                )


if __name__ == "__main__":
    os.environ.setdefault("TQDM_DISABLE", "1")
    main()
