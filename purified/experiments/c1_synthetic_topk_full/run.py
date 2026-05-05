"""C1 full-suite driver — same sweep as agent_filler's c1, routed through
``temp_bench.data.toy_full.api`` (the wasteland-faithful port).

Sweep dimensions (identical to agent_filler):
  - arch ∈ {topk_sae, tsae_paper, tfa, tfa_pos, stacked_sae, txc_base, txc_pro}
  - For stacked_sae: also sweep T ∈ {2, 5} via arch_hparams_override.
  - k_pos ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20}.
  - seed ∈ {1, 2, 42}.

Per-cell metric: feature-recovery AUC against the 20 ground-truth
orthogonal feature directions. Identical scoring to agent_filler.

Datasource: ``toy_markov_n20_d40_full`` — same hyperparameters as
agent_filler's ``toy_markov_n20_d40`` but a distinct ``act_cache_key``
(generator path differs), so leaderboard rows produced by this driver
are kept separate from agent_filler's even though both share
``component='c1'``.

Usage::

    .venv/bin/python -m experiments.c1_synthetic_topk_full.run \\
        --seeds 1 2 42 --n-steps 30000

Smoke::

    .venv/bin/python -m experiments.c1_synthetic_topk_full.run \\
        --archs txc_base --k-poses 5 --seeds 42 --n-steps 200 --smoke
"""
from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch

from temp_bench import runner
from temp_bench.config import instantiate_arch, load_arch, load_datasource
from temp_bench.data.toy_full.api import make_batch_iter, markov_chain_support
from temp_bench.eval.synthetic import feature_recovery
from temp_bench.schemas import TrainingConfig
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c1"
DATASOURCE = "toy_markov_n20_d40_full"
EVAL_PROTOCOL_VERSION = "1.0.0"

ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),
    ("tsae_paper",  None),
    ("tfa",         None),
    ("tfa_pos",     None),
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),
    ("txc_base",    None),
    ("txc_pro",     None),
]

DEFAULT_K_POSES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20)
DEFAULT_SEEDS = (1, 2, 42)

# ── Data plumbing ──────────────────────────────────────────────────────────

_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
    if datasource_name not in _DATA_CACHE:
        spec = load_datasource(datasource_name)
        kwargs = {
            "n_features": int(spec.n_features),
            "d_in": int(spec.d_in),
            "seq_len": int(spec.seq_len),
            "rho_levels": list(spec.rho_levels),
            "pi": float(spec.pi),
            "delta": float(getattr(spec, "delta", 0.0)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
        }
        _DATA_CACHE[datasource_name] = markov_chain_support(**kwargs)
    return _DATA_CACHE[datasource_name]


# ── train_fn / eval_fn ─────────────────────────────────────────────────────


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

    decoder = model.decoder_directions().detach().cpu()       # (d_sae, d_in)
    recov = feature_recovery(decoder, data.features.cpu())

    return {
        "auc": float(recov["auc"]),
        "mean_max_cos": float(recov["mean_max_cos"]),
        "frac_recovered_90": float(recov["frac_recovered_90"]),
        "frac_recovered_80": float(recov["frac_recovered_80"]),
    }, "auc"


# ── Per-arch validity check ────────────────────────────────────────────────


def _is_valid_cell(arch_name: str, t_override: dict | None, k_pos: int) -> bool:
    spec = load_arch(arch_name, component=COMPONENT)
    hp = spec.hparams
    if t_override:
        hp = {**hp, **t_override}
    d_sae = int(hp.get("d_sae", 40))
    if arch_name in ("topk_sae", "tsae_paper", "tfa", "tfa_pos"):
        return k_pos <= d_sae
    if arch_name == "stacked_sae":
        T = int(hp.get("T", 5))
        return k_pos * T <= d_sae
    if arch_name == "txc_base":
        T = int(hp.get("T", 5))
        return k_pos * T <= d_sae
    if arch_name == "txc_pro":
        t_sample = int(hp.get("t_sample", 5))
        h_size = int(hp.get("h_size", d_sae // 5))
        return k_pos * t_sample <= h_size
    return True


# ── Sweep entrypoint ───────────────────────────────────────────────────────


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
            if not _is_valid_cell(arch_name, t_override, int(k_pos)):
                t_label = (
                    "T=" + str(t_override.get("T") or t_override.get("T_max"))
                    if t_override else "default"
                )
                print(f"[c1_full] SKIP invalid {arch_name:12} {t_label:10} k={k_pos:2d}",
                      flush=True)
                continue
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
                t_label = (
                    "T=" + str(t_override.get("T") or t_override.get("T_max"))
                    if t_override else "default"
                )
                eval_cfg = {
                    "k_pos": int(k_pos),
                    "smoke": bool(args.smoke),
                    "_arch_hparams_override": override,
                    "t_label": t_label,
                    "_full_suite": True,
                }
                print(
                    f"[c1_full] {arch_name:12} {t_label:10} k={k_pos:2d} "
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
