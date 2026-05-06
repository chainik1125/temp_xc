"""C2 HUNT driver — find a regime where TXC clearly beats SAE on gAUC.

agent_synth mission 2026-05-06T23:00Z (Han override): hunt across
(p_B, n_parents) parameter space on a noisy + overlap coupled-features
generator (Dmitry-bench-2-style). Phase 1 of the 4-phase mission.

Each invocation processes ONE datasource (sharded across 8 GPUs by the
launcher ``run_hunt.sh``). Inside the process we sweep:
  - 2 archs: ``topk_sae``, ``txc_base`` (T=5)
  - 3 seeds: {1, 2, 42}
  - 6 k_pos: {1, 2, 5, 10, 15, 20}
  - n_steps: 20_000

= 36 cells per process × 8 processes = 288 cells in Phase 1.

Phase 2 (ZOOM) and Phase 3 (ENGINEER) reuse this driver with a wider
arch list and denser k_pos via CLI flags.

Generator: ``temp_bench.data.toy.coupled_noisy:coupled_noisy_hmm`` —
adds per-token Bernoulli emission noise (p_B) on top of OR-gate
coupling. Defaults reproduce Dmitry's Bench 2 (n_parents=5, p_B=0.5,
ρ=0.9) where TXC's gAUC win is largest.
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
from temp_bench.data.toy.coupled_noisy import coupled_noisy_hmm
from temp_bench.data.toy.coupled import make_batch_iter
from temp_bench.eval.synthetic import (
    feature_recovery,
    global_recovery_gAUC,
)
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
EVAL_PROTOCOL_VERSION = "1.0.0"


# Phase 1 (HUNT) defaults — small arch list + coarse k_pos.
# k_pos × T must be ≤ d_sae=40. For T=5 archs (txc_base), k_pos ≤ 8.
# We cap at 8 globally; HUNT keeps a coarse grid below.
HUNT_ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),                          # per-token baseline
    ("txc_base",    None),                          # T=5 default canonical TXC
]
HUNT_K_POSES = (1, 2, 4, 5, 7, 8)
HUNT_SEEDS = (1, 2, 42)

# Phase 2 (ZOOM) defaults — full arch list + dense k_pos.
# Capped at 8 (T=5 limit). Was (1,2,3,4,5,6,8,10,12,15,17,20) but k_pos>8
# crashes for T=5 with k_win=k_pos*T > d_sae=40.
ZOOM_ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),                          # T=5
    ("txc_base",    None),                          # T=5
    ("txc_pro",     {"T_max": 2,  "t_sample": 2}),
    ("txc_pro",     {"T_max": 5,  "t_sample": 2}),
]
ZOOM_K_POSES = (1, 2, 3, 4, 5, 6, 7, 8)
ZOOM_SEEDS = (1, 2, 42)


# ── Data plumbing (process-global cache, datasource-keyed) ────────────────


_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
    """Load coupled_noisy data deterministic on the YAML fields + seed=0.

    Data is built on CPU (Markov chain loop is faster on CPU for small
    seq_len), then pushed to GPU. The batch_iter does an indexing op on
    the GPU tensor — no per-batch CPU→GPU transfer.
    """
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
            "p_A": float(getattr(spec, "p_A", 0.0)),
            "p_B": float(getattr(spec, "p_B", 1.0)),
            "magnitude_dist": getattr(spec, "magnitude_dist", "folded_normal"),
            "magnitude_mean": float(getattr(spec, "magnitude_mean", 1.0)),
            "magnitude_std":  float(getattr(spec, "magnitude_std", 0.15)),
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "seed": 0,
            "device": device,
        }
        _DATA_CACHE[datasource_name] = coupled_noisy_hmm(**kwargs)
    return _DATA_CACHE[datasource_name]


# ── train_fn / eval_fn factories (closures over datasource_name) ──────────


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

        metrics: dict[str, float] = {
            "eauc": float(recov["auc"]),
            "e_mean_max_cos": float(recov["mean_max_cos"]),
            "e_frac_recovered_90": float(recov["frac_recovered_90"]),
            **{k: float(v) for k, v in glob.items()},
        }
        return metrics, "gauc"
    return my_eval_fn


# ── Sweep entrypoint ──────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasource", required=True,
                    help="One of toy_coupled_noisy_K10_M20_d256_pB*_np*. "
                         "The launcher passes one per GPU.")
    ap.add_argument("--phase", choices=["hunt", "zoom"], default="hunt",
                    help="hunt = 2 archs × 6 k_pos (Phase 1). "
                         "zoom = 6 archs × 12 k_pos (Phase 2).")
    ap.add_argument("--archs", nargs="+", default=None,
                    help="Override arch list (still iterates T-overrides).")
    ap.add_argument("--arch-t-idx", type=int, default=None,
                    help="Run ONLY the entry at this index of the phase's "
                         "ARCH_TS list. Used by the ZOOM launcher to fan "
                         "out (arch_t, seed) tuples across 8 GPUs.")
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--k-poses", nargs="+", type=int, default=None)
    ap.add_argument("--n-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true",
                    help="Tag eval_cfg.smoke=True; analysis drops smoke rows.")
    args = ap.parse_args()

    # Pull datasource metadata for tagging eval rows.
    spec = load_datasource(args.datasource)
    p_B = float(getattr(spec, "p_B", 1.0))
    n_parents = int(spec.n_parents)
    rho = float(spec.rho)

    if args.phase == "hunt":
        arch_ts = HUNT_ARCH_TS
        k_poses = HUNT_K_POSES
        seeds = HUNT_SEEDS
        n_steps = 20_000
    else:
        arch_ts = ZOOM_ARCH_TS
        k_poses = ZOOM_K_POSES
        seeds = ZOOM_SEEDS
        n_steps = 30_000

    if args.archs is not None:
        arch_filter = set(args.archs)
        arch_ts = [(a, t) for (a, t) in arch_ts if a in arch_filter]
    if args.arch_t_idx is not None:
        arch_ts = [arch_ts[int(args.arch_t_idx)]]
    if args.k_poses is not None:
        k_poses = tuple(args.k_poses)
    if args.seeds is not None:
        seeds = tuple(args.seeds)
    if args.n_steps is not None:
        n_steps = int(args.n_steps)

    train_fn = make_train_fn(args.datasource)
    eval_fn = make_eval_fn(args.datasource)

    print(
        f"[hunt] datasource={args.datasource} p_B={p_B} n_parents={n_parents} "
        f"ρ={rho} phase={args.phase} archs={[a for a,_ in arch_ts]} "
        f"k_pos={list(k_poses)} seeds={list(seeds)} n_steps={n_steps}",
        flush=True,
    )

    for arch_name, t_override in arch_ts:
        for k_pos in k_poses:
            override: dict[str, Any] = {"k_pos": int(k_pos)}
            if t_override:
                override.update(t_override)
            cfg = TrainingConfig(
                n_steps=int(n_steps),
                batch_size=int(args.batch_size),
                plateau_early_stop=False,
                arch_hparams_override=override,
            )
            for seed in seeds:
                eval_cfg = {
                    "k_pos": int(k_pos),
                    "smoke": bool(args.smoke),
                    "_arch_hparams_override": override,
                    "rho": float(rho),
                    "p_B": float(p_B),
                    "n_parents": int(n_parents),
                    "hunt_phase": args.phase,
                    "n_steps_train": int(n_steps),
                }
                if t_override:
                    eval_cfg["t_label"] = "T=" + str(
                        t_override.get("T_max") or t_override.get("T")
                    )
                else:
                    eval_cfg["t_label"] = "default"
                t_label = eval_cfg["t_label"]
                print(
                    f"[c2 hunt] {arch_name:12} {t_label:10} k={k_pos:2d} seed={seed} "
                    f"p_B={p_B} n_par={n_parents} ρ={rho} "
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
