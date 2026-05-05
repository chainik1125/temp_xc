"""C2 driver — coupled-feature gAUC sweep.

Sweep dimensions:
  - arch ∈ {topk_sae, stacked_sae, txc_base, txc_pro}
  - T (window size) — varied via ``arch_hparams_override``:
      stacked_sae: {T=2, T=5}
      txc_base:    {T=5}             (canonical only)
      txc_pro:     {T_max=2, 5, 12}  (T-modulation sweep)
      topk_sae:    no T (per-token)
  - k_pos ∈ {1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20} — per-cell override.
  - seed ∈ {1, 2, 42}.

Per-cell metrics (`temp_bench.eval.synthetic`):
  - reconstruction_nmse
  - feature_recovery (eAUC, mean_max_cos vs M=20 emission features)
  - global_recovery_gAUC (gAUC vs K=10 hidden features) — paper headline.

Data is generated ONCE per process via ``coupled_hmm`` (deterministic
on datasource fields + a fixed data_seed=0); all cells share the same
underlying coupling matrix + emission/hidden feature directions, only
the SAE init seed and k_pos vary. Per-cell training is fast (~30 sec
on A40 at d_in=256, d_sae=40).
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
from temp_bench.data.toy.coupled import coupled_hmm, make_batch_iter
from temp_bench.eval.synthetic import (
    feature_recovery,
    global_recovery_gAUC,
)
from temp_bench.training.sae_trainer import train_sae

COMPONENT = "c2"
DATASOURCE = "toy_coupled_K10_M20_d256"
EVAL_PROTOCOL_VERSION = "1.0.0"


# ── Sweep configuration ────────────────────────────────────────────────────

# (arch_name, T-override) combinations. arch_hparams_override is merged
# into arch.hparams by runner.run_cell before compute_train_key.
#
# TXC-pro T-modulation sweep: vary T_max ∈ {2, 5, 12}; hold t_sample=2
# constant. Rationale: at C2's toy d_sae=40 + h_size=40, the matryoshka
# topk constraint is k_train = k_pos × t_sample ≤ h_size. Full k-sweep
# (k_pos up to 20) at t_sample=12 requires d_sae ≥ 240, breaking d_sae
# uniformity vs SAE baselines (d_sae=40). t_sample=2 holds encoder load
# constant across T_max → cleaner T-modulation comparison; the cross-
# arch d_sae stays at 40. Document in c2.md caveats.
ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("topk_sae",    None),                          # no T axis (per-token)
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),                          # T=5 default
    ("txc_base",    None),                          # T=5 default (canonical)
    ("txc_pro",     {"T_max": 2,  "t_sample": 2}),  # T=2
    ("txc_pro",     {"T_max": 5,  "t_sample": 2}),  # T=5 (override default T_max=10)
    ("txc_pro",     {"T_max": 12, "t_sample": 2}),  # T=12
]

DEFAULT_K_POSES = (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 17, 20)
DEFAULT_SEEDS = (1, 2, 42)


# ── Data plumbing (process-global cache) ────────────────────────────────────

_DATA_CACHE: dict[str, Any] = {}


def _get_data(datasource_name: str) -> Any:
    """Generate coupled HMM data on first call, cache by datasource_name.

    Deterministic on the datasource YAML fields. data_seed=0 is fixed
    per component (all cells share the same underlying coupling, only
    SAE init seed varies — what we want for cross-arch comparison).
    """
    if datasource_name not in _DATA_CACHE:
        spec = load_datasource(datasource_name)
        # Pull generator hparams from the YAML (extra="allow").
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


# ── train_fn / eval_fn ─────────────────────────────────────────────────────


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg,
                act_cache_key, component):
    data = _get_data(DATASOURCE)
    spec = load_arch(arch_name, component=component)
    # Apply per-cell override on top of YAML+per_component (runner already
    # did this merge for hparams used in train_key; re-apply locally for
    # instantiation, identical merge math).
    if training_cfg.arch_hparams_override:
        merged = {**spec.hparams, **training_cfg.arch_hparams_override}
        spec = spec.model_copy(update={"hparams": merged})
    d_in = data.x.shape[-1]
    model = instantiate_arch(spec, d_in=d_in)

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    # Synthetic data is fast; sample on CPU and the trainer moves to GPU.
    raw_iter = make_batch_iter(data, seed=int(seed))
    result = train_sae(model, raw_iter, training_cfg, device="cuda")
    return result["state_dict"]


def my_eval_fn(*, model=None, eval_cfg, component):
    arch_name = eval_cfg["_arch_name"]
    state = eval_cfg["_state_dict"]
    overrides = eval_cfg.get("_arch_hparams_override")  # may be None

    data = _get_data(DATASOURCE)
    spec = load_arch(arch_name, component=component)
    if overrides:
        merged = {**spec.hparams, **overrides}
        spec = spec.model_copy(update={"hparams": merged})
    d_in = data.x.shape[-1]
    model = instantiate_arch(spec, d_in=d_in).to("cuda").eval()
    model.load_state_dict(state, strict=True)

    # C2 headline: eAUC (vs M=20 emission features) + gAUC (vs K=10
    # hidden features). Both via decoder-direction analysis, paper-
    # faithful per Phase 3. NMSE is not in the C2 metric set (it's C1's
    # axis); skipped here.
    decoder = model.decoder_directions().detach().cpu()  # (d_sae, d_in)
    recov = feature_recovery(decoder, data.emission_features.cpu())
    glob = global_recovery_gAUC(decoder, data.hidden_features.cpu())

    metrics: dict[str, float] = {
        "eauc": float(recov["auc"]),
        "e_mean_max_cos": float(recov["mean_max_cos"]),
        "e_frac_recovered_90": float(recov["frac_recovered_90"]),
        **{k: float(v) for k, v in glob.items()},
    }
    return metrics, "gauc"


# ── Sweep entrypoint ───────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None,
                    help="If set, restrict to these arch names (still "
                         "iterates over all T-overrides for them).")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    ap.add_argument("--k-poses", nargs="+", type=int, default=list(DEFAULT_K_POSES))
    ap.add_argument("--n-steps", type=int, default=30_000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--smoke", action="store_true",
                    help="Tag eval_cfg.smoke=True; agent_paper analysis "
                         "drops smoke rows from the headline.")
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
                }
                if t_override:
                    eval_cfg["t_label"] = "T=" + str(
                        t_override.get("T_max") or t_override.get("T")
                    )
                else:
                    eval_cfg["t_label"] = "default"
                t_label = eval_cfg["t_label"]
                print(
                    f"[c2] {arch_name:12} {t_label:10} k={k_pos:2d} seed={seed} "
                    f"steps={cfg.n_steps}",
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
