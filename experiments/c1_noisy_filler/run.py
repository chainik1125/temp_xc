"""C1-noisy driver — replicate wasteland Phase 2 Experiment 1c noisy emissions.

(decision 2026-05-06) mandate (territory waiver granted): replicate
``docs/legacy/research_logs/2026-03-30-experiment1c-noisy-emissions.md``
on `final` paper infra. Wasteland setup:

  - 20 features, d=40, T=64, ρ=0.7, p_A=0, p_B=0.625 (= γ=0.25)
  - Models: TFA-pos, Stacked T=2,5, TXCDRv2 T=2,5
  - Wasteland headline: TXCDRv2 T=2 hit AUC ≥ 0.98 across k=3..12;
    TXCDRv2 T=5 hit AUC ≈ 0.99 + corr ratio ≈ 1.0 (denoising) at
    k=3..8.

This driver mirrors `experiments/c1_synthetic_topk/run.py` but with:
  - DATASOURCE = "toy_markov_n20_d40_noisy" (Bernoulli p_A/p_B)
  - COMPONENT = "c1_noisy" (separate leaderboard space; doesn't
    pollute the canonical C1 sweep)
  - Same k_pos sweep + arch list (canonical 5: tfa_pos, stacked_sae
    T=2/5, txc_base, txc_pro — the wasteland archs except we use
    txc_base/txc_pro instead of TXCDRv2 since those are the locked
    `final` arch identities for the same architectural family).
  - Eval uses observed support (data.support) for AUC vs
    `data.features` decoder columns. Hidden state (`data.hidden_support`)
    is exposed via MarkovData but not yet consumed by eval — denoising
    metrics (corr ratio, probe ratio) are a follow-up.

Per the prior author's note "with txc_base", txc_base is the primary subject;
the other 4 archs are baselines for cross-arch comparison.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np
import torch

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

from temp_bench import runner  # noqa: E402
from temp_bench.schemas import TrainingConfig  # noqa: E402
from temp_bench.config import instantiate_arch, load_arch, load_datasource  # noqa: E402
from temp_bench.data.toy.markov import markov_chain_support, make_batch_iter  # noqa: E402
from temp_bench.eval.synthetic import feature_recovery  # noqa: E402
from temp_bench.training.sae_trainer import train_sae  # noqa: E402

COMPONENT = "c1_noisy"
DATASOURCE = "toy_markov_n20_d40_noisy"
EVAL_PROTOCOL_VERSION = "1.0.0"


# Wasteland 1c-noisy archs: TFA-pos, Stacked T=2/T=5, TXCDRv2 T=2/T=5.
# In `final` infra, txc_base ≈ TXCDRv2 with the locked anti-dead stack;
# we sweep T={2,5} via arch_hparams_override for both stacked_sae +
# txc_base. Adding txc_pro for completeness against the locked arch
# pair (decisions.md § 1).
ARCH_TS: list[tuple[str, dict[str, Any] | None]] = [
    ("tfa_pos",     None),                 # T=5 default
    ("stacked_sae", {"T": 2}),
    ("stacked_sae", None),                 # T=5 default
    ("txc_base",    {"T": 2}),             # ≈ wasteland TXCDRv2 T=2
    ("txc_base",    None),                 # T=5 default ≈ wasteland TXCDRv2 T=5
    ("txc_pro",     None),                 # T_max=10 t_sample=5 (canonical)
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
            "n_seqs": int(getattr(spec, "n_seqs", 4096)),
            "p_A": float(getattr(spec, "p_A", 0.0)),
            "p_B": float(getattr(spec, "p_B", 1.0)),
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

    metrics: dict[str, float] = {
        "auc": float(recov["auc"]),
        "mean_max_cos": float(recov["mean_max_cos"]),
        "frac_recovered_90": float(recov["frac_recovered_90"]),
        "frac_recovered_80": float(recov["frac_recovered_80"]),
    }
    return metrics, "auc"


# ── Per-arch validity check (mirrors C1's auto-skip) ───────────────────────


def _is_valid_cell(arch_name: str, t_override: dict | None, k_pos: int) -> bool:
    """Skip cells where k_train exceeds the arch budget at d_sae=40."""
    spec = load_arch(arch_name, component=COMPONENT if False else "c1")
    # Reuse C1's per_component_hparams (d_sae=40 etc.) since c1_noisy
    # isn't a registered component in locked_archs.yaml. The arch
    # spec itself is the same; only the data differs.
    hp = spec.hparams
    if t_override:
        hp = {**hp, **t_override}
    d_sae = int(hp.get("d_sae", 40))
    if arch_name in ("topk_sae", "tsae_paper"):
        return k_pos <= d_sae
    if arch_name in ("tfa", "tfa_pos"):
        T = int(hp.get("T", 5))
        return k_pos * T <= d_sae
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
    ap.add_argument("--archs", nargs="+", default=None,
                    help="Restrict to these arch names.")
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
                print(
                    f"[c1_noisy] SKIP invalid cell {arch_name:12} {t_label:10} "
                    f"k={k_pos:2d} (k_train > arch budget at toy scale)",
                    flush=True,
                )
                continue
            # Force toy hparams because c1_noisy is not in
            # locked_archs.yaml's per_component_hparams (only c1/c2/c6/c7
            # are). Without this override, archs default to production
            # d_sae=18432 → OOM at toy n_features=20.
            # Mirror c1's per-arch toy overrides:
            #   - all archs: d_sae=40
            #   - txc_pro:   d_sae=40, h_size=40 (disable matryoshka)
            override: dict[str, Any] = {"k_pos": int(k_pos), "d_sae": 40}
            if arch_name == "txc_pro":
                override["h_size"] = 40
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
                    "_p_A": 0.0,
                    "_p_B": 0.625,
                }
                print(
                    f"[c1_noisy] {arch_name:12} {t_label:10} k={k_pos:2d} "
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
    main()
