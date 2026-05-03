"""C3 sparse-probing component runner.

Thin scaffold per ``experiments/_runner_template.py`` (PROTOCOL.md § 11
*Code reuse contract*). All training-loop / probing logic lives in
the shared modules:

- ``temp_bench.training.train_sae``       — canonical SAE trainer
- ``temp_bench.eval.probing.run_task_suite`` — SAEBench-style probe
- ``temp_bench.runner.run_cell``         — leaderboard append + caching

Task suite: SAEBench+CT (n=38) — see ``decisions.md`` § 11. Tasks are
pre-cached to ``results/probe_cache/<datasource_name>/<task_name>/``
by ``cache_probe_tasks.py`` (TODO — not yet ported). Until that
pipeline lands, this runner supports a ``--smoke`` mode that uses
fake binary labels on the FineWeb cache to validate the full
train→encode→probe pipeline end-to-end with one cell.

Usage::

    # Smoke (1 arch × 1 seed × 1 k_feat × fake labels):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run --smoke

    # Real (when probe cache lands):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run \\
        --archs topk_sae tsae_paper txc_base \\
        --seeds 1 2 42 --k_feats 5 20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from temp_bench import runner
from temp_bench.config import act_cache_dir, instantiate_arch, load_arch
from temp_bench.data import batch_iter_from_act_cache
from temp_bench.eval import probing
from temp_bench.schemas import TrainingConfig
from temp_bench.training import train_sae

# ── Per-component constants ─────────────────────────────────────────────────

COMPONENT = "c3"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
EVAL_PROTOCOL_VERSION = "1.0.0"

DEFAULT_ARCHS = ("topk_sae", "tsae_paper", "txc_base")  # 3 ported; mlc/txc_pro pending
DEFAULT_SEEDS = (1, 2, 42)
DEFAULT_K_FEATS = (5, 20)
DEFAULT_S = 32

# Smoke mode trains a tiny SAE (200 steps) and probes it against fake
# binary labels derived from the FineWeb cache itself.
SMOKE_TRAIN_STEPS = 200
SMOKE_BATCH = 64
SMOKE_N_PROBE_TRAIN = 200
SMOKE_N_PROBE_TEST = 80


# ── Train + eval adapters (called by runner.run_cell) ──────────────────────


def _d_in_from_act_cache(act_cache_key: str) -> int:
    meta = json.loads((act_cache_dir(act_cache_key) / "meta.json").read_text())
    return int(meta["d_in"])


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """Build the arch + train via the shared trainer.

    PROTOCOL.md § 11: never write a training loop here.
    """
    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    model = instantiate_arch(spec, d_in=d_in)

    torch.manual_seed(seed)
    np.random.seed(seed)

    batch_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)
    result = train_sae(model, batch_iter, training_cfg, device="cuda")
    return result["state_dict"]


def my_eval_fn(*, model, eval_cfg, component):
    """Compose eval.probing primitives — never inline a probe routine.

    Reads ``_state_dict`` + ``_arch_name`` + ``_arch_hparams`` from
    eval_cfg (set by ``runner.run_cell``); rebuilds the model on GPU.

    Eval data:
      - If ``eval_cfg["smoke"]`` is True: synthetic binary labels on
        the FineWeb cache (validates pipeline end-to-end).
      - Otherwise: load probe-cache for this datasource (TODO — not
        yet implemented; raises until the probe-cache pipeline lands).
    """
    del model  # we re-instantiate; the runner doesn't pre-load
    arch_name = eval_cfg["_arch_name"]
    state_dict = eval_cfg["_state_dict"]
    act_cache_key = eval_cfg["_act_cache_key"]
    k_feat = int(eval_cfg["k_feat"])
    S = int(eval_cfg.get("S", DEFAULT_S))
    smoke = bool(eval_cfg.get("smoke", False))

    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    m = instantiate_arch(spec, d_in=d_in).cuda().eval()
    m.load_state_dict(state_dict)

    if smoke:
        X_train, y_train, X_test, y_test = _smoke_probe_data(act_cache_key, S=S)
    else:
        raise NotImplementedError(
            "Real probe cache not yet built. Pass smoke=True in eval_cfg "
            "to validate the pipeline, or wait on the probe-cache port "
            "(probe_datasets.py + cache_probe_tasks.py)."
        )

    metrics = probing.s_tail_probe(
        m,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        S=S, k_feat=k_feat,
    )
    return metrics, "auc"


def _smoke_probe_data(act_cache_key: str, *, S: int):
    """Synthetic binary labels: split FineWeb seqs into two arbitrary halves
    after lightly perturbing one half. Validates the pipeline only — AUC is
    not meaningful for paper claims.
    """
    acts = np.load(act_cache_dir(act_cache_key) / "acts.npy", mmap_mode="r")
    rng = np.random.default_rng(0)

    # Sample N_train + N_test sequences, label first half = 1.
    N = SMOKE_N_PROBE_TRAIN + SMOKE_N_PROBE_TEST
    idx = rng.choice(acts.shape[0], size=N, replace=False)
    X = np.ascontiguousarray(acts[idx]).astype(np.float32)  # (N, seq_len, d_in)
    y = np.zeros(N, dtype=np.int64)
    y[: N // 2] = 1
    # Add a small shift to the positive class on a few hidden dims so
    # AUC > 0.5 if the encode + probe pipeline is sane.
    X[y == 1, :, :8] += 0.5
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    return (
        X[: SMOKE_N_PROBE_TRAIN], y[: SMOKE_N_PROBE_TRAIN],
        X[SMOKE_N_PROBE_TRAIN:], y[SMOKE_N_PROBE_TRAIN:],
    )


# ── Runner ─────────────────────────────────────────────────────────────────


def main(*, archs, seeds, k_feats, S, smoke, force_train=False, force_eval=False):
    if smoke:
        # Tiny training cfg for validation
        training_cfg = TrainingConfig(
            n_steps=SMOKE_TRAIN_STEPS,
            batch_size=SMOKE_BATCH,
            learning_rate=3e-4,
            warmup_steps=20,
            precision="bf16",
        )
    else:
        training_cfg = runner.default_training_cfg("topk_sae")

    for arch in archs:
        for seed in seeds:
            for k in k_feats:
                eval_cfg = {"k_feat": k, "S": S, "smoke": smoke}
                # Inject act_cache_key into eval_cfg so eval_fn can read meta.
                # runner.run_cell already injects _state_dict + _arch_name,
                # but we also need the cache key for d_in lookup.
                from temp_bench.config import compute_act_cache_key, load_datasource
                eval_cfg["_act_cache_key"] = compute_act_cache_key(load_datasource(DATASOURCE))

                result = runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    training_cfg=training_cfg,
                    eval_cfg=eval_cfg,
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                    force_train=force_train,
                    force_eval=force_eval,
                )
                tag = "CACHED" if result.cached else "NEW"
                m = result.metrics or {}
                auc = m.get("auc")
                acc = m.get("acc")
                auc_s = f"{auc:.4f}" if auc is not None else "-"
                acc_s = f"{acc:.4f}" if acc is not None else "-"
                print(f"[{tag}] {arch} seed={seed} k_feat={k}  "
                      f"AUC={auc_s}  acc={acc_s}  eval_key={result.eval_key}")


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=list(DEFAULT_ARCHS))
    ap.add_argument("--seeds", type=int, nargs="*", default=list(DEFAULT_SEEDS))
    ap.add_argument("--k_feats", type=int, nargs="*", default=list(DEFAULT_K_FEATS))
    ap.add_argument("--S", type=int, default=DEFAULT_S)
    ap.add_argument("--smoke", action="store_true",
                    help="Run a 1-cell smoke test with fake probe labels.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        # Smoke = 1 arch × 1 seed × 1 k_feat for fast iteration.
        archs = args.archs[:1]
        seeds = args.seeds[:1]
        k_feats = args.k_feats[:1]
    else:
        archs, seeds, k_feats = args.archs, args.seeds, args.k_feats

    main(
        archs=archs, seeds=seeds, k_feats=k_feats, S=args.S,
        smoke=args.smoke,
        force_train=args.force_train, force_eval=args.force_eval,
    )


if __name__ == "__main__":
    cli()
