"""C3 sparse-probing component runner.

Thin scaffold per ``experiments/_runner_template.py`` (PROTOCOL.md § 11
*Code reuse contract*). All training-loop / probing logic lives in
the shared modules:

- ``temp_bench.training.train_sae``           — canonical SAE trainer
- ``temp_bench.eval.probing.s_tail_probe``    — SAEBench-style probe
- ``temp_bench.data.nlp.probe_cache``         — per-task activation cache
- ``temp_bench.runner.run_cell``              — leaderboard append + caching

Task suite: SAEBench+CT (n=38) — see ``decisions.md`` § 11. Tasks live
on disk at ``results/probe_cache/<datasource_name>/<task_name>/`` after
running ``build_probe_cache``. Each task's eval flattens per-task AUC
into the leaderboard row as ``auc__<task>`` keys (38 floats) AND emits
the headline aggregates ``mean_auc`` / ``std_auc`` / ``mean_acc`` /
``std_acc``. Per-task floats let the analysis script compute σ_tasks;
the headline uses ``mean_auc`` as the primary metric.

Usage::

    # Smoke (1 arch × 1 seed × 1 k_feat × fake labels):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run --smoke

    # Real cells (requires probe cache already built):
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing.run \\
        --archs topk_sae tsae_paper txc_base \\
        --seeds 1 2 42 --k_feats 5 20

To build the probe cache first::

    TQDM_DISABLE=1 .venv/bin/python -c "from temp_bench.data.nlp \\
        import build_probe_cache; build_probe_cache('gemma_2_2b_it_l13_fineweb_24k128')"
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
from temp_bench.data.nlp import list_probe_cache, load_probe_cache
from temp_bench.eval import probing
from temp_bench.schemas import TrainingConfig
from temp_bench.training import train_sae

# ── Per-component constants ─────────────────────────────────────────────────

COMPONENT = "c3"
DATASOURCE = "gemma_2_2b_it_l13_fineweb_24k128"
# 1.1.0 (2026-05-04): Phase 7 padding fix — probe cache rebuilt as left-aligned
# (N, S=32, d_in) with per-example first_real metadata; _encode_pool now masks
# padding contributions per row. Old 1.0.0 cells stay in the leaderboard for
# comparison. See docs/han/research_logs/phase7_unification/2026-04-27-URGENT-probing-cache-fix.md
EVAL_PROTOCOL_VERSION = "1.1.0"

DEFAULT_ARCHS = ("topk_sae", "tsae_paper", "txc_base", "txc_pro")  # mlc still pending
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
    import sys
    import time as _time
    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    model = instantiate_arch(spec, d_in=d_in)

    torch.manual_seed(seed)
    np.random.seed(seed)

    raw_iter = batch_iter_from_act_cache(act_cache_key, seed=seed)

    # Wrap to print progress every 1000 steps. The shared trainer is
    # silent by design (PROTOCOL.md § 11). For long autonomous runs we
    # still want visibility — so we count batch_iter calls and emit a
    # one-line progress marker. Adds zero overhead per step.
    state = {"n": 0, "t0": _time.time(), "label": f"{arch_name}/seed={seed}"}

    def progress_iter(bs):
        state["n"] += 1
        if state["n"] % 1000 == 0:
            elapsed = _time.time() - state["t0"]
            steps = state["n"]
            rate = steps / elapsed if elapsed > 0 else 0
            eta = (training_cfg.n_steps - steps) / rate if rate > 0 else 0
            print(
                f"  [TRAIN {state['label']}] step {steps}/{training_cfg.n_steps}  "
                f"({rate:.1f} steps/sec; eta {eta/60:.1f} min)",
                flush=True,
            )
            sys.stdout.flush()
        return raw_iter(bs)

    result = train_sae(model, progress_iter, training_cfg, device="cuda")
    return result["state_dict"]


def my_eval_fn(*, model, eval_cfg, component):
    """Compose eval.probing primitives — never inline a probe routine.

    Reads ``_state_dict`` + ``_arch_name`` + ``_arch_hparams`` from
    eval_cfg (set by ``runner.run_cell``); rebuilds the model on GPU.

    Eval data:
      - If ``eval_cfg["smoke"]`` is True: synthetic binary labels on
        the FineWeb cache (validates pipeline end-to-end with one
        synthetic AUC).
      - Otherwise: iterate every cached SAEBench+CT task, run
        ``s_tail_probe``, return per-task AUCs as ``auc__<task>``
        floats plus aggregates (``mean_auc``, ``std_auc``, ``mean_acc``,
        ``std_acc``). Primary metric is ``mean_auc``.

    Per-task floats are kept on the leaderboard row so analysis can
    compute σ_tasks; the headline uses the aggregate.
    """
    del model  # we re-instantiate; the runner doesn't pre-load
    arch_name = eval_cfg["_arch_name"]
    state_dict = eval_cfg["_state_dict"]
    act_cache_key = eval_cfg["_act_cache_key"]
    datasource_name = eval_cfg["_datasource_name"]
    k_feat = int(eval_cfg["k_feat"])
    S = int(eval_cfg.get("S", DEFAULT_S))
    smoke = bool(eval_cfg.get("smoke", False))

    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)
    m = instantiate_arch(spec, d_in=d_in).cuda().eval()
    m.load_state_dict(state_dict)

    if smoke:
        X_train, y_train, X_test, y_test = _smoke_probe_data(act_cache_key, S=S)
        metrics = probing.s_tail_probe(
            m,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            S=S, k_feat=k_feat,
        )
        return metrics, "auc"

    # ── Real eval: iterate cached SAEBench+CT tasks
    task_names = list_probe_cache(datasource_name)
    if not task_names:
        raise FileNotFoundError(
            f"No probe cache found for datasource {datasource_name!r}. "
            f"Run build_probe_cache(...) first."
        )

    aucs: list[float] = []
    accs: list[float] = []
    per_task_metrics: dict[str, float] = {}
    for tname in task_names:
        task = load_probe_cache(datasource_name, tname)
        r = probing.s_tail_probe(
            m,
            X_train=task["X_train"], y_train=task["y_train"],
            X_test=task["X_test"], y_test=task["y_test"],
            first_real_train=task["first_real_train"],
            first_real_test=task["first_real_test"],
            S=S, k_feat=k_feat,
        )
        per_task_metrics[f"auc__{tname}"] = float(r["auc"])
        per_task_metrics[f"acc__{tname}"] = float(r["acc"])
        aucs.append(float(r["auc"]))
        accs.append(float(r["acc"]))

    aucs_arr = np.asarray(aucs, dtype=np.float64)
    accs_arr = np.asarray(accs, dtype=np.float64)
    metrics = {
        "mean_auc": float(aucs_arr.mean()),
        "std_auc": float(aucs_arr.std(ddof=1)) if len(aucs) > 1 else 0.0,
        "mean_acc": float(accs_arr.mean()),
        "std_acc": float(accs_arr.std(ddof=1)) if len(accs) > 1 else 0.0,
        "n_tasks": float(len(aucs)),
        **per_task_metrics,
    }
    return metrics, "mean_auc"


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


def _real_training_cfg() -> TrainingConfig:
    """Headline-cell TrainingConfig.

    Decision (2026-05-03, agent_nlp autonomous): n_steps=10_000 fits 18
    cells in the 10-hour autonomous window (~25-30 min/cell at H100
    bs=256). Phase 7 reference numbers came from longer runs (~50K),
    but SAE convergence at this token budget (10K × 256 × 128 = 328M
    tokens) is reliable per the temporal_sae paper. If results undershoot
    the Phase 7 leaderboard, re-run with n_steps=30_000 (uses schema
    default; will trigger train-key invalidation since steps is part of
    train_key).
    """
    return TrainingConfig(
        n_steps=10_000,
        batch_size=256,
        learning_rate=3e-4,
        warmup_steps=500,
        precision="bf16",
    )


def main(*, archs, seeds, k_feats, S, smoke, force_train=False, force_eval=False):
    if smoke:
        training_cfg = TrainingConfig(
            n_steps=SMOKE_TRAIN_STEPS,
            batch_size=SMOKE_BATCH,
            learning_rate=3e-4,
            warmup_steps=20,
            precision="bf16",
        )
    else:
        training_cfg = _real_training_cfg()

    from temp_bench.config import compute_act_cache_key, load_datasource
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))

    for arch in archs:
        for seed in seeds:
            for k in k_feats:
                eval_cfg = {
                    "k_feat": k,
                    "S": S,
                    "smoke": smoke,
                    # Inject act_cache_key + datasource_name so eval_fn can
                    # look up d_in and find the probe cache. The runner
                    # already injects _state_dict + _arch_name + _arch_hparams.
                    "_act_cache_key": act_cache_key,
                    "_datasource_name": DATASOURCE,
                }

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
                # Show the headline metric. For smoke runs, that's "auc";
                # for real runs, "mean_auc" + "std_auc" + n_tasks.
                if smoke:
                    auc = m.get("auc")
                    acc = m.get("acc")
                    auc_s = f"{auc:.4f}" if auc is not None else "-"
                    acc_s = f"{acc:.4f}" if acc is not None else "-"
                    print(f"[{tag}] {arch} seed={seed} k_feat={k}  "
                          f"AUC={auc_s}  acc={acc_s}  eval_key={result.eval_key}")
                else:
                    mean_auc = m.get("mean_auc")
                    std_auc = m.get("std_auc")
                    n_tasks = m.get("n_tasks")
                    if mean_auc is not None:
                        print(f"[{tag}] {arch} seed={seed} k_feat={k}  "
                              f"mean_AUC={mean_auc:.4f}±{std_auc:.4f} "
                              f"(n={int(n_tasks or 0)} tasks)  "
                              f"eval_key={result.eval_key}")
                    else:
                        print(f"[{tag}] {arch} seed={seed} k_feat={k}  "
                              f"eval_key={result.eval_key}")


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
