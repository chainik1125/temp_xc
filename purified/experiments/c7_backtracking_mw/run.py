"""C7 multi-window deployment driver.

Replicates ``experiments/c7_backtracking/`` with two changes:

1. **arch_name**: ``txc_base`` / ``txc_pro`` → ``txc_base_mw`` /
   ``txc_pro_mw`` (YAML aliases of TXCBase / TXCPro with
   ``multi_window: true`` baked into hparams, agent_paper commit
   ``ecc4c661``, decisions § 14).
2. **batch_iter**: returns FULL sequences ``(B, seq_len, d_in)`` from
   the activation cache. The MW ``train_step`` tiles internally into
   ``(B*N, T, d_in)`` non-overlapping windows where ``N = seq_len // T``.
   The canonical pre-windowed ``(B, T, d_in)`` batch_iter would defeat
   MW (it'd give ``N = 1``).

Eval pipeline is UNCHANGED. The MW cells reuse ``my_eval_fn`` from
``experiments.c7_backtracking.run`` — Stage A traces, mining,
magnitude grid, Sonnet judge, Δgc + PR-AUC. Train_keys are fresh
because ``arch_name`` is in the train_key hash; no cache collision
with agent_back's canonical cells.

Pattern adapted from agent_em's ``experiments/c6_em_mw/run.py``
(commit 03facd49) — same full-seq batch_iter approach, scaled to
Llama (d_in=4096, d_sae=32768).

Thread cap: inherits the H100-pod profiling from C5 MW (commit
e7b229fd, OQ #1) — OMP/MKL/torch threads=32 mitigates random
fancy indexing cache thrashing on the activation cache. Pure perf
change, bit-identical math.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "32")
os.environ.setdefault("MKL_NUM_THREADS", "32")

import numpy as np
import torch  # noqa: E402

_TORCH_THREADS = int(os.environ.get("TORCH_NUM_THREADS", os.environ["OMP_NUM_THREADS"]))
torch.set_num_threads(_TORCH_THREADS)
torch.set_num_interop_threads(_TORCH_THREADS)

from temp_bench import runner
from temp_bench.config import (
    act_cache_dir, instantiate_arch, load_arch, load_datasource,
)
from temp_bench.schemas import TrainingConfig

# Re-use agent_back's plumbing where possible:
from experiments.c7_backtracking.run import (
    DATASOURCE,
    EVAL_PROTOCOL_VERSION,
    my_eval_fn,                # eval is unchanged for MW
)

log = logging.getLogger("c7.mw.run")


COMPONENT = "c7"
DEFAULT_ARCHS: tuple[str, ...] = ("txc_base_mw", "txc_pro_mw")
DEFAULT_SEEDS: tuple[int, ...] = (42,)


# ── Full-sequence preload cache (process-local) ─────────────────────


# Distinct from c7_backtracking._PRELOADED_C7_ACTS because MW returns
# full sequences while the canonical preload returns the same memory
# but accesses it via T-window slicing. Sharing the dict isn't unsafe
# but would make hits ambiguous; keep them separate for clarity.
_PRELOADED_C7_FULL: dict[str, torch.Tensor] = {}


def _build_full_seq_batch_iter(
    act_cache_key: str,
    *,
    seed: int = 42,
):
    """Return a batch_iter callable ``(n) -> Tensor (n, seq_len, d_in)``.

    Reads from the C7 activation cache (built by agent_back's pipeline,
    file ``resid_post_L10.npy`` per the layer_specs ``key`` field).
    Samples full sequences with replacement across cached rows. The MW
    train_step tiles each sequence into ``N = seq_len // T``
    non-overlapping T-windows internally.

    Preloads the cache into CPU RAM via .clone() once per process per
    cache_path (same pattern as agent_back's _PRELOADED_C7_ACTS).
    """
    cache_dir = act_cache_dir(act_cache_key)
    cache_path = str(cache_dir / "resid_post_L10.npy")

    if cache_path not in _PRELOADED_C7_FULL:
        log.info("[c7.mw.run] preloading acts cache %s into CPU RAM…", cache_path)
        mmapped = np.load(cache_path, mmap_mode="r")
        _PRELOADED_C7_FULL[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
        t = _PRELOADED_C7_FULL[cache_path]
        log.info(
            "[c7.mw.run] preload done: shape=%s dtype=%s ~%.2f GB",
            tuple(t.shape), t.dtype,
            t.element_size() * t.nelement() / 1e9,
        )
    acts = _PRELOADED_C7_FULL[cache_path]
    N, L, d = acts.shape
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        idx = rng.integers(0, N, size=n)
        out = torch.empty((n, L, d), dtype=torch.float32)
        for i in range(n):
            out[i] = acts[int(idx[i])].to(torch.float32)
        return out

    return batch_iter


# ── train_fn: full-sequence batch_iter + bf16 for >1B archs ─────────


def my_train_fn_mw(
    *,
    arch_name: str,
    arch_hparams: dict[str, Any],
    seed: int,
    training_cfg: TrainingConfig,
    act_cache_key: str,
    component: str,
) -> dict[str, Any]:
    """C7 MW train adapter — analogous to ``c7_backtracking.run.my_train_fn``
    with the full-sequence batch_iter swapped in.

    Preserves the bf16 cast for >1B-param archs (txc_base / txc_pro at
    Llama d_in=4096, d_sae=32768) so opt-state fits the H100 cleanly.
    """
    from temp_bench.training.sae_trainer import train_sae

    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])

    log.info("[c7.mw.run] arch=%s seed=%d d_in=%d", arch_name, seed, d_in)

    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = load_arch(arch_name, component=component)
    model = instantiate_arch(spec, d_in=d_in)
    if torch.cuda.is_available():
        model = model.cuda()
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1e9 and torch.cuda.is_available():
        model = model.bfloat16()
        log.info(
            "[c7.mw.run] bf16 cast (%.1fM params; opt-state fits H100 with margin)",
            n_params / 1e6,
        )

    if not getattr(model, "_multi_window", False):
        raise RuntimeError(
            f"Expected multi_window=True on {arch_name!r}; got "
            f"{getattr(model, '_multi_window', None)!r}. Check the "
            "YAML alias in configs/locked_archs.yaml."
        )

    batch_iter = _build_full_seq_batch_iter(act_cache_key, seed=seed)

    result = train_sae(model, batch_iter, training_cfg)
    log.info(
        "[c7.mw.run] done in %d steps; final loss=%.4f",
        result["n_steps"], result["log"]["loss"][-1],
    )

    # Persist train_log alongside agent_back's c7 trainlogs (same naming).
    try:
        log_path = Path("logs") / (
            f"c7_b{training_cfg.batch_size}_{arch_name}_seed{seed}_trainlog.json"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(json.dumps(result.get("log", {})))
        log.info("[c7.mw.run] trainlog saved → %s", log_path)
    except Exception as exc:                                  # noqa: BLE001
        log.warning("[c7.mw.run] trainlog persist failed: %s", exc)

    return result["state_dict"]


# ── Pipeline orchestration ─────────────────────────────────────────


def _run_one_cell(
    arch_name: str,
    *,
    seed: int,
    n_steps: int | None = None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = TrainingConfig(
        batch_size=1024,
        n_steps=20_000 if n_steps is None else n_steps,
        plateau_early_stop=False,
    )
    eval_cfg = {
        "magnitudes": [
            -16, -12, -10, -8, -7, -6, -5, -4, -3, -2, -1, -0.5,
            0, 0.5, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16,
        ],
        "cut_fraction": 0.25,
        "pr_auc_S_grid": [1, 2, 4, 8, 16, 32],
    }
    log.info(
        "[c7.mw.run] CELL: arch=%s seed=%d n_steps=%d eval_protocol=%s",
        arch_name, seed, cfg.n_steps, EVAL_PROTOCOL_VERSION,
    )
    result = runner.run_cell(
        component=COMPONENT,
        arch_name=arch_name,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn_mw,
        eval_fn=my_eval_fn,
        force_train=force_train,
        force_eval=force_eval,
    )
    log.info(
        "[c7.mw.run] CELL DONE: train_key=%s eval_key=%s cached=%s",
        result.train_key, result.eval_key, result.cached,
    )
    return result


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(
        description="C7 multi-window deployment — txc_base_mw + txc_pro_mw "
                    "× seed=42 helper for agent_back."
    )
    p.add_argument("--archs", nargs="+",
                   default=list(DEFAULT_ARCHS),
                   choices=list(DEFAULT_ARCHS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override TrainingConfig.n_steps (default 20000).")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    print(
        f"[c7_mw] sweep archs={args.archs} seeds={args.seeds} "
        f"n_steps={args.n_steps or 20000} eval_protocol={EVAL_PROTOCOL_VERSION}",
        flush=True,
    )

    for arch in args.archs:
        for seed in args.seeds:
            print(
                f"[c7_mw] launching cell arch={arch} seed={seed}",
                flush=True,
            )
            _run_one_cell(
                arch, seed=seed, n_steps=args.n_steps,
                force_train=args.force_train, force_eval=args.force_eval,
            )

    log.info("[c7.mw.run] all cells complete")


if __name__ == "__main__":
    import sys
    sys.exit(main())
