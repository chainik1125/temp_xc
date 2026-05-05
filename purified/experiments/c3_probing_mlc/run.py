"""C3 MLC baseline at L=5 (decisions § 16, paper-faithful).

Trains the multi-layer crosscoder (`mlc` arch) on the L11-L15 stacked
Gemma-2-2b-IT activation cache, then evaluates via SAEBench+CT s-tail
probing on the multi-layer probe_cache.

Custom `my_train_fn_mlc` uses
`temp_bench.data.nlp.cache.preloaded_batch_iter_from_multilayer_cache`
which yields (B, L, d_in) batches matching MLC's
`encode(x: (B, L, d_in)) → (B, 1, d_sae)` signature.

Custom `my_eval_fn_mlc` adapts agent_nlp's s-tail probing pipeline
(`temp_bench.eval.probing.mean_pool_probe`) for 4D probe arrays
`(N, L=5, S=32, d_in)`: encode each (S, L, d_in) frame per-position,
mean-pool with first_real masking, then run the existing top-k +
logistic probe on (N, d_sae) features.

agent_nlp owns the per-token archs (topk_sae, tsae_paper, txc_*) at
T=1 / T=2; MLC is *this* driver's responsibility.

    cd /workspace/temp_xc/purified
    source scripts/set_agent_env.sh agent_em_100k
    TQDM_DISABLE=1 .venv/bin/python -m experiments.c3_probing_mlc.run \\
        --seeds 42 1 2 --k-feats 5 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np
import torch

from experiments.c3_probing.run import (
    COMPONENT,
    DEFAULT_K_FEATS,
    DEFAULT_S,
    EVAL_PROTOCOL_VERSION,
)
from temp_bench import runner
from temp_bench.config import (
    act_cache_dir,
    compute_act_cache_key,
    instantiate_arch,
    load_arch,
    load_datasource,
)
from temp_bench.data.nlp.cache import preloaded_batch_iter_from_multilayer_cache
from temp_bench.data.nlp.probe_cache import list_probe_cache, load_probe_cache
from temp_bench.eval import probing
from temp_bench.schemas import TrainingConfig
from temp_bench.training.sae_trainer import train_sae

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
log = logging.getLogger("c3_mlc.run")

ARCH = "mlc"
DATASOURCE = "gemma_2_2b_it_l11to15_fineweb_24k128"
DEFAULT_SEEDS = (42, 1, 2)

# Per decisions § 16: canonical training config; n_steps mirrors
# agent_nlp's `_real_training_cfg()` for fair comparison with their
# C3 sweep. MLC's L axis comes from the multi-layer datasource, NOT
# from `train_window_size` (kept None).
MLC_TRAINING_CFG = TrainingConfig(
    n_steps=20_000,
    batch_size=1024,
    plateau_early_stop=False,
)


# ── train_fn ──────────────────────────────────────────────────────────


def _d_in_from_act_cache(act_cache_key: str) -> int:
    meta = json.loads((act_cache_dir(act_cache_key) / "meta.json").read_text())
    return int(meta["d_in"])


def my_train_fn_mlc(
    *,
    arch_name: str,
    arch_hparams: dict,
    seed: int,
    training_cfg: TrainingConfig,
    act_cache_key: str,
    component: str,
):
    """MLC training adapter — uses preloaded_batch_iter_from_multilayer_cache.

    The batch_iter yields (B, L, d_in) batches; MLC.train_step accepts
    (B, L, d_in) and returns (loss, info) per the TempBenchArch contract.
    """
    spec = load_arch(arch_name, component=component)
    d_in = _d_in_from_act_cache(act_cache_key)

    log.info("[c3_mlc.train] arch=%s seed=%d d_in=%d", arch_name, seed, d_in)
    log.info("[c3_mlc.train] training_cfg=%s",
             training_cfg.model_dump(exclude_none=True))

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = instantiate_arch(spec, d_in=d_in)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_iter = preloaded_batch_iter_from_multilayer_cache(
        act_cache_key, seed=seed,
    )
    result = train_sae(
        model, batch_iter, training_cfg,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    log.info("[c3_mlc.train] done in %d steps; final loss=%.4f",
             result["n_steps"], result["log"]["loss"][-1])
    return result["state_dict"]


# ── eval_fn ───────────────────────────────────────────────────────────


def _encode_pool_mlc(
    model,
    X: np.ndarray,
    *,
    S: int,
    batch_size: int,
    device: torch.device,
    first_real: np.ndarray | None = None,
) -> np.ndarray:
    """Encode (N, L, S_cache, d_in) → (N, d_sae) via S-tail mean-pool.

    Mirrors `temp_bench.eval.probing._encode_pool` for 4D probe arrays.
    For each example, takes the last-S frame across all L layers,
    iterates positions s ∈ [0, S), encodes (B, L, d_in) → (B, 1, d_sae)
    via MLC.encode (squeeze T-axis), and mean-pools across real
    positions (left-aligned with `first_real` per row).
    """
    N, L, S_cache, d_in = X.shape
    if S_cache < S:
        raise ValueError(f"S_cache={S_cache} < S={S}")
    tail = X[:, :, -S:, :]  # (N, L, S, d_in)
    if first_real is not None:
        first_real = np.asarray(first_real, dtype=np.int64).clip(min=0, max=S)

    out: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = torch.from_numpy(np.ascontiguousarray(tail[start:end])).to(device)
            # (B, L, S, d_in) → (B, S, L, d_in) so contiguous per-position slices
            batch_psn = batch.permute(0, 2, 1, 3).contiguous()
            B = batch_psn.shape[0]
            flat = batch_psn.reshape(B * S, L, d_in)
            z = model.encode(flat)               # (B*S, 1, d_sae)
            z = z.squeeze(1).reshape(B, S, -1)   # (B, S, d_sae)
            if first_real is None:
                pooled = z.mean(dim=1)
            else:
                fr = torch.from_numpy(first_real[start:end]).to(device)
                k_grid = torch.arange(S, device=device).unsqueeze(0)  # (1, S)
                mask = (k_grid >= fr.unsqueeze(1)).to(z.dtype)        # (B, S)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _s_tail_probe_mlc(
    model,
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    S: int,
    k_feat: int,
    first_real_train: np.ndarray | None = None,
    first_real_test: np.ndarray | None = None,
    encode_batch_size: int = 64,
    device: torch.device | None = None,
) -> dict[str, float]:
    """MLC analogue of `probing.s_tail_probe`.

    Expects 4D X arrays (N, L, S_cache, d_in). Encodes via
    `_encode_pool_mlc` to (N, d_sae), then delegates to the standard
    `mean_pool_probe` for top-k + logistic regression.
    """
    if X_train.ndim != 4 or X_test.ndim != 4:
        raise ValueError(
            f"_s_tail_probe_mlc expects (N, L, S_cache, d_in); "
            f"got train={X_train.shape}, test={X_test.shape}"
        )
    if X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"L mismatch: train={X_train.shape[1]} vs test={X_test.shape[1]}"
        )

    device = (
        torch.device(device)
        if device is not None
        else next(model.parameters()).device
    )

    train_feats = _encode_pool_mlc(
        model, X_train, S=S, batch_size=encode_batch_size,
        device=device, first_real=first_real_train,
    )
    test_feats = _encode_pool_mlc(
        model, X_test, S=S, batch_size=encode_batch_size,
        device=device, first_real=first_real_test,
    )
    return probing.mean_pool_probe(
        model,
        X_train=train_feats, y_train=y_train,
        X_test=test_feats, y_test=y_test,
        k_feat=k_feat,
    )


def my_eval_fn_mlc(*, model, eval_cfg, component):
    """MLC eval — multi-layer s-tail probe across SAEBench+CT tasks.

    Mirrors agent_nlp's `experiments/c3_probing/run.py:my_eval_fn`
    structure (smoke-or-real, per-task floats + aggregates) but
    swaps the encode helper for the multi-layer variant.
    """
    del model
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

    log.info("[c3_mlc.eval] arch=%s d_in=%d k_feat=%d S=%d smoke=%s "
             "datasource=%s",
             arch_name, d_in, k_feat, S, smoke, datasource_name)

    if smoke:
        # Synthetic binary labels on the multi-layer act_cache itself —
        # validates the eval pipeline end-to-end without needing the
        # full probe_cache. Mirrors agent_nlp's `_smoke_probe_data`
        # adapted for the multi-layer shape.
        from temp_bench.config import act_cache_dir as _acd
        meta = json.loads((_acd(act_cache_key) / "meta.json").read_text())
        acts_path = _acd(act_cache_key) / "acts.npy"
        arr = np.load(acts_path, mmap_mode="r")  # (N, L, T_seq, d_in)
        if arr.ndim != 4:
            raise ValueError(f"smoke expected 4D act cache; got {arr.shape}")
        N = min(64, arr.shape[0])
        Xs = np.ascontiguousarray(arr[:N, :, -S:, :]).astype(np.float32)  # (N, L, S, d_in)
        ys = (np.arange(N) % 2).astype(np.int64)
        # Split 50/50 train/test
        ntr = N // 2
        X_train, X_test = Xs[:ntr], Xs[ntr:]
        y_train, y_test = ys[:ntr], ys[ntr:]
        # No first_real mask in smoke (full S is "real")
        result = _s_tail_probe_mlc(
            m,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            S=S, k_feat=k_feat,
        )
        log.info("[c3_mlc.eval] smoke result: %s", result)
        return result, "auc"

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
        if task["X_train"].ndim != 4:
            raise ValueError(
                f"Expected 4D probe array for MLC eval; got "
                f"{task['X_train'].shape} on task {tname!r}. "
                f"Datasource {datasource_name} probe_cache may be "
                f"single-layer; rebuild via build_probe_cache(...) "
                f"on the multi-layer datasource."
            )
        r = _s_tail_probe_mlc(
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


# ── Pipeline orchestration ────────────────────────────────────────────


def run_one_cell(
    *,
    seed: int,
    k_feat: int,
    S: int = DEFAULT_S,
    smoke: bool = False,
    training_cfg: TrainingConfig | None = None,
    force_train: bool = False,
    force_eval: bool = False,
):
    cfg = training_cfg if training_cfg is not None else MLC_TRAINING_CFG
    act_cache_key = compute_act_cache_key(load_datasource(DATASOURCE))
    eval_cfg = {
        "k_feat": k_feat,
        "S": S,
        "smoke": smoke,
        "_act_cache_key": act_cache_key,
        "_datasource_name": DATASOURCE,
    }

    log.info(
        "[c3_mlc.run] CELL: arch=%s seed=%d k_feat=%d S=%d "
        "n_steps=%d smoke=%s",
        ARCH, seed, k_feat, S, cfg.n_steps, smoke,
    )

    result = runner.run_cell(
        component=COMPONENT,
        arch_name=ARCH,
        seed=seed,
        datasource_name=DATASOURCE,
        training_cfg=cfg,
        eval_cfg=eval_cfg,
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        train_fn=my_train_fn_mlc,
        eval_fn=my_eval_fn_mlc,
        force_train=force_train,
        force_eval=force_eval,
    )
    log.info(
        "[c3_mlc.run] CELL DONE: arch=%s seed=%d k_feat=%d "
        "train_key=%s eval_key=%s cached=%s",
        ARCH, seed, k_feat,
        result.train_key, result.eval_key, result.cached,
    )
    return result


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="Seeds. Default: 42 1 2.",
    )
    p.add_argument(
        "--k-feats", nargs="+", type=int, default=list(DEFAULT_K_FEATS),
        help="Probing k_feat values. Default: 5 20.",
    )
    p.add_argument("--S", type=int, default=DEFAULT_S)
    p.add_argument("--n-steps", type=int, default=None,
                   help="Override n_steps (e.g. 200 for smoke).")
    p.add_argument("--smoke", action="store_true",
                   help="Pass smoke=True to eval_cfg (synthetic labels).")
    p.add_argument("--force-train", action="store_true")
    p.add_argument("--force-eval", action="store_true")
    args = p.parse_args(argv)

    cfg = MLC_TRAINING_CFG
    if args.n_steps is not None:
        cfg = cfg.model_copy(update={"n_steps": args.n_steps})
        log.info("[c3_mlc.run] training_cfg n_steps override: %d",
                 args.n_steps)

    log.info("[c3_mlc.run] training_cfg=%s",
             cfg.model_dump(exclude_none=True))

    for seed in args.seeds:
        for k in args.k_feats:
            run_one_cell(
                seed=seed,
                k_feat=k,
                S=args.S,
                smoke=args.smoke,
                training_cfg=cfg,
                force_train=args.force_train,
                force_eval=args.force_eval,
            )

    log.info("[c3_mlc.run] all cells complete")


if __name__ == "__main__":
    sys.exit(main())
