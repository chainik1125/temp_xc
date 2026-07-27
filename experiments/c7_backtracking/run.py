"""C7 — Ward Stage B backtracking sweep runner.

Per ``docs/components/c7.md`` and the agent_back briefing, this script
runs the locked-arch sweep for C7:

    7 archs × 3 seeds × 25 magnitudes (cut25 protocol)
        → inducement Δgc (Sonnet 4.6 judge)
        → detection PR-AUC (sparse probe at S ∈ {1,2,4,8,16,32})

Every cell flows through :func:`temp_bench.runner.run_cell` so:

- training is skipped if a checkpoint with that ``train_key`` already exists
- evaluation is skipped if that ``eval_key`` is already in
  ``leaderboard.jsonl``
- the validated leaderboard row is appended on success
- on ephemeral pods, the trained checkpoint is auto-pushed to HF

PROTOCOL.md § 11 *Code reuse contract*: this runner is a thin loop. The
training loop lives in :mod:`temp_bench.training.sae_trainer`; the eval
pipeline lives in :mod:`temp_bench.case_studies.backtracking`. Do not
write either inline here.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from temp_bench import runner
from temp_bench.case_studies.backtracking import (
    BacktrackingCaseStudy,
    DEFAULT_MAGNITUDE_GRID,
    EXTENDED_MAGNITUDE_GRID,
    DEFAULT_PR_AUC_S_GRID,
    SonnetBacktrackingJudge,
    build_cohort,
    load_stage_a,
    run_arch_evaluation,
)
from temp_bench.config import (
    instantiate_arch,
    load_arch,
    load_datasource,
    run_dir,
)
from temp_bench.data.nlp.ward import cache_activations, load_activations
from temp_bench.schemas import TrainingConfig

log = logging.getLogger("c7.run")


# ── Per-component constants ─────────────────────────────────────────────

COMPONENT = "c7"

# Datasource — the NousResearch mirror used to bypass Meta's gated repo
# (see configs/datasources.yaml note + agent_back briefing OQ #4).
DATASOURCE = "llama_3_1_8b_base_l10_ward_nousmirror"

EVAL_PROTOCOL_VERSION = "1.0.0"  # PINNED BACK for the 300K stacked arm
# (2026-07-27 stacked-SAE sprint, branch dmitry-stacked-c7-300k):
# delta_gc peak is a max over the magnitude grid, so a 41-mag (2.0.0)
# peak is not comparable to the paper's 25-mag peaks — the T-SAE
# extended row (0.433 @ +32 vs canonical 0.164 @ +7) proves the tail
# folds in. The paper's printed Fig 4 / Table 2 values are 25-mag
# protocol 1.0.0 rows (origin/300k-tfa:purified/results/leaderboard.jsonl);
# this arm must land on the same grid.
# Earlier protocol versions:
#   1.0.0: original 25-mag sweep (no shuffle ablation)
#   1.1.0: 25-mag + within-window shuffle ablation (no extreme mags)
#   2.0.0: 41-mag extended grid + shuffle ablation
# Each protocol bump → fresh eval_keys; older cells stay in leaderboard
# for diff. Analysis canonicalises on the latest version per train_key.

# All 7 locked archs from docs/components/c7.md (NOT tfa_pos).
DEFAULT_ARCHS = (
    "topk_sae",
    "stacked_sae",
    "tfa",
    "tsae_paper",
    "mlc",
    "txc_base",
    "txc_pro",
)
DEFAULT_SEEDS = (1, 2, 42)


# ── Train-fn adapter: shared trainer + activation cache ────────────────


def _spec_window_size(spec) -> int:
    """Window size needed to satisfy the arch's train_step.

    - Shared-z TXCs (txc_base) want T tokens.
    - txc_pro wants T_max + max(contrastive_shifts) tokens for the
      multi-distance contrastive positives.
    - Stacked-SAE / MLC want T tokens.
    - TFA + TopK + T-SAE handle any seq_len ≥ 1.
    Default seq_len conservatively to ``max(T_window, 16)`` so the
    full sweep doesn't trip on per-arch quirks.
    """
    h = spec.hparams
    T_max = int(h.get("T_max", h.get("T", h.get("n_layers", 5))))
    shifts = h.get("contrastive_shifts") or []
    max_shift = max(shifts) if shifts else 0
    return T_max + max_shift


# Per-process preloaded activation cache for C7. (decision 2026-05-04)
# directive (briefing top): adopt the preloaded `.clone()` pattern from
# agent_nlp's `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache`
# (commit e12dc719) for ~1.4× end-to-end trainer speedup. The helper
# isn't a drop-in for C7 because we sample T-token sliding windows from
# (N, L, d) while the helper returns whole sequences — so we apply the
# `.clone()` pattern locally. Determinism unchanged: same `default_rng(seed)`,
# same fp32 contract → train_keys + checkpoints bit-identical to mmap path.
_PRELOADED_C7_ACTS: dict[str, torch.Tensor] = {}


def _build_batch_iter(act_cache_key: str, *, batch_size: int = 256, T: int = 5,
                      seed: int = 42):
    """Build a batch iterator from a preloaded activation cache.

    Returns a callable ``batch_iter(n) -> Tensor (n, T, d_in)`` matching
    ``temp_bench.training.sae_trainer.train_sae``'s contract. Sliding
    T-token windows are sampled uniformly across the cache. Cache is
    preloaded once per process via ``.clone()`` (load-bearing — without it
    ``torch.from_numpy`` zero-copy wraps the mmap and page-faults persist).
    """
    from temp_bench.config import act_cache_dir, load_datasource, compute_act_cache_key
    ds = load_datasource(DATASOURCE)
    expected_key = compute_act_cache_key(ds)
    if expected_key != act_cache_key:
        raise RuntimeError(
            f"act_cache_key mismatch: expected {expected_key} (from datasource), "
            f"got {act_cache_key} (from runner)."
        )
    cache_dir = act_cache_dir(act_cache_key)
    cache_path = str(cache_dir / "resid_post_L10.npy")

    if cache_path not in _PRELOADED_C7_ACTS:
        mmapped = np.load(cache_path, mmap_mode="r")
        _PRELOADED_C7_ACTS[cache_path] = (
            torch.from_numpy(np.ascontiguousarray(mmapped)).clone()
        )
    acts = _PRELOADED_C7_ACTS[cache_path]
    N, L, d = acts.shape
    if L < T:
        raise RuntimeError(f"seq_len {L} < arch T={T}; bump datasource seq_len.")
    rng = np.random.default_rng(seed)

    def batch_iter(n: int) -> torch.Tensor:
        seq_idx = rng.integers(0, N, size=n)
        pos_idx = rng.integers(0, L - T + 1, size=n)
        out = torch.empty((n, T, d), dtype=torch.float32)
        for i in range(n):
            out[i] = acts[int(seq_idx[i]),
                          int(pos_idx[i]):int(pos_idx[i]) + T].to(torch.float32)
        return out

    return batch_iter


def my_train_fn(*, arch_name, arch_hparams, seed, training_cfg, act_cache_key, component):
    """C7 train-fn adapter. Builds the SAE from the YAML spec + datasource
    d_in, then calls the canonical ``train_sae`` trainer."""
    from temp_bench.training.sae_trainer import train_sae
    spec = load_arch(arch_name, component=component)
    ds = load_datasource(DATASOURCE)
    # d_in resolved via subject_model.config.hidden_size — but we already
    # know it from the cache; read the layer_specs sidecar.
    import json
    from temp_bench.config import act_cache_dir
    cache_dir = act_cache_dir(act_cache_key)
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    model = instantiate_arch(spec, d_in=d_in)
    if torch.cuda.is_available():
        model = model.cuda()
    # Cast heavy archs (>1B params) to bf16 to halve param + grad memory.
    # At d_sae=32768 fp32: txc_pro=42 GB / tfa=37 GB / txc_base=22 GB just for
    # opt-state — won't fit on A40 (47 GB). bf16 brings them to <32 GB.
    # SAE training quality at bf16 is empirically robust per the wasteland
    # phase 7 results; same precision the trainer uses for autocast anyway.
    n_params = sum(p.numel() for p in model.parameters())
    if n_params > 1e9 and torch.cuda.is_available():
        model = model.bfloat16()
        log.info("[c7.run] bf16 cast (%.1fM params → %.1f GB → fits A40)",
                 n_params / 1e6, n_params * (2 + 2 + 4) / 1e9)
    # Use the arch-specific window size (handles T_max + contrastive_shifts).
    T = _spec_window_size(spec)
    batch_iter = _build_batch_iter(act_cache_key, batch_size=training_cfg.batch_size,
                                    T=T, seed=seed)
    result = train_sae(model, batch_iter, training_cfg)
    # Persist train_log for post-cell convergence check (decisions.md
    # § 12: surface if final-1K-step loss drop > 5% of step-N loss
    # flags the cap as binding). Mirrors agent_nlp's c3+c4 pattern
    # (commit 033a3eb6) + agent_steer's 8953f6e4. The runner's
    # save_checkpoint discards the per-step log so we save it
    # ourselves alongside other run logs.
    try:
        import json as _json
        from pathlib import Path as _Path
        log_path = _Path("logs") / (
            f"c7_b{training_cfg.batch_size}_{arch_name}_seed{seed}_trainlog.json"
        )
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(_json.dumps(result.get("log", {})))
        log.info("[c7.run] trainlog saved → %s", log_path)
    except Exception as exc:
        log.warning("[c7.run] trainlog persist failed: %s", exc)
    return result["state_dict"]


# ── Eval-fn adapter: BacktrackingCaseStudy ─────────────────────────────


def _instantiate_from_state(arch_name: str, state_dict, component: str = COMPONENT):
    spec = load_arch(arch_name, component=component)
    ds = load_datasource(DATASOURCE)
    import json
    from temp_bench.config import act_cache_dir, compute_act_cache_key
    cache_dir = act_cache_dir(compute_act_cache_key(ds))
    specs = json.loads((cache_dir / "layer_specs.json").read_text())
    d_in = int(specs["d_model"])
    model = instantiate_arch(spec, d_in=d_in)
    model.load_state_dict(state_dict)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def my_eval_fn(*, model, eval_cfg, component):
    """C7 eval-fn adapter. Instantiates the SAE from the cached state_dict,
    auto-loads cohort + sentence acts from disk, forwards to
    :func:`run_arch_evaluation`.

    The runner passes the SAE's ``_state_dict`` + ``_arch_name`` via
    ``eval_cfg`` (see ``temp_bench.runner.run_cell``); we instantiate
    the module here and pass it to the case study.

    Sentence acts are loaded from
    ``results/c7_backtracking/stage_a/sentence_acts_L10.npz`` (built
    by :func:`temp_bench.case_studies.backtracking.extract_labeled_sentence_acts`).
    Caller may override by passing ``feature_mining_acts`` /
    ``sentence_acts`` / ``sentence_labels`` / ``sentence_qids`` in
    ``eval_cfg`` directly (e.g. for smoke tests).
    """
    from pathlib import Path
    import numpy as np
    from temp_bench.case_studies.backtracking import (
        extract_labeled_sentence_acts, split_pos_neg,
    )

    from temp_bench.config import compute_eval_key
    arch_name = eval_cfg["_arch_name"]
    seed = eval_cfg.get("seed", 42)
    state_dict = eval_cfg["_state_dict"]
    arch_module = _instantiate_from_state(arch_name, state_dict, component=component)

    # Per-(train_key × eval_cfg) workspace so cells with different magnitude
    # grids / cut_fractions don't pollute each other's judge cache. Strip
    # private/underscore-prefixed keys + numpy-array kwargs that don't
    # belong in the deterministic eval_key.
    _hash_eval_cfg = {
        k: v for k, v in eval_cfg.items()
        if not k.startswith("_") and k not in (
            "feature_mining_acts", "sentence_acts", "sentence_labels", "sentence_qids",
        )
    }
    eval_key = compute_eval_key(
        train_key=eval_cfg["_train_key"],
        eval_protocol_version=EVAL_PROTOCOL_VERSION,
        eval_cfg=_hash_eval_cfg,
    )
    workspace = run_dir(eval_key)
    workspace.mkdir(parents=True, exist_ok=True)
    judge = SonnetBacktrackingJudge(workspace=workspace)

    from temp_bench.case_studies.backtracking import build_cohort, load_stage_a
    cohort = build_cohort()
    stage_a = load_stage_a()

    # Mining + PR-AUC inputs: load (or extract if missing) from cache.
    if eval_cfg.get("feature_mining_acts") is not None:
        pos_neg = eval_cfg["feature_mining_acts"]
        pr_X = eval_cfg.get("sentence_acts")
        pr_y = eval_cfg.get("sentence_labels")
        pr_qids = eval_cfg.get("sentence_qids")
    else:
        sa = extract_labeled_sentence_acts()  # idempotent — cache hit on disk
        pos_neg = split_pos_neg(sa)
        pr_X = sa["X"]
        pr_y = sa["is_bt"].astype(int)
        pr_qids = np.array([k.split("|")[0] for k in sa["keys"]], dtype=object)

    result = run_arch_evaluation(
        arch=arch_module,
        seed=seed,
        cohort=cohort,
        stage_a=stage_a,
        workspace=workspace,
        judge=judge,
        magnitudes=tuple(eval_cfg.get("magnitudes", DEFAULT_MAGNITUDE_GRID)),
        cut_fraction=eval_cfg.get("cut_fraction", 0.25),
        arch_name=arch_name,
        feature_mining_acts=pos_neg,
        sentence_acts=pr_X,
        sentence_labels=pr_y,
        sentence_qids=pr_qids,
        pr_auc_S_grid=tuple(eval_cfg.get("pr_auc_S_grid", DEFAULT_PR_AUC_S_GRID)),
        max_new_tokens=eval_cfg.get("max_new_tokens", 1024),
        gen_batch_size=eval_cfg.get("gen_batch_size", 8),
    )
    return result.metrics, result.primary_metric


# ── Main loop ──────────────────────────────────────────────────────────


def main(*, archs=None, seeds=DEFAULT_SEEDS, build_cache_only: bool = False,
         force_train: bool = False, force_eval: bool = False):
    archs = archs or DEFAULT_ARCHS
    log.info("[c7.run] datasource=%s", DATASOURCE)

    # Step 0: ensure activation cache exists. cache_activations is idempotent.
    cache_dir = cache_activations(DATASOURCE)
    log.info("[c7.run] act cache at %s", cache_dir)
    if build_cache_only:
        return 0

    for arch in archs:
        for seed in seeds:
            log.info("[c7.run] cell arch=%s seed=%d", arch, seed)
            try:
                runner.run_cell(
                    component=COMPONENT,
                    arch_name=arch,
                    seed=seed,
                    datasource_name=DATASOURCE,
                    # (2026-07-27 stacked-SAE sprint) paper-scale override:
                    # n_steps=300_000 to match the published 300K arms
                    # (whose checkpoints are lost from git+HF; this arm is
                    # the scale-matched stacked row for Fig 4 / Table 2,
                    # judge-drift caveat recorded in the sprint log).
                    training_cfg=TrainingConfig(n_steps=300_000),
                    eval_cfg={
                        # 25-mag canonical grid — matches the printed
                        # Fig 4 / Table 2 peaks (see EVAL_PROTOCOL_VERSION
                        # note above).
                        "magnitudes": list(DEFAULT_MAGNITUDE_GRID),
                        "cut_fraction": 0.25,
                        "pr_auc_S_grid": list(DEFAULT_PR_AUC_S_GRID),
                    },
                    eval_protocol_version=EVAL_PROTOCOL_VERSION,
                    train_fn=my_train_fn,
                    eval_fn=my_eval_fn,
                    force_train=force_train,
                    force_eval=force_eval,
                )
            except Exception as e:
                log.exception("[c7.run] cell failed arch=%s seed=%d: %s", arch, seed, e)
            finally:
                # Aggressive GPU cleanup between cells — without this,
                # PyTorch's caching allocator holds onto dead-cell tensors
                # and the next cell OOMs (witnessed on tfa after a txc_pro
                # failure). gc.collect() drops Python refs; empty_cache()
                # releases reserved blocks back to the driver.
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    return 0


def cli():
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=DEFAULT_SEEDS)
    ap.add_argument("--build-cache-only", action="store_true",
                    help="Build the Llama BASE L10 activation cache and exit.")
    ap.add_argument("--force-train", action="store_true")
    ap.add_argument("--force-eval", action="store_true")
    args = ap.parse_args()
    raise SystemExit(main(
        archs=args.archs,
        seeds=tuple(args.seeds),
        build_cache_only=args.build_cache_only,
        force_train=args.force_train,
        force_eval=args.force_eval,
    ))


if __name__ == "__main__":
    cli()
