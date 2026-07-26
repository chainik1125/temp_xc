"""§ 5.1 — Sparse probing on Gemma-2-2B-IT layer 13.

For each (arch, seed) trained on the Gemma activation cache, run the
38-task SAEBench+CT probing suite (36 upstream SAEBench binary tasks +
WinoGrande + WSC). Headline = mean ROC AUC over tasks.

Port of the paper protocol from
``origin/final:purified/experiments/c3_probing/run.py`` (my_eval_fn) +
``origin/final:purified/src/temp_bench/eval/probing.py`` (s_tail_probe /
mean_pool_probe / _encode_pool). Protocol semantics follow the v1
``1.1.0`` eval (Phase 7 padding fix: probe cache is left-aligned
``(N, S=32, d_in)`` with per-example ``first_real``; padding
contributions are masked per row).

``1.2.0`` (ACTMIX, 2026-07-26) is a strictly additive extension — the
ordered-path numbers are computed exactly as in 1.1.0:

- **Shuffle control** (``extra["shuffle"] = "within_window"``): after
  fitting each task's probe on ORDERED train features, the same fixed
  probe is scored on test features encoded from windows whose token
  order is permuted per row (``temp_bench.utils.shuffles``, Aniket's
  cross-task convention: per-row seeded permutation inside each
  T-window; the probe itself is never refit). Per-token archs (T = 1)
  are exactly invariant — reported equal by construction
  (``shuffle_identity = 1``), the control's own control.
- **Realized L0**: mean count of NONZERO latents per code unit (per
  token for T = 1 archs, per window for T > 1), measured on the
  ordered test encodes (and shuffled encodes separately). Nonzero —
  not positive — counting follows mac-a's btk-only convention
  (fired ⇔ z != 0; negative survivors are alive); identical for
  relu-mix codes, which are nonnegative. This is the
  activation-mixing fingerprint required on every ACTMIX cell.

Probe protocol (Kantamneni et al. § 2.2 + Phase 7 S-tail):

1. Feature aggregation: per-token archs encode every position and
   mean-pool latents over the real-token region of the S-tail; window
   archs slide a T-window over the S-tail, encode each window to one
   code, and mean-pool codes over windows fully inside the real region.
2. Top-k feature selection on TRAIN only: ``|mean(z|y=1) - mean(z|y=0)|``.
3. L1 logistic regression (liblinear, C=1.0, random_state=0); test ROC
   AUC + accuracy.

Determinism: given a checkpoint and ``extra``, results are exact.
Shuffle permutations are seeded per encode micro-batch as
``shuffle_seed * 1_000_003 + row_start`` so they do not depend on task
iteration order; ``encode_batch_size`` therefore participates in the
protocol and is pinned in ``extra`` (default 64).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator
from temp_bench.utils.shuffles import shuffle_within_window

DEFAULT_S = 32
DEFAULT_ENCODE_BATCH = 64
_SEED_STRIDE = 1_000_003


# ── Probe primitives (ported from origin/final eval/probing.py) ────────────


def _fit_probe(train_feats: np.ndarray, y_train: np.ndarray, k_feat: int):
    """Top-k feature selection by class-mean |diff| + L1 logistic regression.

    Returns ``(clf, top_idx)`` so the same fixed probe can be scored on
    multiple feature sets (ordered / shuffled test).
    """
    from sklearn.linear_model import LogisticRegression

    y_train = np.asarray(y_train).astype(int).ravel()
    if train_feats.ndim != 2:
        raise ValueError(f"expected (N, d_sae) train feats; got {train_feats.shape}")
    d_sae = train_feats.shape[1]
    if k_feat > d_sae:
        raise ValueError(f"k_feat={k_feat} > d_sae={d_sae}")

    pos = y_train == 1
    neg = y_train == 0
    if not pos.any() or not neg.any():
        raise ValueError("Train labels must contain both classes (0 and 1).")
    diff = np.abs(train_feats[pos].mean(axis=0) - train_feats[neg].mean(axis=0))
    top_idx = np.argsort(diff)[-k_feat:]

    import warnings as _warn
    with _warn.catch_warnings():
        _warn.simplefilter("ignore", FutureWarning)
        _warn.simplefilter("ignore", UserWarning)
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=1.0, max_iter=1000, random_state=0,
        )
        clf.fit(train_feats[:, top_idx], y_train)
    return clf, top_idx


def _score_probe(clf, top_idx, feats: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, roc_auc_score

    y = np.asarray(y).astype(int).ravel()
    proba = clf.predict_proba(feats[:, top_idx])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "acc": float(accuracy_score(y, pred)),
    }


def _encode_pool(
    model: TempBenchArch,
    X: np.ndarray,
    *,
    S: int,
    batch_size: int,
    device: torch.device,
    first_real: np.ndarray | None = None,
    shuffle_seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Encode the last-S tokens of each sequence and mean-pool latents.

    Returns ``(pooled (N, d_sae), realized_l0)`` where realized_l0 is the
    mean count of nonzero latent entries per code unit (token for
    T = 1, window for T > 1; fired ⇔ z != 0 per the btk-only
    convention), averaged over the units that contribute to the pool
    (padding-masked ones excluded).

    ``shuffle_seed``: if not None and ``model.T > 1``, each T-window's
    token order is permuted per row (fixed, seeded) before encoding —
    the ACTMIX shuffle control. For T = 1 the permutation is the
    identity, so callers skip re-encoding.

    Padding handling follows the Phase 7 recipe verbatim: real tokens
    occupy ``[first_real[i], S-1]``; per-token pooling masks earlier
    positions; window pooling keeps windows whose left edge is
    ``>= first_real[i]``, falling back to all windows for rows with
    fewer than T real tokens (no NaNs; noisy signal accepted).
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 3:
        raise ValueError(f"_encode_pool expects (N, seq_len, d_in); got {X.shape}")
    if X.shape[1] < S:
        raise ValueError(f"seq_len={X.shape[1]} < S={S}")
    if S < model.T:
        raise ValueError(f"S={S} < T={model.T}; S-tail protocol requires S >= T.")

    N = X.shape[0]
    tail = X[:, -S:, :]
    if first_real is not None:
        first_real = np.asarray(first_real, dtype=np.int64).clip(min=0, max=S)

    out: list[np.ndarray] = []
    l0_sum = 0.0
    l0_units = 0.0

    # Dispatch on the arch's CONSUMPTION CONTRACT, not on T: a window
    # arch at T=1 (the controlled-limit anchor) asserts (B, 1, d_in)
    # windows and rejects flat (B, S, d_in) batches. Routing it through
    # the window path with T=1 is mathematically identical to the
    # per-token path (length-1 windows, same first_real mask, and the
    # within-window shuffle of a length-1 window is the identity —
    # exactly the T=1 anchor semantics). Per-token/sequence archs
    # (consumes "token"/"sequence") take the flat path as in v1.
    window_arch = getattr(model, "consumes", "token") == "window"

    model.eval()
    with torch.no_grad():
        if not window_arch:
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.from_numpy(tail[start:end]).to(device)
                z = model.encode(batch)                    # (B, S, d_sae)
                if first_real is None:
                    mask = torch.ones(z.shape[:2], dtype=z.dtype, device=device)
                else:
                    fr = torch.from_numpy(first_real[start:end]).to(device)
                    k_grid = torch.arange(S, device=device).unsqueeze(0)
                    mask = (k_grid >= fr.unsqueeze(1)).to(z.dtype)   # (B, S)
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
                l0_sum += float(((z != 0).to(z.dtype).sum(dim=-1) * mask).sum())
                l0_units += float(mask.sum())
                out.append(pooled.float().cpu().numpy())
        else:
            T = model.T
            n_windows = S - T + 1
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                batch = torch.from_numpy(tail[start:end]).to(device)
                wins = torch.stack(
                    [batch[:, w:w + T] for w in range(n_windows)], dim=1
                )                                          # (B, n_windows, T, d_in)
                B = wins.shape[0]
                wins_flat = wins.reshape(B * n_windows, T, batch.shape[-1])
                if shuffle_seed is not None:
                    wins_flat = shuffle_within_window(
                        wins_flat, T=T,
                        seed=shuffle_seed * _SEED_STRIDE + start,
                        per_row=True,
                    )
                z = model.encode(wins_flat)                # (B*n_windows, 1, d_sae)
                z = z.squeeze(1).reshape(B, n_windows, -1)
                if first_real is None:
                    mask = torch.ones(z.shape[:2], dtype=z.dtype, device=device)
                else:
                    fr = torch.from_numpy(first_real[start:end]).to(device)
                    w_grid = torch.arange(n_windows, device=device).unsqueeze(0)
                    mask = (w_grid >= fr.unsqueeze(1)).to(z.dtype)   # (B, n_windows)
                    fallback = mask.sum(dim=1) == 0
                    if fallback.any():
                        mask[fallback] = 1.0
                counts = mask.sum(dim=1).clamp(min=1.0)
                pooled = (z * mask.unsqueeze(-1)).sum(dim=1) / counts.unsqueeze(-1)
                l0_sum += float(((z != 0).to(z.dtype).sum(dim=-1) * mask).sum())
                l0_units += float(mask.sum())
                out.append(pooled.float().cpu().numpy())

    realized_l0 = l0_sum / max(l0_units, 1.0)
    return np.concatenate(out, axis=0), realized_l0


# ── Evaluator ──────────────────────────────────────────────────────────────


class ProbingEval(Evaluator):
    """SAEBench+CT 38-task sparse probing on Gemma (+ ACTMIX shuffle control)."""

    name = "probing"
    protocol_version = "1.2.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        extra = dict(spec.extra or {})
        k_feat = int(extra.get("k_feat", 20))
        S = int(extra.get("S", DEFAULT_S))
        batch_size = int(extra.get("encode_batch_size", DEFAULT_ENCODE_BATCH))
        shuffle_mode = str(extra.get("shuffle", "within_window"))
        shuffle_seed = int(extra.get("shuffle_seed", 0))
        if shuffle_mode not in ("none", "within_window"):
            raise ValueError(f"Unknown shuffle mode {shuffle_mode!r}")

        # Defensive device pinning: the runner's train path returns a
        # CUDA-resident model, but its checkpoint-cache path loads on
        # CPU (contract gap flagged in the ACTMIX LOG — core fix is not
        # this plugin's to make). A CPU-resident 38-task encode is a
        # silent multi-hour stall, so pin here.
        if torch.cuda.is_available() and next(model.parameters()).device.type == "cpu":
            model = model.cuda()
        device = next(model.parameters()).device
        if spec.smoke:
            task_names = ["smoke_planted"]
        else:
            from temp_bench.data.probe_cache import list_probe_cache
            task_names = list_probe_cache(spec.datasource)
            if not task_names:
                raise FileNotFoundError(
                    f"No probe cache found for datasource {spec.datasource!r}. "
                    "Sync the canonical cache from HF "
                    "(han1823123123/temp-bench-data:probe_cache/<datasource>/) "
                    "into results/probe_cache/ first."
                )

        aucs, accs, aucs_sh, accs_sh, l0s, l0s_sh = [], [], [], [], [], []
        per_task: dict[str, float] = {}
        do_shuffle = shuffle_mode == "within_window"
        identity = model.T == 1   # T=1 within-window shuffle is the identity

        for tname in task_names:
            if spec.smoke:
                task = _smoke_task(d_in=model.config.d_in, S=S)
            else:
                from temp_bench.data.probe_cache import load_probe_cache
                task = load_probe_cache(spec.datasource, tname)

            train_feats, _ = _encode_pool(
                model, task["X_train"], S=S, batch_size=batch_size,
                device=device, first_real=task.get("first_real_train"),
            )
            clf, top_idx = _fit_probe(train_feats, task["y_train"], k_feat)

            test_feats, l0 = _encode_pool(
                model, task["X_test"], S=S, batch_size=batch_size,
                device=device, first_real=task.get("first_real_test"),
            )
            r = _score_probe(clf, top_idx, test_feats, task["y_test"])

            if do_shuffle and not identity:
                sh_feats, l0_sh = _encode_pool(
                    model, task["X_test"], S=S, batch_size=batch_size,
                    device=device, first_real=task.get("first_real_test"),
                    shuffle_seed=shuffle_seed,
                )
                r_sh = _score_probe(clf, top_idx, sh_feats, task["y_test"])
            else:
                r_sh, l0_sh = dict(r), l0   # exact invariance at T=1

            per_task[f"auc__{tname}"] = r["auc"]
            per_task[f"acc__{tname}"] = r["acc"]
            per_task[f"l0__{tname}"] = float(l0)
            aucs.append(r["auc"]); accs.append(r["acc"]); l0s.append(l0)
            if do_shuffle:
                per_task[f"auc_shuf__{tname}"] = r_sh["auc"]
                aucs_sh.append(r_sh["auc"]); accs_sh.append(r_sh["acc"])
                l0s_sh.append(l0_sh)

        def _agg(v: list[float]) -> tuple[float, float]:
            a = np.asarray(v, dtype=np.float64)
            return float(a.mean()), (float(a.std(ddof=1)) if len(v) > 1 else 0.0)

        mean_auc, std_auc = _agg(aucs)
        mean_acc, std_acc = _agg(accs)
        metrics: dict[str, float] = {
            "mean_auc": mean_auc, "std_auc": std_auc,
            "mean_acc": mean_acc, "std_acc": std_acc,
            "n_tasks": float(len(aucs)),
            "realized_l0": float(np.mean(l0s)),
            "realized_l0_min_task": float(np.min(l0s)),
            "realized_l0_max_task": float(np.max(l0s)),
            "T": float(model.T),
        }
        if do_shuffle:
            mean_auc_sh, std_auc_sh = _agg(aucs_sh)
            mean_acc_sh, _ = _agg(accs_sh)
            metrics.update({
                "mean_auc_shuf": mean_auc_sh, "std_auc_shuf": std_auc_sh,
                "mean_acc_shuf": mean_acc_sh,
                "delta_auc_shuf": mean_auc - mean_auc_sh,
                "realized_l0_shuf": float(np.mean(l0s_sh)),
                "shuffle_identity": 1.0 if identity else 0.0,
            })
        metrics.update(per_task)
        return metrics


def _smoke_task(*, d_in: int, S: int, n_train: int = 200, n_test: int = 80) -> dict[str, Any]:
    """Hermetic planted-signal task: positive class carries a +0.5 shift
    on the first 8 input dims. Validates encode + pool + probe + shuffle
    end-to-end without any disk dependency; AUC is meaningless for claims.
    """
    rng = np.random.default_rng(0)
    N = n_train + n_test
    X = rng.standard_normal((N, S, d_in)).astype(np.float32)
    y = np.zeros(N, dtype=np.int64)
    y[: N // 2] = 1
    X[y == 1, :, : min(8, d_in)] += 0.5
    perm = rng.permutation(N)
    X, y = X[perm], y[perm]
    first_real = np.zeros(N, dtype=np.int64)
    return {
        "task_name": "smoke_planted",
        "X_train": X[:n_train], "y_train": y[:n_train],
        "X_test": X[n_train:], "y_test": y[n_train:],
        "first_real_train": first_real[:n_train],
        "first_real_test": first_real[n_train:],
    }
