"""Sparse-probing runner for Phase 5 — v2 using split-cache format.

Cache format (from build_probe_cache.py):
    acts_anchor.npz:  train_acts (N, LAST_N=20, d), train_last_idx, train_labels +
                      test_*  — L13 tail. Used by TopK/TXCDR/Stacked/Matryoshka.
    acts_mlc.npz:     train_acts (N, 5, d), train_labels + test_*.  MLC-specific.

Probing protocol (matches Kantamneni et al. §2.2):
    - Top-k latents by abs(mean_train[y=1] - mean_train[y=0])
    - Fit L1 logistic regression on those k features
    - AUC on held-out test set

Baselines:
    - L2 LR on raw last-token L13 activations.
    - Attention-pooled (Eq. 2) over the tail-20 L13 activations.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.architectures.topk_sae import TopKSAE
from src.architectures.stacked_sae import StackedSAE
from src.architectures.crosscoder import TemporalCrosscoder
from src.architectures.mlc import MultiLayerCrosscoder

from experiments.phase5_downstream_utility.probing.attn_pooled_probe import (
    train_attn_probe, AttnProbeConfig,
)


REPO = Path("/workspace/temp_xc")
PROBE_CACHE = REPO / "experiments/phase5_downstream_utility/results/probe_cache"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
OUT_JSONL = REPO / "experiments/phase5_downstream_utility/results/probing_results.jsonl"
PREDICTIONS_DIR = REPO / "experiments/phase5_downstream_utility/results/predictions"

K_VALUES = [1, 2, 5, 20]

# Phase 5.7 autoresearch val/test split.
# Per task: 3040 train -> train' (2432) + val (608), 80/20, deterministic
# seed from dataset_key. Test (760) untouched. The base aggregations
# (last_position, mean_pool) fit on full train, evaluate on test. The
# *_val variants fit on train', evaluate on val so the agent can
# iterate without peeking at test.
VAL_FRAC = 0.20
BASE_AGGREGATIONS = ("last_position", "full_window", "mean_pool")
VAL_AGGREGATIONS = ("last_position_val", "mean_pool_val")
ALL_AGGREGATIONS = BASE_AGGREGATIONS + VAL_AGGREGATIONS


def _split_indices_for_dataset(dataset_key: str, n_train: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic train'/val index split keyed on dataset_key.

    Same dataset_key -> same split, regardless of arch / seed.
    """
    seed = abs(hash(dataset_key)) % (2 ** 32)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    n_val = int(round(n_train * VAL_FRAC))
    val_idx = np.sort(perm[:n_val])
    train_idx = np.sort(perm[n_val:])
    return train_idx, val_idx


def _load_task_cache(task_dir: Path) -> dict[str, Any]:
    anchor = np.load(task_dir / "acts_anchor.npz")
    mlc = np.load(task_dir / "acts_mlc.npz")
    meta = json.loads((task_dir / "meta.json").read_text())
    out = {
        "anchor_train": anchor["train_acts"].astype(np.float32),  # (N, LAST_N, d)
        "anchor_test": anchor["test_acts"].astype(np.float32),
        "train_last_idx": anchor["train_last_idx"],
        "test_last_idx": anchor["test_last_idx"],
        "mlc_train": mlc["train_acts"].astype(np.float32),  # (N, 5, d)
        "mlc_test": mlc["test_acts"].astype(np.float32),
        "train_labels": anchor["train_labels"],
        "test_labels": anchor["test_labels"],
        "meta": meta,
    }
    # Optional: multi-layer × tail-20 cache for time_layer_crosscoder probing.
    mlc_tail_path = task_dir / "acts_mlc_tail.npz"
    if mlc_tail_path.exists():
        mlc_tail = np.load(mlc_tail_path)
        out["mlc_tail_train"] = mlc_tail["train_acts"].astype(np.float32)  # (N, LAST_N, L, d)
        out["mlc_tail_test"] = mlc_tail["test_acts"].astype(np.float32)
    return out


def _last_token(acts, last_idx):
    """(N, LAST_N, d) + last_idx -> (N, d) at each sample's last real pos."""
    N = acts.shape[0]
    out = np.empty((N, acts.shape[-1]), dtype=np.float32)
    for i in range(N):
        out[i] = acts[i, int(last_idx[i])]
    return out


def _window_at_last(acts, last_idx, T):
    """(N, LAST_N, d) + last_idx -> (N, T, d) left-clamped window ending at last_idx."""
    N = acts.shape[0]
    d = acts.shape[-1]
    out = np.empty((N, T, d), dtype=np.float32)
    for i in range(N):
        li = int(last_idx[i])
        start = max(0, li - T + 1)
        win = acts[i, start:li + 1]
        if win.shape[0] < T:
            pad = np.broadcast_to(win[0:1], (T - win.shape[0], d))
            win = np.concatenate([pad, win], axis=0)
        out[i] = win
    return out


# ─── encoders per arch ───

def _slide_windows(anchor_acts, T):
    """(N, LAST_N, d) -> (N, K, T, d) by sliding a T-window across LAST_N.
    K = LAST_N - T + 1 (1 when T == LAST_N, LAST_N when T == 1)."""
    N, LN, d = anchor_acts.shape
    if T == 1:
        return anchor_acts[:, :, None, :]  # (N, LN, 1, d)
    K = LN - T + 1
    # Advanced indexing: for each sample and each slide-start, grab T positions
    starts = np.arange(K)  # (K,)
    offsets = np.arange(T)  # (T,)
    idx = starts[:, None] + offsets[None, :]  # (K, T)
    return anchor_acts[:, idx, :]  # (N, K, T, d)


def _encode_per_token(model_encode, X_tokens, device, batch: int = 2048):
    """Encode a flat (M, d_or_T_d) array through `model_encode`, batched.

    Returns (M, d_sae) numpy.
    """
    outs = []
    for i in range(0, X_tokens.shape[0], batch):
        Xb = torch.from_numpy(X_tokens[i:i + batch]).to(device)
        with torch.no_grad():
            zb = model_encode(Xb)
        outs.append(zb.cpu().numpy())
    return np.concatenate(outs, axis=0)


def _encode_topk(model, anchor_acts, last_idx, device, aggregation="last_position"):
    if aggregation == "last_position":
        X = _last_token(anchor_acts, last_idx)
        return _encode_per_token(model.encode, X, device)
    # full_window: encode every tail position, flatten (N, LAST_N * d_sae)
    N, LN, d = anchor_acts.shape
    flat = anchor_acts.reshape(N * LN, d)
    z = _encode_per_token(model.encode, flat, device)
    return z.reshape(N, LN * z.shape[-1])


def _encode_mlc(model, mlc_acts, device, aggregation="last_position",
                mlc_tail=None):
    """MLC encoding.

    last_position: use (N, L, d) mlc_acts at prompt's last real token.
    full_window:   if mlc_tail (N, TAIL, L, d) is available, encode each
                   of the TAIL positions and flatten; else last_position.
    """
    if aggregation == "last_position" or mlc_tail is None:
        return _encode_per_token(model.encode, mlc_acts, device)
    N, TAIL, L, d = mlc_tail.shape
    flat = mlc_tail.reshape(N * TAIL, L, d)
    z = _encode_per_token(model.encode, flat, device)  # (N*TAIL, d_sae)
    return z.reshape(N, TAIL * z.shape[-1])


def _encode_txcdr(model, anchor_acts, last_idx, T, device,
                  aggregation="last_position"):
    if aggregation == "last_position":
        X = _window_at_last(anchor_acts, last_idx, T)
        return _encode_per_token(model.encode, X, device)
    # full_window: slide T-window across tail-20, encode each, flatten.
    wins = _slide_windows(anchor_acts, T)  # (N, K, T, d)
    N, K, _, d = wins.shape
    flat = wins.reshape(N * K, T, d)
    z = _encode_per_token(model.encode, flat, device)  # (N*K, d_sae)
    return z.reshape(N, K * z.shape[-1])


def _encode_stacked_last(model, anchor_acts, last_idx, T, device,
                         aggregation="last_position"):
    """Stacked probing: use the last SAE in the stack (saes[-1]).
    last_position = anchor at last real token; full_window = slide."""
    enc = model.saes[-1].encode
    if aggregation == "last_position":
        X = _last_token(anchor_acts, last_idx)
        return _encode_per_token(enc, X, device)
    N, LN, d = anchor_acts.shape
    flat = anchor_acts.reshape(N * LN, d)
    z = _encode_per_token(enc, flat, device)
    return z.reshape(N, LN * z.shape[-1])


def _encode_matryoshka(model, anchor_acts, last_idx, T, device,
                       aggregation="last_position"):
    if aggregation == "last_position":
        X = _window_at_last(anchor_acts, last_idx, T)
        return _encode_per_token(model.encode, X, device)
    wins = _slide_windows(anchor_acts, T)
    N, K, _, d = wins.shape
    flat = wins.reshape(N * K, T, d)
    z = _encode_per_token(model.encode, flat, device)
    return z.reshape(N, K * z.shape[-1])


# ─── feature selection + probe ───


def top_k_by_class_sep(Z_train, y_train, k):
    pos_mean = Z_train[y_train == 1].mean(axis=0)
    neg_mean = Z_train[y_train == 0].mean(axis=0)
    diff = np.abs(pos_mean - neg_mean)
    k = min(k, Z_train.shape[1])
    top = np.argpartition(-diff, k - 1)[:k]
    return np.sort(top)


def sae_probe_metrics(Z_train, y_train, Z_test, y_test, k, save_path=None):
    """Return (auc, acc). Uses top-k class-sep + L1 LR.
    If save_path is given, dumps (example_id, y_true, decision_score, y_pred) .npz."""
    idx = top_k_by_class_sep(Z_train, y_train, k)
    Xtr = Z_train[:, idx]
    Xte = Z_test[:, idx]
    if Xtr.shape[0] < 5 or len(np.unique(y_train)) < 2:
        return 0.5, 0.5
    scaler = StandardScaler(with_mean=False)
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)
    try:
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", max_iter=2000, C=1.0,
        )
        clf.fit(Xtr_s, y_train)
    except Exception:
        return 0.5, 0.5
    if len(np.unique(y_test)) < 2:
        return 0.5, 0.5
    preds_score = clf.decision_function(Xte_s)
    preds_cls = clf.predict(Xte_s)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            save_path,
            example_id=np.arange(len(y_test), dtype=np.int32),
            y_true=np.asarray(y_test, dtype=np.int8),
            decision_score=preds_score.astype(np.float32),
            y_pred=preds_cls.astype(np.int8),
        )
    return (
        float(roc_auc_score(y_test, preds_score)),
        float(accuracy_score(y_test, preds_cls)),
    )


def last_token_lr_metrics(X_train, y_train, X_test, y_test):
    """Return (auc, acc) for the L2-LR-on-raw-last-token baseline."""
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    try:
        clf = LogisticRegression(
            penalty="l2", solver="lbfgs", max_iter=2000, C=1.0,
        )
        clf.fit(Xtr, y_train)
    except Exception:
        return 0.5, 0.5
    if len(np.unique(y_test)) < 2:
        return 0.5, 0.5
    preds_score = clf.decision_function(Xte)
    preds_cls = clf.predict(Xte)
    return (
        float(roc_auc_score(y_test, preds_score)),
        float(accuracy_score(y_test, preds_cls)),
    )


def attn_pool_metrics(acts_tr, last_tr, y_tr, acts_te, last_te, y_te, device,
                      seed: int = 42):
    """acts_* shape: (N, LAST_N, d); L13 only (anchor). Returns (auc, acc)."""
    mask_tr = np.arange(acts_tr.shape[1])[None, :] <= last_tr[:, None]
    mask_te = np.arange(acts_te.shape[1])[None, :] <= last_te[:, None]
    try:
        out = train_attn_probe(
            X_train=torch.from_numpy(acts_tr).float(),
            y_train=torch.from_numpy(y_tr),
            X_test=torch.from_numpy(acts_te).float(),
            y_test=torch.from_numpy(y_te),
            mask_train=torch.from_numpy(mask_tr),
            mask_test=torch.from_numpy(mask_te),
            cfg=AttnProbeConfig(seed=seed),
            device=device,
        )
        return out["test_auc"], out["test_acc"]
    except Exception as e:
        print(f"    attn-pool FAIL: {e}")
        return 0.5, 0.5


def _iter_ckpts(run_ids):
    if run_ids:
        return [(rid, CKPT_DIR / f"{rid}.pt") for rid in run_ids]
    return [(p.stem, p) for p in sorted(CKPT_DIR.glob("*.pt"))]


def _load_model_for_run(run_id, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = state["meta"]
    arch = state["arch"]
    d_sae = 18_432
    d_in = 2304
    if arch == "topk_sae":
        model = TopKSAE(d_in, d_sae, k=meta["k_pos"]).to(device)
    elif arch == "mlc":
        model = MultiLayerCrosscoder(d_in, d_sae, n_layers=5, k=meta["k_pos"]).to(device)
    elif arch == "mlc_contrastive":
        from src.architectures.mlc_contrastive import MLCContrastive
        model = MLCContrastive(
            d_in, d_sae, n_layers=5,
            k=meta["k_pos"], h=meta.get("h", d_sae // 2),
        ).to(device)
    elif arch in ("txcdr_t2", "txcdr_t3", "txcdr_t5", "txcdr_t8",
                  "txcdr_t10", "txcdr_t15", "txcdr_t20"):
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TemporalCrosscoder(d_in, d_sae, T, k_eff).to(device)
    elif arch == "txcdr_contrastive_t5":
        from src.architectures.txcdr_contrastive import TXCDRContrastive
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TXCDRContrastive(
            d_in, d_sae, T, k_eff, h=meta.get("h", d_sae // 2),
        ).to(device)
    elif arch == "txcdr_rotational_t5":
        from src.architectures.txcdr_rotational import TXCDRRotational
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TXCDRRotational(
            d_in, d_sae, T, k_eff, K_rank=meta.get("K_rank", 8),
        ).to(device)
    elif arch == "matryoshka_txcdr_contrastive_t5":
        from src.architectures.matryoshka_txcdr_contrastive import (
            MatryoshkaTXCDRContrastive,
        )
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = MatryoshkaTXCDRContrastive(d_in, d_sae, T, k_eff).to(device)
    elif arch == "mlc_temporal_t3":
        from src.architectures.mlc_temporal import MLCTemporal
        T = meta["T"]
        L = meta.get("n_layers", 5)
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = MLCTemporal(d_in, d_sae, T, L, k_eff).to(device)
    elif arch == "txcdr_basis_expansion_t5":
        from src.architectures.txcdr_basis import TXCDRBasisExpansion
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TXCDRBasisExpansion(
            d_in, d_sae, T, k_eff, K_basis=meta.get("K_basis", 3),
        ).to(device)
    elif arch == "time_layer_contrastive_t5":
        from src.architectures.time_layer_contrastive import TimeLayerContrastive
        T = meta["T"]
        L = meta.get("n_layers", 5)
        d_sae_eff = meta.get("d_sae", 8192)
        k_total = meta["k_pos"] * T * L
        model = TimeLayerContrastive(
            d_in, d_sae_eff, T, L, k_total,
            h=meta.get("h", d_sae_eff // 2),
        ).to(device)
    elif arch == "txcdr_dynamics_t5":
        from src.architectures.txcdr_dynamics import TXCDRDynamics
        T = meta["T"]
        # Per-position k (not k * T). Stored as k_pos.
        model = TXCDRDynamics(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    elif arch.startswith("stacked_t"):
        T = meta["T"]
        model = StackedSAE(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    elif arch == "matryoshka_t5":
        from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = PositionMatryoshkaTXCDR(d_in, d_sae, T, k_eff).to(device)
    elif arch == "shared_perpos_t5":
        from experiments.phase5_downstream_utility.train_primary_archs import (
            SharedPerPositionSAE,
        )
        T = meta["T"]
        model = SharedPerPositionSAE(d_in, d_sae, T, k=meta["k_pos"]).to(device)
    elif arch in ("tfa", "tfa_pos", "tfa_small", "tfa_pos_small"):
        from src.architectures._tfa_module import TemporalSAE
        d_sae_eff = meta.get("d_sae", d_sae)
        use_pos = bool(meta.get("use_pos", arch.startswith("tfa_pos")))
        model = TemporalSAE(
            dimin=d_in, width=d_sae_eff, n_heads=4,
            sae_diff_type="topk", kval_topk=meta["k_pos"],
            tied_weights=True, n_attn_layers=1,
            bottleneck_factor=4, use_pos_encoding=use_pos,
        ).to(device)
    elif arch in (
        "txcdr_shared_dec_t5", "txcdr_shared_enc_t5",
        "txcdr_tied_t5", "txcdr_pos_t5", "txcdr_causal_t5",
        "txcdr_block_sparse_t5", "txcdr_lowrank_dec_t5",
        "txcdr_rank_k_dec_t5",
    ):
        from src.architectures.txcdr_variants import (
            TXCDRSharedDec, TXCDRSharedEnc, TXCDRTied, TXCDRPos, TXCDRCausal,
            TXCDRBlockSparseTopK, TXCDRLowRankDec, TXCDRRankKFeature,
        )
        cls = {
            "txcdr_shared_dec_t5": TXCDRSharedDec,
            "txcdr_shared_enc_t5": TXCDRSharedEnc,
            "txcdr_tied_t5": TXCDRTied,
            "txcdr_pos_t5": TXCDRPos,
            "txcdr_causal_t5": TXCDRCausal,
            "txcdr_block_sparse_t5": TXCDRBlockSparseTopK,
            "txcdr_lowrank_dec_t5": TXCDRLowRankDec,
            "txcdr_rank_k_dec_t5": TXCDRRankKFeature,
        }[arch]
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = cls(d_in, d_sae, T, k_eff).to(device)
    elif arch == "time_layer_crosscoder_t5":
        from src.architectures.time_layer_crosscoder import TimeLayerCrosscoder
        T = meta["T"]
        L = meta.get("n_layers", 5)
        d_sae_eff = meta.get("d_sae", d_sae)
        # k_win not stored for TxL (k is TopK count over flattened TxLxd_sae).
        k_total = meta["k_pos"] * T * L
        model = TimeLayerCrosscoder(d_in, d_sae_eff, T, L, k_total).to(device)
    elif arch == "temporal_contrastive":
        from src.architectures.temporal_contrastive_sae import (
            TemporalContrastiveSAE,
        )
        model = TemporalContrastiveSAE(d_in, d_sae, k=meta["k_pos"]).to(device)
    else:
        raise ValueError(f"Unknown arch {arch}")
    cast_state = {
        k: v.to(torch.float32) if v.dtype == torch.float16 else v
        for k, v in state["state_dict"].items()
    }
    model.load_state_dict(cast_state)
    model.eval()
    return model, arch, meta


def _encode_for_probe(
    model, arch, meta,
    tc: dict, split: str,
    device,
    aggregation: str = "last_position",
):
    # *_val aggregations encode the same way as their base; the train'/val
    # split is applied AFTER encoding inside `run_probing`. Strip the
    # suffix here so encoder dispatch is identical.
    if aggregation.endswith("_val"):
        aggregation = aggregation[: -len("_val")]
    # `mean_pool` reuses the `full_window` slide-and-encode path but
    # averages the K per-slide d_sae vectors instead of concatenating,
    # matching SAEBench / Kantamneni convention (probe sees one d_sae
    # feature vector per example, averaged over the sequence tail).
    # One extra np.reshape+mean — no additional GPU forward.
    if aggregation == "mean_pool":
        Z = _encode_for_probe(
            model, arch, meta, tc, split, device,
            aggregation="full_window",
        )
        N = Z.shape[0]
        d_sae = int(meta.get("d_sae", 18_432))
        K, rem = divmod(Z.shape[1], d_sae)
        assert rem == 0, (
            f"mean_pool reshape expected K*d_sae; got {Z.shape[1]} "
            f"not divisible by d_sae={d_sae} for arch={arch}"
        )
        return Z.reshape(N, K, d_sae).mean(axis=1)

    if split == "train":
        anchor = tc["anchor_train"]
        mlc = tc["mlc_train"]
        li = tc["train_last_idx"]
        mlc_tail = tc.get("mlc_tail_train")
    else:
        anchor = tc["anchor_test"]
        mlc = tc["mlc_test"]
        li = tc["test_last_idx"]
        mlc_tail = tc.get("mlc_tail_test")

    if arch == "topk_sae":
        return _encode_topk(model, anchor, li, device, aggregation)
    if arch in ("mlc", "mlc_contrastive"):
        # mlc_contrastive has identical encode API — subclass of MLC.
        # last_position: use mlc (N, L, d) at last real token.
        # full_window:   use mlc_tail (N, TAIL=20, L, d) if available.
        return _encode_mlc(model, mlc, device, aggregation, mlc_tail=mlc_tail)
    if (arch.startswith("txcdr_t")
            or arch in ("txcdr_contrastive_t5", "txcdr_rotational_t5",
                        "txcdr_basis_expansion_t5")):
        T = meta["T"]
        return _encode_txcdr(model, anchor, li, T, device, aggregation)
    if arch == "txcdr_dynamics_t5":
        # Dynamics model: encode returns z_last (B, d_sae). For mean_pool
        # we want all T intermediate states averaged — use encode_sequence.
        T = meta["T"]
        if aggregation == "last_position":
            X = _window_at_last(anchor, li, T)
            return _encode_per_token(model.encode, X, device)
        # full_window/mean_pool: slide T-window across tail-20, get all T
        # latents per slide, flatten. (mean_pool path in _encode_for_probe
        # later averages over the K slides.)
        wins = _slide_windows(anchor, T)  # (N, K, T, d)
        N, K, _, d = wins.shape
        flat = wins.reshape(N * K, T, d)
        # encode_sequence -> (N*K, T, d_sae). Flatten to (N, K * T * d_sae)
        # for full_window; mean_pool will reshape to (N, K, T*d_sae) and
        # then average... actually simpler: return (N, K, d_sae) using
        # z_last per slide so it matches the other archs' full_window
        # semantics (one d_sae vector per slide).
        outs = []
        batch = 512
        for i in range(0, flat.shape[0], batch):
            Xb = torch.from_numpy(flat[i:i + batch]).to(device)
            with torch.no_grad():
                z_seq = model.encode_sequence(Xb)   # (B, T, d_sae)
            outs.append(z_seq[:, -1, :].cpu().numpy())  # take last position
        Z_last = np.concatenate(outs, axis=0)          # (N*K, d_sae)
        return Z_last.reshape(N, K * Z_last.shape[-1])
    if arch.startswith("stacked_t"):
        T = meta["T"]
        return _encode_stacked_last(model, anchor, li, T, device, aggregation)
    if arch in ("matryoshka_t5", "matryoshka_txcdr_contrastive_t5"):
        T = meta["T"]
        return _encode_matryoshka(model, anchor, li, T, device, aggregation)
    if arch == "mlc_temporal_t3":
        # Input shape (B, T, L, d_in). Use mlc_tail cache which is
        # (N, TAIL_MLC_N, L, d). For last_position take the last T positions;
        # for mean_pool slide a T-window across the tail.
        T = meta["T"]
        L = meta.get("n_layers", 5)
        center_l = L // 2  # unused but kept for symmetry
        mlc_tail_key = "mlc_tail_train" if split == "train" else "mlc_tail_test"
        if mlc_tail_key not in tc:
            raise ValueError(
                f"mlc_temporal_t3 requires mlc_tail cache; not found for {split}"
            )
        mlc_tail = tc[mlc_tail_key]  # (N, TAIL, L, d)
        N, TAIL, _, d = mlc_tail.shape
        if aggregation == "last_position":
            X_win = mlc_tail[:, -T:, :, :].astype(np.float32)  # (N, T, L, d)
            outs = []
            for i in range(0, N, 256):
                Xb = torch.from_numpy(X_win[i:i + 256]).to(device)
                with torch.no_grad():
                    zb = model.encode(Xb)  # (B, d_sae)
                outs.append(zb.cpu().numpy())
            return np.concatenate(outs, axis=0)
        # full_window / mean_pool: slide T-window across TAIL.
        K = TAIL - T + 1
        if K <= 0:
            raise ValueError(
                f"mlc_temporal_t3 needs TAIL >= T; TAIL={TAIL}, T={T}"
            )
        d_sae_eff = int(meta.get("d_sae", 18_432))
        z_out = np.empty((N, K, d_sae_eff), dtype=np.float32)
        for k in range(K):
            win_k = mlc_tail[:, k:k + T, :, :]  # (N, T, L, d)
            for i in range(0, N, 256):
                Xb = torch.from_numpy(
                    win_k[i:i + 256].astype(np.float32)
                ).to(device)
                with torch.no_grad():
                    zb = model.encode(Xb)  # (B, d_sae)
                z_out[i:i + Xb.shape[0], k, :] = zb.cpu().numpy()
        return z_out.reshape(N, K * d_sae_eff)
    if arch == "shared_perpos_t5":
        # Kept for forward-compat; we do not train this arch in 5.1.
        X = _last_token(anchor, li)
        with torch.no_grad():
            z = model.encode(
                torch.from_numpy(X).to(device).unsqueeze(1)
            )[:, 0, :]
        return z.cpu().numpy()
    if arch in ("txcdr_shared_dec_t5", "txcdr_shared_enc_t5",
                "txcdr_tied_t5", "txcdr_pos_t5", "txcdr_lowrank_dec_t5",
                "txcdr_rank_k_dec_t5"):
        # Same (B, T, d) -> (B, d_sae) API as vanilla TXCDR.
        T = meta["T"]
        return _encode_txcdr(model, anchor, li, T, device, aggregation)
    if arch == "txcdr_block_sparse_t5":
        T = meta["T"]
        # Block-sparse emits (B, T, d_sae). Last position probing = take
        # position T-1 of the window ending at the last real token.
        if aggregation == "last_position":
            X = _window_at_last(anchor, li, T)
            outs = []
            for i in range(0, X.shape[0], 512):
                Xb = torch.from_numpy(X[i:i + 512]).to(device)
                with torch.no_grad():
                    Zb = model.encode(Xb)
                outs.append(Zb[:, -1, :].cpu().numpy())
            return np.concatenate(outs, axis=0)
        wins = _slide_windows(anchor, T)
        N, K, _, d = wins.shape
        flat = wins.reshape(N * K, T, d)
        outs = []
        for i in range(0, flat.shape[0], 512):
            Xb = torch.from_numpy(flat[i:i + 512]).to(device)
            with torch.no_grad():
                Zb = model.encode(Xb)
            outs.append(Zb[:, -1, :].cpu().numpy())
        Z_last = np.concatenate(outs, axis=0)
        return Z_last.reshape(N, K * Z_last.shape[-1])
    if arch in ("time_layer_crosscoder_t5", "time_layer_contrastive_t5"):
        T = meta["T"]
        L = meta.get("n_layers", 5)
        center_l = L // 2
        mlc_tail_key = "mlc_tail_train" if split == "train" else "mlc_tail_test"
        if mlc_tail_key in tc:
            # PROPER: mlc_tail shape (N, TAIL_MLC_N=5, L, d). With T=5 and
            # TAIL_MLC_N=5, the whole tail is the window — one (T, L, d)
            # slab per sample.
            mlc_tail = tc[mlc_tail_key]  # (N, 5, L, d)
            N, LN, _, d = mlc_tail.shape
            if aggregation == "last_position" or LN == T:
                # Pass the tail directly as the T-window.
                X_win = mlc_tail[:, -T:, :, :].astype(np.float32)
                outs = []
                for i in range(0, N, 256):
                    Xb = torch.from_numpy(X_win[i:i + 256]).to(device)
                    with torch.no_grad():
                        Zb = model.encode(Xb)
                    outs.append(Zb[:, -1, center_l, :].cpu().numpy())
                return np.concatenate(outs, axis=0)
            # full_window when LN > T: slide a T-window across LN.
            # STREAMING: iterate slide-by-slide and pre-allocate the
            # output to avoid materializing the full (N, K, T, L, d)
            # fancy-index copy (~22 GB at TAIL_MLC_N=20 × T=5 × N=3040).
            # Peak per slide is (N, T, L, d) ≈ 1.4 GB.
            K = LN - T + 1
            d_sae = model.encode(torch.from_numpy(
                mlc_tail[:1, :T, :, :].astype(np.float32)
            ).to(device)).shape[-1]
            z_out = np.empty((N, K, d_sae), dtype=np.float32)
            for k in range(K):
                win_k = mlc_tail[:, k:k + T, :, :]  # (N, T, L, d) — contiguous slice, ~1.4 GB
                for i in range(0, N, 256):
                    Xb = torch.from_numpy(
                        win_k[i:i + 256].astype(np.float32)
                    ).to(device)
                    with torch.no_grad():
                        Zb = model.encode(Xb)
                    z_out[i:i + Xb.shape[0], k, :] = (
                        Zb[:, -1, center_l, :].cpu().numpy()
                    )
            return z_out.reshape(N, K * d_sae)
        # Legacy fallback: T-1 zero-pad of last-token multilayer cache.
        X_1 = mlc[:, None, :, :]
        X_pad = np.concatenate(
            [np.zeros_like(X_1) for _ in range(T - 1)] + [X_1], axis=1
        )
        outs = []
        for i in range(0, X_pad.shape[0], 512):
            Xb = torch.from_numpy(X_pad[i:i + 512]).to(device)
            with torch.no_grad():
                Zb = model.encode(Xb)
            outs.append(Zb[:, -1, center_l, :].cpu().numpy())
        return np.concatenate(outs, axis=0)
    if arch == "txcdr_causal_t5":
        T = meta["T"]
        # Causal returns (B, T, d_sae). last_position = position T-1 of
        # the window ending at the last real token. Batched to avoid GPU
        # OOM — causal encoder stacks T intermediate tensors internally.
        if aggregation == "last_position":
            X = _window_at_last(anchor, li, T)
            outs = []
            for i in range(0, X.shape[0], 512):
                Xb = torch.from_numpy(X[i:i + 512]).to(device)
                with torch.no_grad():
                    Zb = model.encode(Xb)
                outs.append(Zb[:, -1, :].cpu().numpy())
            return np.concatenate(outs, axis=0)
        wins = _slide_windows(anchor, T)  # (N, K, T, d)
        N, K, _, d = wins.shape
        flat = wins.reshape(N * K, T, d)
        outs = []
        for i in range(0, flat.shape[0], 512):
            Xb = torch.from_numpy(flat[i:i + 512]).to(device)
            with torch.no_grad():
                Zb = model.encode(Xb)
            outs.append(Zb[:, -1, :].cpu().numpy())
        Z_last = np.concatenate(outs, axis=0)  # (N*K, d_sae)
        return Z_last.reshape(N, K * Z_last.shape[-1])
    if arch in ("tfa", "tfa_pos", "tfa_small", "tfa_pos_small"):
        # TFA takes (B, T, d); novel_codes live at inter["novel_codes"].
        # Scaling factor must match training; saved in meta as `scale`.
        scale = float(meta.get("scale", 1.0))
        outs = []
        for i in range(0, anchor.shape[0], 256):
            Xb = torch.from_numpy(anchor[i:i + 256]).float().to(device) * scale
            with torch.no_grad():
                _, inter = model(Xb)
            novel = inter["novel_codes"]  # (B, 20, d_sae)
            if aggregation == "last_position":
                lib = torch.from_numpy(li[i:i + 256]).to(device)
                outs.append(novel[torch.arange(novel.shape[0], device=device), lib].cpu().numpy())
            else:
                outs.append(novel.reshape(novel.shape[0], -1).cpu().numpy())
        return np.concatenate(outs, axis=0)
    if arch == "temporal_contrastive":
        # Token-level SAE; encode last token directly.
        if aggregation == "last_position":
            X = _last_token(anchor, li)
            return _encode_per_token(model.encode, X, device)
        # full_window: encode every tail position (matches topk full_window).
        N, LN, d = anchor.shape
        flat = anchor.reshape(N * LN, d)
        z = _encode_per_token(model.encode, flat, device)
        return z.reshape(N, LN * z.shape[-1])
    raise ValueError(f"Unknown arch {arch}")


def run_probing(
    run_ids=None, task_names=None, k_values=None,
    include_baselines: bool = True,
    aggregation: str = "last_position",
    save_predictions: bool = False,
):
    assert aggregation in ALL_AGGREGATIONS, aggregation
    is_val = aggregation.endswith("_val")
    k_values = k_values or K_VALUES
    device = torch.device("cuda")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)
    import gc

    task_dirs = [d for d in sorted(PROBE_CACHE.iterdir()) if d.is_dir()]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "acts_mlc.npz").exists()
    ]
    # STREAMING: load each task only when probing it. This caps RAM at
    # ~1 task cache + model + Z tensors (~5 GB peak), fitting in a 46 GB
    # container. Previously loaded all 36 task caches upfront (~18 GB fp32)
    # which combined with encoding peaks caused OOM kills.
    print(f"Probing {len(task_dirs)} tasks  aggregation={aggregation}")

    if include_baselines:
        with OUT_JSONL.open("a") as out_f:
            for task_dir in task_dirs:
                task_name = task_dir.name
                tc = _load_task_cache(task_dir)
                dkey = tc["meta"]["dataset_key"]

                # last-token LR baseline is single-token by design; we keep
                # the same X regardless of aggregation — both aggregation
                # records tagged so downstream filters work uniformly.
                X_last_tr_full = _last_token(tc["anchor_train"], tc["train_last_idx"])
                if is_val:
                    # Baselines also fit on train', evaluate on val — never
                    # touch the held-out test split when in val mode.
                    train_idx, val_idx = _split_indices_for_dataset(
                        dkey, n_train=tc["train_labels"].shape[0],
                    )
                    X_last_tr = X_last_tr_full[train_idx]
                    X_last_te = X_last_tr_full[val_idx]
                    ytr = tc["train_labels"][train_idx]
                    yte = tc["train_labels"][val_idx]
                    anchor_for_attn_tr = tc["anchor_train"][train_idx]
                    anchor_for_attn_te = tc["anchor_train"][val_idx]
                    last_idx_attn_tr = tc["train_last_idx"][train_idx]
                    last_idx_attn_te = tc["train_last_idx"][val_idx]
                else:
                    X_last_tr = X_last_tr_full
                    X_last_te = _last_token(tc["anchor_test"], tc["test_last_idx"])
                    ytr = tc["train_labels"]
                    yte = tc["test_labels"]
                    anchor_for_attn_tr = tc["anchor_train"]
                    anchor_for_attn_te = tc["anchor_test"]
                    last_idx_attn_tr = tc["train_last_idx"]
                    last_idx_attn_te = tc["test_last_idx"]

                t0 = time.time()
                last_auc, last_acc = last_token_lr_metrics(
                    X_last_tr, ytr, X_last_te, yte
                )
                out_f.write(json.dumps({
                    "run_id": "BASELINE_last_token_lr",
                    "arch": "baseline_last_token_lr",
                    "task_name": task_name, "dataset_key": dkey,
                    "aggregation": aggregation,
                    "k_feat": None,
                    "test_auc": last_auc, "test_acc": last_acc,
                    "n_train": int(ytr.size), "n_test": int(yte.size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                out_f.flush()
                print(f"  {task_name} last-LR  auc={last_auc:.4f} acc={last_acc:.4f}")

                t0 = time.time()
                ap_auc, ap_acc = attn_pool_metrics(
                    anchor_for_attn_tr, last_idx_attn_tr, ytr,
                    anchor_for_attn_te, last_idx_attn_te, yte, device,
                )
                out_f.write(json.dumps({
                    "run_id": "BASELINE_attn_pool",
                    "arch": "baseline_attn_pool",
                    "task_name": task_name, "dataset_key": dkey,
                    "aggregation": aggregation,
                    "k_feat": None,
                    "test_auc": ap_auc, "test_acc": ap_acc,
                    "n_train": int(ytr.size), "n_test": int(yte.size),
                    "elapsed_s": time.time() - t0,
                }) + "\n")
                out_f.flush()
                print(f"  {task_name} attn-pool auc={ap_auc:.4f} acc={ap_acc:.4f}")
                del tc, X_last_tr, X_last_te
                gc.collect()

    with OUT_JSONL.open("a") as out_f:
        for run_id, ckpt_path in _iter_ckpts(run_ids):
            if not ckpt_path.exists():
                print(f"  {run_id}: ckpt missing, skip")
                continue
            try:
                model, arch, meta = _load_model_for_run(run_id, ckpt_path, device)
            except Exception as e:
                print(f"  {run_id}: load FAIL: {e}")
                continue
            mlc_tail_aggs = ("full_window", "mean_pool", "mean_pool_val")
            # mlc_temporal_t3 always needs the multi-layer tail cache (even
            # at last_position, since its input shape is (B, T, L, d)).
            if arch == "mlc_temporal_t3" and aggregation in (
                "last_position", "last_position_val",
                *mlc_tail_aggs,
            ):
                sample_tail = task_dirs[0] / "acts_mlc_tail.npz"
                if not sample_tail.exists():
                    print(f"  {run_id}: acts_mlc_tail.npz absent — run build_probe_cache.py first, skip")
                    del model
                    torch.cuda.empty_cache()
                    continue
            if (arch in ("mlc", "mlc_contrastive",
                         "time_layer_crosscoder_t5",
                         "time_layer_contrastive_t5")
                    and aggregation in mlc_tail_aggs):
                # Both archs need the tail × multi-layer cache
                # (acts_mlc_tail.npz) for full_window; mean_pool delegates
                # to the full_window path internally so it needs the same
                # cache.
                sample_tail = task_dirs[0] / "acts_mlc_tail.npz"
                if not sample_tail.exists():
                    print(f"  {run_id}: acts_mlc_tail.npz absent — run build_probe_cache.py first, skip")
                    del model
                    torch.cuda.empty_cache()
                    continue
            print(f"=== {run_id} ({arch})  aggregation={aggregation} ===")
            for task_dir in task_dirs:
                task_name = task_dir.name
                tc = _load_task_cache(task_dir)
                try:
                    t0 = time.time()
                    if is_val:
                        # VAL MODE: encode train only, split into train' + val.
                        # Test split is NOT loaded or encoded — train acts only.
                        Z_full = _encode_for_probe(
                            model, arch, meta, tc, "train", device,
                            aggregation=aggregation,
                        )
                        y_full = tc["train_labels"]
                        train_idx, val_idx = _split_indices_for_dataset(
                            tc["meta"]["dataset_key"], n_train=Z_full.shape[0],
                        )
                        Ztr = Z_full[train_idx]
                        ytr_split = y_full[train_idx]
                        Zte = Z_full[val_idx]
                        yte_split = y_full[val_idx]
                        n_train_used = int(ytr_split.size)
                        n_test_used = int(yte_split.size)
                    else:
                        Ztr = _encode_for_probe(
                            model, arch, meta, tc, "train", device,
                            aggregation=aggregation,
                        )
                        Zte = _encode_for_probe(
                            model, arch, meta, tc, "test", device,
                            aggregation=aggregation,
                        )
                        ytr_split = tc["train_labels"]
                        yte_split = tc["test_labels"]
                        n_train_used = int(ytr_split.size)
                        n_test_used = int(yte_split.size)
                    for k in k_values:
                        save_path = None
                        if save_predictions:
                            save_path = (
                                PREDICTIONS_DIR
                                / f"{run_id}__{aggregation}__{task_name}__k{k}.npz"
                            )
                        auc, acc = sae_probe_metrics(
                            Ztr, ytr_split,
                            Zte, yte_split, k,
                            save_path=save_path,
                        )
                        out_f.write(json.dumps({
                            "run_id": run_id, "arch": arch,
                            "task_name": task_name,
                            "dataset_key": tc["meta"]["dataset_key"],
                            "aggregation": aggregation,
                            "k_feat": k,
                            "test_auc": auc, "test_acc": acc,
                            "n_train": n_train_used,
                            "n_test": n_test_used,
                            "elapsed_s": time.time() - t0,
                        }) + "\n")
                    out_f.flush()
                    print(f"  {task_name}: [{(time.time()-t0):.1f}s]", flush=True)
                    del Ztr, Zte
                except Exception as e:
                    print(f"  {task_name} FAIL: {e}", flush=True)
                    out_f.write(json.dumps({
                        "run_id": run_id, "arch": arch,
                        "task_name": task_name, "error": str(e),
                    }) + "\n")
                    out_f.flush()
                out_f.flush()
                del tc
                gc.collect()
                torch.cuda.empty_cache()
            del model
            gc.collect()
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--save-predictions", action="store_true",
                    help="Dump per-example predictions to results/predictions/ for confusion-matrix analysis.")
    ap.add_argument(
        "--aggregation",
        choices=list(ALL_AGGREGATIONS),
        default="last_position",
        help=("Base aggregations probe full-train→test. *_val variants "
              "split train (3040) into train' (2432) + val (608) keyed on "
              "dataset_key — test split is never read."),
    )
    args = ap.parse_args()
    run_probing(
        run_ids=args.run_ids, task_names=args.tasks,
        k_values=args.k_values, include_baselines=not args.skip_baselines,
        aggregation=args.aggregation,
        save_predictions=args.save_predictions,
    )


if __name__ == "__main__":
    main()
