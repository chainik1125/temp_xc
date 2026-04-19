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

K_VALUES = [1, 2, 5, 20]


def _load_task_cache(task_dir: Path) -> dict[str, Any]:
    anchor = np.load(task_dir / "acts_anchor.npz")
    mlc = np.load(task_dir / "acts_mlc.npz")
    meta = json.loads((task_dir / "meta.json").read_text())
    return {
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


def _encode_mlc(model, mlc_acts, device, aggregation="last_position"):
    # MLC cache is last-token only; no slide possible. last_position only.
    return _encode_per_token(model.encode, mlc_acts, device)


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


def sae_probe_metrics(Z_train, y_train, Z_test, y_test, k):
    """Return (auc, acc). Uses top-k class-sep + L1 LR."""
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
    elif arch.startswith("txcdr_t"):
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = TemporalCrosscoder(d_in, d_sae, T, k_eff).to(device)
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
    if split == "train":
        anchor = tc["anchor_train"]
        mlc = tc["mlc_train"]
        li = tc["train_last_idx"]
    else:
        anchor = tc["anchor_test"]
        mlc = tc["mlc_test"]
        li = tc["test_last_idx"]

    if arch == "topk_sae":
        return _encode_topk(model, anchor, li, device, aggregation)
    if arch == "mlc":
        # MLC probe cache is last-token-only (no per-position slide possible
        # without rebuilding the probe cache for all 20 tail positions ×
        # 5 layers, which would bust the MooseFS quota). Emit last_position
        # only; probing runner skips MLC records for full_window.
        return _encode_mlc(model, mlc, device, aggregation)
    if arch.startswith("txcdr_t"):
        T = meta["T"]
        return _encode_txcdr(model, anchor, li, T, device, aggregation)
    if arch.startswith("stacked_t"):
        T = meta["T"]
        return _encode_stacked_last(model, anchor, li, T, device, aggregation)
    if arch == "matryoshka_t5":
        T = meta["T"]
        return _encode_matryoshka(model, anchor, li, T, device, aggregation)
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
            with torch.no_grad():
                Z = model.encode(torch.from_numpy(X).to(device))
            return Z[:, -1, :].cpu().numpy()
        wins = _slide_windows(anchor, T)
        N, K, _, d = wins.shape
        flat = wins.reshape(N * K, T, d)
        with torch.no_grad():
            Z = model.encode(torch.from_numpy(flat).to(device))
        Z_last = Z[:, -1, :]
        return Z_last.reshape(N, K * Z_last.shape[-1]).cpu().numpy()
    if arch == "time_layer_crosscoder_t5":
        # TxL probing fallback: use last-token multi-layer cache with a
        # T=1 degenerate window. (B, 1, L, d) through the T=5-trained
        # encoder uses only W_enc[0, :, :, :]. Genuine (B, T, L, d)
        # probing requires a rebuilt cache; deferred.
        T = meta["T"]
        L = meta.get("n_layers", 5)
        d_sae_eff = meta.get("d_sae", 8192)
        X_1 = mlc[:, None, :, :]  # (N, 1, L, d)
        X_pad = np.concatenate(
            [np.zeros_like(X_1) for _ in range(T - 1)] + [X_1], axis=1
        )  # (N, T, L, d) — fill earlier T-1 slots with zeros
        with torch.no_grad():
            Z = model.encode(torch.from_numpy(X_pad).to(device))
        # Take the last (time, layer) slab — features at (T-1, center_layer)
        center_l = L // 2
        z_last = Z[:, -1, center_l, :]  # (N, d_sae_eff)
        return z_last.cpu().numpy()
    if arch == "txcdr_causal_t5":
        T = meta["T"]
        # Causal returns (B, T, d_sae). last_position = position T-1 of
        # the window ending at the last real token.
        if aggregation == "last_position":
            X = _window_at_last(anchor, li, T)
            with torch.no_grad():
                Z = model.encode(torch.from_numpy(X).to(device))
            return Z[:, -1, :].cpu().numpy()
        # full_window: slide T-window across tail-20, flatten per-position
        wins = _slide_windows(anchor, T)  # (N, K, T, d)
        N, K, _, d = wins.shape
        flat = wins.reshape(N * K, T, d)
        with torch.no_grad():
            Z = model.encode(torch.from_numpy(flat).to(device))  # (N*K, T, d_sae)
        # Take the last position of each window (t = T-1 is "current")
        Z_last = Z[:, -1, :]  # (N*K, d_sae)
        return Z_last.reshape(N, K * Z_last.shape[-1]).cpu().numpy()
    if arch in ("tfa", "tfa_pos", "tfa_small", "tfa_pos_small"):
        # TFA takes (B, T, d); novel_codes live at inter["novel_codes"].
        # Scaling factor must match training; saved in meta as `scale`.
        scale = float(meta.get("scale", 1.0))
        if aggregation == "last_position":
            X = anchor  # (N, 20, d)
            X_t = torch.from_numpy(X).float().to(device) * scale
            with torch.no_grad():
                _, inter = model(X_t)
            novel = inter["novel_codes"]  # (N, 20, d_sae)
            out = novel[torch.arange(novel.shape[0], device=device),
                        torch.from_numpy(li).to(device)]
            return out.cpu().numpy()
        # full_window: flatten all 20 positions
        X_t = torch.from_numpy(anchor).float().to(device) * scale
        with torch.no_grad():
            _, inter = model(X_t)
        novel = inter["novel_codes"]  # (N, 20, d_sae)
        return novel.reshape(novel.shape[0], -1).cpu().numpy()
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
):
    assert aggregation in ("last_position", "full_window"), aggregation
    k_values = k_values or K_VALUES
    device = torch.device("cuda")
    OUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

    task_dirs = [d for d in sorted(PROBE_CACHE.iterdir()) if d.is_dir()]
    if task_names:
        task_dirs = [d for d in task_dirs if d.name in task_names]
    # Only load dirs that have both caches
    task_dirs = [
        d for d in task_dirs
        if (d / "acts_anchor.npz").exists() and (d / "acts_mlc.npz").exists()
    ]
    task_cache = {d.name: _load_task_cache(d) for d in task_dirs}
    print(f"Probing {len(task_cache)} tasks  aggregation={aggregation}")

    if include_baselines:
        with OUT_JSONL.open("a") as out_f:
            for task_name, tc in task_cache.items():
                ytr, yte = tc["train_labels"], tc["test_labels"]
                dkey = tc["meta"]["dataset_key"]

                # last-token LR baseline is single-token by design; we keep
                # the same X regardless of aggregation — both aggregation
                # records tagged so downstream filters work uniformly.
                X_last_tr = _last_token(tc["anchor_train"], tc["train_last_idx"])
                X_last_te = _last_token(tc["anchor_test"], tc["test_last_idx"])

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
                print(f"  {task_name} last-LR  auc={last_auc:.4f} acc={last_acc:.4f}")

                t0 = time.time()
                ap_auc, ap_acc = attn_pool_metrics(
                    tc["anchor_train"], tc["train_last_idx"], ytr,
                    tc["anchor_test"], tc["test_last_idx"], yte, device,
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
                print(f"  {task_name} attn-pool auc={ap_auc:.4f} acc={ap_acc:.4f}")

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
            if arch == "mlc" and aggregation == "full_window":
                print(f"  {run_id}: MLC has no full_window (cache is last-token only), skip")
                del model
                torch.cuda.empty_cache()
                continue
            print(f"=== {run_id} ({arch})  aggregation={aggregation} ===")
            for task_name, tc in task_cache.items():
                try:
                    t0 = time.time()
                    Ztr = _encode_for_probe(
                        model, arch, meta, tc, "train", device,
                        aggregation=aggregation,
                    )
                    Zte = _encode_for_probe(
                        model, arch, meta, tc, "test", device,
                        aggregation=aggregation,
                    )
                    for k in k_values:
                        auc, acc = sae_probe_metrics(
                            Ztr, tc["train_labels"],
                            Zte, tc["test_labels"], k,
                        )
                        out_f.write(json.dumps({
                            "run_id": run_id, "arch": arch,
                            "task_name": task_name,
                            "dataset_key": tc["meta"]["dataset_key"],
                            "aggregation": aggregation,
                            "k_feat": k,
                            "test_auc": auc, "test_acc": acc,
                            "n_train": int(tc["train_labels"].size),
                            "n_test": int(tc["test_labels"].size),
                            "elapsed_s": time.time() - t0,
                        }) + "\n")
                    print(f"  {task_name}: [{(time.time()-t0):.1f}s]")
                except Exception as e:
                    print(f"  {task_name} FAIL: {e}")
                    out_f.write(json.dumps({
                        "run_id": run_id, "arch": arch,
                        "task_name": task_name, "error": str(e),
                    }) + "\n")
            del model
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-ids", nargs="+", default=None)
    ap.add_argument("--tasks", nargs="+", default=None)
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument(
        "--aggregation", choices=["last_position", "full_window"],
        default="last_position",
    )
    args = ap.parse_args()
    run_probing(
        run_ids=args.run_ids, task_names=args.tasks,
        k_values=args.k_values, include_baselines=not args.skip_baselines,
        aggregation=args.aggregation,
    )


if __name__ == "__main__":
    main()
