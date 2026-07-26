"""ACTMIX RLHF — shared decomposition + metrics (BOTH arms use this).

One implementation so the paper-match case-study and the btk-only
evaluator cannot diverge:

- `aggregate_response_mean` — the paper's aggregation (023d52c24
  `_arch_utils.encode_per_position` + `decompose_hh_rlhf.py`): per-token
  archs encode every position; window archs slide T-windows stride 1
  and attribute the code to the RIGHT edge (positions 0..T-2 zero);
  per-example mean over response_mask positions. Optional within-window
  shuffle (per-sliding-window independent permutation, seed 42 —
  protocol semantics of `evals/em.py` / Aniket's `shuffles.py`; for
  T = 1 shuffle is the identity BY CONSTRUCTION and is not simulated).
- `preference_auc` — the ablation's primary quantitative currency
  (the paper's own table is judge-graded semantics + mass; this is the
  computable head): 5-fold CV over pairs (seeded); per fold rank
  features by |mean_rejected − mean_chosen| on TRAIN pairs, project
  each side onto the signed top-K diff vector, AUC = fraction of TEST
  pairs with score(rejected) > score(chosen) (ties 0.5).
- `mass_at_k` — |diff|-mass concentration of the top-K (the paper
  table's "% mass" column, judge-free).
- `length_pearson_topk` — the paper's length-spurious diagnostic:
  per top-K feature, Pearson r between per-pair feature diff and
  response-length diff.
- `realized_l0` — mean nonzero code entries per encode unit (token or
  window) over RESPONSE-attributed positions.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _shuffle_windows(w: torch.Tensor, seed: int) -> torch.Tensor:
    """Per-row independent permutation of the T axis of (B, T, d)."""
    g = torch.Generator().manual_seed(seed)
    perms = torch.argsort(torch.rand(w.shape[0], w.shape[1], generator=g),
                          dim=1)
    idx = perms.unsqueeze(-1).expand(-1, -1, w.shape[2]).to(w.device)
    return torch.gather(w, 1, idx)


@torch.no_grad()
def aggregate_response_mean(encode_fn, acts: np.ndarray, resp_mask: np.ndarray,
                            *, T: int, d_sae: int, device,
                            shuffle_seed: int | None = None,
                            batch: int = 8,
                            dtype=torch.float32):
    """Return ((N, d_sae) response-mean features, realized-l0 stats).

    `encode_fn`: (B, T, d_in)->(B, d_sae) for window archs (T>1), or
    (B, d_in)->(B, d_sae) per-token (T==1). acts: (N, S, d_in) fp16.
    """
    N, S, d_in = acts.shape
    out = np.zeros((N, d_sae), dtype=np.float32)
    nnz_sum, unit_count = 0.0, 0
    for i in range(0, N, batch):
        j = min(i + batch, N)
        x = torch.from_numpy(np.asarray(acts[i:j])).to(device=device,
                                                       dtype=dtype)
        m = torch.from_numpy(resp_mask[i:j]).to(device)          # (B, S)
        B = x.shape[0]
        if T == 1:
            z = encode_fn(x.reshape(B * S, d_in)).reshape(B, S, -1)
        else:
            w = x.unfold(1, T, 1).movedim(-1, 2)                 # (B, K, T, d)
            K = w.shape[1]
            wf = w.reshape(B * K, T, d_in)
            if shuffle_seed is not None:
                wf = _shuffle_windows(wf, seed=shuffle_seed + i)
            zw = encode_fn(wf)                                   # (B*K, d_sae)
            if zw.dim() == 3:
                zw = zw.squeeze(1)
            z = torch.zeros((B, S, zw.shape[-1]), dtype=zw.dtype,
                            device=zw.device)
            z[:, T - 1: T - 1 + K] = zw.view(B, K, -1)
        mf = m.float().unsqueeze(-1)
        sums = (z * mf).sum(dim=1)
        counts = mf.sum(dim=1).clamp(min=1)
        out[i:j] = (sums / counts).float().cpu().numpy()
        # realized l0 over response-attributed encode units
        resp_units = (z != 0).float().sum(-1) * m.float()
        nnz_sum += float(resp_units.sum().item())
        unit_count += int(m.sum().item())
        del x, z, m
    l0 = nnz_sum / max(unit_count, 1)
    return out, {"l0_per_unit": l0, "n_units": unit_count}


def preference_auc(chosen_pe: np.ndarray, rejected_pe: np.ndarray,
                   valid: np.ndarray, *, k: int = 20, n_folds: int = 5,
                   seed: int = 42) -> dict:
    idx = np.flatnonzero(valid)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(idx))
    folds = np.array_split(order, n_folds)
    aucs, top_sets = [], []
    for f in range(n_folds):
        te = idx[folds[f]]
        tr = idx[np.concatenate([folds[g] for g in range(n_folds)
                                 if g != f])]
        diff = rejected_pe[tr].mean(0) - chosen_pe[tr].mean(0)
        top = np.argsort(-np.abs(diff))[:k]
        w = np.zeros_like(diff)
        w[top] = diff[top]
        s_r = rejected_pe[te] @ w
        s_c = chosen_pe[te] @ w
        auc = float(((s_r > s_c).sum() + 0.5 * (s_r == s_c).sum())
                    / len(te))
        aucs.append(auc)
        top_sets.append(set(int(t) for t in top))
    inter = set.intersection(*top_sets) if top_sets else set()
    return {"auc_mean": float(np.mean(aucs)),
            "auc_folds": [float(a) for a in aucs],
            "top_overlap_frac": len(inter) / max(k, 1),
            "k": k, "n_folds": n_folds}


def mass_at_k(chosen_pe, rejected_pe, valid, *, k: int = 20) -> float:
    diff = rejected_pe[valid].mean(0) - chosen_pe[valid].mean(0)
    a = np.abs(diff)
    top = np.argsort(-a)[:k]
    denom = float(a.sum())
    return float(a[top].sum() / denom) if denom > 0 else 0.0


def length_pearson_topk(chosen_pe, rejected_pe, chosen_len, rejected_len,
                        valid, *, k: int = 20) -> dict:
    from scipy.stats import pearsonr
    diff = rejected_pe[valid].mean(0) - chosen_pe[valid].mean(0)
    top = np.argsort(-np.abs(diff))[:k]
    ld = (rejected_len[valid] - chosen_len[valid]).astype(np.float64)
    rs = []
    for j in top:
        fd = rejected_pe[valid, j] - chosen_pe[valid, j]
        rs.append(0.0 if fd.std() < 1e-8 else float(pearsonr(fd, ld)[0]))
    rs = np.array(rs)
    return {"top_k_length_r": rs.tolist(),
            "mean_abs_r": float(np.abs(rs).mean()),
            "n_spurious_r_gt_0.5": int((np.abs(rs) > 0.5).sum()),
            "top_k_indices": [int(t) for t in top]}
