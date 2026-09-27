"""§ 5.4 — HH-RLHF preference decomposition (protocol 2.0.0 port).

Code-faithful port of the paper's RLHF decomposition —
`origin/han-phase7-agent-c @ 023d52c24`
(`case_studies/hh_rlhf/decompose_hh_rlhf.py` +
`case_studies/_arch_utils.encode_per_position`), pinned by
`task_hunt/COMPOSITION_AUDIT.md § 6` — with the quantitative head the
ACTMIX ablation needs (the paper's own table column "N/20 semantic"
is judge-graded and out of scope; its "% mass" column and the
length-spurious diagnostic are computed here verbatim-equivalent):

1. Reads the rebuilt HH-RLHF cache (`$TEMP_BENCH_HH_RLHF_DIR`,
   default `/workspace/caches/rlhf/cached_hh_rlhf`) — built by
   `experiments.explorations.actmix_rlhf.build_cache`, integrity-gated
   against phase-7's recorded response-length t-test.
2. Encodes each side per position: per-token archs encode every
   position; window archs (arch.config.T > 1) slide stride-1 T-windows
   with RIGHT-EDGE attribution (audit § 6 convention; positions
   0..T-2 zero). Response-mask mean → (N, d_sae) per side.
3. Metrics: `preference_auc_k{20,50}` (5-fold seeded CV: rank by
   train-fold |mean_rejected − mean_chosen|, signed top-K projection,
   held-out paired ordering AUC), `mass_at_20` (paper's mass
   concentration), top-20 length-Pearson diagnostics (the paper's
   length-spurious signature), realized l0 per encode unit over
   response positions.
4. Within-window shuffle twin for T > 1 (per-sliding-window
   independent input permutation, seed 42 — protocol semantics shared
   with `evals/em.py` / Aniket's `shuffles.py`); for T = 1 the shuffle
   is the identity BY CONSTRUCTION and is not simulated.

The shared implementation lives HERE so the canonical-runner arm and
the paper-match case-study (`experiments/explorations/actmix_rlhf/`)
cannot diverge — the exploration's `decomp.py` re-exports these
functions.

Smoke mode returns ``{"smoke_ok": 1.0}`` without touching the cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from temp_bench.interfaces.architecture import TempBenchArch
from temp_bench.interfaces.evaluator import EvalSpec, Evaluator

DEFAULT_CACHE_DIR = "/workspace/caches/rlhf/cached_hh_rlhf"
SHUFFLE_SEED = 42
K_PRIMARY = 20
N_FOLDS = 5
CV_SEED = 42


# ────────────────────────────── shared implementation (both arms) ──

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
    """((N, d_sae) response-mean features, realized-l0 stats).

    encode_fn: (B, T, d_in)->(B, d_sae) window archs (T>1), or
    (B, d_in)->(B, d_sae) per-token (T==1). acts: (N, S, d_in) fp16.
    """
    N, S, d_in = acts.shape
    out = np.zeros((N, d_sae), dtype=np.float32)
    nnz_sum, unit_count = 0.0, 0
    for i in range(0, N, batch):
        j = min(i + batch, N)
        x = torch.from_numpy(np.asarray(acts[i:j])).to(device=device,
                                                       dtype=dtype)
        m = torch.from_numpy(resp_mask[i:j]).to(device)
        B = x.shape[0]
        if T == 1:
            z = encode_fn(x.reshape(B * S, d_in))
            if z.dim() == 3:
                z = z.squeeze(1)
            z = z.reshape(B, S, -1)
        else:
            w = x.unfold(1, T, 1).movedim(-1, 2)
            K = w.shape[1]
            wf = w.reshape(B * K, T, d_in)
            if shuffle_seed is not None:
                wf = _shuffle_windows(wf, seed=shuffle_seed + i)
            zw = encode_fn(wf)
            if zw.dim() == 3:
                zw = zw.squeeze(1)
            z = torch.zeros((B, S, zw.shape[-1]), dtype=zw.dtype,
                            device=zw.device)
            z[:, T - 1: T - 1 + K] = zw.view(B, K, -1)
        mf = m.float().unsqueeze(-1)
        sums = (z * mf).sum(dim=1)
        counts = mf.sum(dim=1).clamp(min=1)
        out[i:j] = (sums / counts).float().cpu().numpy()
        resp_units = (z != 0).float().sum(-1) * m.float()
        nnz_sum += float(resp_units.sum().item())
        unit_count += int(m.sum().item())
        del x, z, m
    l0 = nnz_sum / max(unit_count, 1)
    return out, {"l0_per_unit": l0, "n_units": unit_count}


def preference_auc(chosen_pe, rejected_pe, valid, *, k: int = K_PRIMARY,
                   n_folds: int = N_FOLDS, seed: int = CV_SEED) -> dict:
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
        aucs.append(float(((s_r > s_c).sum() + 0.5 * (s_r == s_c).sum())
                          / len(te)))
        top_sets.append(set(int(t) for t in top))
    inter = set.intersection(*top_sets) if top_sets else set()
    return {"auc_mean": float(np.mean(aucs)),
            "auc_folds": [float(a) for a in aucs],
            "top_overlap_frac": len(inter) / max(k, 1),
            "k": k, "n_folds": n_folds}


def mass_at_k(chosen_pe, rejected_pe, valid, *, k: int = K_PRIMARY) -> float:
    diff = rejected_pe[valid].mean(0) - chosen_pe[valid].mean(0)
    a = np.abs(diff)
    top = np.argsort(-a)[:k]
    denom = float(a.sum())
    return float(a[top].sum() / denom) if denom > 0 else 0.0


def length_pearson_topk(chosen_pe, rejected_pe, chosen_len, rejected_len,
                        valid, *, k: int = K_PRIMARY) -> dict:
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


# ────────────────────────────── the evaluator ──────────────────────

class RLHFEval(Evaluator):
    """§ 5.4 — HH-RLHF preference decomposition (paper aggregation +
    computable quantitative head + shuffle control)."""

    name = "rlhf"
    protocol_version = "2.0.0"

    def eval(self, model: TempBenchArch, spec: EvalSpec) -> dict[str, float]:
        if spec.smoke:
            return {"smoke_ok": 1.0}

        cache = Path(os.environ.get("TEMP_BENCH_HH_RLHF_DIR",
                                    DEFAULT_CACHE_DIR))
        if not (cache / "chosen.npz").exists():
            raise FileNotFoundError(
                f"HH-RLHF cache missing at {cache}; run "
                "experiments.explorations.actmix_rlhf.build_cache first.")
        chosen = np.load(cache / "chosen.npz")
        rejected = np.load(cache / "rejected.npz")
        c_acts, r_acts = chosen["acts"], rejected["acts"]
        c_mask, r_mask = chosen["response_mask"], rejected["response_mask"]
        c_len = chosen["response_len"].astype(np.float64)
        r_len = rejected["response_len"].astype(np.float64)
        valid = (c_len > 0) & (r_len > 0)

        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        device = next(model.parameters()).device
        T = int(model.config.T)
        d_sae = int(model.config.d_sae)
        enc = lambda x: model.encode(x)

        metrics: dict[str, float] = {
            "n_pairs": float(len(c_len)),
            "n_valid": float(valid.sum()),
            "T": float(T),
        }

        def run_variant(shuffle_seed):
            c_pe, c_l0 = aggregate_response_mean(
                enc, c_acts, c_mask, T=T, d_sae=d_sae, device=device,
                shuffle_seed=shuffle_seed)
            r_pe, r_l0 = aggregate_response_mean(
                enc, r_acts, r_mask, T=T, d_sae=d_sae, device=device,
                shuffle_seed=shuffle_seed)
            auc20 = preference_auc(c_pe, r_pe, valid, k=20)
            auc50 = preference_auc(c_pe, r_pe, valid, k=50)
            lp = length_pearson_topk(c_pe, r_pe, c_len, r_len, valid)
            return {
                "preference_auc_k20": auc20["auc_mean"],
                "preference_auc_k50": auc50["auc_mean"],
                "auc_k20_fold_min": min(auc20["auc_folds"]),
                "auc_k20_fold_max": max(auc20["auc_folds"]),
                "top20_overlap_frac": auc20["top_overlap_frac"],
                "mass_at_20": mass_at_k(c_pe, r_pe, valid),
                "len_mean_abs_r": lp["mean_abs_r"],
                "len_n_spurious": float(lp["n_spurious_r_gt_0.5"]),
                "l0_per_unit": 0.5 * (c_l0["l0_per_unit"]
                                      + r_l0["l0_per_unit"]),
            }

        plain = run_variant(None)
        metrics.update(plain)
        if T > 1:
            sh = run_variant(SHUFFLE_SEED)
            for k, v in sh.items():
                metrics[f"shuffled_{k}"] = v
            metrics["shuffle_gap_auc_k20"] = (
                plain["preference_auc_k20"] - sh["preference_auc_k20"])
            metrics["shuffle_gap_mass_at_20"] = (
                plain["mass_at_20"] - sh["mass_at_20"])
        return metrics

    def primary_metric(self) -> str:
        return "preference_auc_k20"
