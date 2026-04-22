"""Compute paper Table 1 metrics for Phase 6 archs.

Metrics (from Ye et al. 2025 §4.3):

  FVE (Fraction Variance Explained)  = 1 - Var(x - xhat) / Var(x)
  Cos Sim                            = mean cosine(x, xhat)
  Fraction Alive                     = fraction of features firing ≥1× on sample
  Activation Smoothness S            = mean_s 1/n'_s sum_i
                                            max_t |f_i(x_t) - f_i(x_{t-1})|
                                            / ||x_t - x_{t-1}||_2
                                       averaged over sequences (paper §4.3)

Paper Table 1 targets on Gemma-2-2b:
  T-SAE     : FVE 0.75, Cos 0.88, Alive 0.78, S 0.10 (H)
  Matryoshka: FVE 0.75, Cos 0.89, Alive 0.76, S 0.14
  BatchTopK : FVE 0.76, Cos 0.89, Alive 0.66, S 0.13
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CACHE = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
CKPT_DIR = REPO / "experiments/phase5_downstream_utility/results/ckpts"
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results"

D_IN = 2304
D_SAE = 18_432


def _sample_sequences(n_seqs: int, seed: int = 0) -> np.ndarray:
    """Return n_seqs × 128-length L13 activation sequences (fp32)."""
    arr = np.load(CACHE / "resid_L13.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    idx = rng.choice(arr.shape[0], size=n_seqs, replace=False)
    return np.asarray(arr[idx], dtype=np.float32)  # (N, L, d)


def load_arch_model(arch: str, device):
    """Load arch for reconstruction eval. Returns (model, meta, arch_kind)."""
    state = torch.load(CKPT_DIR / f"{arch}__seed42.pt",
                       map_location=device, weights_only=False)
    meta = state["meta"]
    kind = None
    if arch == "tsae_paper":
        from src.architectures.tsae_paper import TemporalMatryoshkaBatchTopKSAE
        model = TemporalMatryoshkaBatchTopKSAE(
            D_IN, int(meta["d_sae"]), k=int(meta["k_pos"]),
            group_sizes=list(meta["group_sizes"]),
        ).to(device); kind = "tsae_paper"
    elif arch == "tsae_ours":
        from src.architectures.tsae_ours import TSAEOurs
        model = TSAEOurs(D_IN, D_SAE, k=meta["k_pos"]).to(device)
        kind = "tsae_ours"
    elif arch == "agentic_txc_02":
        from src.architectures.matryoshka_txcdr_contrastive_multiscale import (
            MatryoshkaTXCDRContrastiveMultiscale,
        )
        T = meta["T"]
        k_eff = meta["k_win"] or (meta["k_pos"] * T)
        model = MatryoshkaTXCDRContrastiveMultiscale(
            D_IN, D_SAE, T, k_eff,
            n_contr_scales=meta.get("n_contr_scales", 3),
            gamma=meta.get("gamma", 0.5),
        ).to(device); kind = "txc"
    elif arch == "agentic_mlc_08":
        from src.architectures.mlc_contrastive_multiscale import MLCContrastiveMultiscale
        model = MLCContrastiveMultiscale(
            D_IN, D_SAE, n_layers=5, k=meta["k_pos"],
            gamma=meta.get("gamma", 0.5),
        ).to(device); kind = "mlc"
    elif arch == "tfa_big":
        from src.architectures._tfa_module import TemporalSAE
        model = TemporalSAE(
            dimin=D_IN, width=D_SAE, n_heads=4,
            sae_diff_type="topk", kval_topk=meta["k_pos"],
            tied_weights=True, n_attn_layers=1,
            bottleneck_factor=4, use_pos_encoding=False,
        ).to(device); kind = "tfa"
    else:
        raise ValueError(arch)
    cast = {k: v.to(torch.float32) if v.dtype == torch.float16 else v
            for k, v in state["state_dict"].items()}
    model.load_state_dict(cast)
    model.eval()
    return model, meta, kind


@torch.no_grad()
def reconstruct(model, arch_kind: str, seqs: np.ndarray, device):
    """Return (xhat, z) for the per-token L13 input.

    For archs that need context (TXC windows, TFA seq, MLC stack) we
    approximate: TXC uses edge-padded T=5 windows; TFA feeds whole seqs;
    MLC would need multi-layer activations which we don't have
    here, so we skip MLC in this eval.
    """
    N, L, d = seqs.shape
    if arch_kind == "mlc":
        # Needs L11-L15 activations. Skip for this eval.
        return None, None
    x = torch.from_numpy(seqs).to(device)  # (N, L, d)
    if arch_kind in ("tsae_paper",):
        flat = x.reshape(-1, d)
        z = model.encode(flat, use_threshold=True)
        xhat = model.decode(z).reshape(N, L, d)
        z = z.reshape(N, L, -1)
    elif arch_kind == "tsae_ours":
        flat = x.reshape(-1, d)
        z = model.encode(flat)
        xhat = (z @ model.W_dec + model.b_dec).reshape(N, L, d)
        z = z.reshape(N, L, -1)
    elif arch_kind == "txc":
        # T=5 window, edge-pad, assign center-token z
        T = 5; half = T // 2
        # build shifted stacks: (N, L, T, d) by taking x[:, t-2:t+3] with edge pad
        xpad = torch.nn.functional.pad(
            x.permute(0, 2, 1), (half, half), mode="replicate"
        ).permute(0, 2, 1)  # (N, L+T-1, d)
        windows = xpad.unfold(1, T, 1)  # (N, L, d, T)
        windows = windows.permute(0, 1, 3, 2).contiguous()  # (N, L, T, d)
        # Encode in batches
        BATCH = 256
        z_rows = []; xhat_rows = []
        flat = windows.reshape(-1, T, d)  # (N*L, T, d)
        for b0 in range(0, flat.shape[0], BATCH):
            b1 = min(b0 + BATCH, flat.shape[0])
            zi = model.encode(flat[b0:b1])  # (B, d_sae)
            # Full decoder on highest scale to reconstruct the T-token window center
            xhat_i = model.decode_scale(zi, model.T - 1)  # (B, T, d)
            center = xhat_i[:, T // 2, :]
            z_rows.append(zi)
            xhat_rows.append(center)
        z = torch.cat(z_rows, dim=0).reshape(N, L, -1)
        xhat = torch.cat(xhat_rows, dim=0).reshape(N, L, d)
    elif arch_kind == "tfa":
        seq = 128
        # whole seq at once
        xhat_list = []; z_list = []
        BATCH = 8
        for b0 in range(0, N, BATCH):
            b1 = min(b0 + BATCH, N)
            xhat_i, rd = model(x[b0:b1])
            xhat_list.append(xhat_i)
            z_list.append(rd["novel_codes"])
        xhat = torch.cat(xhat_list, dim=0)
        z = torch.cat(z_list, dim=0)
    return xhat, z


@torch.no_grad()
def paper_metrics(arch: str, device, n_seqs: int = 256) -> dict:
    print(f"[{arch}] sampling {n_seqs} sequences...")
    seqs = _sample_sequences(n_seqs)
    model, meta, kind = load_arch_model(arch, device)
    xhat, z = reconstruct(model, kind, seqs, device)
    if xhat is None:
        print(f"  [{arch}] skipped (multi-layer input not available locally)")
        return {"arch": arch, "skipped": True}

    x = torch.from_numpy(seqs).to(device)
    # FVE
    var_x = x.float().var(dim=(0, 1))  # (d,)
    var_err = (x - xhat).float().var(dim=(0, 1))
    fve = float(1 - (var_err.mean() / var_x.mean()))

    # Cos Sim
    x_flat = x.reshape(-1, x.shape[-1])
    xh_flat = xhat.reshape(-1, xhat.shape[-1])
    cs = torch.nn.functional.cosine_similarity(x_flat, xh_flat, dim=-1).mean()
    cos_sim = float(cs)

    # Alive fraction on these sequences' latents
    z_flat = z.reshape(-1, z.shape[-1])
    alive = float((z_flat > 0).any(dim=0).float().mean())
    l0 = float((z_flat > 0).sum(dim=-1).float().mean())

    # Smoothness S — paper §4.3
    # For each seq, active features = features that fire ≥1× on that seq.
    # Delta_s = mean_i max_t |f_i(x_t) - f_i(x_{t-1})| / ||x_t - x_{t-1}||_2
    # S = mean over seqs. We restrict to the high-level prefix if applicable.
    d_sae = z.shape[-1]
    if kind == "tsae_paper":
        hi = int(meta["group_sizes"][0])  # 3686
    elif kind in ("mlc", "tsae_ours"):
        hi = d_sae // 2
    elif kind == "txc":
        hi = d_sae // 5  # scale-1 = d_sae / T = 3686
    else:
        hi = d_sae  # tfa: no H/L split
    z_hi = z[..., :hi]  # (N, L, hi)
    dz = (z_hi[:, 1:, :] - z_hi[:, :-1, :]).abs()  # (N, L-1, hi)
    dx = (x[:, 1:, :] - x[:, :-1, :]).float().norm(dim=-1).clamp(min=1e-6)  # (N, L-1)
    # Per-feature active mask on each seq: active if ≥1 activation in seq
    active_mask = (z_hi > 0).any(dim=1).float()  # (N, hi)
    n_active = active_mask.sum(dim=-1).clamp(min=1.0)  # (N,)
    # Per-feature max delta normalized by dx
    max_dz_over_dx = (dz / dx.unsqueeze(-1)).max(dim=1).values  # (N, hi)
    per_seq = (max_dz_over_dx * active_mask).sum(dim=-1) / n_active  # (N,)
    S = float(per_seq.mean())

    print(f"  FVE={fve:.4f}  CosSim={cos_sim:.4f}  Alive={alive:.4f}  S(H)={S:.4f}")
    return {
        "arch": arch, "fve": fve, "cos_sim": cos_sim, "alive": alive,
        "l0": l0, "smoothness_H": S, "n_seqs": n_seqs,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours", "tfa_big"])
    p.add_argument("--n-seqs", type=int, default=256)
    args = p.parse_args()

    device = torch.device("cuda")
    results = []
    for arch in args.archs:
        if not (CKPT_DIR / f"{arch}__seed42.pt").exists():
            print(f"skip {arch} (no ckpt)")
            continue
        try:
            results.append(paper_metrics(arch, device, args.n_seqs))
        except Exception as e:
            print(f"  ERR {arch}: {e}")
            results.append({"arch": arch, "error": str(e)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "paper_metrics.json").write_text(json.dumps(results, indent=2))
    print(f"\nwrote {OUT_DIR / 'paper_metrics.json'}")
    print("\n" + "=" * 75)
    print(f"{'arch':<18}  {'FVE':>7}  {'CosSim':>7}  {'Alive':>7}  {'L0':>7}  {'S(H)':>7}")
    print("-" * 75)
    for r in results:
        if "error" in r or r.get("skipped"):
            print(f"{r['arch']:<18}  (skipped/error)")
            continue
        print(f"{r['arch']:<18}  {r['fve']:>7.4f}  {r['cos_sim']:>7.4f}"
              f"  {r['alive']:>7.4f}  {r['l0']:>7.2f}  {r['smoothness_H']:>7.4f}")


if __name__ == "__main__":
    main()
