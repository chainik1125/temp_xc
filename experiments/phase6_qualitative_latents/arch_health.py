"""Phase 6 arch-health diagnostics: per-arch alive fraction, L0, decoder cos sim.

Implements CLAUDE.md rule #5 sanity checks for each Phase 6 arch so the
summary can cite concrete numbers instead of implying things look fine.

Usage:
    TQDM_DISABLE=1 PYTHONPATH=. .venv/bin/python \
        experiments/phase6_qualitative_latents/arch_health.py \
        [--archs agentic_txc_02 agentic_mlc_08 tsae_paper tsae_ours] \
        [--n-tokens 2048]
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
OUT_DIR = REPO / "experiments/phase6_qualitative_latents/results"

D_IN = 2304
D_SAE = 18_432


def sample_L13(n_tokens: int, seed: int = 0) -> torch.Tensor:
    """Return a (n_tokens, d_in) sample from the L13 cache."""
    arr = np.load(CACHE / "resid_L13.npy", mmap_mode="r")
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, arr.shape[0], n_tokens)
    toks = rng.integers(0, arr.shape[1], n_tokens)
    sample = np.stack(
        [arr[s, t] for s, t in zip(seqs, toks)], axis=0
    ).astype(np.float32)
    return torch.from_numpy(sample)


def sample_multilayer(n_tokens: int, seed: int = 0) -> torch.Tensor | None:
    """Return (n_tokens, 5, d_in) stack if L11-L15 are all cached; else None."""
    lks = [f"resid_L{L}.npy" for L in (11, 12, 13, 14, 15)]
    if not all((CACHE / lk).exists() for lk in lks):
        return None
    rng = np.random.default_rng(seed)
    seqs = rng.integers(0, np.load(CACHE / lks[0], mmap_mode="r").shape[0],
                        n_tokens)
    toks = rng.integers(0, np.load(CACHE / lks[0], mmap_mode="r").shape[1],
                        n_tokens)
    layers = []
    for lk in lks:
        arr = np.load(CACHE / lk, mmap_mode="r")
        a = np.stack([arr[s, t] for s, t in zip(seqs, toks)], axis=0).astype(np.float32)
        layers.append(a)
    return torch.from_numpy(np.stack(layers, axis=1))  # (N, 5, d)


def load_arch(arch: str, device):
    """Shared loader; mirrors encode_archs.load_arch."""
    import sys
    sys.path.insert(0, str(REPO))
    from experiments.phase6_qualitative_latents.encode_archs import load_arch as _la
    return _la(arch, device)


@torch.no_grad()
def diagnose(arch: str, device, n_tokens: int = 2048) -> dict:
    print(f"\n=== {arch} ===")
    model, meta = load_arch(arch, device)
    W = None
    # decoder shape may vary per arch
    for attr in ("W_dec",):
        if hasattr(model, attr):
            W = getattr(model, attr)
    if W is None and hasattr(model, "D"):
        W = model.D  # TFA stores dict as D (d_sae, d_in)
    if W is None:
        print(f"  couldn't find decoder on {arch}; skipping cos sim")
    else:
        # flatten any layer axis for per-feature comparison (MLC has (d_sae, L, d))
        if W.dim() == 3:
            W_flat = W.reshape(W.shape[0], -1)  # (d_sae, L*d)
        else:
            W_flat = W
        Wn = torch.nn.functional.normalize(W_flat, dim=-1)
        idx = torch.randperm(Wn.shape[0], device=device)[:2000]
        sim = (Wn[idx] @ Wn[idx].T).abs()
        sim.fill_diagonal_(0)
        dec_cos = float(sim.sum() / (sim.shape[0] * (sim.shape[0] - 1)))
        dec_norms = (W if W.dim() == 2 else W_flat).norm(dim=-1)
        print(f"  decoder: shape={tuple(W.shape)}  norm mean={dec_norms.mean():.4f}"
              f"  std={dec_norms.std():.4f}")
        print(f"  decoder mean|cos| (2000 pairs off-diag): {dec_cos:.4f}")

    # alive fraction: run sample through encoder, check per-feature activation counts
    if arch == "agentic_mlc_08":
        x = sample_multilayer(n_tokens, seed=0)
        if x is None:
            print("  (can't sample multilayer; only L13 cached. SKIPPING MLC alive)")
            return {"arch": arch}
        x = x.to(device)
    else:
        x = sample_L13(n_tokens, seed=0).to(device)

    if arch == "agentic_txc_02":
        # T=5 window: need (B, T, d). Build non-overlapping windows by reshape.
        # Trim to multiple of T=5.
        T = 5
        n_full = (n_tokens // T) * T
        x = x[:n_full].reshape(-1, T, D_IN)
        z = model.encode(x)
    elif arch == "tsae_paper":
        z = model.encode(x, use_threshold=True)
    elif arch == "tfa_big":
        # TFA needs seq-shaped input; reshape to 128-chunks and use novel_codes
        seq = int(meta.get("T", 128))
        n_full = (n_tokens // seq) * seq
        if n_full == 0:
            print("  too few tokens for TFA seq; skip")
            return {"arch": arch}
        x_seq = x[:n_full].reshape(-1, seq, D_IN)
        _, rd = model(x_seq)
        z = rd["novel_codes"].reshape(-1, D_SAE)
    else:
        z = model.encode(x)

    alive = (z > 0).any(dim=0).float().mean().item()
    per_tok_l0 = (z > 0).float().sum(dim=-1).mean().item()
    max_act = z.max().item()
    n_tokens_used = z.shape[0]

    print(f"  z shape: {tuple(z.shape)}  max activation: {max_act:.3f}")
    print(f"  alive fraction (≥1 activation in {n_tokens_used} tokens): {alive:.4f}")
    print(f"  mean L0 per token: {per_tok_l0:.1f}")

    return {
        "arch": arch,
        "alive_fraction": alive,
        "l0_per_token": per_tok_l0,
        "n_tokens": n_tokens_used,
        "decoder_mean_abs_cos": dec_cos if W is not None else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--archs", type=str, nargs="+",
                   default=["agentic_txc_02", "agentic_mlc_08",
                            "tsae_paper", "tsae_ours", "tfa_big"])
    p.add_argument("--n-tokens", type=int, default=2048)
    args = p.parse_args()

    device = torch.device("cuda")
    results = []
    for arch in args.archs:
        ckpt = REPO / "experiments/phase5_downstream_utility/results/ckpts" / f"{arch}__seed42.pt"
        if not ckpt.exists():
            print(f"\n=== {arch} (SKIP: no ckpt at {ckpt}) ===")
            continue
        try:
            results.append(diagnose(arch, device, args.n_tokens))
        except Exception as e:
            print(f"  ERROR on {arch}: {e}")
            results.append({"arch": arch, "error": str(e)})

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "arch_health.json").write_text(json.dumps(results, indent=2))
    print("\n" + "=" * 60)
    print(f"arch health summary: {OUT_DIR / 'arch_health.json'}")
    print("=" * 60)
    for r in results:
        if "error" in r:
            print(f"  {r['arch']:<20}  ERROR")
        else:
            print(f"  {r['arch']:<20}  alive={r.get('alive_fraction', '?'):.4f}"
                  f"  L0={r.get('l0_per_token', '?'):>6.2f}"
                  f"  dec|cos|={r.get('decoder_mean_abs_cos', float('nan')):.4f}")


if __name__ == "__main__":
    main()
