"""Phase 1a follow-up: decoder alignment filtered to ALIVE features only.

A feature is "alive" if it activates on some fraction of eval tokens.
For each architecture, we run the model on the held-out eval split,
compute per-feature activation frequency, and filter out dead features
before doing cross-arch alignment.

For TFA we consider novel_codes and pred_codes separately since they
have very different activation patterns (novel sparse, pred dense).

Outputs: results/analysis/decoder_alignment/alive_*.{png,json}
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, "/home/elysium/temp_xc")


CKPT_DIR = "/home/elysium/temp_xc/results/nlp_sweep/gemma/ckpts"
OUT_DIR = "/home/elysium/temp_xc/results/analysis/decoder_alignment"
D_IN = 2304
D_SAE = 18432
T_WIN = 5
K = 100
LAYER = "resid_L25"
ACT_PATH = f"/home/elysium/temp_xc/data/cached_activations/gemma-2-2b-it/fineweb/{LAYER}.npy"
ALIVE_FIRING_THRESHOLD = 0.0001  # must fire on at least 0.01% of tokens to count as alive


def unit_norm(x: torch.Tensor) -> torch.Tensor:
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return x / norms


def best_match(a: torch.Tensor, b: torch.Tensor, batch: int = 512) -> torch.Tensor:
    """Max |cos| between each row in a and every row in b."""
    a = unit_norm(a).float().cuda()
    b = unit_norm(b).float().cuda()
    out = torch.zeros(a.shape[0], device="cuda")
    for s in range(0, a.shape[0], batch):
        sims = a[s : s + batch] @ b.T
        out[s : s + batch] = sims.abs().max(dim=-1).values
    return out.cpu()


@torch.no_grad()
def compute_activation_freq_stacked(state: dict, eval_x: torch.Tensor) -> torch.Tensor:
    """Per-feature fraction of eval tokens where Stacked SAE fires.

    eval_x: (N, 128, d_in). Stacked processes 5-token windows per position.
    We sweep all windows, record which features fire per (token, position)
    average, then return a mean firing rate per feature across all window positions.
    """
    d_sae = D_SAE
    seq_len = eval_x.shape[1]
    n_windows = seq_len - T_WIN + 1

    # Per-position encoders: W_enc[t] = (d_sae, d_in), b_enc[t], b_dec[t]
    # Feature i fires at window w position t if TopK(W_enc[t] @ (x - b_dec[t]) + b_enc[t])[i] > 0
    freq = torch.zeros(d_sae)
    total_positions = 0

    for s in range(0, eval_x.shape[0], 64):
        x = eval_x[s : s + 64].cuda()  # (B, 128, d)
        B = x.shape[0]

        for w in range(n_windows):
            window = x[:, w : w + T_WIN, :]  # (B, T, d)
            for t in range(T_WIN):
                tok = window[:, t, :]  # (B, d)
                W_enc = state[f"saes.{t}.W_enc"].cuda()  # (d_sae, d_in)
                b_enc = state[f"saes.{t}.b_enc"].cuda()  # (d_sae,)
                W_dec = state[f"saes.{t}.W_dec"].cuda()  # (d_in, d_sae)
                b_dec = state[f"saes.{t}.b_dec"].cuda()  # (d_in,)
                pre = (tok - b_dec) @ W_enc.T + b_enc  # (B, d_sae)
                _, idx = pre.topk(K, dim=-1)
                mask = torch.zeros_like(pre)
                mask.scatter_(-1, idx, 1.0)
                # Only count positive activations (after ReLU(topk) they're >0)
                active = (mask > 0) & (pre > 0)
                freq += active.sum(dim=0).cpu().float()
                total_positions += B

    return freq / max(total_positions, 1)


@torch.no_grad()
def compute_activation_freq_txcdr(state: dict, eval_x: torch.Tensor) -> torch.Tensor:
    """Per-feature fraction of eval windows where TXCDR fires."""
    # FastTemporalCrosscoder: encode expands with W_enc (T, d_in, d_sae)
    # pre = einsum('btd,tds->bs', x, W_enc) + b_enc
    W_enc = state["W_enc"].cuda()  # (T, d_in, d_sae)
    b_enc = state["b_enc"].cuda()  # (d_sae,)

    seq_len = eval_x.shape[1]
    n_windows = seq_len - T_WIN + 1
    k_total = K * T_WIN

    freq = torch.zeros(D_SAE)
    total = 0
    for s in range(0, eval_x.shape[0], 64):
        x = eval_x[s : s + 64].cuda()
        B = x.shape[0]
        for w in range(n_windows):
            window = x[:, w : w + T_WIN, :]  # (B, T, d)
            pre = torch.einsum("btd,tds->bs", window, W_enc) + b_enc
            _, idx = pre.topk(k_total, dim=-1)
            mask = torch.zeros_like(pre, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            active = mask & (pre > 0)
            freq += active.sum(dim=0).cpu().float()
            total += B
    return freq / max(total, 1)


@torch.no_grad()
def compute_activation_freq_tfa(
    state: dict, eval_x: torch.Tensor, use_pos: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (novel_freq, pred_freq) for each of 18432 features."""
    from src.architectures._tfa_module import TemporalSAE
    import math

    model = TemporalSAE(
        dimin=D_IN, width=D_SAE, n_heads=4, sae_diff_type="topk", kval_topk=K,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=8,
        use_pos_encoding=use_pos, max_seq_len=512,
    ).cuda()
    model.load_state_dict(state)
    model.eval()

    # Compute scaling factor from a sample batch
    sample = eval_x[:16].cuda()
    scaling_factor = math.sqrt(D_IN) / sample.norm(dim=-1).mean().item()

    novel_freq = torch.zeros(D_SAE)
    pred_freq = torch.zeros(D_SAE)
    total_tokens = 0

    for s in range(0, eval_x.shape[0], 16):
        x = (eval_x[s : s + 16].cuda() * scaling_factor)
        B, T, _ = x.shape
        _, inter = model(x)
        novel = inter["novel_codes"]  # (B, T, d_sae), sparse
        pred = inter["pred_codes"]    # (B, T, d_sae), dense
        novel_active = (novel > 0)
        pred_active = (pred.abs() > 1e-6)
        novel_freq += novel_active.reshape(-1, D_SAE).sum(dim=0).cpu().float()
        pred_freq += pred_active.reshape(-1, D_SAE).sum(dim=0).cpu().float()
        total_tokens += B * T

    return novel_freq / max(total_tokens, 1), pred_freq / max(total_tokens, 1)


def load_stacked_decoder():
    state = torch.load(
        f"{CKPT_DIR}/stacked_sae__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_dec_list = [state[f"saes.{t}.W_dec"] for t in range(T_WIN)]
    W_dec = torch.stack(W_dec_list).mean(dim=0)
    return W_dec.T, state  # (d_sae, d_in)


def load_txcdr_decoder():
    state = torch.load(
        f"{CKPT_DIR}/crosscoder__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    W_dec = state["W_dec"].mean(dim=1)  # (d_sae, d_in)
    return W_dec, state


def load_tfa_decoder():
    state = torch.load(
        f"{CKPT_DIR}/tfa_pos__gemma-2-2b-it__fineweb__{LAYER}__k{K}__seed42.pt",
        map_location="cpu", weights_only=True,
    )
    return state["D"], state  # (d_sae, d_in)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading 2000 held-out eval sequences...")
    arr = np.load(ACT_PATH, mmap_mode="r")
    eval_x = torch.from_numpy(np.array(arr[-2000:])).float()  # (2000, 128, 2304)
    print(f"  eval_x: {list(eval_x.shape)}")

    print("\nLoading decoders + computing activation frequencies...")

    D_stacked, state_stacked = load_stacked_decoder()
    D_txcdr, state_txcdr = load_txcdr_decoder()
    D_tfa, state_tfa = load_tfa_decoder()

    print("  Stacked firing freq (per-token positive-activation rate)...")
    freq_stacked = compute_activation_freq_stacked(state_stacked, eval_x)
    print(f"    alive@>={ALIVE_FIRING_THRESHOLD}: {(freq_stacked > ALIVE_FIRING_THRESHOLD).sum().item()} / {D_SAE}")

    print("  TXCDR firing freq...")
    freq_txcdr = compute_activation_freq_txcdr(state_txcdr, eval_x)
    print(f"    alive: {(freq_txcdr > ALIVE_FIRING_THRESHOLD).sum().item()} / {D_SAE}")

    print("  TFA firing freq (novel and pred)...")
    freq_tfa_novel, freq_tfa_pred = compute_activation_freq_tfa(state_tfa, eval_x)
    print(f"    TFA novel alive: {(freq_tfa_novel > ALIVE_FIRING_THRESHOLD).sum().item()} / {D_SAE}")
    print(f"    TFA pred alive:  {(freq_tfa_pred > ALIVE_FIRING_THRESHOLD).sum().item()} / {D_SAE}")

    # Save freqs for downstream use
    torch.save({
        "freq_stacked": freq_stacked,
        "freq_txcdr": freq_txcdr,
        "freq_tfa_novel": freq_tfa_novel,
        "freq_tfa_pred": freq_tfa_pred,
    }, f"{OUT_DIR}/firing_frequencies.pt")

    # Alive masks
    alive_stacked = freq_stacked > ALIVE_FIRING_THRESHOLD
    alive_txcdr = freq_txcdr > ALIVE_FIRING_THRESHOLD
    alive_tfa_novel = freq_tfa_novel > ALIVE_FIRING_THRESHOLD
    alive_tfa_pred = freq_tfa_pred > ALIVE_FIRING_THRESHOLD
    alive_tfa_any = alive_tfa_novel | alive_tfa_pred

    # Union: features alive via any TFA code-type
    print(f"\n  TFA alive (novel OR pred): {alive_tfa_any.sum().item()} / {D_SAE}")

    # Filtered decoders (alive features only)
    D_stacked_alive = D_stacked[alive_stacked]
    D_txcdr_alive = D_txcdr[alive_txcdr]
    D_tfa_alive = D_tfa[alive_tfa_any]

    print(f"\nFiltered decoder shapes:")
    print(f"  Stacked alive:  {list(D_stacked_alive.shape)}")
    print(f"  TXCDR alive:    {list(D_txcdr_alive.shape)}")
    print(f"  TFA alive:      {list(D_tfa_alive.shape)}")

    print("\nPairwise alignment (alive only, each feature's best match in OTHER arch)...")
    alignment = {
        "tfa_vs_txcdr":     best_match(D_tfa_alive, D_txcdr_alive),
        "tfa_vs_stacked":   best_match(D_tfa_alive, D_stacked_alive),
        "txcdr_vs_tfa":     best_match(D_txcdr_alive, D_tfa_alive),
        "txcdr_vs_stacked": best_match(D_txcdr_alive, D_stacked_alive),
        "stacked_vs_tfa":   best_match(D_stacked_alive, D_tfa_alive),
        "stacked_vs_txcdr": best_match(D_stacked_alive, D_txcdr_alive),
    }

    summary = {}
    for key, vals in alignment.items():
        summary[key] = {
            "n": int(len(vals)),
            "mean": float(vals.mean()),
            "median": float(vals.median()),
            "p10": float(vals.quantile(0.10)),
            "p90": float(vals.quantile(0.90)),
            "frac_above_0.5": float((vals >= 0.5).float().mean()),
            "frac_above_0.7": float((vals >= 0.7).float().mean()),
            "frac_above_0.9": float((vals >= 0.9).float().mean()),
            "frac_below_0.3": float((vals <= 0.3).float().mean()),
        }
        print(f"  {key:20s} (n={summary[key]['n']:5d}): median={summary[key]['median']:.3f}  "
              f">=0.5: {summary[key]['frac_above_0.5']:.3f}  "
              f">=0.7: {summary[key]['frac_above_0.7']:.3f}  "
              f"<=0.3: {summary[key]['frac_below_0.3']:.3f}")

    with open(f"{OUT_DIR}/alive_alignment_summary.json", "w") as f:
        json.dump({
            "thresholds": {"alive_firing_threshold": ALIVE_FIRING_THRESHOLD},
            "alive_counts": {
                "stacked": int(alive_stacked.sum()),
                "txcdr": int(alive_txcdr.sum()),
                "tfa_novel": int(alive_tfa_novel.sum()),
                "tfa_pred": int(alive_tfa_pred.sum()),
                "tfa_any": int(alive_tfa_any.sum()),
            },
            "alignment": summary,
        }, f, indent=2)

    # ── plot: histograms ────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    pairs = [
        ("TFA ↔ TXCDR", alignment["tfa_vs_txcdr"], alignment["txcdr_vs_tfa"], "TFA→TXCDR", "TXCDR→TFA"),
        ("TFA ↔ Stacked", alignment["tfa_vs_stacked"], alignment["stacked_vs_tfa"], "TFA→Stacked", "Stacked→TFA"),
        ("TXCDR ↔ Stacked", alignment["txcdr_vs_stacked"], alignment["stacked_vs_txcdr"], "TXCDR→Stacked", "Stacked→TXCDR"),
    ]
    for ax, (title, a, b, la, lb) in zip(axes, pairs):
        ax.hist(a.numpy(), bins=50, alpha=0.6, label=la)
        ax.hist(b.numpy(), bins=50, alpha=0.6, label=lb)
        ax.set_title(title)
        ax.set_xlabel("best |cos sim|")
        ax.set_ylabel("# alive features")
        ax.legend(fontsize=8)
        ax.axvline(0.3, color="red", linestyle="--", alpha=0.3, label="0.3")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/alive_best_match_hist.png", dpi=120)
    print(f"\n  -> {OUT_DIR}/alive_best_match_hist.png")

    # ── Save per-arch unique features (low alignment = unique) ──────────────
    # Use original feature indices, not the alive-filtered ones
    stacked_idx = torch.where(alive_stacked)[0]
    txcdr_idx = torch.where(alive_txcdr)[0]
    tfa_idx = torch.where(alive_tfa_any)[0]

    tfa_uniqueness = torch.max(alignment["tfa_vs_txcdr"], alignment["tfa_vs_stacked"])
    txcdr_uniqueness = torch.max(alignment["txcdr_vs_tfa"], alignment["txcdr_vs_stacked"])
    stacked_uniqueness = torch.max(alignment["stacked_vs_tfa"], alignment["stacked_vs_txcdr"])

    N_TOP = 50
    sel = {
        "tfa_pos": {
            "feature_indices": tfa_idx[torch.argsort(tfa_uniqueness)[:N_TOP]].tolist(),
            "max_cos_with_other_archs": torch.sort(tfa_uniqueness).values[:N_TOP].tolist(),
        },
        "crosscoder": {
            "feature_indices": txcdr_idx[torch.argsort(txcdr_uniqueness)[:N_TOP]].tolist(),
            "max_cos_with_other_archs": torch.sort(txcdr_uniqueness).values[:N_TOP].tolist(),
        },
        "stacked_sae": {
            "feature_indices": stacked_idx[torch.argsort(stacked_uniqueness)[:N_TOP]].tolist(),
            "max_cos_with_other_archs": torch.sort(stacked_uniqueness).values[:N_TOP].tolist(),
        },
    }
    with open(f"{OUT_DIR}/alive_top_unique.json", "w") as f:
        json.dump(sel, f, indent=2)
    print(f"  -> {OUT_DIR}/alive_top_unique.json")

    # Print top-5 per arch for sanity
    print("\nTop 5 unique alive features per architecture:")
    for arch, d in sel.items():
        print(f"  {arch}:")
        for fi, sc in list(zip(d["feature_indices"], d["max_cos_with_other_archs"]))[:5]:
            print(f"    feat_{fi:5d}  max-cos={sc:.3f}")


if __name__ == "__main__":
    main()
