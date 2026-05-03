"""V15 — Attention-weighted broadcast — hill-climb iter 2.

V7/V9 broadcast a SINGLE uniform δ across all T positions in a window.
But for TXCSoftMaxPool (Galaxy 23 = T=5), the encoder uses softmax-weighted
pooling: position weights for feature j are softmax(pre_pos[:, j] / τ_j).

V15: at intervention time, USE the encoder's actual per-position softmax
weights to write the decoded δ proportionally. Position t in block gets
δ × softmax_weight[t, feat].

This is "content-aware position weighting" — the steering follows the
encoder's own per-position attention pattern for this feature in this
context. Predicts: matches the model's natural representation of the
feature.

For non-softmaxpool archs, falls back to encoder L2-magnitude weighting
(same as V10) — but those would just recover V10's failure mode for
Galaxy 23. So this is intended specifically for the softmax-pool family.

Block stride: 2 (= T // 2 at T=5), inheriting V9's denser coverage.
"""
from __future__ import annotations
import argparse, gc, json, os, time
from pathlib import Path
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

# Same imports as V9 — reuse the same scaffolding
import sys
sys.path.insert(0, "/workspace/temp_xc")
from experiments.phase7_unification.case_studies.protocol_hillclimb.intervene_v9_sliding_tb import (
    steer_for_arch as _v9_steer_for_arch, main as _v9_main, _decode_full_window,
    PAPER_STRENGTHS, S_NORMS_DEFAULT,
)


OUT_SUBDIR = "steering_protocol_hillclimb_v15_attn_weighted_tb"


def _v15_build_hook(sae, src_class, T, strengths_t, state):
    """V15: stride-T/2 sliding tiled-broadcast with attention-weighted writes
    using the SoftMaxPool encoder's actual per-position softmax weights."""
    sae_dtype = torch.float32
    has_log_tau = hasattr(sae, "log_tau")  # SoftMaxPool family

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(sae_dtype)
        h_steered = h_f.clone()

        STRIDE = max(1, T // 2)
        block_starts = list(range(0, S - T + 1, STRIDE))
        if not block_starts or block_starts[-1] != S - T:
            block_starts.append(S - T)
        block_starts = sorted(set(block_starts))

        windows = torch.stack([h_f[:, s:s + T, :] for s in block_starts], dim=1)
        n_blocks = windows.shape[1]
        flat = windows.reshape(Bh * n_blocks, T, d_in)
        with torch.no_grad():
            # Compute per-position attention weights for this feature in each window
            # pre_pos[b, t, j] = (W_enc[t] @ x[t])[j], shape (Bh*n_blocks, T, d_sae)
            pre_pos = torch.einsum("btd,tds->bts", flat, sae.W_enc)
            if has_log_tau:
                tau = sae.log_tau[feat].exp().clamp(min=0.05, max=20.0)
                attn_w = torch.softmax(pre_pos[:, :, feat] / tau, dim=1)  # (Bh*n_blocks, T)
            else:
                # Fallback: encoder L2 magnitude weighting
                pos_mag = sae.W_enc[:, :, feat].norm(dim=1)  # (T,)
                attn_w = (pos_mag / pos_mag.sum().clamp(min=1e-8)).view(1, T).expand(pre_pos.size(0), T)
            # Normalize attn_w to sum to T (so total mass matches V7 uniform-T)
            attn_w = attn_w / attn_w.sum(dim=1, keepdim=True).clamp(min=1e-8) * T

            z = sae.encode(flat)
            x_hat_orig = _decode_full_window(sae, src_class, z, T)
            z_c = z.clone().reshape(Bh, n_blocks, -1)
            z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, n_blocks)
            z_c = z_c.reshape(Bh * n_blocks, -1)
            x_hat_steer = _decode_full_window(sae, src_class, z_c, T)
            delta_per_pos = (x_hat_steer - x_hat_orig).reshape(Bh, n_blocks, T, d_in)
            delta_avg = delta_per_pos.mean(dim=2)  # (Bh, n_blocks, d_in)

            # attn_w shape: (Bh*n_blocks, T) → reshape to (Bh, n_blocks, T)
            attn_w_b = attn_w.reshape(Bh, n_blocks, T)

            accum = torch.zeros_like(h_f)
            count = torch.zeros(S, device=h_f.device, dtype=sae_dtype)
            for bi, s in enumerate(block_starts):
                if s + T > S:
                    continue
                # Per-position write: δ_avg × attn_w[t]
                # delta_avg[bi]: (Bh, d_in); attn_w_b[:, bi, :]: (Bh, T)
                # → weighted: (Bh, T, d_in)
                weighted = delta_avg[:, bi:bi+1, :] * attn_w_b[:, bi, :].unsqueeze(-1)
                accum[:, s:s + T, :] += weighted
                count[s:s + T] += 1.0
            count = count.clamp(min=1.0).view(1, S, 1)
            h_steered = h_f + accum / count

        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


# Monkey-patch the V9 hook with V15's hook
import experiments.phase7_unification.case_studies.protocol_hillclimb.intervene_v9_sliding_tb as _v9_mod
_v9_mod._build_hook = _v15_build_hook
_v9_mod.OUT_SUBDIR = OUT_SUBDIR


def main():
    """Wrapper that calls v9's main() but with V15 hook + output dir."""
    _v9_main()


if __name__ == "__main__":
    main()
