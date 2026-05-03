"""V14 — Multi-scale tiled-broadcast — hill-climb iter 2.

Combines THREE scales of tiled-broadcast simultaneously:
- T-scale: V7 with native window-T (here T=5)
- T_mid-scale: tile with T_mid = max(2, T//2) = 2 (V9-like, but at scale 2)
- 1-scale: per-position write (degenerate to V2 PP at T=1, single-token)

For each scale, compute uniform-within-block δ. Sum the three δs at each
position (each scaled by 1/3 to keep total magnitude matched).

Hypothesis: different scales capture different feature dynamics.
Multi-scale concepts (T=5 window features) get the T-scale write;
fine-grained per-token features get the 1-scale write; intermediate
features get the T_mid-scale write.

Risk: at T=1 the encoder uses only 1 position which doesn't give a
useful TXC encoding (TXC was trained at T=5). For the 1-scale step,
fall back to the average decoder direction × strength (V6 dec-broadcast
style) as a degenerate single-pos signal.
"""
from __future__ import annotations
import argparse, gc, json, os, time
from pathlib import Path
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

import sys
sys.path.insert(0, "/workspace/temp_xc")
from experiments.phase7_unification.case_studies.protocol_hillclimb.intervene_v9_sliding_tb import (
    _decode_full_window, main as _v9_main,
)


OUT_SUBDIR = "steering_protocol_hillclimb_v14_multiscale_tb"


def _v14_build_hook(sae, src_class, T, strengths_t, state):
    """V14: sum of 3 tiled-broadcast scales: T-blocks + (T//2)-blocks + V6-style."""
    sae_dtype = torch.float32

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

        # Helper: tile + uniform-broadcast at a given block size
        def tile_broadcast(block_T, stride):
            block_starts = list(range(0, S - block_T + 1, stride))
            if not block_starts or block_starts[-1] != S - block_T:
                block_starts.append(S - block_T)
            block_starts = sorted(set(block_starts))
            windows = torch.stack([h_f[:, s:s + block_T, :] for s in block_starts], dim=1)
            n_blocks = windows.shape[1]
            flat = windows.reshape(Bh * n_blocks, block_T, d_in)
            with torch.no_grad():
                # Encoder may need T positions; pad if block_T < T by repeating last
                if block_T < T:
                    pad = flat[:, -1:, :].expand(-1, T - block_T, -1)
                    enc_input = torch.cat([flat, pad], dim=1)
                else:
                    enc_input = flat
                z = sae.encode(enc_input)
                x_hat_orig = _decode_full_window(sae, src_class, z, T)
                z_c = z.clone().reshape(Bh, n_blocks, -1)
                z_c[:, :, feat] = strengths_t.view(Bh, 1).expand(Bh, n_blocks)
                z_c = z_c.reshape(Bh * n_blocks, -1)
                x_hat_steer = _decode_full_window(sae, src_class, z_c, T)
                # delta over the T positions; if block_T < T, take only first block_T
                delta_per_pos = (x_hat_steer - x_hat_orig).reshape(Bh, n_blocks, T, d_in)
                delta_per_pos = delta_per_pos[:, :, :block_T, :]  # (Bh, n_blocks, block_T, d_in)
                delta_avg = delta_per_pos.mean(dim=2)  # (Bh, n_blocks, d_in)

            accum = torch.zeros_like(h_f)
            count = torch.zeros(S, device=h_f.device, dtype=sae_dtype)
            for bi, s in enumerate(block_starts):
                if s + block_T > S:
                    continue
                accum[:, s:s + block_T, :] += delta_avg[:, bi:bi+1, :]
                count[s:s + block_T] += 1.0
            count = count.clamp(min=1.0).view(1, S, 1)
            return accum / count  # (Bh, S, d_in) per-position averaged δ

        # Three scales
        T_mid = max(2, T // 2)  # T=5 → 2
        scale_T = tile_broadcast(T, stride=max(1, T // 2))   # T-scale, V9-stride
        scale_mid = tile_broadcast(T_mid, stride=1)           # T/2-scale
        # Skip 1-scale at T=1: would require encoder at single pos which is undefined.
        # Use static decoder direction broadcast instead.
        with torch.no_grad():
            # mean over T of W_dec[feat] → (d_in,)
            if hasattr(sae, "W_dec") and sae.W_dec.dim() == 3:
                dir_d_in = sae.W_dec[feat, :, :].mean(dim=0).detach()  # (d_in,)
            else:
                dir_d_in = torch.zeros(d_in, device=h_f.device)
            scale_1_per_batch = strengths_t.view(Bh, 1) * dir_d_in.unsqueeze(0)  # (Bh, d_in)
        scale_1 = scale_1_per_batch.unsqueeze(1).expand(Bh, S, d_in)

        # Sum the three scales (each contributes 1/3 of total δ magnitude)
        h_steered = h_f + (scale_T + scale_mid + scale_1) / 3.0

        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


import experiments.phase7_unification.case_studies.protocol_hillclimb.intervene_v9_sliding_tb as _v9_mod
_v9_mod._build_hook = _v14_build_hook
_v9_mod.OUT_SUBDIR = OUT_SUBDIR


def main():
    _v9_main()


if __name__ == "__main__":
    main()
