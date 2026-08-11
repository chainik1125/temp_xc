"""The denoise-after-steer hook must behave identically under KV-cached decode
and under a single full-sequence forward.

That equivalence is the whole correctness claim of the manual window buffer:
during generation the hook only ever sees (B, 1, d) after prefill, so if the
buffer is off by one the projector silently reads a shifted window and every
delta-gc number downstream is measuring the wrong thing while still looking
perfectly plausible.

    uv run --no-sync --with torch --with numpy python \
        experiments/backtracking_steering_dsm/test_w6_hook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.backtracking_steering_dsm.w6 import (  # noqa: E402
    SteerDenoiseHook, denoise_w6,
)
from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE  # noqa: E402

D, T, H, B, S_PRE, S_TOT = 8, 6, 16, 2, 10, 14
ARCH = "topk_sae_w6"


def _reference(sae, x_full, mags, vec):
    """Steer every position, then project every position that has a full
    trailing-T window of STEERED (unprojected) activations."""
    steered = x_full + mags.view(-1, 1, 1) * vec
    out = steered.clone()
    for j in range(T - 1, x_full.shape[1]):
        win = steered[:, j - T + 1:j + 1, :].reshape(B, T * D)
        rec = denoise_w6(sae, ARCH, win).reshape(B, T, D)
        out[:, j, :] = rec[:, -1, :]
    return steered, out


def main() -> int:
    torch.manual_seed(0)
    sae = TopKSAE(T * D, H, 4).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    vec = torch.randn(D)
    mags = torch.tensor([1.0, -2.0])
    x_full = torch.randn(B, S_TOT, D)

    steered, expected = _reference(sae, x_full, mags, vec)

    hook = SteerDenoiseHook(vec, proj=sae, proj_arch=ARCH, T=T)
    hook.magnitudes = mags
    chunks = [hook(None, None, x_full[:, :S_PRE, :])]
    for j in range(S_PRE, S_TOT):                       # KV-cached decode steps
        chunks.append(hook(None, None, x_full[:, j:j + 1, :]))
    got = torch.cat(chunks, dim=1)

    assert got.shape == expected.shape, (got.shape, expected.shape)
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-4)

    # The first T-1 positions have no history and must pass through steered
    # but unprojected.
    torch.testing.assert_close(got[:, :T - 1, :], steered[:, :T - 1, :])
    # ...and everything after must actually have been rewritten.
    assert not torch.allclose(got[:, T - 1:, :], steered[:, T - 1:, :])

    n_expected = B * (S_TOT - (T - 1))
    assert hook.n_projected == n_expected, (hook.n_projected, n_expected)
    assert hook.n_passthrough == B * (T - 1), hook.n_passthrough

    # A prefill resets the buffer: replaying the same panel must reproduce the
    # same output, otherwise state leaks between _generate_panels chunks.
    hook.n_projected = hook.n_passthrough = 0
    again = torch.cat(
        [hook(None, None, x_full[:, :S_PRE, :])]
        + [hook(None, None, x_full[:, j:j + 1, :]) for j in range(S_PRE, S_TOT)],
        dim=1)
    torch.testing.assert_close(again, expected, atol=1e-5, rtol=1e-4)

    # Magnitude 0 with no projector is the reference _Hook no-op.
    plain = SteerDenoiseHook(vec)
    plain.magnitudes = torch.zeros(B)
    torch.testing.assert_close(plain(None, None, x_full), x_full)

    print("OK: decode-time buffer matches full-sequence projection "
          f"({hook.n_projected} positions projected, "
          f"{hook.n_passthrough} passed through)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
