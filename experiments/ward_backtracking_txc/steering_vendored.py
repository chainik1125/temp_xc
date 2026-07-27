"""The paper's steering hooks, copied verbatim, plus one new reduction.

`_build_v7_hook` and `_build_pp_hook` are copied CHARACTER-FOR-CHARACTER from
`origin/temp-bench-anon:src/temp_bench/case_studies/steering.py` (only the `TempBenchArch` type
annotation is dropped, since that framework is not vendored). They are not reimplemented. The
first version of this experiment did reimplement them and the run failed on three of my own
bugs; this repo's `lessons_learned.md` already says to delegate to the reference's function
rather than rewrite the prep, and that rule applies to baselines as much as to recipes.

`_build_slab_hook` is the ONE new thing: identical to `_build_v7_hook` in windowing, clamping
and dtype handling, differing in a single line -- it writes the per-position delta where v7
writes the window mean broadcast. Keeping it a copy-with-one-line-changed is deliberate, so
that any difference in the results is attributable to that line and not to incidental
differences in how the two hooks tile, clamp or cast.

`TXCAdapter` supplies `encode`/`decode` copied from
`origin/temp-bench-anon:src/temp_bench/architectures/txc_base.py`, so the code the hooks call
is also the reference's. Note `encode` returns `(B, 1, d_sae)` and `decode` adds a PER-POSITION
`b_dec` of shape `(T, d_in)`; the bias cancels in `x_hat_steer - x_hat_orig`, which is why the
delta is still exactly `(s - z_j) * W_dec[j]`.
"""
from typing import Any, Callable

import torch
import torch.nn.functional as F


class TXCAdapter(torch.nn.Module):
    """Minimal carrier for a saved TXCBase checkpoint: encode/decode verbatim from txc_base.py."""

    def __init__(self, W_enc, b_enc, W_dec, b_dec, k_win):
        super().__init__()
        # nn.Parameter, not plain attributes: the vendored hooks call
        # `next(arch.parameters()).dtype` to decide their compute dtype, which raises
        # StopIteration on a module with no registered parameters.
        P = lambda t: torch.nn.Parameter(t, requires_grad=False)
        self.W_enc, self.b_enc = P(W_enc), P(b_enc)
        self.W_dec, self.b_dec = P(W_dec), P(b_dec)
        self.k_win = int(k_win)
        self._T = W_dec.shape[1]

    # ── verbatim from txc_base.py ──
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
        if x.shape[1] != self._T:
            raise ValueError(
                f"TXCBase.encode expects (B, T={self._T}, d_in); got T_input={x.shape[1]}.")
        pre = torch.einsum("btd,tds->bs", x, self.W_enc) + self.b_enc
        vals, idx = pre.topk(self.k_win, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(1, idx, F.relu(vals))
        return z.unsqueeze(1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 3:
            if z.shape[1] != 1:
                raise ValueError(f"TXCBase.decode expects (B, 1, d_sae); got T={z.shape[1]}.")
            z = z.squeeze(1)
        return torch.einsum("bs,std->btd", z, self.W_dec) + self.b_dec


# ─────────────────────────────────────────────────────────────────────────────
# VERBATIM from temp_bench/case_studies/steering.py
# ─────────────────────────────────────────────────────────────────────────────
def _build_v7_hook(arch, *, T: int, strengths_t: torch.Tensor,
                   state: dict[str, Any]) -> Callable:
    """Return a forward hook that implements V7 tiled-broadcast steering."""
    arch_dtype = next(arch.parameters()).dtype

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(arch_dtype)

        n_full = S // T
        block_starts = [i * T for i in range(n_full)]
        if S % T > 0:
            block_starts.append(S - T)

        windows = torch.stack([h_f[:, s:s + T, :] for s in block_starts], dim=1)
        n_blocks = windows.shape[1]
        flat = windows.reshape(Bh * n_blocks, T, d_in)
        with torch.no_grad():
            z = arch.encode(flat)
            x_hat_orig = arch.decode(z)
            z_c = z.clone()
            if z_c.dim() == 3:
                z_c[:, :, feat] = (strengths_t.view(Bh, 1, 1)
                                   .expand(Bh, n_blocks, z_c.shape[-2])
                                   .reshape(Bh * n_blocks, z_c.shape[-2]))
            else:
                z_c[:, feat] = (strengths_t.view(Bh, 1).expand(Bh, n_blocks)
                                .reshape(Bh * n_blocks))
            x_hat_steer = arch.decode(z_c)
            delta_per_pos = (x_hat_steer - x_hat_orig).reshape(Bh, n_blocks, T, d_in)
            delta_avg = delta_per_pos.mean(dim=2)
            state.setdefault("cs", []).append(float(delta_per_pos.norm()))

        h_steered = h_f.clone()
        for bi, s in enumerate(block_starts):
            if s + T > S:
                continue
            h_steered[:, s:s + T, :] = h_f[:, s:s + T, :] + delta_avg[:, bi:bi + 1, :]

        state.setdefault("norms", []).append(float((h_steered - h_f).norm(dim=(1, 2)).mean()))
        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


def _build_pp_hook(arch, *, T: int, strengths_t: torch.Tensor,
                   state: dict[str, Any]) -> Callable:
    """Return a forward hook that implements PP per-position steering."""
    arch_dtype = next(arch.parameters()).dtype

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(arch_dtype)

        K = S - T + 1
        windows = h_f.unfold(dimension=1, size=T, step=1).movedim(-1, 2)
        flat = windows.reshape(Bh * K, T, d_in)
        with torch.no_grad():
            z = arch.encode(flat)
            x_hat_orig = arch.decode(z)
            z_c = z.clone()
            if z_c.dim() == 3:
                z_c[:, :, feat] = (strengths_t.view(Bh, 1, 1).expand(Bh, K, z_c.shape[-2])
                                   .reshape(Bh * K, z_c.shape[-2]))
            else:
                z_c[:, feat] = strengths_t.view(Bh, 1).expand(Bh, K).reshape(Bh * K)
            x_hat_steer = arch.decode(z_c)
            delta = (x_hat_steer - x_hat_orig).reshape(Bh, K, T, d_in)

        h_steered = h_f.clone()
        accum = torch.zeros_like(h_f)
        counts = torch.zeros((Bh, S, 1), dtype=arch_dtype, device=h_f.device)
        for w in range(K):
            accum[:, w:w + T, :] += delta[:, w, :, :]
            counts[:, w:w + T, :] += 1.0
        counts = counts.clamp(min=1.0)
        h_steered = h_f + accum / counts
        state.setdefault("norms", []).append(float((h_steered - h_f).norm(dim=(1, 2)).mean()))
        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# NEW. A copy of _build_v7_hook with exactly one line changed.
# ─────────────────────────────────────────────────────────────────────────────
def _build_slab_hook(arch, *, T: int, strengths_t: torch.Tensor,
                     state: dict[str, Any]) -> Callable:
    """V7's windowing and clamp, writing the PER-POSITION delta instead of its window mean.

    The only difference from `_build_v7_hook` is marked below. Everything else -- the block
    tiling, the right-aligned remainder window, the clamp, the dtype round-trip -- is identical,
    so a difference in outcome isolates the reduction.
    """
    arch_dtype = next(arch.parameters()).dtype

    def hook(module, inp, output):
        h = output[0] if isinstance(output, tuple) else output
        feat = state["feature_idx"]
        if feat is None:
            return None
        Bh, S, d_in = h.shape
        if S < T:
            return None
        b_dtype = h.dtype
        h_f = h.to(arch_dtype)

        n_full = S // T
        block_starts = [i * T for i in range(n_full)]
        if S % T > 0:
            block_starts.append(S - T)

        windows = torch.stack([h_f[:, s:s + T, :] for s in block_starts], dim=1)
        n_blocks = windows.shape[1]
        flat = windows.reshape(Bh * n_blocks, T, d_in)
        with torch.no_grad():
            z = arch.encode(flat)
            x_hat_orig = arch.decode(z)
            z_c = z.clone()
            if z_c.dim() == 3:
                z_c[:, :, feat] = (strengths_t.view(Bh, 1, 1)
                                   .expand(Bh, n_blocks, z_c.shape[-2])
                                   .reshape(Bh * n_blocks, z_c.shape[-2]))
            else:
                z_c[:, feat] = (strengths_t.view(Bh, 1).expand(Bh, n_blocks)
                                .reshape(Bh * n_blocks))
            x_hat_steer = arch.decode(z_c)
            delta_per_pos = (x_hat_steer - x_hat_orig).reshape(Bh, n_blocks, T, d_in)

        h_steered = h_f.clone()
        for bi, s in enumerate(block_starts):
            if s + T > S:
                continue
            # ── THE ONLY CHANGED LINE: per-position rows, not the broadcast window mean ──
            h_steered[:, s:s + T, :] = h_f[:, s:s + T, :] + delta_per_pos[:, bi, :, :]

        state.setdefault("norms", []).append(float((h_steered - h_f).norm(dim=(1, 2)).mean()))
        h_steered = h_steered.to(b_dtype)
        if isinstance(output, tuple):
            return (h_steered,) + output[1:]
        return h_steered

    return hook


def build_steering_hook(arch, *, protocol: str, T: int, strengths_t, state):
    if protocol == "v7":
        return _build_v7_hook(arch, T=T, strengths_t=strengths_t, state=state)
    if protocol == "pp":
        return _build_pp_hook(arch, T=T, strengths_t=strengths_t, state=state)
    if protocol == "slab":
        return _build_slab_hook(arch, T=T, strengths_t=strengths_t, state=state)
    raise ValueError(f"unknown protocol {protocol!r}; expected 'v7', 'pp' or 'slab'")
