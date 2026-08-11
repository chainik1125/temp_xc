"""Wave-2 window dictionaries (T=6 on resid_L10) and the denoise-after-steer hook.

The wave-2 trio is trained by experiments/diffusion_txc/psc/psc_train_sae.py via
psc/modal_txc.py: `--hook resid10 --window 6 --k 96 --H 16384` on
NousResearch/Meta-Llama-3.1-8B. Window layout is the trainer's own (line 251-254
of psc_train_sae.py):

    a.unfold(1, T, 1).permute(0, 1, 3, 2).reshape(-1, T * d)

i.e. TIME-MAJOR (T, d), slot 0 = oldest position in the window, slot T-1 = the
current one. d_eff = 6 * 4096 = 24576. `recon` and `dsm` are TopKSAE(d_eff,
16384, 96); `bayes_gate` is BayesGateSAE(d_eff, 16384) whose encode(x, u)
returns (z, g) -- evaluated at u = 0, the clean-input setting.

The hookpoint identity was verified by the predecessor and re-derived here:
psc_train_sae.py::make_hook reads `hidden_states[layer + 1]` for `resid10`,
which is the output of `model.model.layers[10]` -- the same tensor
b1_steer_eval::_Hook is registered on. Steering site and dictionary site are
the same tensor, not merely the same layer index.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.backtracking_steering_dsm.protocol import (
    CKPT_EVERY, D_MODEL, PLAN_STEPS, SETTLE_STEPS, T_WINDOW, TRUNC_MIN_STEP,
)

D_EFF = D_MODEL * T_WINDOW
H_SAE = 16384
K_WINDOW = 96

# Running statistics the trainer keeps for its own bookkeeping and that play no
# part in encode/decode, so a checkpoint may legitimately lack them:
#   steps_since_fire - dead-latent counter, drives AuxK revival during training.
#   rate_ema         - EMA of per-latent gate rate, used only in BayesGateSAE's
#                      rate-KL sparsity term. It is a later, uncommitted
#                      addition to psc_train_sae.py than the running wave-2 job,
#                      so that job's checkpoints will never contain it.
# Anything else missing is a genuine architecture mismatch and still raises.
TRAIN_ONLY_BUFFERS = {"steps_since_fire", "rate_ema"}

W6_ARMS = [
    {"name": "w6_recon", "dir": "txc_w6/txc_recon", "stem": "recon_s0"},
    {"name": "w6_dsm", "dir": "txc_w6/txc_dsm", "stem": "dsm_s0"},
    {"name": "w6_bayes", "dir": "txc_w6/txc_bayes", "stem": "bayes_gate_s0"},
]


def load_w6(ckpt_path: str | Path, device: str = "cuda"):
    """-> (model, meta). Architecture is inferred from the state_dict keys:
    BayesGateSAE carries the gate parameters, TopKSAE does not."""
    import sys
    sys.path.insert(0, "/work")
    sd = torch.load(Path(ckpt_path), map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    is_bayes = "theta0" in sd
    from experiments.diffusion_txc.psc.psc_train_sae import BayesGateSAE
    from experiments.diffusion_txc.topk_vs_topkdiff.sae import TopKSAE
    d_eff, H = sd["W_enc"].shape
    if is_bayes:
        m = BayesGateSAE(d_eff, H)
    else:
        m = TopKSAE(d_eff, H, K_WINDOW)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    bad = [k for k in missing if k not in TRAIN_ONLY_BUFFERS]
    if bad:
        raise RuntimeError(f"{ckpt_path}: missing keys {bad}")
    m.eval().to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m, {"arch": "bayes_gate_w6" if is_bayes else "topk_sae_w6",
               "d_eff": int(d_eff), "d_sae": int(H), "T": T_WINDOW,
               "k": None if is_bayes else K_WINDOW,
               "unexpected_keys": list(unexpected),
               "absent_train_buffers": sorted(missing)}


@torch.no_grad()
def encode_w6(model, arch: str, x_flat: torch.Tensor) -> torch.Tensor:
    """(N, d_eff) -> (N, d_sae). BayesGate is read at u = 0 (clean input)."""
    if arch == "bayes_gate_w6":
        z, _ = model.encode(x_flat, 0.0)
        return z
    return model.encode(x_flat)


@torch.no_grad()
def decode_w6(model, z: torch.Tensor) -> torch.Tensor:
    return z @ model.W_dec + model.b_dec


@torch.no_grad()
def denoise_w6(model, arch: str, x_flat: torch.Tensor) -> torch.Tensor:
    """The dictionary's denoising map: (N, d_eff) -> (N, d_eff)."""
    return decode_w6(model, encode_w6(model, arch, x_flat))


def decoder_dirs_w6(model) -> dict:
    """W_dec is (d_sae, T*d) time-major -> per-slot decoder rows.

    pos0 is slot 0 (the oldest position in the window), matching
    config.yaml mining.steer_decoder_pos = 0 and architectures.py's
    `W_dec[:, 0, :]` for the Stage B TXC.
    """
    W = model.W_dec.detach().float().cpu()                 # (H, T*d)
    W = W.reshape(W.shape[0], T_WINDOW, D_MODEL)           # (H, T, d)
    return {"pos0": W[:, 0, :], "union": W.mean(dim=1), "per_pos": W}


# --------------------------------------------------------------------------
# steering hook with optional denoise-after-steer projection
# --------------------------------------------------------------------------

class SteerDenoiseHook:
    """b1_steer_eval::_Hook, plus an optional projection through a window
    dictionary's denoising map.

    Steering is the reference add, unchanged: `magnitudes[:, None, None] * vec`
    broadcast over every position of the layer-10 output, including the prompt.

    Projection (only when `proj` is set):
      - a rolling buffer holds the trailing T-1 PRE-PROJECTION activations, i.e.
        the steered but unprojected trajectory. The projector therefore always
        reads the raw steered signal and never its own output; the alternative
        (feeding the projection back in) compounds the map across a 1500-token
        generation and is a different experiment.
      - for each position with a full trailing-T window, the window is flattened
        time-major, denoised, and slot T-1 of the reconstruction REPLACES that
        position. Earlier slots are discarded: only the current position is
        rewritten.
      - the first T-1 positions of a prefill have no history and pass through
        unprojected.

    Under KV-cached decode the hook sees (B, 1, d) after prefill, so the window
    cannot be read off the tensor and is maintained here. A forward pass with
    seq_len > 1 is a prefill and resets the buffer, which is what makes the hook
    safe to leave registered across the many generate() calls
    _generate_panels issues.
    """

    def __init__(self, vec: torch.Tensor, proj=None, proj_arch: str | None = None,
                 T: int = T_WINDOW):
        self._raw = vec.detach()
        self._cached: torch.Tensor | None = None
        self.magnitude = 0.0
        self.magnitudes: torch.Tensor | None = None
        self.proj = proj
        self.proj_arch = proj_arch
        self.T = T
        self.buf: torch.Tensor | None = None
        self.n_projected = 0
        self.n_passthrough = 0

    # -- steering (identical to _Hook) --------------------------------------
    def _materialize(self, ref: torch.Tensor) -> torch.Tensor:
        if (self._cached is None or self._cached.device != ref.device
                or self._cached.dtype != ref.dtype):
            self._cached = self._raw.to(device=ref.device, dtype=ref.dtype)
        return self._cached

    def _delta(self, x: torch.Tensor) -> torch.Tensor:
        v = self._materialize(x)
        if self.magnitudes is not None:
            mags = self.magnitudes.to(device=x.device, dtype=x.dtype)
            return mags.view(-1, 1, 1) * v
        return self.magnitude * v

    def _steer(self, x: torch.Tensor) -> torch.Tensor:
        if self.magnitudes is None and self.magnitude == 0.0:
            return x
        if self.magnitudes is not None and torch.count_nonzero(self.magnitudes) == 0:
            return x
        return x + self._delta(x)

    # -- projection ---------------------------------------------------------
    @torch.no_grad()
    def _project(self, x: torch.Tensor) -> torch.Tensor:
        B, S, d = x.shape
        if S > 1 or self.buf is None or self.buf.shape[0] != B:
            self.buf = None                       # prefill / new panel chunk
        hist = x if self.buf is None else torch.cat([self.buf, x], dim=1)
        L = hist.shape[1] - S
        first = max(0, self.T - 1 - L)             # first index in x with history
        if first >= S:
            self.n_passthrough += B * S
            self.buf = hist[:, -(self.T - 1):].detach()
            return x

        idx = torch.arange(first, S, device=x.device)
        # windows[:, j] = hist[:, L + idx[j] - T + 1 : L + idx[j] + 1]
        starts = L + idx - self.T + 1
        gather = starts[:, None] + torch.arange(self.T, device=x.device)[None, :]
        win = hist[:, gather.reshape(-1), :].reshape(B, len(idx), self.T, d)
        flat = win.reshape(-1, self.T * d).float()
        rec = denoise_w6(self.proj, self.proj_arch, flat)
        cur = rec.reshape(-1, self.T, d)[:, -1, :].reshape(B, len(idx), d)

        out = x.clone()
        out[:, first:, :] = cur.to(x.dtype)
        self.n_projected += B * len(idx)
        self.n_passthrough += B * first
        self.buf = hist[:, -(self.T - 1):].detach()
        return out

    def __call__(self, _m, _i, output):
        is_tuple = isinstance(output, tuple)
        x = output[0] if is_tuple else output
        x = self._steer(x)
        if self.proj is not None:
            x = self._project(x)
        return (x,) + output[1:] if is_tuple else x


# --------------------------------------------------------------------------
# projector pre-flight
# --------------------------------------------------------------------------

@torch.no_grad()
def preflight(model, arch: str, X: torch.Tensor, batch: int = 512) -> dict:
    """Live-latent fraction and NMSE of the projector on the supplied windows.

    NMSE is psc_train_sae.py's own definition -- sum |x - recon|^2 over
    sum |x - mean|^2 -- so the number is comparable to the training log's
    `nmse` field rather than being a new quantity.
    """
    N = X.shape[0]
    mu = X.mean(0)
    num = den = 0.0
    fired = torch.zeros(H_SAE, dtype=torch.bool, device=X.device
                        if X.is_cuda else "cpu")
    l0s = []
    for s in range(0, N, batch):
        xb = X[s:s + batch].float()
        z = encode_w6(model, arch, xb)
        rec = decode_w6(model, z)
        num += float((xb - rec).pow(2).sum())
        den += float((xb - mu).pow(2).sum())
        live = z > 0
        fired |= live.any(0).to(fired.device)
        l0s.append(live.float().sum(1))
    l0 = torch.cat(l0s)
    n_live = int(fired.sum())
    return {"arch": arch, "n_windows": int(N),
            "nmse": num / max(den, 1e-12),
            "live_latents": n_live, "d_sae": int(fired.numel()),
            "live_fraction": n_live / float(fired.numel()),
            "dead_fraction": 1.0 - n_live / float(fired.numel()),
            "mean_l0": float(l0.mean()), "std_l0": float(l0.std()),
            "median_l0": float(l0.median())}


def find_ckpt(vol_root: str, arm: dict, allow_partial: bool = False) -> Path | None:
    """Locate an arm's checkpoint.

    psc_train_sae.py writes <stem>.pt every 5000 steps and again at the end,
    but writes <stem>_final.json only on a clean finish -- so the JSON, not the
    .pt, is the completion marker.

    `allow_partial` accepts the most recent periodic .pt when the run was cut
    short. The three wave-2 arms are projected to need 6.3-6.8 h at their
    observed 1.14-1.23 s/step against modal_txc.py's 6 h function timeout, so a
    partial checkpoint is the expected outcome rather than a rare one. Callers
    that use it must record `train_status` from `ckpt_status` in their output;
    a 15k-step dictionary is usable for an internal comparison but must not be
    labelled as the 20k-step arm.
    """
    d = Path(vol_root) / arm["dir"]
    ckpt = d / f"{arm['stem']}.pt"
    if (d / f"{arm['stem']}_final.json").exists() and ckpt.exists():
        return ckpt
    return ckpt if (allow_partial and ckpt.exists()) else None


def ckpt_status(vol_root: str, arm: dict) -> dict:
    """Completion state of one arm, read off the trainer's own artifacts."""
    d = Path(vol_root) / arm["dir"]
    ckpt = d / f"{arm['stem']}.pt"
    final = d / f"{arm['stem']}_final.json"
    log = d / f"{arm['stem']}.jsonl"
    last = None
    if log.exists():
        for line in log.read_text().splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    last = json.loads(line)
                except ValueError:
                    pass
    step = (last or {}).get("step")
    # The .pt on disk is the last multiple of CKPT_EVERY, not the last logged
    # step: psc_train_sae.py logs every 250 steps but rewrites the checkpoint
    # every 5000.
    ckpt_step = (step // CKPT_EVERY) * CKPT_EVERY if step else None
    return {"arm": arm["name"], "ckpt_exists": ckpt.exists(),
            "finished": final.exists(),
            "last_logged_step": step,
            "ckpt_step": PLAN_STEPS if final.exists() else ckpt_step,
            "last_nmse_clean": (last or {}).get("nmse_clean"),
            "last_dead_frac": (last or {}).get("dead_frac"),
            "train_status": "complete" if final.exists() else
                            ("partial" if ckpt.exists() else "missing")}


def resolve_ckpt(vol_root: str, arm: dict, min_step: int = TRUNC_MIN_STEP,
                 settle: int = SETTLE_STEPS) -> tuple[Path | None, dict]:
    """-> (checkpoint path or None, status). The timeout-truncation ruling.

    modal_txc.py gives the trainer a 6 h function timeout against a 20 000-step
    plan it cannot reach at the observed ~1.27 s/step, so `_final.json` is never
    written and find_ckpt's clean-finish path never fires. What does exist is the
    periodic .pt, rewritten every CKPT_EVERY steps; the last one to land before
    the timeout is step 15 000.

    Two things are checked rather than one. `ckpt_step >= min_step` is the ruling
    itself. The `settle` margin on top of it covers the gap between the trainer
    writing the .pt and modal_txc.py's 300 s committer thread making it visible
    on the volume: accepting the instant the log crosses 15 000 could load a
    still-uncommitted step-10 000 checkpoint while labelling it 15 000. At the
    observed rate `settle` steps is ~10 min, longer than one commit interval.

    Callers must label the arm by the returned `ckpt_step`, never by PLAN_STEPS.
    """
    st = ckpt_status(vol_root, arm)
    ckpt = Path(vol_root) / arm["dir"] / f"{arm['stem']}.pt"
    if st["finished"] and st["ckpt_exists"]:
        return ckpt, st
    step, cstep = st["last_logged_step"] or 0, st["ckpt_step"] or 0
    if st["ckpt_exists"] and cstep >= min_step and step >= cstep + settle:
        return ckpt, st
    return None, st


def read_final(vol_root: str, arm: dict) -> dict | None:
    p = Path(vol_root) / arm["dir"] / f"{arm['stem']}_final.json"
    return json.loads(p.read_text()) if p.exists() else None
