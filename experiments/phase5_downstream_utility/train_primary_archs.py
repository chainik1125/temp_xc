"""Train the primary architectures for Phase 5 sub-phase 5.1.

v2: **preloads activation subset onto GPU** to eliminate the MooseFS
mmap disk bottleneck that was throttling training to 0.5 s/step.

Strategy:
    - Once per run, load the first `PRELOAD_SEQS = 6000` sequences of
      the anchor-layer activation cache into GPU memory (fp16, ~3.5 GB).
    - Sample from the GPU tensor for every training step. Zero disk
      access during training.
    - For MLC, stack all 5 layers up-front (~18 GB on GPU; fits in 48 GB
      A40 alongside the 18432x2304 SAE).

Protocol (same as v1):
    - Adam, lr 3e-4, batch 1024, grad_clip 1.0, log every 200 steps.
    - Plateau stop: <2% loss drop per 1000 steps; max 25 000 steps.

Output: CKPT_DIR/<run_id>.pt + LOGS_DIR/<run_id>.json + INDEX_PATH row.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

os.environ.setdefault("TQDM_DISABLE", "1")

REPO = Path("/workspace/temp_xc")
CACHE_DIR = REPO / "data/cached_activations/gemma-2-2b-it/fineweb"
OUT_DIR = REPO / "experiments/phase5_downstream_utility/results"
CKPT_DIR = OUT_DIR / "ckpts"
LOGS_DIR = OUT_DIR / "training_logs"
INDEX_PATH = OUT_DIR / "training_index.jsonl"

DEFAULT_D_SAE = 18_432
ANCHOR_LAYER_KEY = "resid_L13"
MLC_LAYER_KEYS = ("resid_L11", "resid_L12", "resid_L13", "resid_L14", "resid_L15")
PRELOAD_SEQS = 6_000         # size of the in-GPU activation buffer
TRAIN_FRAC = 0.9             # remainder reserved for eval slice (not used here)


# ──────────────────────────────────────────────── data preload


def _preload_single(layer_key: str, device: torch.device,
                    n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
    arr = np.load(CACHE_DIR / f"{layer_key}.npy", mmap_mode="r")
    sub = np.asarray(arr[:n_seqs], dtype=np.float16)
    return torch.from_numpy(sub).to(device)


def _preload_multilayer(device: torch.device,
                        n_seqs: int = PRELOAD_SEQS) -> torch.Tensor:
    """(N, L, n_layers, d) stacked on GPU."""
    arrs = []
    for lk in MLC_LAYER_KEYS:
        a = np.asarray(np.load(CACHE_DIR / f"{lk}.npy", mmap_mode="r")[:n_seqs],
                       dtype=np.float16)
        arrs.append(torch.from_numpy(a))
    stacked = torch.stack(arrs, dim=2)  # (N, L, n_layers, d)
    return stacked.to(device)


# ──────────────────────────────────────────────── GPU-resident generators


def make_flat_gen_gpu(buf: torch.Tensor):
    """`buf` shape (N, L, d). flat sampling: (B, d)."""
    N, L, d = buf.shape

    def gen(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, N, (batch_size,), device=buf.device)
        tok = torch.randint(0, L, (batch_size,), device=buf.device)
        return buf[idx, tok].float()
    return gen


def make_window_gen_gpu(buf: torch.Tensor, T: int):
    """`buf` shape (N, L, d). window sampling: (B, T, d)."""
    N, L, d = buf.shape
    n_wins = L - T + 1

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        off = torch.randint(0, n_wins, (batch_size,), device=buf.device)
        # Build offsets tensor to use fancy indexing.
        # Shape (batch_size, T): each row is off[i]+[0..T-1]
        rng = torch.arange(T, device=buf.device)
        pos = off.unsqueeze(1) + rng.unsqueeze(0)  # (B, T)
        return buf[seq.unsqueeze(1).expand(-1, T), pos].float()
    return gen


def make_multilayer_gen_gpu(buf: torch.Tensor):
    """`buf` shape (N, L, n_layers, d). sample: (B, n_layers, d)."""
    N, L, n_layers, d = buf.shape

    def gen(batch_size: int) -> torch.Tensor:
        seq = torch.randint(0, N, (batch_size,), device=buf.device)
        tok = torch.randint(0, L, (batch_size,), device=buf.device)
        return buf[seq, tok].float()
    return gen


def make_seq_gen_gpu(buf: torch.Tensor):
    """Full-sequence sampling for TFA: (B, L, d)."""
    N, L, d = buf.shape

    def gen(batch_size: int) -> torch.Tensor:
        idx = torch.randint(0, N, (batch_size,), device=buf.device)
        return buf[idx].float()
    return gen


# ─────────────────────────────────────────── shared-per-pos SAE


class SharedPerPositionSAE(nn.Module):
    def __init__(self, d_in: int, d_sae: int, T: int, k: int | None):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.k = k

        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.W_enc = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_in, d_sae))
        nn.init.kaiming_uniform_(self.W_enc)
        nn.init.kaiming_uniform_(self.W_dec)
        self._normalize_decoder()

    @torch.no_grad()
    def _normalize_decoder(self):
        norms = self.W_dec.norm(dim=0, keepdim=True).clamp(min=1e-8)
        self.W_dec.data = self.W_dec.data / norms

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.b_dec
        pre = torch.einsum("btd,sd->bts", xc, self.W_enc) + self.b_enc
        if self.k is not None:
            vals, idx = pre.topk(self.k, dim=-1)
            z = torch.zeros_like(pre)
            z.scatter_(-1, idx, F.relu(vals))
        else:
            z = F.relu(pre)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bts,ds->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon = (x_hat - x).pow(2).sum(dim=-1).mean()
        return recon, x_hat, z


# ─────────────────────────────────────────── plateau + trainer


def compute_plateau(losses: list[float], window: int = 5) -> float | None:
    if len(losses) < 2 * window:
        return None
    recent = sum(losses[-window:]) / window
    prior = sum(losses[-2 * window:-window]) / window
    if prior == 0:
        return None
    return (prior - recent) / abs(prior)


@dataclass
class TrainCfg:
    lr: float = 3e-4
    batch_size: int = 1024
    max_steps: int = 25_000
    log_every: int = 200
    grad_clip: float = 1.0
    plateau_threshold: float = 0.02
    min_steps: int = 3_000
    seed: int = 42


def _iterate_train(
    model: nn.Module,
    gen_fn: Callable[[int], torch.Tensor],
    cfg: TrainCfg,
    device: torch.device,
    normalize_decoder: Callable | None = None,
    latent_l0_fn: Callable[[torch.Tensor], float] | None = None,
    tfa_mode: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()

    losses: list[float] = []
    l0s: list[float] = []
    steps_logged: list[int] = []
    converged = False
    plateau_val: float | None = None

    t0 = time.time()
    for step in range(cfg.max_steps):
        x = gen_fn(cfg.batch_size)
        if tfa_mode:
            recons, inter = model(x)
            flat_x = x.reshape(-1, x.shape[-1])
            flat_r = recons.reshape(-1, recons.shape[-1])
            loss = F.mse_loss(flat_r, flat_x, reduction="sum") / flat_x.shape[0]
            z = inter["novel_codes"]
        else:
            loss, _, z = model(x)

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if normalize_decoder is not None:
            normalize_decoder()

        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (
                    latent_l0_fn(z) if latent_l0_fn is not None
                    else (z > 0).float().sum(dim=-1).mean().item()
                )
            losses.append(loss.item())
            l0s.append(l0)
            steps_logged.append(step)
            plateau_val = compute_plateau(losses, window=5)
            if (
                plateau_val is not None
                and plateau_val < cfg.plateau_threshold
                and step >= cfg.min_steps
            ):
                converged = True
                break

    elapsed = time.time() - t0
    model.eval()
    return {
        "loss": losses,
        "l0": l0s,
        "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged,
        "plateau_last": plateau_val,
        "elapsed_s": elapsed,
    }


# ─────────────────────────────────────────── arch-specific trainers


def train_topk_sae(cfg, device, k, d_sae=DEFAULT_D_SAE,
                   buf: torch.Tensor | None = None):
    from src.architectures.topk_sae import TopKSAE
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    model = TopKSAE(buf.shape[-1], d_sae, k).to(device)
    gen = make_flat_gen_gpu(buf)

    def norm(): model._normalize_decoder()
    log = _iterate_train(model, gen, cfg, device, normalize_decoder=norm)
    return model, log


def train_mlc(cfg, device, k, d_sae=DEFAULT_D_SAE,
              buf: torch.Tensor | None = None):
    from src.architectures.mlc import MultiLayerCrosscoder
    buf = buf if buf is not None else _preload_multilayer(device)
    d_in = buf.shape[-1]
    n_layers = buf.shape[2]
    model = MultiLayerCrosscoder(d_in, d_sae, n_layers=n_layers, k=k).to(device)
    gen = make_multilayer_gen_gpu(buf)

    def norm(): model._normalize_decoder()
    log = _iterate_train(model, gen, cfg, device, normalize_decoder=norm)
    return model, log


def train_txcdr(cfg, device, k, T, match_budget=True,
                d_sae=DEFAULT_D_SAE, buf=None):
    from src.architectures.crosscoder import TemporalCrosscoder
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    k_eff = k * T if match_budget else k
    model = TemporalCrosscoder(buf.shape[-1], d_sae, T, k_eff).to(device)
    gen = make_window_gen_gpu(buf, T)

    def norm(): model._normalize_decoder()
    log = _iterate_train(model, gen, cfg, device, normalize_decoder=norm)
    return model, log


def train_stacked(cfg, device, k, T, d_sae=DEFAULT_D_SAE, buf=None):
    from src.architectures.stacked_sae import StackedSAE
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    model = StackedSAE(buf.shape[-1], d_sae, T, k).to(device)
    gen = make_window_gen_gpu(buf, T)

    def norm(): model._normalize_decoder()
    def l0(z): return (z > 0).float().sum(dim=-1).mean().item() * T
    log = _iterate_train(
        model, gen, cfg, device, normalize_decoder=norm, latent_l0_fn=l0
    )
    return model, log


def train_shared_perpos(cfg, device, k, T, d_sae=DEFAULT_D_SAE, buf=None):
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    model = SharedPerPositionSAE(buf.shape[-1], d_sae, T, k).to(device)
    gen = make_window_gen_gpu(buf, T)

    def norm(): model._normalize_decoder()
    def l0(z): return (z > 0).float().sum(dim=-1).mean().item() * T
    log = _iterate_train(
        model, gen, cfg, device, normalize_decoder=norm, latent_l0_fn=l0
    )
    return model, log


def train_matryoshka_txcdr(cfg, device, k, T, d_sae=DEFAULT_D_SAE, buf=None):
    from src.architectures.matryoshka_txcdr import PositionMatryoshkaTXCDR
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    k_eff = k * T
    model = PositionMatryoshkaTXCDR(buf.shape[-1], d_sae, T, k_eff).to(device)
    gen = make_window_gen_gpu(buf, T)

    def norm(): model._normalize_decoder()
    log = _iterate_train(model, gen, cfg, device, normalize_decoder=norm)
    return model, log


def train_tfa(cfg, device, k, use_pos, d_sae=DEFAULT_D_SAE, buf=None):
    from src.architectures._tfa_module import TemporalSAE
    buf = buf if buf is not None else _preload_single(ANCHOR_LAYER_KEY, device)
    d_in = buf.shape[-1]

    # Compute TFA input scale once from a CPU-side sample.
    cpu_sample = buf[:64].float().cpu().numpy().reshape(-1, d_in)
    mean_norm = np.linalg.norm(cpu_sample, axis=-1).mean()
    scale = math.sqrt(d_in) / max(mean_norm, 1e-8)
    model = TemporalSAE(
        dimin=d_in, width=d_sae, n_heads=4,
        sae_diff_type="topk", kval_topk=k,
        tied_weights=True, n_attn_layers=1,
        bottleneck_factor=4, use_pos_encoding=use_pos,
    ).to(device)

    base_seq_gen = make_seq_gen_gpu(buf)

    def gen(batch_size: int) -> torch.Tensor:
        bs = min(batch_size, 32)
        return base_seq_gen(bs) * scale

    def norm():
        with torch.no_grad():
            norms = model.D.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
            model.D.data.div_(norms)

    def l0(z): return (z > 0).float().sum(dim=-1).mean().item()
    tfa_cfg = TrainCfg(
        lr=3e-4, batch_size=32, max_steps=cfg.max_steps,
        log_every=cfg.log_every, grad_clip=cfg.grad_clip,
        plateau_threshold=cfg.plateau_threshold,
        min_steps=cfg.min_steps, seed=cfg.seed,
    )
    log = _iterate_train(
        model, gen, tfa_cfg, device,
        normalize_decoder=norm, latent_l0_fn=l0, tfa_mode=True,
    )
    return model, log


# ─────────────────────────────────────────── runner


def _save_run(run_id, arch, model, log, meta):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    log_path = LOGS_DIR / f"{run_id}.json"
    # Save in fp16 to halve disk cost — our quota is tight. Reconstructed
    # model in run_probing.py casts the fp16 state_dict back into the
    # newly-built fp32 module via load_state_dict's implicit casting.
    fp16_state = {
        k: v.to(torch.float16) if v.dtype == torch.float32 else v
        for k, v in model.state_dict().items()
    }
    torch.save(
        {"state_dict": fp16_state, "arch": arch, "meta": meta,
         "state_dict_dtype": "float16"},
        ckpt_path,
    )
    log_path.write_text(json.dumps({
        "run_id": run_id, "arch": arch, **meta,
        **{k: v for k, v in log.items() if k != "model"},
    }, indent=2, default=str))
    row = {
        "run_id": run_id, "arch": arch, **meta,
        "final_step": log["final_step"],
        "converged": log["converged"],
        "final_loss": log["loss"][-1] if log["loss"] else None,
        "final_l0": log["l0"][-1] if log["l0"] else None,
        "plateau_last": log["plateau_last"],
        "elapsed_s": log["elapsed_s"],
    }
    with INDEX_PATH.open("a") as f:
        f.write(json.dumps(row, default=str) + "\n")
    print(f"  -> {ckpt_path} ({log['elapsed_s']:.1f}s conv={log['converged']})")


def run_all(seeds, max_steps, archs=None):
    device = torch.device("cuda")
    # Headline 5.1 sweep: 7 architectures, all trainable within the 48 h budget.
    #
    # Deferred to sub-phase 5.6 (bonus):
    #   - shared_perpos_t5: mathematically identical to topk_sae under the
    #     last_position probing aggregation; adds no new information to the
    #     headline comparison, only to training-dynamics analysis.
    #   - tfa / tfa_pos: self-attention on 128-token sequences at
    #     d_sae=18432 costs ~20 s per step; 25k steps ≈ 5.8 days. Including
    #     TFA in 5.1 would consume the entire 5.1 budget on one arch.
    DEFAULT_ARCHS = [
        "topk_sae",
        "mlc",
        "txcdr_t5",
        "txcdr_t20",
        "stacked_t5",
        "stacked_t20",
        "matryoshka_t5",
    ]
    archs_to_run = archs or DEFAULT_ARCHS

    # Preload buffers once per invocation — amortize the MooseFS I/O.
    anchor_buf: torch.Tensor | None = None
    ml_buf: torch.Tensor | None = None

    def get_anchor():
        nonlocal anchor_buf
        if anchor_buf is None:
            print("  [data] preloading anchor layer to GPU...")
            t0 = time.time()
            anchor_buf = _preload_single(ANCHOR_LAYER_KEY, device)
            print(f"  [data] anchor loaded: {anchor_buf.shape} "
                  f"({anchor_buf.element_size() * anchor_buf.numel() / 1e9:.1f}GB) "
                  f"in {time.time() - t0:.1f}s")
        return anchor_buf

    def get_ml():
        nonlocal ml_buf
        if ml_buf is None:
            print("  [data] preloading MLC layers to GPU...")
            t0 = time.time()
            ml_buf = _preload_multilayer(device)
            print(f"  [data] ml loaded: {ml_buf.shape} "
                  f"({ml_buf.element_size() * ml_buf.numel() / 1e9:.1f}GB) "
                  f"in {time.time() - t0:.1f}s")
        return ml_buf

    for seed in seeds:
        for arch in archs_to_run:
            cfg = TrainCfg(seed=seed, max_steps=max_steps)
            run_id = f"{arch}__seed{seed}"
            print(f"=== {run_id} ===")

            if arch == "topk_sae":
                model, log = train_topk_sae(cfg, device, k=100, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=None, T=1, layer=13)
            elif arch == "mlc":
                model, log = train_mlc(cfg, device, k=100, buf=get_ml())
                meta = dict(seed=seed, k_pos=100, k_win=None, T=1,
                            layers="L11-L15")
            elif arch == "txcdr_t5":
                model, log = train_txcdr(cfg, device, k=100, T=5, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=500, T=5,
                            match_budget=True, layer=13)
            elif arch == "txcdr_t20":
                model, log = train_txcdr(cfg, device, k=100, T=20, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=2000, T=20,
                            match_budget=True, layer=13)
            elif arch == "stacked_t5":
                model, log = train_stacked(cfg, device, k=100, T=5, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=500, T=5, layer=13)
            elif arch == "stacked_t20":
                model, log = train_stacked(cfg, device, k=100, T=20, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=2000, T=20, layer=13)
            elif arch == "shared_perpos_t5":
                model, log = train_shared_perpos(cfg, device, k=100, T=5, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=500, T=5, layer=13)
            elif arch == "tfa":
                model, log = train_tfa(cfg, device, k=100, use_pos=False, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=None, T=128,
                            use_pos=False, layer=13)
            elif arch == "tfa_pos":
                model, log = train_tfa(cfg, device, k=100, use_pos=True, buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=None, T=128,
                            use_pos=True, layer=13)
            elif arch == "matryoshka_t5":
                model, log = train_matryoshka_txcdr(cfg, device, k=100, T=5,
                                                    buf=get_anchor())
                meta = dict(seed=seed, k_pos=100, k_win=500, T=5, layer=13)
            else:
                print(f"  unknown arch: {arch}")
                continue

            _save_run(run_id, arch, model, log, meta)
            del model
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--max-steps", type=int, default=25_000)
    p.add_argument("--archs", type=str, nargs="+", default=None)
    args = p.parse_args()
    run_all(seeds=args.seeds, max_steps=args.max_steps, archs=args.archs)


if __name__ == "__main__":
    main()
