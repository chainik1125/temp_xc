"""Phase 2 — train one dictionary-learning model per cached hookpoint.

Originally trained only the `TemporalCrosscoder`. Now dispatches on
`config.txc.arch_list` so we can also train flat TopK SAEs, StackedSAEs,
and Han's TemporalSAE on the same cached activations as paper-budget
baselines (see `architectures.py`).

Reads the on-disk activation cache from Phase 1 (one float16 .npy per
hookpoint), loads it onto the GPU, and trains each requested arch over a
sliding window of length T (config.txc.T).

Logs train loss / FVU / window-L0 / dead-feature count at LOG_INTERVAL
and writes one checkpoint per (arch, hookpoint) to `paths.ckpt_dir`. The
checkpoint filename is `txc_<hookpoint>.pt` for backwards compatibility
when arch="txc"; for other architectures it's `<arch>_<hookpoint>.pt`.
Per-step metrics are dumped as a jsonl in
`paths.logs_dir/<arch>_<hookpoint>_train.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm.auto import tqdm

from experiments.ward_backtracking_txc.architectures import (
    build_arch, arch_forward, arch_param_count,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("ward_txc.train")


class _ActivationLoader:
    """Streams (B, T, d) windows from a single hookpoint's cached npy."""

    def __init__(self, path: Path, T: int, batch_size: int, device: str = "cuda"):
        self.T = T
        self.batch_size = batch_size
        self.device = device
        # mmap on disk; copy to GPU in chunks so we don't blow host RAM.
        log.info("[loader] mmap %s", path)
        arr = np.load(path, mmap_mode="r")
        self.num_seq, self.seq_len, self.d = arr.shape
        log.info("[loader] shape=%s d=%d", arr.shape, self.d)
        # Load fully to GPU as float16 — for our scoped run, ~1M tokens * 4096
        # * 2B = 8 GB, fits next to the TXC params.
        chunks = []
        chunk_n = 256
        for i in tqdm(range(0, self.num_seq, chunk_n), desc="GPU upload"):
            end = min(i + chunk_n, self.num_seq)
            chunks.append(torch.from_numpy(arr[i:end].copy()).to(device))
        self.data = torch.cat(chunks, dim=0)  # (N, L, d) float16 GPU
        del chunks
        torch.cuda.empty_cache()
        log.info("[loader] uploaded to %s (%.2f GB)",
                 device, self.data.element_size() * self.data.nelement() / 1e9)

    def sample(self) -> torch.Tensor:
        max_start = self.seq_len - self.T
        chain_idx = torch.randint(0, self.num_seq, (self.batch_size,), device=self.device)
        start_idx = torch.randint(0, max_start + 1, (self.batch_size,), device=self.device)
        offsets = torch.arange(self.T, device=self.device).unsqueeze(0)
        pos_idx = start_idx.unsqueeze(1) + offsets
        chain_exp = chain_idx.unsqueeze(1).expand(-1, self.T)
        return self.data[chain_exp, pos_idx].to(torch.float32)  # (B, T, d) fp32 for stable training


def _ckpt_filename(arch: str, hookpoint_key: str) -> str:
    """Backwards-compat filename: TXC keeps the old `txc_<key>.pt` name."""
    if arch == "txc":
        return f"txc_{hookpoint_key}.pt"
    return f"{arch}_{hookpoint_key}.pt"


def _log_filename(arch: str, hookpoint_key: str) -> str:
    if arch == "txc":
        return f"{hookpoint_key}_train.jsonl"
    return f"{arch}_{hookpoint_key}_train.jsonl"


def _train_one(arch: str, hookpoint: dict, cfg: dict) -> dict:
    """Train one (arch, hookpoint) cell. Skips silently if checkpoint exists."""
    key = hookpoint["key"]
    paths = cfg["paths"]
    txc_cfg = cfg["txc"]

    acts_path = Path(paths["acts_dir"]) / f"{key}.npy"
    if not acts_path.exists():
        log.warning("[skip] %s/%s: no cached activations at %s", arch, key, acts_path)
        return {"arch": arch, "key": key, "skipped": True}

    ckpt_dir = Path(paths["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / _ckpt_filename(arch, key)
    log_path = logs_dir / _log_filename(arch, key)

    if ckpt_path.exists():
        log.info("[skip] checkpoint %s exists — remove to retrain", ckpt_path)
        return {"arch": arch, "key": key, "skipped": True, "ckpt": str(ckpt_path)}

    seed = int(txc_cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    T = int(txc_cfg["T"])
    d_sae = int(txc_cfg["d_sae"])
    d_in = int(txc_cfg["d_model"])
    k_per_pos = int(txc_cfg["k_per_position"])

    loader = _ActivationLoader(
        acts_path, T=T,
        batch_size=int(txc_cfg["batch_size"]),
    )

    arch_kwargs = (txc_cfg.get("arch_kwargs", {}) or {}).get(arch, {})
    model = build_arch(arch, d_in=d_in, d_sae=d_sae, T=T, k=k_per_pos, **arch_kwargs).to("cuda")
    if hasattr(model, "_normalize_decoder"):
        with torch.no_grad():
            model._normalize_decoder()

    log.info("[arch=%s] params=%.1fM", arch, arch_param_count(model) / 1e6)

    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(txc_cfg["learning_rate"]),
        betas=(0.9, 0.999),
    )
    n_steps = int(txc_cfg["train_steps"])
    log_interval = int(txc_cfg["log_interval"])

    history: list[dict] = []
    pbar = tqdm(range(n_steps), desc=f"train {arch}/{key}", mininterval=2.0)

    feat_active_count = torch.zeros(d_sae, device="cuda")

    log_f = open(log_path, "w")
    try:
        for step in pbar:
            x = loader.sample()                    # (B, T, d) fp32
            loss, lat = arch_forward(arch, model, x)
            z = lat["window"]                      # (B, d_sae)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(txc_cfg["grad_clip"]))
            optim.step()
            with torch.no_grad():
                if hasattr(model, "_normalize_decoder"):
                    model._normalize_decoder()
                feat_active_count += (z > 0).float().sum(dim=0)

            if step % log_interval == 0 or step == n_steps - 1:
                with torch.no_grad():
                    # FVU normalizes per-token MSE by per-token variance so
                    # losses are arch-comparable. Recompute the recon for
                    # FVU using the same forward (no grad).
                    _, lat_eval = arch_forward(arch, model, x)
                    z_eval = lat_eval["window"]
                    # Per-token recon variance — recompute via arch's forward.
                    # Compute FVU on the per-token residual relative to the
                    # per-token variance of x; cheap enough.
                    var = x.var(dim=0).sum().clamp_min(1e-8)
                    fvu = (loss / var).item()
                    window_l0 = (z_eval > 0).float().sum(dim=-1).mean().item()
                    n_dead = int((feat_active_count == 0).sum().item())
                row = {
                    "step": step, "arch": arch,
                    "loss": float(loss.item()),
                    "fvu": fvu,
                    "window_l0": window_l0,
                    "n_dead": n_dead,
                }
                history.append(row)
                log_f.write(json.dumps(row) + "\n")
                log_f.flush()
                pbar.set_postfix(loss=f"{loss.item():.3f}",
                                 fvu=f"{fvu:.3f}",
                                 L0=f"{window_l0:.0f}",
                                 dead=n_dead)
    finally:
        log_f.close()

    torch.save({
        "state_dict": model.state_dict(),
        "config": {
            "arch": arch,
            "d_in": d_in,
            "d_sae": d_sae,
            "T": T,
            "k_per_position": k_per_pos,
            "arch_kwargs": arch_kwargs,
            "hookpoint": hookpoint,
            "seed": seed,
        },
        "history": history,
    }, ckpt_path)
    log.info("[saved] %s", ckpt_path)
    # Free GPU before next hookpoint.
    del model, loader, optim
    torch.cuda.empty_cache()
    return {"key": key, "ckpt": str(ckpt_path), "n_steps": n_steps}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=Path(__file__).parent / "config.yaml")
    p.add_argument("--only", type=str, default=None,
                   help="train only this hookpoint key")
    p.add_argument("--arch", type=str, nargs="+", default=None,
                   help="restrict to these architectures (default: cfg.txc.arch_list)")
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    hookpoints = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    if args.only:
        hookpoints = [hp for hp in hookpoints if hp["key"] == args.only]
        if not hookpoints:
            log.error("hookpoint %s not in config", args.only); return 1

    arch_list = args.arch or cfg["txc"].get("arch_list", ["txc"])
    log.info("[arch_list] %s", arch_list)

    summary = []
    for arch in arch_list:
        for hp in hookpoints:
            log.info("=" * 70); log.info("[arch=%s hookpoint=%s]", arch, hp["key"])
            summary.append(_train_one(arch, hp, cfg))
    log.info("[done] %d (arch, hookpoint) cells trained", len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
