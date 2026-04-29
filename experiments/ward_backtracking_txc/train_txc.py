"""Phase 2 — train one TemporalCrosscoder per cached hookpoint.

Reads the on-disk activation cache from Phase 1 (one float16 .npy per
hookpoint), loads it onto the GPU, and trains a TemporalCrosscoder over a
sliding window of length T (config.txc.T).

Logs train loss / FVU / window-L0 / dead-feature count at LOG_INTERVAL
and writes one checkpoint per hookpoint to `paths.ckpt_dir`. Per-step
metrics are dumped as a jsonl in `paths.logs_dir/<key>_train.jsonl` —
the training-curves plot reads them.
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

from temporal_crosscoders.models import TemporalCrosscoder

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


def _train_one(hookpoint: dict, cfg: dict) -> dict:
    key = hookpoint["key"]
    paths = cfg["paths"]
    txc_cfg = cfg["txc"]

    acts_path = Path(paths["acts_dir"]) / f"{key}.npy"
    if not acts_path.exists():
        log.warning("[skip] %s: no cached activations at %s", key, acts_path)
        return {"key": key, "skipped": True}

    ckpt_dir = Path(paths["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(paths["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"txc_{key}.pt"
    log_path = logs_dir / f"{key}_train.jsonl"

    if ckpt_path.exists():
        log.info("[skip] checkpoint %s exists — remove to retrain", ckpt_path)
        return {"key": key, "skipped": True, "ckpt": str(ckpt_path)}

    torch.manual_seed(int(txc_cfg["seed"]))
    np.random.seed(int(txc_cfg["seed"]))

    loader = _ActivationLoader(
        acts_path, T=int(txc_cfg["T"]),
        batch_size=int(txc_cfg["batch_size"]),
    )

    model = TemporalCrosscoder(
        d_in=int(txc_cfg["d_model"]),
        d_sae=int(txc_cfg["d_sae"]),
        T=int(txc_cfg["T"]),
        k=int(txc_cfg["k_per_position"]),
    ).to("cuda")
    with torch.no_grad():
        model._normalize_decoder()

    optim = torch.optim.Adam(
        model.parameters(),
        lr=float(txc_cfg["learning_rate"]),
        betas=(0.9, 0.999),
    )
    n_steps = int(txc_cfg["train_steps"])
    log_interval = int(txc_cfg["log_interval"])

    history: list[dict] = []
    pbar = tqdm(range(n_steps), desc=f"train {key}", mininterval=2.0)

    # Track activation count for dead-feature counting.
    feat_active_count = torch.zeros(int(txc_cfg["d_sae"]), device="cuda")

    log_f = open(log_path, "w")
    try:
        for step in pbar:
            x = loader.sample()  # (B, T, d) fp32
            loss, x_hat, z = model(x)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), float(txc_cfg["grad_clip"]))
            optim.step()
            with torch.no_grad():
                model._normalize_decoder()
                feat_active_count += (z > 0).float().sum(dim=0)

            if step % log_interval == 0 or step == n_steps - 1:
                with torch.no_grad():
                    fvu = ((x_hat - x).pow(2).sum(dim=-1).mean()
                           / x.var(dim=0).sum().clamp_min(1e-8)).item()
                    window_l0 = (z > 0).float().sum(dim=-1).mean().item()
                    n_dead = int((feat_active_count == 0).sum().item())
                row = {
                    "step": step,
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
            "d_in": int(txc_cfg["d_model"]),
            "d_sae": int(txc_cfg["d_sae"]),
            "T": int(txc_cfg["T"]),
            "k_per_position": int(txc_cfg["k_per_position"]),
            "hookpoint": hookpoint,
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
    args = p.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    hookpoints = [hp for hp in cfg["hookpoints"] if hp.get("enabled", True)]
    if args.only:
        hookpoints = [hp for hp in hookpoints if hp["key"] == args.only]
        if not hookpoints:
            log.error("hookpoint %s not in config", args.only); return 1

    summary = []
    for hp in hookpoints:
        log.info("=" * 70); log.info("[hookpoint] %s", hp["key"])
        summary.append(_train_one(hp, cfg))
    log.info("[done] %d hookpoints trained", len(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
