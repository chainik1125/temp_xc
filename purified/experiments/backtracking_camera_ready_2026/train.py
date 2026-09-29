"""Seeded historical C7 training with bounded, exact-state restart checkpoints.

The default full-batch update is the pinned historical trainer's update. Only
TopK/Stacked support gradient microbatches; batch-dependent TXC/T-SAE losses
must use the full logical batch. Short --stop-after runs never receive a
completed receipt. No pro implementation is exposed here.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import os
import random
import signal
import sys
import time
from pathlib import Path

HISTORICAL_COMMIT = "284a8bf5e3e5a7cc094dd68c6fa5a92a9fd4eec3"
PROTOCOL_VERSION = "c7-camera-ready-300k-v1"
ACT_CACHE_KEY = "fb2a74be884e512a"
ACT_CACHE_SHA256 = "dc34dfb117f77abddef4b4396d0d00afc707c39876d0ee36015de1e7b8406914"
DATASOURCE = "llama_3_1_8b_base_l10_ward_nousmirror"
ARCHS = ("txc_base", "topk_sae", "tsae_paper", "stacked_sae")
FINAL_STEPS = 300_000
HISTORICAL_FILES = {
    "configs/locked_archs.yaml": "92f7994f3d4ac1cd7983e33dda64633cc35c75410c925f7e8ce07f4a3a7dc2fa",
    "configs/datasources.yaml": "ad8e50d79865030a723b94497d96b0392db1ae4229f5437bcdf6d6c57d7bb23d",
    "src/temp_bench/config.py": "e0da59015da8b883177ad3e64ceb42cd619e8d423af0d30e4b9acf8f3bcdc430",
    "src/temp_bench/schemas.py": "936510830d9d3139c0e429adf605b96d8aa93c71c0c1cb733915d39b19312700",
    "src/temp_bench/training/sae_trainer.py": "7891ad345fcabf6635834bffb686871f356e65f7857190dc61542cc20bcc8980",
    "src/temp_bench/utils/seed.py": "8e3e576e0efb0d455925a4956c882e16586426c94a44c07ea6d55790d2ceb0db",
    "src/temp_bench/architectures/base.py": "d5fd0a5724aeefc99e023dc5363446b20912ee1fcfeaf11ed2ad1eab5da93159",
    "src/temp_bench/architectures/topk_sae.py": "b55d25c8eb710cbcf0504181e0202ee3321bc22812b32b79998b3a782ebc9acb",
    "src/temp_bench/architectures/txc_base.py": "09e9afddaf1681347f171a7640959761cd2fe0483564f5ee6ef18aa165830aad",
    "src/temp_bench/architectures/stacked_sae.py": "6076860fde5437258b46596ae055521b4234ba65341bc44e046ca38b23dacedc",
    "src/temp_bench/architectures/tsae.py": "8805e4f574134e4530a539dbd05c1c353bf40c03ea4570295ac6a4baebf4b34b",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch_save(path, payload):
    import torch
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def verify_historical_source(root):
    root = Path(root).resolve()
    if (root / "HISTORICAL_COMMIT").read_text().strip() != HISTORICAL_COMMIT:
        raise RuntimeError("historical commit marker mismatch")
    for relative, expected in HISTORICAL_FILES.items():
        if sha256(root / relative) != expected:
            raise RuntimeError(f"historical source SHA-256 mismatch: {relative}")
    return hashlib.sha256(json.dumps(HISTORICAL_FILES, sort_keys=True).encode()).hexdigest()


def validate_microbatch(arch, logical_batch, microbatch):
    if microbatch < 1 or microbatch > logical_batch:
        raise ValueError("microbatch must be between 1 and logical batch")
    if arch not in ("topk_sae", "stacked_sae") and microbatch != logical_batch:
        raise ValueError(f"{arch} has batch-dependent loss/state; use full logical batch")


def capture_rng(sampler):
    import numpy as np
    import torch
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "window_sampler": copy.deepcopy(sampler.bit_generator.state),
    }


def restore_rng(state, sampler):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA RNG device count changed across resume")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    sampler.bit_generator.state = state["window_sampler"]


class WindowSampler:
    """The July corrected runner's identical two-draw NumPy sample order."""
    def __init__(self, acts, seed, window=5):
        import numpy as np
        self.acts, self.window = acts, window
        self.rng = np.random.default_rng(seed)

    def __call__(self, batch_size):
        import torch
        n, length, _ = self.acts.shape
        sequences = self.rng.integers(0, n, size=batch_size)
        positions = self.rng.integers(0, length - self.window + 1, size=batch_size)
        seq = torch.as_tensor(sequences, dtype=torch.int64, device=self.acts.device)
        pos = torch.as_tensor(positions, dtype=torch.int64, device=self.acts.device)
        offsets = torch.arange(self.window, device=self.acts.device)
        return self.acts[seq[:, None], pos[:, None] + offsets[None, :]].float()


def one_update(model, optimizer, batch, cfg, step, *, arch, microbatch_size):
    """Historical Adam/warmup/clipping/post_step order; no loss rewrites."""
    import torch
    from temp_bench.training.sae_trainer import _autocast_dtype, _lr_at, _set_lr
    validate_microbatch(arch, len(batch), microbatch_size)
    _set_lr(optimizer, _lr_at(step, cfg))
    dtype = _autocast_dtype(cfg.precision)

    def forward(chunk):
        ctx = torch.autocast("cuda", dtype=dtype) if dtype and batch.is_cuda else contextlib.nullcontext()
        with ctx:
            return model.train_step(chunk)

    if microbatch_size == len(batch):
        loss, info = forward(batch)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        values = {key: float(info[key]) for key in ("mse", "l0", "dead", "threshold") if key in info}
        values["loss"] = float(loss.detach())
    else:
        # Only independent token/position reconstruction losses reach here.
        # A smaller last microbatch receives its actual fraction of the batch.
        optimizer.zero_grad(set_to_none=True)
        values = {"loss": 0.0, "mse": 0.0, "l0": 0.0}
        for start in range(0, len(batch), microbatch_size):
            chunk = batch[start:start + microbatch_size]
            loss, info = forward(chunk)
            weight = len(chunk) / len(batch)
            (loss * weight).backward()
            values["loss"] += float(loss.detach()) * weight
            values["mse"] += float(info.get("mse", loss.detach())) * weight
            values["l0"] += float(info.get("l0", 0.0)) * weight
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    if not torch.isfinite(grad_norm) or not all(__import__("math").isfinite(v) for v in values.values()):
        raise FloatingPointError(f"nonfinite update at step {step}; latest checkpoint remains intact")
    optimizer.step()
    model.post_step()
    values.update(lr=_lr_at(step, cfg), grad_norm=float(grad_norm))
    return values


def resume_payload(model, optimizer, sampler, *, identity, completed, metrics, elapsed):
    return {
        "format_version": 1, "identity": identity, "completed_steps": completed,
        "model": model.state_dict(), "optimizer": optimizer.state_dict(),
        "rng": capture_rng(sampler.rng), "metrics": metrics, "elapsed_seconds": elapsed,
    }


def restore_payload(payload, model, optimizer, sampler, identity):
    if payload.get("format_version") != 1 or payload.get("identity") != identity:
        raise RuntimeError("resume identity mismatch (source/config/runtime/microbatch)")
    completed = payload["completed_steps"]
    if not isinstance(completed, int) or not 0 <= completed <= identity["training_cfg"]["n_steps"]:
        raise RuntimeError("invalid completed-step count in resume checkpoint")
    model.load_state_dict(payload["model"], strict=True)
    optimizer.load_state_dict(payload["optimizer"])
    restore_rng(payload["rng"], sampler.rng)
    return completed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-root", type=Path, required=True)
    p.add_argument("--cache-file", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--arch", choices=ARCHS, required=True)
    p.add_argument("--d-sae", type=int, default=32768)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--n-steps", type=int, default=FINAL_STEPS)
    p.add_argument("--microbatch-size", type=int, default=1024)
    p.add_argument("--checkpoint-every", type=int, default=5000)
    p.add_argument("--progress-every", type=int, default=250)
    p.add_argument("--stop-after", type=int, help="Stop after this total update count; incomplete, resumable only")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--cache-device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--preflight-only", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.n_steps != FINAL_STEPS or args.batch_size != 1024:
        raise ValueError("final protocol is exactly 300000 optimizer steps, logical batch 1024")
    if args.d_sae != 32768 and not (args.arch == "tsae_paper" and args.d_sae == 16384):
        raise ValueError("only width 32768, or the T-SAE 16384 sensitivity, is allowed")
    if args.seed not in (1, 2, 42):
        raise ValueError("locked seeds are 1, 2, 42")
    validate_microbatch(args.arch, args.batch_size, args.microbatch_size)
    if min(args.checkpoint_every, args.progress_every) < 1:
        raise ValueError("checkpoint/progress intervals must be positive")
    if args.stop_after is not None and not 1 <= args.stop_after <= FINAL_STEPS:
        raise ValueError("stop-after must be in [1,300000]")
    historical_root = args.historical_root.resolve()
    source_digest = verify_historical_source(historical_root)
    if sha256(args.cache_file) != ACT_CACHE_SHA256:
        raise RuntimeError("activation-cache SHA-256 mismatch")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["TEMP_BENCH_ROOT"] = str(historical_root)
    sys.path.insert(0, str(historical_root / "src"))
    import numpy as np
    import torch
    from temp_bench.config import compute_act_cache_key, compute_train_key, instantiate_arch, load_arch, load_datasource
    from temp_bench.schemas import TrainingConfig
    from temp_bench.training.sae_trainer import _make_optimizer
    from temp_bench.utils.seed import set_seed
    from safetensors.torch import save_file
    import temp_bench.config as loaded_config
    if Path(loaded_config.__file__).resolve() != historical_root / "src/temp_bench/config.py":
        raise RuntimeError("temp_bench was imported from a different source tree")
    if compute_act_cache_key(load_datasource(DATASOURCE)) != ACT_CACHE_KEY:
        raise RuntimeError("historical datasource key mismatch")
    cfg = TrainingConfig(n_steps=FINAL_STEPS, batch_size=1024)
    spec = load_arch(args.arch, component="c7")
    spec = spec.model_copy(update={"hparams": {**spec.hparams, "d_sae": args.d_sae}})
    train_key = compute_train_key(arch=spec, seed=args.seed, training_cfg=cfg, act_cache_key=ACT_CACHE_KEY)
    acts_np = np.load(args.cache_file, mmap_mode="r")
    if acts_np.shape != (4044, 128, 4096) or acts_np.dtype != np.float16:
        raise RuntimeError("training cache shape/dtype mismatch")
    identity = {
        "protocol_version": PROTOCOL_VERSION, "historical_commit": HISTORICAL_COMMIT,
        "historical_source_sha256": source_digest, "runner_sha256": sha256(__file__),
        "arch": args.arch, "arch_version": spec.arch_version, "hparams": spec.hparams,
        "d_sae": args.d_sae, "d_in": 4096, "k_pos": 20, "window_size": 5,
        "seed": args.seed, "train_key": train_key, "act_cache_key": ACT_CACHE_KEY,
        "act_cache_sha256": ACT_CACHE_SHA256, "training_cfg": cfg.model_dump(),
        "microbatch_size": args.microbatch_size, "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda, "device_type": torch.device(args.device).type,
        "logical_batch_unit": "5-token source windows",
    }
    if args.preflight_only:
        print(json.dumps({**identity, "status": "preflight-complete"}, sort_keys=True))
        return 0
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cell = args.output_root.resolve() / "cells" / f"{args.arch}_d{args.d_sae}_seed{args.seed}"
    cell.mkdir(parents=True, exist_ok=True)
    # An OS lock is released automatically on crash; two workers may not train
    # or replace the same latest checkpoint concurrently.
    import fcntl
    lock = (cell / ".train.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    latest = cell / "latest-resume.pt"
    final = cell / "checkpoint"
    if (final / "config.json").exists():
        finished = json.loads((final / "config.json").read_text())
        if any(finished.get(k) != v for k, v in identity.items()) or finished.get("n_steps_completed") != FINAL_STEPS:
            raise RuntimeError("existing final receipt does not match this cell")
        if finished.get("status") != "complete" or sha256(final / "model.safetensors") != finished.get("model_sha256"):
            raise RuntimeError("existing final checkpoint failed integrity check")
        print(f"verified completed checkpoint: {final}", flush=True)
        return 0
    if (cell / "manifest.json").exists() and not args.resume:
        raise RuntimeError("existing cell requires --resume; refusing overwrite")
    if args.resume and not latest.exists():
        raise RuntimeError("--resume requested but latest-resume.pt is missing")
    set_seed(args.seed)
    model = instantiate_arch(spec, d_in=4096).to(device)
    n_parameters = sum(p.numel() for p in model.parameters())
    if n_parameters > 1_000_000_000 and device.type == "cuda":
        model = model.bfloat16()
    model.train()
    optimizer = _make_optimizer(model, cfg)
    acts = torch.from_numpy(np.array(acts_np, copy=True)).to(args.cache_device)
    sampler = WindowSampler(acts, args.seed)
    completed, previous_elapsed, metrics = 0, 0.0, {}
    if args.resume:
        # This file is written only by this runner on the trusted local volume.
        payload = torch.load(latest, map_location="cpu", weights_only=False)
        completed = restore_payload(payload, model, optimizer, sampler, identity)
        previous_elapsed, metrics = payload["elapsed_seconds"], payload["metrics"]
        del payload
    started = time.monotonic()
    stop_requested = []
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda signum, frame: stop_requested.append(signum))
    metadata = {**identity, "status": "running", "n_parameters": n_parameters,
                "parameter_dtype": str(next(model.parameters()).dtype),
                "arch_window": int(model.config.T), "host": os.uname().nodename,
                "seed_correction": "Python, NumPy, CPU Torch, all CUDA RNGs seeded before init"}
    atomic_json(cell / "manifest.json", metadata)
    log_path = cell / "training_metrics.jsonl"
    if args.resume and log_path.exists():
        retained = [line for line in log_path.read_text().splitlines() if json.loads(line)["step"] <= completed]
        log_path.write_text("\n".join(retained) + ("\n" if retained else ""))

    def receipt(status):
        elapsed = previous_elapsed + time.monotonic() - started
        return {"status": status, "step": completed, "n_steps_completed": completed,
                "n_steps": FINAL_STEPS, "elapsed_seconds": elapsed,
                "source_windows_seen": completed * 1024,
                "source_token_positions_seen": completed * 1024 * 5,
                "reconstructed_token_positions": completed * 1024 * (1 if args.arch == "tsae_paper" else 5),
                "temporal_pair_positions": completed * 1024 * 2 if args.arch == "tsae_paper" else 0,
                "metrics": metrics}

    def snapshot():
        row = receipt("running")
        atomic_torch_save(latest, resume_payload(model, optimizer, sampler, identity=identity,
                          completed=completed, metrics=metrics, elapsed=row["elapsed_seconds"]))
        atomic_json(cell / "resume_receipt.json", {**row, "path": latest.name, "bytes": latest.stat().st_size})

    if not args.resume:
        # A failed first training block can restart from the exact initialized
        # model without deleting the cell or reconstructing its initial RNGs.
        snapshot()
    target = args.stop_after or FINAL_STEPS
    while completed < target and not stop_requested:
        batch = sampler(cfg.batch_size).to(device)
        metrics = one_update(model, optimizer, batch, cfg, completed,
                             arch=args.arch, microbatch_size=args.microbatch_size)
        completed += 1
        del batch
        if completed == 1 or completed % args.progress_every == 0:
            row = receipt("running")
            atomic_json(cell / "progress.json", row)
            with log_path.open("a") as handle:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            print(json.dumps(row, sort_keys=True), flush=True)
        if completed % args.checkpoint_every == 0:
            snapshot()
    snapshot()
    if completed != FINAL_STEPS:
        atomic_json(cell / "progress.json", receipt("incomplete"))
        print(f"saved resumable incomplete cell at {completed}/{FINAL_STEPS}", flush=True)
        return 0
    final.mkdir(exist_ok=True)
    model_path = final / "model.safetensors"
    temporary = final / "model.safetensors.tmp"
    save_file({k: v.detach().contiguous().cpu() for k, v in model.state_dict().items()}, str(temporary))
    os.replace(temporary, model_path)
    digest = sha256(model_path)
    final_metadata = {**metadata, **receipt("complete"), "model_sha256": digest,
                      "checkpoint_sha256": digest, "finished_unix": time.time()}
    atomic_json(final / "config.json", final_metadata)
    atomic_json(cell / "progress.json", receipt("complete"))
    atomic_json(cell / "manifest.json", final_metadata)
    print(f"complete train_key={train_key} checkpoint={final} sha256={digest}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
