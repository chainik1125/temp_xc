"""Bit-exact reproducibility test for chunked vs continuous training.

Runs each architecture in two modes and verifies the per-step loss
trajectories agree:

  Continuous: train arch from scratch for ``--total_steps`` (e.g. 300)
              steps, log loss every step.
  Chunked:    train ``--chunk_size`` steps (e.g. 100), save snapshot,
              start a new process, --resume_from snapshot, train another
              ``--chunk_size``, repeat until --total_steps. Log loss every
              step.

Then compare losses[i] for each i; pass if all step losses agree to
``--rel_tol`` (default 1e-4 relative).

    uv run python -m experiments.em_features.test_chunked_vs_continuous \\
        --arch han --total_steps 300 --chunk_size 100
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["sae", "han"], required=True)
    p.add_argument("--config", type=Path,
                   default=REPO_ROOT / "experiments/em_features/config.yaml")
    p.add_argument("--total_steps", type=int, default=300)
    p.add_argument("--chunk_size", type=int, default=100)
    p.add_argument("--workdir", type=Path, default=Path("/root/em_features/test_chunked"))
    p.add_argument("--rel_tol", type=float, default=1e-4)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def parse_losses_from_log(log_path: Path, prefix: str) -> dict[int, float]:
    """Returns step -> loss for lines like '[prefix] step  X/Y  loss=Z ...' (any whitespace)."""
    out: dict[int, float] = {}
    if not log_path.exists():
        return out
    for line in log_path.read_text(errors="ignore").splitlines():
        if prefix not in line or "loss=" not in line:
            continue
        # extract step
        try:
            after_step = line.split("step", 1)[1].lstrip()
            step_str = after_step.split("/", 1)[0].strip()
            step = int(step_str)
            loss_str = line.split("loss=")[1].split()[0]
            out[step] = float(loss_str)
        except Exception:
            continue
    return out


def run_subprocess(cmd: list[str], log_path: Path, env: dict | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {' '.join(shlex.quote(c) for c in cmd)}\n  → log: {log_path}", flush=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    return proc.returncode


def main():
    args = parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)

    if args.total_steps % args.chunk_size != 0:
        sys.exit(f"--total_steps ({args.total_steps}) must be divisible by --chunk_size ({args.chunk_size})")

    if args.arch == "sae":
        log_prefix = "[sae]"
        train_module = "experiments.em_features.run_training_sae_custom"
        common = [
            "--config", str(args.config),
            "--hookpoint", "resid_post", "--layer", "15",
            "--d_sae", "1024", "--k", "16", "--batch_size", "32", "--lr", "3e-4",
            "--log_every", "1",
            "--device", args.device,
        ]
    elif args.arch == "han":
        log_prefix = "[han_champ]"
        train_module = "experiments.em_features.run_training_han_champion"
        common = [
            "--config", str(args.config),
            "--d_sae", "1024", "--T", "5", "--k", "16",
            "--batch_size", "32", "--lr", "3e-4",
            "--log_every", "1",
            "--device", args.device,
        ]
    else:
        raise ValueError(args.arch)

    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([
        str(REPO_ROOT),
        str(REPO_ROOT / "experiments/separation_scaling/vendor/src"),
    ])

    # ---------- Continuous run ----------
    cont_pfx = args.workdir / "cont"
    cont_log = args.workdir / "cont.log"
    cmd_cont = ["python", "-m", train_module,
                "--out_prefix", str(cont_pfx),
                "--total_steps", str(args.total_steps),
                "--snapshot_at", str(args.total_steps),
                ] + common
    rc = run_subprocess(cmd_cont, cont_log, env=env)
    if rc != 0:
        sys.exit(f"continuous run failed with rc={rc}; see {cont_log}")

    # ---------- Chunked run ----------
    chunked_pfx = args.workdir / "chunked"
    chunked_log = args.workdir / "chunked.log"
    # Concatenate logs across chunks into one file for easy parsing
    chunked_log.write_text("")
    prev_ckpt = None
    n_chunks = args.total_steps // args.chunk_size
    for i in range(n_chunks):
        target = (i + 1) * args.chunk_size
        cmd = ["python", "-m", train_module,
               "--out_prefix", str(chunked_pfx),
               "--total_steps", str(target),
               "--snapshot_at", str(target),
               ] + common
        if prev_ckpt is not None:
            cmd += ["--resume_from", str(prev_ckpt)]
        chunk_log = args.workdir / f"chunked_chunk{i+1}.log"
        rc = run_subprocess(cmd, chunk_log, env=env)
        if rc != 0:
            sys.exit(f"chunk {i+1} failed with rc={rc}; see {chunk_log}")
        # Append chunk log to combined chunked.log
        with chunked_log.open("a") as f:
            f.write(chunk_log.read_text())
        prev_ckpt = chunked_pfx.with_name(f"{chunked_pfx.name}_step{target}.pt")

    # ---------- Compare ----------
    cont_losses = parse_losses_from_log(cont_log, log_prefix)
    chunk_losses = parse_losses_from_log(chunked_log, log_prefix)

    print(f"\n=== Comparison ({args.arch}, {args.total_steps} steps, chunk={args.chunk_size}) ===", flush=True)
    print(f"  continuous: {len(cont_losses)} step-loss entries")
    print(f"  chunked:    {len(chunk_losses)} step-loss entries")

    n_match = n_diverge = n_missing = 0
    biggest_rel = 0.0
    biggest_step = -1
    for step in sorted(cont_losses):
        if step not in chunk_losses:
            n_missing += 1
            continue
        a, b = cont_losses[step], chunk_losses[step]
        denom = max(abs(a), abs(b), 1e-9)
        rel = abs(a - b) / denom
        if rel > biggest_rel:
            biggest_rel = rel
            biggest_step = step
        if rel <= args.rel_tol:
            n_match += 1
        else:
            n_diverge += 1
            if n_diverge <= 5:
                print(f"  step {step}: cont={a:.6f}  chunked={b:.6f}  rel_err={rel:.3e}")

    print(f"\n  matched: {n_match}/{len(cont_losses)}  diverged: {n_diverge}  missing: {n_missing}")
    print(f"  largest rel error: {biggest_rel:.3e} at step {biggest_step}")
    if n_diverge == 0 and n_missing == 0:
        print("\nPASS: chunked == continuous within tolerance")
        sys.exit(0)
    else:
        print("\nFAIL: chunked != continuous")
        sys.exit(1)


if __name__ == "__main__":
    main()
