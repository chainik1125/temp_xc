"""Train ONE SubseqH8 cell at seed=42 — spawn-per-cell pattern.

Each invocation = fresh Python process = clean CUDA context, so an OOM
in cell A can't poison cell B. Wrap with bash to run multiple cells
sequentially, e.g.:

    for spec in "14 5 18000" "20 2 18000" "16 2 18000"; do
      .venv/bin/python -m experiments.phase7_unification.hill_climb._run_one_subseq $spec
    done
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from experiments.phase7_unification._paths import (
    banner, CKPT_DIR, DEFAULT_D_IN, DEFAULT_D_SAE,
)
from experiments.phase7_unification._train_utils import (
    TrainCfg, preload_single, make_multidistance_pair_gen_gpu, compute_plateau,
)
from experiments.phase7_unification.train_phase7 import (
    _save_run, _meta_from_arch, _hf_push_ckpt,
)


def make_arch(T_max: int, t_sample: int) -> dict:
    arch_id = f"hill_subseq_h8_T{T_max}_s{t_sample}"
    return {
        "row": 300 + (T_max * 10) + t_sample,
        "arch_id": arch_id,
        "group": 99,
        "T": None,
        "T_max": T_max,
        "t_sample": t_sample,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T_max),
        "shifts": None,
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": f"hill-climb subseq H8 T_max={T_max} t_sample={t_sample}",
        "purpose": "round3 hill-climb: separate-process",
    }


def train_subseq_h8_verbose(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8

    T_max = arch["T_max"]
    t_sample = arch["t_sample"]
    k = arch["k_win"]
    h = int(DEFAULT_D_SAE * 0.2)
    raw_shifts = (1, max(1, T_max // 4), max(1, T_max // 2))
    shifts = tuple(sorted(set(s for s in raw_shifts if 1 <= s <= T_max - 1)))
    model = SubseqH8(
        DEFAULT_D_IN, DEFAULT_D_SAE, T_max=T_max, k=k, t_sample=t_sample,
        contiguous=False, shifts=shifts, weights=None,
        matryoshka_h_size=h, alpha=1.0,
    ).to("cuda")
    gen = make_multidistance_pair_gen_gpu(buf_anchor, T_max, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]
    if hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    print(f"  [train] start T_max={T_max} t_sample={t_sample} shifts={shifts} "
          f"max_steps={cfg.max_steps} log_every={cfg.log_every} "
          f"min_steps={cfg.min_steps} plateau_thr={cfg.plateau_threshold}",
          flush=True)
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        if hasattr(model, "_normalize_decoder"):
            model._normalize_decoder()
        if step % cfg.log_every == 0 or step == cfg.max_steps - 1:
            with torch.no_grad():
                l0 = (z > 0).float().sum(dim=-1).mean().item()
            losses.append(loss.item())
            l0s.append(l0)
            steps_logged.append(step)
            plateau_val = compute_plateau(losses, window=5)
            elapsed_min = (time.time() - t0) / 60.0
            print(f"  [train] step={step:5d} loss={loss.item():.2f} l0={l0:.2f} "
                  f"plateau={plateau_val} elapsed={elapsed_min:.1f}m",
                  flush=True)
            if (
                plateau_val is not None
                and plateau_val < cfg.plateau_threshold
                and step >= cfg.min_steps
            ):
                converged = True
                print(f"  [train] CONVERGED at step={step} "
                      f"plateau={plateau_val:.4f} < {cfg.plateau_threshold}",
                      flush=True)
                break
    elapsed = time.time() - t0
    model.eval()
    log = {
        "loss": losses, "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_max": T_max, "t_sample": t_sample, "shifts": list(shifts),
        "matryoshka_h_size": h, "alpha": 1.0,
    }
    return model, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("T_max", type=int)
    ap.add_argument("t_sample", type=int)
    ap.add_argument("n_seqs", type=int)
    ap.add_argument(
        "--ctx", type=int, default=128,
        help="crop L axis of the cache to this length (default 128 = full). "
             "Cache stores 128 tokens per sequence; gen samples a window of "
             "T+max_shift tokens from random offsets within the L axis. "
             "Cropping L to e.g. 64 halves the buffer (14→7 GB) while still "
             "leaving ~30 valid offsets per sequence for typical T+max_shift "
             "≤ 30 — large memory win on the 5090.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=8000)
    args = ap.parse_args()
    banner(__file__)

    seed = args.seed
    cfg = TrainCfg(seed=seed, max_steps=args.max_steps)

    arch = make_arch(T_max=args.T_max, t_sample=args.t_sample)
    run_id = f"{arch['arch_id']}__seed{seed}"
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    if ckpt_path.exists():
        print(f"=== {run_id}: ckpt already exists, skip ===", flush=True)
        return

    print(f"=== {run_id} (T_max={args.T_max}, t_sample={args.t_sample}, "
          f"n_seqs={args.n_seqs}, ctx={args.ctx}) ===",
          flush=True)
    print(f"Preloading anchor buffer to GPU (n_seqs={args.n_seqs}, "
          f"ctx={args.ctx})...",
          flush=True)
    t0 = time.time()
    buf_anchor = preload_single(n_seqs=args.n_seqs)
    if args.ctx < buf_anchor.shape[1]:
        buf_anchor = buf_anchor[:, :args.ctx, :].contiguous()
        torch.cuda.empty_cache()
    print(f"  shape={tuple(buf_anchor.shape)} dtype={buf_anchor.dtype} "
          f"device={buf_anchor.device} "
          f"size={buf_anchor.element_size() * buf_anchor.numel() / 1e9:.1f} GB "
          f"(load {time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    torch.manual_seed(seed)
    try:
        model, log = train_subseq_h8_verbose(arch, cfg, buf_anchor)
    except Exception as e:
        print(f"  TRAIN FAIL {type(e).__name__}: {e}", flush=True)
        import traceback; traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"  train done in {elapsed/60:.1f} min "
          f"(final_step={log.get('final_step')}, "
          f"converged={log.get('converged')}, "
          f"plateau={log.get('plateau_last')})",
          flush=True)

    log["n_seqs_used"] = args.n_seqs
    log["ctx_used"] = args.ctx
    meta = _meta_from_arch(arch, seed)
    out_path = _save_run(model, log, run_id, meta)
    print(f"  ckpt: {out_path}", flush=True)
    ok = _hf_push_ckpt(out_path, run_id)
    print(f"  HF push: {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
