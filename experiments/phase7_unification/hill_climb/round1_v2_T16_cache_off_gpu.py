"""Hill-climbing round 1 V2: SubseqH8 T_max=16 on local 5090.

Runs ONLY the V2 cell (T_max=16) so it doesn't overlap V1 (already
trained by Agent A as `hill_subseq_h8_T12_s5__seed42`). V3 (T_max=20)
deferred to H200.

Memory strategy: a first attempt (2026-04-29) used cache-off-GPU with
the full 24k-seq buffer pinned in CPU RAM. That hit ~9 sec/step on
the 5090 — synchronous PCIe copies of the 4-shift fancy-indexed batch
saturated a single CPU core. At max_steps=8000 it would have run for
~20 hours, impractical.

**Switched to cache-ON-GPU with n_seqs=12000** (vs the paper-canonical
24000). New budget on the 5090's 32 GB:
  7 GB cache + 16 GB W+Adam + 3 GB workspace = 26 GB.
Step rate now matches the H200 baseline (~2 sec/step).

The n_seqs reduction is a documented deviation from
`paper_archs.json:training_constants.preload_seqs=24000`. Justification:
max_steps=8000 × batch=4096 = 32M token-samples — even at n_seqs=12000
(=1.5M unique tokens) the buffer is oversampled ~21×, well past the
saturating regime. Plateau threshold, min_steps, batch_size, lr,
max_steps unchanged.

Run from repo root:
    .venv/bin/python -m experiments.phase7_unification.hill_climb.round1_v2_T16_cache_off_gpu
"""
from __future__ import annotations

import time

import torch

from experiments.phase7_unification._paths import (
    banner, DEFAULT_D_IN, DEFAULT_D_SAE,
)
from experiments.phase7_unification._train_utils import (
    preload_single, make_multidistance_pair_gen_gpu,
)
from experiments.phase7_unification.train_phase7 import (
    TrainCfg, _save_run, _meta_from_arch, _hf_push_ckpt,
)


# Z's 5090 deviation from paper-canonical preload_seqs=24000 — see header.
# 6000 instead of 12000 — first attempt at 12k saturated the 32 GB VRAM
# (228 MiB free) and made per-step pathologically slow (~10 min, no
# step=200 print after 12 min). 6k gives ~5 GB of headroom for kernel
# workspace and matryoshka sort buffers.
N_SEQS_5090 = 6_000


def make_arch(T_max: int, row: int) -> dict:
    arch_id = f"hill_subseq_h8_T{T_max}_s5"
    return {
        "row": row,
        "arch_id": arch_id,
        "group": 99,
        "T": None,
        "T_max": T_max,
        "t_sample": 5,
        "n_layers": None,
        "k_win": 500,
        "k_pos": int(500 / T_max),
        "shifts": None,
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.phase5b_subseq_sampling_txcdr",
        "src_class": "SubseqH8",
        "recipe": f"hill-climb subseq H8 at T_max={T_max} (5090 n_seqs={N_SEQS_5090})",
        "purpose": "round1 hill-climb V2: T_max=16 on local 5090",
    }


def train_subseq_h8_gpu_buf(arch: dict, cfg: TrainCfg, buf_gpu) -> tuple:
    """Mirror of train_phase7.train_subseq_h8 + train_phase7._contrastive_train,
    with verbose per-log_every step output (printed + flushed) so a crash
    mid-training leaves a visible trail in the log file. Cache buf is on GPU.
    """
    import time as _time
    import numpy as _np
    import torch.nn as _nn
    from src.architectures.phase5b_subseq_sampling_txcdr import SubseqH8
    from experiments.phase7_unification._train_utils import compute_plateau

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
    gen = make_multidistance_pair_gen_gpu(buf_gpu, T_max, list(shifts))
    # b_dec geometric-median init from the unshifted slice of one batch.
    init_x = gen(cfg.batch_size)[:, 0]
    if hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x)

    # Verbose contrastive training loop — same semantics as
    # train_phase7._contrastive_train, just with per-log_every prints + flush.
    torch.manual_seed(cfg.seed)
    _np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = _time.time()
    print(f"  [train] start max_steps={cfg.max_steps} log_every={cfg.log_every} "
          f"min_steps={cfg.min_steps} plateau_thr={cfg.plateau_threshold}",
          flush=True)
    for step in range(cfg.max_steps):
        x = gen(cfg.batch_size)
        loss, _, z = model(x, alpha=1.0)
        opt.zero_grad()
        loss.backward()
        if hasattr(model, "remove_gradient_parallel_to_decoder"):
            model.remove_gradient_parallel_to_decoder()
        _nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
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
            elapsed_min = (_time.time() - t0) / 60.0
            print(f"  [train] step={step:5d} loss={loss.item():.2f} l0={l0:.2f} "
                  f"plateau={plateau_val} elapsed={elapsed_min:.1f}m",
                  flush=True)
            if (
                plateau_val is not None
                and plateau_val < cfg.plateau_threshold
                and step >= cfg.min_steps
            ):
                converged = True
                print(f"  [train] CONVERGED at step={step} plateau={plateau_val:.4f} "
                      f"< {cfg.plateau_threshold}", flush=True)
                break
    elapsed = _time.time() - t0
    model.eval()
    log = {
        "loss": losses, "l0": l0s, "steps_logged": steps_logged,
        "final_step": steps_logged[-1] if steps_logged else 0,
        "converged": converged, "plateau_last": plateau_val,
        "elapsed_s": elapsed,
        "T_max": T_max, "t_sample": t_sample, "shifts": list(shifts),
        "matryoshka_h_size": h,
        "n_seqs_used": N_SEQS_5090,
        "deviation_note": (
            "preload_seqs reduced from paper-canonical 24000 to 12000 "
            "to fit cache-on-GPU on the 5090's 32 GB VRAM at T_max=16 "
            "without the slow cache-off-GPU path."
        ),
    }
    return model, log


def main() -> None:
    banner(__file__)
    seed = 42
    cfg = TrainCfg(seed=seed, max_steps=8000)

    print(f"Preloading anchor buffer to GPU (n_seqs={N_SEQS_5090}, "
          "5090 deviation from paper canon)...")
    t0 = time.time()
    buf_gpu = preload_single(n_seqs=N_SEQS_5090)
    print(f"  shape={tuple(buf_gpu.shape)} dtype={buf_gpu.dtype} "
          f"device={buf_gpu.device} "
          f"size={buf_gpu.element_size() * buf_gpu.numel() / 1e9:.1f} GB "
          f"(load {time.time()-t0:.1f}s)")

    arch = make_arch(T_max=16, row=101)
    run_id = f"{arch['arch_id']}__seed{seed}"
    print(f"\n=== {run_id} (T_max={arch['T_max']}, t_sample={arch['t_sample']}) ===")
    t0 = time.time()
    torch.manual_seed(seed)
    try:
        model, log = train_subseq_h8_gpu_buf(arch, cfg, buf_gpu)
    except Exception as e:
        print(f"  TRAIN FAIL {type(e).__name__}: {e}")
        import traceback; traceback.print_exc()
        return
    elapsed = time.time() - t0
    print(f"  train done in {elapsed/60:.1f} min "
          f"(final_step={log.get('final_step')}, converged={log.get('converged')}, "
          f"plateau={log.get('plateau_last')})")

    meta = _meta_from_arch(arch, seed)
    ckpt_path = _save_run(model, log, run_id, meta)
    print(f"  ckpt: {ckpt_path}")
    ok = _hf_push_ckpt(ckpt_path, run_id)
    print(f"  HF push: {'OK' if ok else 'FAIL'}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
