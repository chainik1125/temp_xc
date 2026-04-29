"""Train ONE Z-variant cell (rank-routed or shared) — spawn-per-cell pattern.

Each invocation = fresh Python process = clean CUDA context, so an OOM
in cell A can't poison cell B (per the spawn-per-cell pattern Z fell
into in the previous round). Wrap with bash to run multiple cells
sequentially:

    for spec in "ranked 20 5" "ranked 16 4" "shared 20 -" "shared 24 -"; do
      read variant T_max t_sample <<< "$spec"
      args="--variant $variant --T_max $T_max"
      [ "$t_sample" != "-" ] && args+=" --t_sample $t_sample"
      .venv/bin/python -m experiments.phase7_unification.hill_climb._run_one_z_variant $args
    done

Cache slicing: by default --ctx=64 to fit the 5090's 32 GB budget at
T_max=20 (last 64 positions; left-padded cache means [-ctx:] is the
right direction — Z's prior session bug was the first-N slice). Pass
--ctx 128 for full canonical buffer (safer but borderline OOM at
T_max=20).
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


def make_arch_ranked(T_max: int, t_sample: int) -> dict:
    arch_id = f"hill_z_ranked_T{T_max}_s{t_sample}"
    return {
        "row": 400 + (T_max * 10) + t_sample,
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
        "src_module": "src.architectures.phase7_subseq_z_variants",
        "src_class": "SubseqRankedH8",
        "recipe": (
            f"Z-variant rank-routed SubseqH8 T_max={T_max} t_sample={t_sample}. "
            f"Encoder/decoder slabs = t_sample (not T_max), so encoder "
            f"forward only stores activations for t_sample positions. "
            f"Probe-time uses equally-spaced positions routed by rank."
        ),
        "purpose": "round4 hill-climb: per-sampled-slot to fit 5090 at large T_max",
    }


def make_arch_shared(T_max: int) -> dict:
    arch_id = f"hill_z_shared_T{T_max}"
    return {
        "row": 500 + T_max,
        "arch_id": arch_id,
        "group": 99,
        "T": None,
        "T_max": T_max,
        "t_sample": None,
        "n_layers": None,
        "k_win": 500,
        "k_pos": None,
        "shifts": None,
        "alpha": 1.0,
        "gamma": None,
        "n_scales": None,
        "src_module": "src.architectures.phase7_subseq_z_variants",
        "src_class": "SubseqSharedH8",
        "recipe": (
            f"Z-variant shared-encoder SubseqH8 T_max={T_max}. Single (d_in, "
            f"d_sae) encoder + (d_sae, d_in) decoder; pre-act = W_enc @ "
            f"sum-pooled(window). Multi-distance InfoNCE provides the "
            f"temporal signal. Trivially fits 5090 at any T_max."
        ),
        "purpose": "round4 hill-climb: shared-encoder ablation vs ranked variant",
    }


def train_z_variant_verbose(arch: dict, cfg: TrainCfg, buf_anchor) -> tuple:
    """Verbose contrastive loop with flushed prints, mirroring
    train_phase7._contrastive_train semantics. Cache buf is on GPU.
    """
    src_class = arch["src_class"]
    T_max = arch["T_max"]
    k_win = arch["k_win"]
    h = int(DEFAULT_D_SAE * 0.2)
    raw_shifts = (1, max(1, T_max // 4), max(1, T_max // 2))
    shifts = tuple(sorted(set(s for s in raw_shifts if 1 <= s <= T_max - 1)))

    if src_class == "SubseqRankedH8":
        from src.architectures.phase7_subseq_z_variants import SubseqRankedH8
        model = SubseqRankedH8(
            DEFAULT_D_IN, DEFAULT_D_SAE,
            T_max=T_max, t_sample=arch["t_sample"], k=k_win,
            shifts=shifts, weights=None,
            matryoshka_h_size=h, alpha=1.0, contiguous=False,
        ).to("cuda")
    elif src_class == "SubseqSharedH8":
        from src.architectures.phase7_subseq_z_variants import SubseqSharedH8
        model = SubseqSharedH8(
            DEFAULT_D_IN, DEFAULT_D_SAE,
            T_max=T_max, k=k_win,
            shifts=shifts, weights=None,
            matryoshka_h_size=h, alpha=1.0, sum_pool=True,
        ).to("cuda")
    else:
        raise ValueError(f"Unknown src_class: {src_class}")

    gen = make_multidistance_pair_gen_gpu(buf_anchor, T_max, list(shifts))
    init_x = gen(cfg.batch_size)[:, 0]  # (B, T_max, d) — variant init handles routing
    if hasattr(model, "init_b_dec_geometric_median"):
        model.init_b_dec_geometric_median(init_x)
    del init_x
    torch.cuda.empty_cache()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    model.train()
    losses, l0s, steps_logged = [], [], []
    converged, plateau_val = False, None
    t0 = time.time()
    print(f"  [train] start variant={src_class} T_max={T_max} "
          f"shifts={shifts} max_steps={cfg.max_steps} "
          f"log_every={cfg.log_every} min_steps={cfg.min_steps} "
          f"plateau_thr={cfg.plateau_threshold}",
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
            print(f"  [train] step={step:5d} loss={loss.item():.2f} "
                  f"l0={l0:.2f} plateau={plateau_val} "
                  f"elapsed={elapsed_min:.1f}m",
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
        "T_max": T_max, "shifts": list(shifts),
        "matryoshka_h_size": h, "alpha": 1.0,
        "src_class": src_class,
    }
    if "t_sample" in arch and arch["t_sample"] is not None:
        log["t_sample"] = arch["t_sample"]
    return model, log


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--variant", required=True, choices=["ranked", "shared"],
        help="ranked = SubseqRankedH8 (t_sample slabs); "
             "shared = SubseqSharedH8 (1 slab).",
    )
    ap.add_argument("--T_max", type=int, required=True)
    ap.add_argument(
        "--t_sample", type=int, default=None,
        help="Required for --variant ranked; ignored for --variant shared.",
    )
    ap.add_argument(
        "--ctx", type=int, default=128,
        help="Crop L axis of the cache to last ctx positions "
             "(default 128 = full L). Slice direction is [:, -ctx:, :] "
             "(left-padded fineweb cache; min n_real=37, so ctx≤32 is "
             "guaranteed all-real, ctx=64 is mostly-real). NEVER "
             "[:, :ctx, :] — that's the prior-session bug. Recommended: "
             "--ctx 64 for T_max=20 (cache 7 GB; 34 valid offsets/seq).",
    )
    ap.add_argument(
        "--n_seqs", type=int, default=24_000,
        help="Sequences to preload (default paper canon 24000).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_steps", type=int, default=8_000)
    args = ap.parse_args()
    banner(__file__)

    if args.variant == "ranked":
        if args.t_sample is None:
            raise SystemExit("--variant ranked requires --t_sample")
        arch = make_arch_ranked(T_max=args.T_max, t_sample=args.t_sample)
    else:
        arch = make_arch_shared(T_max=args.T_max)

    seed = args.seed
    cfg = TrainCfg(seed=seed, max_steps=args.max_steps)
    run_id = f"{arch['arch_id']}__seed{seed}"
    ckpt_path = CKPT_DIR / f"{run_id}.pt"
    if ckpt_path.exists():
        print(f"=== {run_id}: ckpt already exists, skip ===", flush=True)
        return

    print(f"=== {run_id} (variant={args.variant} T_max={args.T_max}, "
          f"t_sample={args.t_sample}, n_seqs={args.n_seqs}, ctx={args.ctx}) ===",
          flush=True)
    print(f"Preloading anchor buffer to GPU (n_seqs={args.n_seqs}, "
          f"ctx={args.ctx})...", flush=True)
    t0 = time.time()
    buf_anchor = preload_single(n_seqs=args.n_seqs)
    if args.ctx < buf_anchor.shape[1]:
        # LAST ctx positions — see header docstring on slice direction.
        buf_anchor = buf_anchor[:, -args.ctx:, :].contiguous()
        torch.cuda.empty_cache()
    print(f"  shape={tuple(buf_anchor.shape)} dtype={buf_anchor.dtype} "
          f"device={buf_anchor.device} "
          f"size={buf_anchor.element_size() * buf_anchor.numel() / 1e9:.1f} GB "
          f"(load {time.time()-t0:.1f}s)",
          flush=True)

    t0 = time.time()
    torch.manual_seed(seed)
    try:
        model, log = train_z_variant_verbose(arch, cfg, buf_anchor)
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
    log["ctx_slice_direction"] = "last" if args.ctx < 128 else "full"
    meta = _meta_from_arch(arch, seed)
    out_path = _save_run(model, log, run_id, meta)
    print(f"  ckpt: {out_path}", flush=True)
    ok = _hf_push_ckpt(out_path, run_id)
    print(f"  HF push: {'OK' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
