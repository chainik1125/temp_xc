"""TFA v4: small LR + large batch — robust convergence."""
from __future__ import annotations

import math, sys, time, json, os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, "/home/elysium/temp_xc")

from src.bench.architectures._tfa_module import TemporalSAE


def run_trial(
    *, total_steps, lr, grad_clip, batch_size=64, bottleneck_factor=8,
    normalize_decoder=True, log_every=200, trial_name="default",
):
    print(f"\n{'=' * 70}")
    print(f"TRIAL: {trial_name} lr={lr} clip={grad_clip} bs={batch_size} "
          f"norm_dec={normalize_decoder} bf={bottleneck_factor} steps={total_steps}")
    print("=" * 70)

    arr = np.load(
        "/home/elysium/temp_xc/data/cached_activations/"
        "gemma-2-2b-it/fineweb/resid_L25.npy", mmap_mode="r",
    )
    torch.manual_seed(42); np.random.seed(42)

    model = TemporalSAE(
        dimin=2304, width=18432, n_heads=4, sae_diff_type="topk", kval_topk=50,
        tied_weights=True, n_attn_layers=1, bottleneck_factor=bottleneck_factor,
        use_pos_encoding=True, max_seq_len=512,
    ).to("cuda")

    idx0 = np.random.randint(0, 22000, (batch_size,))
    x0 = torch.from_numpy(np.array(arr[idx0])).float().cuda()
    scaling_factor = math.sqrt(x0.shape[-1]) / x0.norm(dim=-1).mean().item()
    del x0
    print(f"  scaling_factor = {scaling_factor:.4f}")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [{"params": decay_params, "weight_decay": 1e-4},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )

    warmup = min(500, total_steps // 10)
    min_lr = lr * 0.1
    model.train()

    skipped = 0
    log_history = []
    t0 = time.time()
    best_eval_loss = float("inf")

    for step in range(total_steps):
        if step < warmup:
            current_lr = lr * step / max(1, warmup)
        else:
            dec = (step - warmup) / max(1, total_steps - warmup)
            coeff = 0.5 * (1.0 + math.cos(math.pi * dec))
            current_lr = min_lr + coeff * (lr - min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        idx = np.random.randint(0, 22000, (batch_size,))
        batch = torch.from_numpy(np.array(arr[idx])).float().cuda() * scaling_factor

        recons, inter = model(batch)
        loss = F.mse_loss(
            recons.reshape(-1, recons.shape[-1]),
            batch.reshape(-1, batch.shape[-1]),
            reduction="sum",
        ) / (batch.shape[0] * batch.shape[1])

        if loss.isnan().any() or loss.isinf().any():
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            if any(p.isnan().any() for p in model.parameters()):
                print(f"  step {step}: PARAMS NAN. HALTING.")
                break
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        if grad_norm.isnan() or grad_norm.isinf():
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        if normalize_decoder:
            with torch.no_grad():
                norms = model.D.data.norm(dim=1, keepdim=True).clamp(min=1e-8)
                model.D.data.div_(norms)

        if step % log_every == 0 or step == total_steps - 1:
            with torch.no_grad():
                novel_l0 = (inter["novel_codes"] > 0).float().sum(dim=-1).mean().item()
                D_nan = model.D.isnan().any().item()
                D_norm = model.D.norm().item()
            elapsed = time.time() - t0
            log_history.append(dict(
                step=step, loss=loss.item(), L0=novel_l0, grad=grad_norm.item(),
                lr=current_lr, D_norm=D_norm, elapsed=elapsed, skipped=skipped,
            ))
            print(
                f"  step {step:5d} {elapsed/60:5.1f}m loss={loss.item():>8.2f} "
                f"L0={novel_l0:>5.1f} grad={grad_norm.item():>9.1e} "
                f"lr={current_lr:.1e} D={D_norm:>7.1f} skip={skipped}",
                flush=True,
            )
            if D_nan:
                break

    total_time = time.time() - t0
    final_bad = any(p.isnan().any() for p in model.parameters())
    print(f"\n  RESULT: {'SUCCESS' if not final_bad else 'FAILED'} "
          f"time={total_time/60:.1f}m skipped={skipped}")

    print("  Eval on 200 sequences...")
    model.eval()
    with torch.no_grad():
        eval_x = torch.from_numpy(np.array(arr[-200:])).float().cuda() * scaling_factor
        recons, _ = model(eval_x)
        se = (recons - eval_x).pow(2).sum().item()
        signal = eval_x.pow(2).sum().item()
        nmse = se / signal
    print(f"  Eval NMSE: {nmse:.4f}")

    return dict(
        trial=trial_name, success=not final_bad, total_time=total_time,
        skipped=skipped, eval_nmse=nmse, log=log_history,
    )


if __name__ == "__main__":
    # Fix v4: small LR + large batch, longer training
    result = run_trial(
        total_steps=8000,
        lr=3e-4,             # 3x smaller
        grad_clip=1.0,
        batch_size=64,
        normalize_decoder=True,
        trial_name="fix_v4_lr3e-4_bs64",
    )

    os.makedirs("logs", exist_ok=True)
    with open("/home/elysium/temp_xc/logs/tfa_debug_fix_v4.json", "w") as f:
        json.dump([result], f, indent=2)
