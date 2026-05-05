"""Benchmark: TXCPro multi_window=False vs True on a synthetic task.

Verifies the two claims that motivated decisions.md § 14:
1. Multi-window mode processes ~N× more tokens per step at roughly
   the same wall-time (encoder is GPU-underutilized at small batch×T;
   the bigger (B*N, T, d) shape is GEMM-friendly).
2. Multi-window mode converges faster *in step count* (each step
   trains on N× more data) and ideally also in wall-time.

Synthetic task: each input sequence is (seq_len, d_in) iid Gaussian
plus a low-rank "concept" component shared across all positions in
the sequence. Both modes minimize MSE reconstruction; the bigger
per-step data feed of multi-window should produce a faster loss
descent on a wall-time-vs-loss plot.

Usage:
    .venv/bin/python -m scripts.bench_txc_multi_window
"""

from __future__ import annotations

import time
import torch

from temp_bench.architectures.txc_pro import TXCPro

# Small dims for fast benchmarking on local 5090 (32 GB VRAM)
D_IN = 512
D_SAE = 2048
T_MAX = 10
T_SAMPLE = 5
SHIFTS = (1, 2)
K_POS = 8

B = 64
SEQ_LEN = 64                 # → multi_window: N = seq_len // (T_max + max_shift) = 64 // 12 = 5
N_STEPS = 500
LR = 3e-4
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _make_synthetic_data(n_seqs: int) -> torch.Tensor:
    """Random sequences with a low-rank shared component per sequence
    so the SAE has something to compress."""
    torch.manual_seed(SEED)
    # Per-sequence "concept" vector (rank K)
    K = 16
    concept_dirs = torch.randn(D_IN, K, device=DEVICE) / (D_IN ** 0.5)
    concept_coefs = torch.randn(n_seqs, K, device=DEVICE)
    concept = (concept_coefs @ concept_dirs.T).unsqueeze(1)  # (n_seqs, 1, d_in)
    noise = torch.randn(n_seqs, SEQ_LEN, D_IN, device=DEVICE) * 0.5
    return concept + noise  # (n_seqs, seq_len, d_in)


def _build_arch(multi_window: bool) -> TXCPro:
    torch.manual_seed(SEED)  # match init across modes
    return TXCPro(
        d_in=D_IN,
        d_sae=D_SAE,
        T_max=T_MAX,
        t_sample=T_SAMPLE,
        k_pos=K_POS,
        contrastive_shifts=SHIFTS,
        contrastive_alpha=1.0,           # full TXC-pro recipe
        auxk_alpha=0.0,                  # disable AuxK (no dead features in 500 iters)
        bdec_geom_median_init=False,     # not needed for synthetic
        multi_window=multi_window,
    ).to(DEVICE)


def _run_one(multi_window: bool, data: torch.Tensor) -> dict:
    """Train one arch for N_STEPS, return timing + loss curve."""
    model = _build_arch(multi_window=multi_window)
    optim = torch.optim.Adam(model.parameters(), lr=LR)
    n_seqs = data.shape[0]
    rng = torch.Generator(device=DEVICE).manual_seed(SEED)

    # n_tokens_per_step depends on the mode (this is what we want to verify):
    #   non-MW: 1 anchor + len(shifts) positives, each (B, T_max, d), but
    #           t_sample positions are summed → effective B*t_sample tokens per anchor
    #   MW:     N anchors per row (with shifts) → B*N*t_sample
    max_shift = max(SHIFTS)
    if multi_window:
        N = SEQ_LEN // (T_MAX + max_shift)
        n_anchors_per_step = B * N
    else:
        n_anchors_per_step = B
    # Each anchor contributes its own t_sample-token recon (encode is
    # subset-summed). Positives also contribute t_sample tokens each.
    tokens_per_step = n_anchors_per_step * T_SAMPLE * (1 + len(SHIFTS))

    losses: list[float] = []
    times: list[float] = []
    cum_tokens: list[int] = []
    cum_t = 0
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t_start = time.perf_counter()

    for step in range(N_STEPS):
        idx = torch.randint(0, n_seqs, (B,), generator=rng, device=DEVICE)
        x = data[idx]
        loss, _ = model.train_step(x)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        cum_t += tokens_per_step
        if (step + 1) % 25 == 0:
            if DEVICE == "cuda":
                torch.cuda.synchronize()
            t_elapsed = time.perf_counter() - t_start
            losses.append(float(loss.detach()))
            times.append(t_elapsed)
            cum_tokens.append(cum_t)

    if DEVICE == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - t_start
    return {
        "multi_window": multi_window,
        "total_wall_s": total_time,
        "tokens_per_step": tokens_per_step,
        "total_tokens_seen": cum_t,
        "loss_curve": losses,
        "time_curve": times,
        "cum_tokens_curve": cum_tokens,
        "final_loss": losses[-1] if losses else float("nan"),
    }


def main():
    print(f"[bench] device={DEVICE}, d_in={D_IN}, d_sae={D_SAE}, "
          f"B={B}, seq_len={SEQ_LEN}, T_max={T_MAX}, t_sample={T_SAMPLE}, "
          f"n_steps={N_STEPS}")
    if DEVICE == "cuda":
        print(f"[bench] gpu={torch.cuda.get_device_name(0)}, "
              f"vram={torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data = _make_synthetic_data(n_seqs=2048)

    print("\n[bench] warm-up (50 steps each, untimed)…")
    for mw in (False, True):
        m = _build_arch(multi_window=mw)
        op = torch.optim.Adam(m.parameters(), lr=LR)
        for _ in range(50):
            idx = torch.randint(0, data.shape[0], (B,), device=DEVICE)
            loss, _ = m.train_step(data[idx])
            op.zero_grad(set_to_none=True); loss.backward(); op.step()
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    print("\n[bench] timed runs (500 steps each):")
    print("=" * 78)
    print("    mw=False (single anchor per row)")
    r_off = _run_one(multi_window=False, data=data)
    print(f"    wall={r_off['total_wall_s']:.1f}s  "
          f"tokens/step={r_off['tokens_per_step']}  "
          f"total_tokens={r_off['total_tokens_seen']:_}  "
          f"final_loss={r_off['final_loss']:.4f}")

    print("\n    mw=True (stride-min_seq tiled, N=5 anchors per row)")
    r_on = _run_one(multi_window=True, data=data)
    print(f"    wall={r_on['total_wall_s']:.1f}s  "
          f"tokens/step={r_on['tokens_per_step']}  "
          f"total_tokens={r_on['total_tokens_seen']:_}  "
          f"final_loss={r_on['final_loss']:.4f}")
    print("=" * 78)

    print("\n[bench] verdict:")
    speedup_per_step = r_off["tokens_per_step"]
    n_factor = r_on["tokens_per_step"] / r_off["tokens_per_step"]
    wall_ratio = r_on["total_wall_s"] / r_off["total_wall_s"]
    print(f"    tokens/step ratio (MW / non-MW) = {n_factor:.2f}×")
    print(f"    wall-time ratio (MW / non-MW)   = {wall_ratio:.2f}×  "
          f"(if << {n_factor:.1f}× then we're recovering parallelism)")
    print(f"    tokens-per-wall-second ratio    = "
          f"{(r_on['total_tokens_seen']/r_on['total_wall_s']) / (r_off['total_tokens_seen']/r_off['total_wall_s']):.2f}×")
    print(f"    final-loss difference (lower is better) = "
          f"{r_off['final_loss'] - r_on['final_loss']:+.4f}")
    if r_on["final_loss"] < r_off["final_loss"]:
        print("    ✓ MW converged to a LOWER loss in the same step count")
    else:
        print("    ✗ MW did NOT converge to a lower loss in same step count")

    print("\n[bench] loss curve (every 25 steps):")
    print(f"    {'step':>5} {'mw=False':>12} {'mw=True':>12} {'Δ':>10}")
    for i, ((s_off, l_off, t_off), (s_on, l_on, t_on)) in enumerate(zip(
        zip(range(25, N_STEPS + 1, 25), r_off["loss_curve"], r_off["time_curve"]),
        zip(range(25, N_STEPS + 1, 25), r_on["loss_curve"], r_on["time_curve"]),
    )):
        print(f"    {s_off:>5} {l_off:>12.4f} {l_on:>12.4f} {l_off - l_on:>+10.4f}")


if __name__ == "__main__":
    main()
