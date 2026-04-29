"""CS2 controls — long FineWeb-edu text + width sweep + position ablation.

Extends `smoke_masked_recon.py` with the controls listed in the
2026-04-29-y-cs2-masked-recon.md "What needs to happen before this
becomes a paper case study" section, plus hill-climb variants:

  (1) confirm on long natural text (FineWeb-edu, S=256), not the
      150-sentence Q1.1 cache;
  (2) compare against naive baselines (predict-zero, predict-mean,
      predict-x[t-w], predict-mean-of-context-window);
  (3) ablate the held-out position within the T-window for window
      archs (leftmost / centre / rightmost);
  (4) hill-climb: corruption width sweep w in {1, 2, 3} — per-token
      archs cannot use context, so their FVE stays at 0 regardless
      of width; window archs degrade gracefully with width and the
      absolute gap to per-token grows;
  (5) more TXC-family archs (txc_bare_antidead_t5, txcdr_t10, ...).

We aggregate everything into one JSON + one plot. The plot has
three panels:

  - Panel A: baseline + held-out FVE per arch (width=1 right-edge),
    matching the smoke-test format but on long text.
  - Panel B: held-out FVE vs corruption width per arch — the
    hill-climb story.
  - Panel C: held-out FVE vs within-window held-out position per
    window arch — direction-robustness story.

Output:
  results/case_studies/cs2_masked_recon/controls_masked_recon.{png,json}
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("HF_HOME", "/workspace/hf_cache")
os.environ.setdefault("TQDM_DISABLE", "1")

from experiments.phase7_unification._paths import OUT_DIR
from experiments.phase7_unification.case_studies._arch_utils import (
    window_T, _d_sae_of, MLC_CLASSES,
    load_phase7_model_safe as _load_phase7_model,
)
from experiments.phase7_unification.case_studies._paths import (
    CASE_STUDIES_DIR, SUBJECT_MODEL, ANCHOR_LAYER, DEFAULT_D_IN,
)


# ─────────────────────────────────────────────────────────── config ──

N_PASSAGES = 30
MAX_LEN = 256
ARCHS = [
    ("topk_sae", "per-token (k=500)"),
    ("tsae_paper_k20", "per-token (k=20)"),
    ("agentic_txc_02", "TXC matryoshka (T=5)"),
    ("phase5b_subseq_h8", "SubseqH8 (T_max=10)"),
]
WIDTHS = [1, 2, 3]                              # corruption widths to sweep
POSITIONS = ("right", "center", "left")          # within-window ablation (w=1)
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/cs2_masked_recon")
)
LONG_CACHE = OUT_SUBDIR / "_l12_long_cache.npz"  # gitignored


# ───────────────────────────────────────────── L12 cache from text ──

def _pull_fineweb(n: int) -> list[str]:
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      streaming=True, split="train")
    out = []
    for row in ds:
        out.append(row["text"])
        if len(out) >= n:
            break
    return out


def _capture_l12(passages: list[str], device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"  loading {SUBJECT_MODEL} (bf16)...")
    tokenizer = AutoTokenizer.from_pretrained(SUBJECT_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        SUBJECT_MODEL, torch_dtype=torch.bfloat16, device_map="cuda",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    n = len(passages)
    acts = np.zeros((n, MAX_LEN, DEFAULT_D_IN), dtype=np.float16)
    attn = np.zeros((n, MAX_LEN), dtype=np.int8)
    captured = {}

    def hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["x"] = h.detach().cpu()

    handle = model.model.layers[ANCHOR_LAYER].register_forward_hook(hook)
    try:
        for start in range(0, n, 4):
            end = min(start + 4, n)
            chunk = passages[start:end]
            enc = tokenizer(chunk, return_tensors="pt", padding="max_length",
                            truncation=True, max_length=MAX_LEN)
            captured.clear()
            with torch.no_grad():
                model(enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
            h = captured["x"]
            if h.shape[-1] != DEFAULT_D_IN:
                h = h[..., :DEFAULT_D_IN]
            acts[start:end] = h.to(torch.float16).numpy()
            attn[start:end] = enc["attention_mask"].to(torch.int8).numpy()
    finally:
        handle.remove()
        del model
        torch.cuda.empty_cache()
    return acts, attn


def _ensure_cache(device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    if LONG_CACHE.exists():
        with np.load(LONG_CACHE, allow_pickle=False) as z:
            return z["l12"].copy(), z["attn"].copy()
    print(f"CS2 controls — pulling {N_PASSAGES} FineWeb-edu passages")
    passages = _pull_fineweb(N_PASSAGES)
    print(f"  capturing L{ANCHOR_LAYER} from {SUBJECT_MODEL}")
    acts, attn = _capture_l12(passages, device)
    np.savez(LONG_CACHE, l12=acts, attn=attn)
    print(f"  cached -> {LONG_CACHE} ({acts.nbytes / 1e6:.0f} MB)")
    return acts, attn


# ────────────────────────────────────────────── decoder helpers ──

def _decode_full_window(sae, src_class: str, z: torch.Tensor, T: int) -> torch.Tensor:
    if hasattr(sae, "decode") and not src_class.startswith("Matryoshka"):
        return sae.decode(z)
    if hasattr(sae, "decode_scale"):
        return sae.decode_scale(z, T - 1)
    raise AttributeError(f"no decode/decode_scale on {src_class}")


def _encode(sae, src_class: str, sub: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        if src_class == "TemporalMatryoshkaBatchTopKSAE":
            z = sae.encode(sub, use_threshold=True)
            if isinstance(z, tuple):
                z = z[0]
        else:
            z = sae.encode(sub)
    return z


# ────────────────────────────────────────── per-arch held-out runs ──

def _zero_input_recon(sae, src_class: str, T: int, d_in: int,
                      device: torch.device) -> torch.Tensor:
    """Reconstruction when the encoder input is all-zero. Used as the
    per-token holdout output (input is the (1, d_in) zero vector for
    per-token archs, the (1, T, d_in) zero tensor for window archs)."""
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        zero = torch.zeros((1, d_in), device=device)
        z = _encode(sae, src_class, zero)
        with torch.no_grad():
            return sae.decode(z).squeeze(0)
    zero = torch.zeros((1, T, d_in), device=device)
    z = _encode(sae, src_class, zero)
    with torch.no_grad():
        out = _decode_full_window(sae, src_class, z, T)
    return out.squeeze(0)            # (T, d_in)


def _per_token_baseline_and_holdout(sae, src_class: str, x: torch.Tensor,
                                    d_in: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token archs: baseline encode/decode at every position;
    holdout = constant ~b_dec (zero input). Returns (x_hat_baseline,
    x_hat_holdout_const) where x_hat_baseline is shape (N, S, d_in) and
    x_hat_holdout_const is shape (d_in,) (the same value at every
    position)."""
    N, S = x.shape[:2]
    flat = x.reshape(N * S, d_in)
    z = _encode(sae, src_class, flat)
    with torch.no_grad():
        x_hat = sae.decode(z).reshape(N, S, d_in)
    x_hat_zero = _zero_input_recon(sae, src_class, T=1, d_in=d_in, device=device)
    return x_hat, x_hat_zero


def _window_recon_at_position(sae, src_class: str, x: torch.Tensor, T: int,
                              hold_pos: int, hold_width: int, device: torch.device,
                              bs: int = 8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct at all valid window centres.
    For each window of length T, hold out positions [hold_pos..hold_pos+hold_width-1]
    of the window (zero them), encode + decode, and read the
    held-out positions of the reconstruction.

    Returns three tensors of shape (N, S, d_in):
      - x_hat_baseline: full-window reconstruction at every position,
        attributed via right-edge mapping (window centre at position
        t covers [t-T+1..t]; the rightmost output slice maps to t).
      - x_hat_holdout: held-out reconstruction at the
        held-out positions. Position t gets the reconstruction of the
        held-out slot inside the window centred at... we adopt the
        convention that window k covers passage positions [k..k+T-1],
        and we report reconstruction at passage position k + hold_pos
        (the held-out passage position for window k).
      - mask: (N, S) bool -- True at positions where x_hat_holdout is
        defined.

    Note on accounting: holding out position [hold_pos..hold_pos+w-1]
    of window k = passage positions [k+hold_pos..k+hold_pos+w-1]. We
    write reconstruction[k+hold_pos+i] = full_recon[k, hold_pos+i, :]
    for i in [0..w-1] (the FIRST window with that passage position
    held out is what we use; later windows that also span that
    position are ignored).
    """
    N, S, d_in = x.shape
    K = S - T + 1
    if K <= 0:
        empty = torch.zeros_like(x)
        mask_empty = torch.zeros((N, S), dtype=torch.bool, device=device)
        return empty, empty, mask_empty
    pos_idx = torch.arange(S, device=device)
    x_hat_baseline = torch.zeros_like(x)
    x_hat_holdout = torch.zeros_like(x)
    mask_holdout = torch.zeros((N, S), dtype=torch.bool, device=device)

    # Process each passage independently to bound memory.
    for n_idx in range(N):
        x_n = x[n_idx]                              # (S, d_in)
        windows = x_n.unfold(0, T, 1).movedim(-1, 1).contiguous()   # (K, T, d_in)
        windows_holdout = windows.clone()
        windows_holdout[:, hold_pos:hold_pos + hold_width, :] = 0.0
        # Baseline.
        for i in range(0, K, bs):
            j = min(i + bs, K)
            sub_b = windows[i:j]
            z_b = _encode(sae, src_class, sub_b)
            with torch.no_grad():
                full_b = _decode_full_window(sae, src_class, z_b, T)
            # Right-edge mapping: window k -> passage position k + T - 1
            x_hat_baseline[n_idx, i + T - 1: j + T - 1] = full_b[:, -1, :]
            del z_b, full_b
            # Holdout.
            sub_h = windows_holdout[i:j]
            z_h = _encode(sae, src_class, sub_h)
            with torch.no_grad():
                full_h = _decode_full_window(sae, src_class, z_h, T)
            # Holdout output for held-out positions:
            # window k covers passage positions [i + 0 .. i + T-1] for window i,
            # i.e. window starting at passage index i. Position hold_pos in
            # window i = passage position i + hold_pos.
            for w_idx in range(j - i):
                k_passage_start = i + w_idx
                t_held = k_passage_start + hold_pos          # leftmost held-out passage pos
                if t_held + hold_width > S:
                    continue
                if mask_holdout[n_idx, t_held]:
                    continue
                # write w slots starting at t_held.
                x_hat_holdout[n_idx, t_held:t_held + hold_width] = (
                    full_h[w_idx, hold_pos:hold_pos + hold_width, :]
                )
                mask_holdout[n_idx, t_held:t_held + hold_width] = True
            del z_h, full_h
    return x_hat_baseline, x_hat_holdout, mask_holdout


# ───────────────────────────────────────────── per-arch driver ──

def _fve(err_sum: torch.Tensor, var_sum: torch.Tensor, mask: torch.Tensor) -> tuple[float, float, float]:
    """Compute FVE, MSE, signal-variance over a boolean mask of (N, S)."""
    if mask.sum() == 0:
        return 0.0, 0.0, 0.0
    mse = float(err_sum[mask].mean().item())
    sig = float(var_sum[mask].mean().item())
    return 1.0 - mse / max(sig, 1e-9), mse, sig


def _run_arch(arch_id: str, x_full: torch.Tensor, attn: torch.Tensor,
              device: torch.device) -> dict:
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    if not ckpt_path.exists():
        return {"skipped": f"missing ckpt {ckpt_path.name}"}
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        return {"src_class": src_class, "skipped": "MLC needs multi-layer cache"}
    print(f"  loading {arch_id} ({src_class})...")
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    T = window_T(sae, src_class, meta)
    d_sae = _d_sae_of(sae, src_class)
    print(f"    T={T} d_sae={d_sae}")

    N, S, d_in = x_full.shape
    pos_idx = torch.arange(S, device=device)
    var_sum = (x_full ** 2).sum(dim=-1)                        # (N, S)
    valid_text = (attn > 0)                                    # (N, S)

    out: dict = {"src_class": src_class, "T": T, "d_sae": d_sae}

    # ─── Baseline reconstruction (no mask).
    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        x_hat_baseline, x_hat_zero = _per_token_baseline_and_holdout(
            sae, src_class, x_full, d_in, device,
        )
    else:
        # Window arch: baseline is right-edge attribution; reuse the
        # rightmost held-out call but discard the holdout output.
        x_hat_baseline, _, _ = _window_recon_at_position(
            sae, src_class, x_full, T, hold_pos=T - 1, hold_width=1,
            device=device,
        )
    err_b = ((x_hat_baseline - x_full) ** 2).sum(dim=-1)
    valid_b = valid_text.clone()
    if T > 1:
        valid_b = valid_b & (pos_idx.unsqueeze(0) >= T - 1)
    fve_b, mse_b, sig_b = _fve(err_b, var_sum, valid_b)
    out["baseline"] = {"fve": fve_b, "mse": mse_b, "n_valid": int(valid_b.sum().item())}
    print(f"    baseline FVE = {fve_b:.3f}")

    # ─── Width sweep (rightmost held-out).
    out["width_sweep"] = {}
    for w in WIDTHS:
        if T == 1 and w > 1:
            # Per-token arch: holding out >1 consecutive tokens
            # produces the SAME constant b_dec at each held-out
            # position. FVE is computed by treating each held-out
            # position independently.
            err = ((x_hat_zero.unsqueeze(0).unsqueeze(0) - x_full) ** 2).sum(dim=-1)
            valid_w = valid_text & (pos_idx.unsqueeze(0) >= w - 1)
            fve_h, mse_h, _ = _fve(err, var_sum, valid_w)
            out["width_sweep"][w] = {"fve": fve_h, "mse": mse_h,
                                     "n_valid": int(valid_w.sum().item())}
            continue
        if T == 1 and w == 1:
            err = ((x_hat_zero.unsqueeze(0).unsqueeze(0) - x_full) ** 2).sum(dim=-1)
            fve_h, mse_h, _ = _fve(err, var_sum, valid_text)
            out["width_sweep"][1] = {"fve": fve_h, "mse": mse_h,
                                     "n_valid": int(valid_text.sum().item())}
            continue
        if w >= T:
            out["width_sweep"][w] = {"fve": float("nan"),
                                     "skipped": f"width {w} >= T {T}"}
            continue
        # Window arch: hold out the rightmost w positions.
        _, x_hat_h, mask_h = _window_recon_at_position(
            sae, src_class, x_full, T,
            hold_pos=T - w, hold_width=w, device=device,
        )
        err = ((x_hat_h - x_full) ** 2).sum(dim=-1)
        valid_w = valid_text & mask_h
        fve_h, mse_h, _ = _fve(err, var_sum, valid_w)
        out["width_sweep"][w] = {"fve": fve_h, "mse": mse_h,
                                 "n_valid": int(valid_w.sum().item())}
        print(f"    held-out width={w} (right-edge): FVE = {fve_h:.3f}  "
              f"(n={int(valid_w.sum().item())})")

    # ─── Within-window position ablation at width=1.
    out["position_ablation"] = {}
    if T == 1:
        out["position_ablation"]["right"] = out["width_sweep"][1]
        out["position_ablation"]["center"] = out["width_sweep"][1]
        out["position_ablation"]["left"] = out["width_sweep"][1]
    else:
        offsets = {"right": T - 1, "center": T // 2, "left": 0}
        for name, offset in offsets.items():
            _, x_hat_h, mask_h = _window_recon_at_position(
                sae, src_class, x_full, T,
                hold_pos=offset, hold_width=1, device=device,
            )
            err = ((x_hat_h - x_full) ** 2).sum(dim=-1)
            valid_p = valid_text & mask_h
            fve_h, mse_h, _ = _fve(err, var_sum, valid_p)
            out["position_ablation"][name] = {
                "fve": fve_h, "mse": mse_h,
                "n_valid": int(valid_p.sum().item()),
                "window_offset": offset,
            }
            print(f"    held-out pos={name} (offset {offset}): FVE = {fve_h:.3f}")

    del sae
    torch.cuda.empty_cache()
    return out


# ─────────────────────────────────────────── naive baselines ──

def _naive_baselines(x_full: torch.Tensor, attn: torch.Tensor,
                     device: torch.device) -> dict:
    """Compute FVE for non-SAE baselines:
       - predict-zero               (x_hat[t] = 0)
       - predict-mean-residual      (x_hat[t] = mean(x[v]) over valid v)
       - predict-x[t-1]             (x_hat[t] = x[t-1] if t-1 valid)
       - predict-mean-context-T5    (x_hat[t] = mean(x[t-T+1..t-1]) over T-1=4 left context)
       - predict-mean-context-T10   (same with T=10 -> 9 left context)
    Each baseline is FVE-evaluated over its own valid mask.
    """
    N, S, d_in = x_full.shape
    var_sum = (x_full ** 2).sum(dim=-1)
    valid = attn > 0
    out: dict = {}

    # zero
    err = (x_full ** 2).sum(dim=-1)
    fve, mse, _ = _fve(err, var_sum, valid)
    out["predict_zero"] = {"fve": fve, "mse": mse, "n_valid": int(valid.sum().item())}

    # mean residual
    flat_valid = x_full[valid]
    mean_resid = flat_valid.mean(dim=0)                    # (d_in,)
    err = ((mean_resid.unsqueeze(0).unsqueeze(0) - x_full) ** 2).sum(dim=-1)
    fve, mse, _ = _fve(err, var_sum, valid)
    out["predict_mean_residual"] = {"fve": fve, "mse": mse, "n_valid": int(valid.sum().item())}

    # predict x[t-1]
    pos_idx = torch.arange(S, device=device)
    valid_t1 = valid & (pos_idx.unsqueeze(0) >= 1)
    valid_t1 = valid_t1 & torch.cat(
        [torch.zeros((N, 1), dtype=torch.bool, device=device), valid[:, :-1]], dim=1,
    )
    x_hat = torch.zeros_like(x_full)
    x_hat[:, 1:, :] = x_full[:, :-1, :]
    err = ((x_hat - x_full) ** 2).sum(dim=-1)
    fve, mse, _ = _fve(err, var_sum, valid_t1)
    out["predict_x_t_minus_1"] = {"fve": fve, "mse": mse, "n_valid": int(valid_t1.sum().item())}

    # predict mean of left context, for T = 5 and T = 10 (the two
    # window sizes in the arch shortlist).
    for T_ctx in (5, 10):
        if S < T_ctx:
            continue
        x_hat = torch.zeros_like(x_full)
        # x_hat[t] = mean(x[t-T_ctx+1 .. t-1]) when those positions are valid.
        valid_left_count = torch.zeros((N, S), device=device)
        for offset in range(1, T_ctx):
            shifted = torch.zeros_like(x_full)
            shifted[:, offset:, :] = x_full[:, :-offset, :]
            valid_shift = torch.zeros((N, S), dtype=torch.bool, device=device)
            valid_shift[:, offset:] = valid[:, :-offset]
            x_hat = x_hat + shifted * valid_shift.unsqueeze(-1).float()
            valid_left_count = valid_left_count + valid_shift.float()
        valid_ctx = valid & (valid_left_count >= (T_ctx - 1))
        x_hat = x_hat / valid_left_count.clamp(min=1.0).unsqueeze(-1)
        err = ((x_hat - x_full) ** 2).sum(dim=-1)
        fve, mse, _ = _fve(err, var_sum, valid_ctx)
        out[f"predict_mean_context_T{T_ctx}"] = {
            "fve": fve, "mse": mse, "n_valid": int(valid_ctx.sum().item()),
        }
    return out


# ─────────────────────────────────────────── plot ──

def _plot(payload: dict, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k20": "#d62728",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
        "txc_bare_antidead_t5": "#9467bd",
        "txcdr_t10": "#8c564b",
    }
    label = {
        "topk_sae": "TopKSAE per-token (k=500)",
        "tsae_paper_k20": "T-SAE per-token (k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
        "txc_bare_antidead_t5": "TXC-bare anti-dead (T=5)",
        "txcdr_t10": "TXCDR (T=10)",
    }

    archs = [a for a, _ in ARCHS if a in payload["archs"]
             and "skipped" not in payload["archs"][a]]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_a, ax_b, ax_c = axes

    # Panel A: baseline + width-1 holdout per arch (right-edge).
    x = np.arange(len(archs)); width = 0.4
    base = [payload["archs"][a]["baseline"]["fve"] for a in archs]
    hold = [payload["archs"][a]["width_sweep"][1]["fve"] for a in archs]
    colors = [palette.get(a, "#888") for a in archs]
    ax_a.bar(x - width / 2, base, width, label="baseline", color=colors,
             alpha=0.85, edgecolor="black")
    ax_a.bar(x + width / 2, hold, width, label="held-out width=1",
             color=colors, alpha=0.40, edgecolor="black", hatch="//")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([label.get(a, a) for a in archs],
                         rotation=15, ha="right", fontsize=8)
    ax_a.set_ylabel("L12 fraction of variance explained")
    ax_a.axhline(0, color="black", lw=0.5)
    ax_a.set_title("(A) Baseline + held-out FVE on long FineWeb-edu text")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_a.grid(True, axis="y", ls=":", alpha=0.4)

    # Panel B: width sweep (corruption hill-climb).
    for a in archs:
        c = palette.get(a, "#888")
        ws = sorted([w for w in payload["archs"][a]["width_sweep"]
                     if not isinstance(payload["archs"][a]["width_sweep"][w], dict)
                     or "skipped" not in payload["archs"][a]["width_sweep"][w]])
        ws = [int(w) for w in ws]
        fves = [payload["archs"][a]["width_sweep"][w]["fve"] for w in ws]
        ax_b.plot(ws, fves, "-o", color=c, label=label.get(a, a), lw=2)
    # Add the "predict-zero" reference line.
    ax_b.axhline(0, color="black", ls="--", lw=0.7,
                 label="predict-zero baseline")
    # Add naive baselines as horizontal lines (FVE doesn't depend on
    # corruption width — they're text-only references).
    nb = payload.get("naive_baselines", {})
    refs = [
        ("predict_x_t_minus_1", "predict x[t-1]", "#c0392b"),
        ("predict_mean_context_T5", "predict mean(x[t-T+1..t-1]) T=5", "#16a085"),
    ]
    for key, lbl, c in refs:
        if key in nb:
            ax_b.axhline(nb[key]["fve"], color=c, ls=":", lw=1.0,
                         label=f"{lbl} ({nb[key]['fve']:.2f})")
    ax_b.set_xlabel("corruption width w (consecutive zeroed positions)")
    ax_b.set_ylabel("held-out FVE")
    ax_b.set_title("(B) Corruption-width sweep — TXC's structural prior in regime w<T")
    ax_b.legend(loc="lower left", fontsize=7)
    ax_b.grid(True, ls=":", alpha=0.4)

    # Panel C: within-window position ablation.
    win_archs = [a for a in archs if payload["archs"][a]["T"] > 1]
    pos_order = ["left", "center", "right"]
    x = np.arange(len(pos_order))
    bw = 0.8 / max(len(win_archs), 1)
    for k, a in enumerate(win_archs):
        c = palette.get(a, "#888")
        fves = [payload["archs"][a]["position_ablation"][p]["fve"]
                for p in pos_order]
        ax_c.bar(x + k * bw - 0.4 + bw / 2, fves, bw,
                 label=label.get(a, a), color=c, alpha=0.85, edgecolor="black")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([f"held out\n{p} of window" for p in pos_order],
                         fontsize=9)
    ax_c.set_ylabel("held-out FVE (width=1)")
    ax_c.axhline(0, color="black", lw=0.5)
    ax_c.set_title("(C) Within-window position ablation — TXC win is direction-robust?")
    ax_c.legend(loc="upper right", fontsize=8)
    ax_c.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        f"CS2 controls — held-out reconstruction on long natural text "
        f"({payload['n_passages']} FineWeb-edu × {payload['max_len']} tokens)",
        fontsize=12,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    save_figure(fig, str(out_path))
    plt.close(fig)


# ──────────────────────────────────────────────────── main ──

def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    acts_l12, attn = _ensure_cache(device)
    print(f"\nCS2 controls — long cache shape {acts_l12.shape}, "
          f"valid tokens / passage mean {attn.sum(axis=1).mean():.1f}")

    x_full = torch.from_numpy(acts_l12).float().to(device)
    attn_t = torch.from_numpy(attn).bool().to(device)

    archs_data: dict = {}
    t0 = time.time()
    for arch_id, _label in ARCHS:
        print(f"\n=== {arch_id} ===")
        archs_data[arch_id] = _run_arch(arch_id, x_full, attn_t, device)

    print("\n=== naive baselines ===")
    naive = _naive_baselines(x_full, attn_t, device)
    for k, v in naive.items():
        print(f"  {k:<32}  FVE = {v['fve']:.3f}  (n={v['n_valid']})")

    payload = {
        "n_passages": int(acts_l12.shape[0]),
        "max_len": int(acts_l12.shape[1]),
        "archs": archs_data,
        "naive_baselines": naive,
        "widths": WIDTHS,
        "wall_time_seconds": time.time() - t0,
    }
    json_path = OUT_SUBDIR / "controls_masked_recon.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {json_path}")
    png_path = OUT_SUBDIR / "controls_masked_recon.png"
    _plot(payload, png_path)
    print(f"wrote {png_path}  (+ thumb)")


if __name__ == "__main__":
    main()
