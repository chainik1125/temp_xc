"""CS2 hill-climb — downstream LM-CE under held-out-then-impute.

Smoke + controls showed TXC matryoshka beats per-token SAEs on
held-out MSE (FVE 0.28 vs ~0.00) but **loses** to a trivial
predict-mean-context baseline (FVE 0.35). MSE-FVE is a smoothing-
friendly metric: an averaged residual minimises squared error
even though it lies off the residual manifold.

Hill-climb: switch to a *behavioural* metric. Patch L12 at one
held-out position with x_hat[t] from each method, run the rest of
the LM forward (L13..L25), and measure cross-entropy at position
t+1 conditioned on the patched residual.

Hypothesis:
  - SAE reconstructions (TXC matryoshka in particular) are on-manifold
    (`b_dec + W_dec @ z` is what real residuals look like under SAE
    training);
  - mean-context smoothing is OFF-manifold (a low-magnitude average
    of nearby residuals); the LM is sensitive to magnitude / direction
    in ways squared-error doesn't measure;
  - per-token SAE reconstruction at zero input is ~b_dec (a constant)
    — even more off-distribution than mean-context;
  - TXC's reconstruction will preserve LM next-token CE better than
    naive baselines despite higher MSE.

Output:
  results/case_studies/cs2_masked_recon/behavioral_masked_recon.{json,png}
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


N_PASSAGES = 30
MAX_LEN = 256
N_HOLDOUT_PER_PASSAGE = 16            # random positions per passage to test
RNG_SEED = 42
ARCH_RECONS = [
    "topk_sae",
    "tsae_paper_k20",
    "agentic_txc_02",
    "phase5b_subseq_h8",
]
NAIVE_METHODS = ("zero", "mean_residual", "x_t_minus_1", "mean_context_T5")
LONG_CACHE = Path(
    "experiments/phase7_unification/results/case_studies/"
    "cs2_masked_recon/_l12_long_cache.npz"
)
OUT_SUBDIR = (
    Path("experiments/phase7_unification/results/case_studies/cs2_masked_recon")
)
RECON_CACHE = OUT_SUBDIR / "_x_hat_holdout_cache.npz"     # gitignored


# ─────────────────── load cache + sample held-out positions ──

def _load_cache() -> tuple[np.ndarray, np.ndarray]:
    if not LONG_CACHE.exists():
        raise FileNotFoundError(
            f"missing {LONG_CACHE}; run controls_masked_recon first to build it"
        )
    with np.load(LONG_CACHE, allow_pickle=False) as z:
        return z["l12"].copy(), z["attn"].copy()


def _sample_holdout_positions(attn: np.ndarray, T_min: int = 5, K: int = N_HOLDOUT_PER_PASSAGE,
                              S_max_for_t1: int = MAX_LEN - 1) -> np.ndarray:
    """Sample K held-out positions per passage. Each position must be a
    valid (non-pad) token, must have at least T_min-1 valid left-context
    positions, and must have a valid t+1 position (so we can read CE at
    t+1)."""
    rng = np.random.default_rng(RNG_SEED)
    N, S = attn.shape
    positions = -np.ones((N, K), dtype=np.int64)
    for n in range(N):
        valid_t = np.where(attn[n] > 0)[0]
        # require t in [T_min-1, ..., t_max-1] AND t+1 valid.
        ok = []
        for t in valid_t:
            if t < T_min - 1:
                continue
            if t + 1 >= S:
                continue
            if attn[n, t + 1] <= 0:
                continue
            ok.append(t)
        ok = np.array(ok, dtype=np.int64)
        if ok.size == 0:
            continue
        if ok.size < K:
            chosen = ok
        else:
            chosen = rng.choice(ok, size=K, replace=False)
        positions[n, :chosen.size] = chosen
    return positions


# ──────────────────────── precompute reconstructions ──

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


def _arch_holdout_recon(arch_id: str, x_full: torch.Tensor,
                        positions: np.ndarray, device: torch.device) -> np.ndarray:
    """Compute x_hat[n, t, :] at each held-out position, from each
    arch's SAE with that position zeroed at the encoder. Returns
    (N, K, d_in) numpy array."""
    log_path = OUT_DIR / "training_logs" / f"{arch_id}__seed42.json"
    ckpt_path = OUT_DIR / "ckpts" / f"{arch_id}__seed42.pt"
    meta = json.loads(log_path.read_text())
    src_class = meta["src_class"]
    if src_class in MLC_CLASSES:
        return None
    sae, _ = _load_phase7_model(meta, ckpt_path, device)
    sae.eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    T = window_T(sae, src_class, meta)

    N, S, d_in = x_full.shape
    K = positions.shape[1]
    out = np.zeros((N, K, d_in), dtype=np.float32)

    if src_class in {"TopKSAE", "TemporalMatryoshkaBatchTopKSAE", "TemporalSAE"}:
        # Per-token: x_hat = decode(encode(zero_vec)). Constant per arch.
        zero = torch.zeros((1, d_in), device=device)
        z = _encode(sae, src_class, zero)
        with torch.no_grad():
            x_hat_const = sae.decode(z).squeeze(0).cpu().numpy()
        for n in range(N):
            for k in range(K):
                if positions[n, k] >= 0:
                    out[n, k] = x_hat_const
    else:
        # Window: build a window for each (n, t) where the rightmost
        # position is t and zero out the rightmost slot. Encode + decode
        # full window; read out the rightmost slice as x_hat[t].
        bs_n = 1     # passage-by-passage
        for n in range(N):
            x_n = x_full[n]                         # (S, d_in)
            for k in range(K):
                t = int(positions[n, k])
                if t < 0 or t < T - 1:
                    continue
                window = x_n[t - T + 1:t + 1].clone()    # (T, d_in)
                window[-1] = 0.0
                window = window.unsqueeze(0)             # (1, T, d_in)
                z = _encode(sae, src_class, window)
                with torch.no_grad():
                    full = _decode_full_window(sae, src_class, z, T)  # (1, T, d_in)
                out[n, k] = full[0, -1, :].cpu().numpy()
    del sae
    torch.cuda.empty_cache()
    return out


def _naive_holdout_recon(method: str, x_full: torch.Tensor, attn: np.ndarray,
                         positions: np.ndarray, device: torch.device) -> np.ndarray:
    N, S, d_in = x_full.shape
    K = positions.shape[1]
    out = np.zeros((N, K, d_in), dtype=np.float32)
    if method == "zero":
        return out
    if method == "mean_residual":
        attn_t = torch.from_numpy(attn.astype(bool)).to(device)
        flat_valid = x_full[attn_t]
        mean_resid = flat_valid.mean(dim=0).cpu().numpy()
        for n in range(N):
            for k in range(K):
                if positions[n, k] >= 0:
                    out[n, k] = mean_resid
        return out
    if method == "x_t_minus_1":
        for n in range(N):
            for k in range(K):
                t = int(positions[n, k])
                if t >= 1 and attn[n, t - 1] > 0:
                    out[n, k] = x_full[n, t - 1].cpu().numpy()
        return out
    if method == "mean_context_T5":
        T_ctx = 5
        for n in range(N):
            for k in range(K):
                t = int(positions[n, k])
                if t < T_ctx - 1:
                    continue
                ctx = x_full[n, t - T_ctx + 1:t]    # (T-1, d_in)
                ctx_attn = attn[n, t - T_ctx + 1:t]
                if ctx_attn.sum() > 0:
                    valid = torch.from_numpy(ctx_attn.astype(bool)).to(device)
                    mean = ctx[valid].mean(dim=0)
                    out[n, k] = mean.cpu().numpy()
        return out
    raise ValueError(f"unknown method {method}")


# ──────────────────────── LM forward with patching ──

def _lm_forward_with_patch(input_ids: torch.Tensor, attention_mask: torch.Tensor,
                            l12_orig: torch.Tensor,
                            patch_value: torch.Tensor, patch_pos: int,
                            model, layer_idx: int = ANCHOR_LAYER) -> torch.Tensor:
    """Run a single LM forward with the L12 residual replaced at
    `patch_pos` by `patch_value` (1, d_in tensor). Returns logits
    (1, S, V).

    The hook overwrites the OUTPUT of layer `layer_idx`. We pass in
    `l12_orig` to enforce that other positions are exactly the original
    residual (in case rare numerics differ across runs)."""
    captured = {}

    def hook(_module, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        # h: (1, S, d_model). We assume d_in matches d_model on a slice
        # because the cache was sliced to DEFAULT_D_IN. Defensive.
        d_in = patch_value.shape[-1]
        h_patched = h.clone()
        # Replace ALL positions with the precomputed l12_orig (for
        # determinism vs the cached forward), then overwrite the patch
        # position.
        h_patched[..., :, :d_in] = l12_orig[None, :, :d_in].to(h.dtype).to(h.device)
        h_patched[0, patch_pos, :d_in] = patch_value.to(h.dtype).to(h.device)
        if isinstance(output, tuple):
            return (h_patched,) + output[1:]
        return h_patched

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    finally:
        handle.remove()
    return logits


def _ce_at_position(logits: torch.Tensor, input_ids: torch.Tensor, t: int) -> float:
    """Return cross-entropy of token at position t+1 given logits at
    position t. logits: (1, S, V), input_ids: (1, S)."""
    log_probs = torch.log_softmax(logits[0, t, :].float(), dim=-1)
    target = int(input_ids[0, t + 1].item())
    return -float(log_probs[target].item())


# ──────────────────────────────────────────────────────── main ──

def main() -> None:
    OUT_SUBDIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda")
    print(f"loading L12 long cache from {LONG_CACHE}")
    acts_l12, attn = _load_cache()
    print(f"  shape {acts_l12.shape}; valid tokens / passage mean "
          f"{attn.sum(axis=1).mean():.1f}")

    positions = _sample_holdout_positions(attn, T_min=5, K=N_HOLDOUT_PER_PASSAGE)
    n_total_pos = int((positions >= 0).sum())
    print(f"  sampled {n_total_pos} held-out positions across {acts_l12.shape[0]} passages")

    x_full = torch.from_numpy(acts_l12).float().to(device)

    # Step 1: precompute x_hat for every method.
    print("\n=== precomputing reconstructions ===")
    recons: dict[str, np.ndarray] = {}
    for arch_id in ARCH_RECONS:
        print(f"  arch {arch_id}...")
        r = _arch_holdout_recon(arch_id, x_full, positions, device)
        if r is None:
            print(f"    skipped (MLC)")
            continue
        recons[arch_id] = r
    for method in NAIVE_METHODS:
        print(f"  naive {method}...")
        recons[method] = _naive_holdout_recon(method, x_full, attn, positions, device)

    # Free x_full to make room for the LM.
    del x_full
    torch.cuda.empty_cache()

    # Step 2: load Gemma-2-2b and re-tokenize the cached passages.
    # We need the input_ids; the cache only saved attn + L12. Re-tokenize
    # by pulling FineWeb-edu in the same order as the cache was built.
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    print("\nloading Gemma-2-2b for behavioural sweep...")
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

    print("  re-pulling FineWeb-edu passages (same RNG order as cache build)...")
    ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT",
                      streaming=True, split="train")
    passages = []
    for row in ds:
        passages.append(row["text"])
        if len(passages) >= acts_l12.shape[0]:
            break

    enc = tokenizer(passages, return_tensors="pt", padding="max_length",
                    truncation=True, max_length=MAX_LEN)
    input_ids_full = enc["input_ids"].to(device)            # (N, S)
    attn_mask_full = enc["attention_mask"].to(device)       # (N, S)

    # Verify: this attn vs the cached attn must match.
    cached_attn_match = (attn_mask_full.cpu().numpy() == attn.astype(np.int64)).all()
    print(f"  attn-mask match vs cache: {bool(cached_attn_match)}")

    # Step 3: original (clean) CE per held-out (n, t), and patched CE
    # per (method, n, t).
    N, S, d_in = acts_l12.shape
    K = positions.shape[1]

    # First, compute and cache the "original" L12 residual per passage
    # by running a forward with no patch — but we already HAVE that
    # in acts_l12. We need to verify: the LM forward's L12 output should
    # equal acts_l12 (modulo the sliced d_in). We assume so.

    l12_orig_by_n = [torch.from_numpy(acts_l12[n]).float() for n in range(N)]

    methods_all = list(recons.keys())
    ce_results: dict[str, list[float]] = {m: [] for m in methods_all}
    ce_clean: list[float] = []

    print(f"\n=== running behavioural sweep over {n_total_pos} held-out positions × "
          f"{len(methods_all)} methods ===")
    t0 = time.time()
    sweep_count = 0
    for n in range(N):
        l12_orig = l12_orig_by_n[n]
        ids_n = input_ids_full[n:n + 1]
        attn_n = attn_mask_full[n:n + 1]
        for k in range(K):
            t = int(positions[n, k])
            if t < 0:
                continue

            # Clean reference: patch position t with the ORIGINAL residual
            # (no-op — but we run it to get a deterministic CE under the
            # same hook structure).
            x_orig_t = torch.from_numpy(acts_l12[n, t]).float().to(device)
            logits_clean = _lm_forward_with_patch(
                ids_n, attn_n, l12_orig, x_orig_t, t, model,
            )
            ce_c = _ce_at_position(logits_clean, ids_n, t)
            ce_clean.append(ce_c)

            for method in methods_all:
                x_hat_t = torch.from_numpy(recons[method][n, k]).float().to(device)
                logits = _lm_forward_with_patch(
                    ids_n, attn_n, l12_orig, x_hat_t, t, model,
                )
                ce = _ce_at_position(logits, ids_n, t)
                ce_results[method].append(ce)
            sweep_count += 1
            if sweep_count % 32 == 0:
                print(f"  done {sweep_count}/{n_total_pos}  "
                      f"({time.time() - t0:.0f}s elapsed)")

    print(f"  sweep wall time: {time.time() - t0:.0f}s")

    # Step 4: aggregate and write.
    ce_clean_arr = np.array(ce_clean, dtype=np.float64)
    summary = {"n_holdout_positions": int(ce_clean_arr.size),
               "ce_clean": {"mean": float(ce_clean_arr.mean()),
                            "median": float(np.median(ce_clean_arr)),
                            "std": float(ce_clean_arr.std())}}
    for method, vals in ce_results.items():
        arr = np.array(vals, dtype=np.float64)
        delta = arr - ce_clean_arr
        summary[method] = {
            "ce_mean": float(arr.mean()),
            "ce_median": float(np.median(arr)),
            "ce_std": float(arr.std()),
            "delta_ce_mean": float(delta.mean()),
            "delta_ce_median": float(np.median(delta)),
            "delta_ce_std": float(delta.std()),
            "delta_ce_p25": float(np.percentile(delta, 25)),
            "delta_ce_p75": float(np.percentile(delta, 75)),
        }
        print(f"  {method:<24}  ce={arr.mean():.3f}  "
              f"Δce={delta.mean():+.3f}  (median Δ={np.median(delta):+.3f})")

    json_path = OUT_SUBDIR / "behavioral_masked_recon.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {json_path}")

    _plot(summary, ce_results, ce_clean_arr,
          OUT_SUBDIR / "behavioral_masked_recon.png")


def _plot(summary: dict, ce_results: dict, ce_clean: np.ndarray,
          out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.plotting.save_figure import save_figure

    palette = {
        "topk_sae": "#1f77b4",
        "tsae_paper_k20": "#d62728",
        "agentic_txc_02": "#2ca02c",
        "phase5b_subseq_h8": "#17becf",
        "zero": "#7f7f7f",
        "mean_residual": "#bcbd22",
        "x_t_minus_1": "#e377c2",
        "mean_context_T5": "#16a085",
    }
    label = {
        "topk_sae": "TopKSAE per-token (k=500)",
        "tsae_paper_k20": "T-SAE per-token (k=20)",
        "agentic_txc_02": "TXC matryoshka (T=5)",
        "phase5b_subseq_h8": "SubseqH8 (T_max=10)",
        "zero": "predict-zero",
        "mean_residual": "predict-mean-residual",
        "x_t_minus_1": "predict-x[t-1]",
        "mean_context_T5": "predict-mean-context (T=5)",
    }

    methods = [m for m in [
        "topk_sae", "tsae_paper_k20", "agentic_txc_02", "phase5b_subseq_h8",
        "zero", "mean_residual", "x_t_minus_1", "mean_context_T5",
    ] if m in ce_results]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    ax_a, ax_b = axes

    # Panel A: ΔCE box per method.
    deltas = [np.array(ce_results[m]) - ce_clean for m in methods]
    box = ax_a.boxplot(deltas, tick_labels=[label[m] for m in methods],
                       patch_artist=True, showfliers=False, widths=0.5)
    for patch, m in zip(box["boxes"], methods):
        patch.set_facecolor(palette[m])
        patch.set_alpha(0.6)
    ax_a.axhline(0, color="black", lw=0.5, ls="--")
    ax_a.set_ylabel("ΔCE (patched − clean) at position t+1")
    ax_a.set_title(
        f"(A) Downstream LM ΔCE under held-out-and-impute "
        f"(n={summary['n_holdout_positions']} positions)\n"
        "lower = imputation preserves LM behaviour better"
    )
    ax_a.tick_params(axis="x", labelrotation=20, labelsize=8)
    ax_a.grid(True, axis="y", ls=":", alpha=0.4)

    # Panel B: mean ΔCE bar with error bars.
    means = [summary[m]["delta_ce_mean"] for m in methods]
    stds = [summary[m]["delta_ce_std"] / np.sqrt(summary["n_holdout_positions"])
            for m in methods]
    x = np.arange(len(methods))
    colors = [palette[m] for m in methods]
    ax_b.bar(x, means, yerr=stds, capsize=4, color=colors, alpha=0.85,
             edgecolor="black")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([label[m] for m in methods], rotation=15,
                         ha="right", fontsize=8)
    ax_b.axhline(0, color="black", lw=0.5)
    ax_b.set_ylabel("mean ΔCE (patched − clean)")
    ax_b.set_title("(B) Mean ΔCE with SEM error bars — TXC vs naive baselines")
    ax_b.grid(True, axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        "CS2 hill-climb — held-out reconstruction quality measured by\n"
        "downstream LM next-token CE (lower ΔCE = on-manifold reconstruction)",
        fontsize=11,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, str(out_path))
    plt.close(fig)
    print(f"wrote {out_path}  (+ thumb)")


if __name__ == "__main__":
    main()
