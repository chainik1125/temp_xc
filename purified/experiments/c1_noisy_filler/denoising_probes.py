"""Linear probe + single-latent correlation denoising eval for c1_noisy.

Reproduces wasteland 1c-noisy analysis ((decision 2026-05-06) directive).

Wasteland sources (commit 3c44f530babf6378c909a30a1796c792478ae5ba):
- ``docs/han/research_logs/2026-03-30-experiment1c-noisy-emissions.md``
- ``src/v2_temporal_schemeC/run_exp1c_linear_probe.py``
- ``src/v2_temporal_schemeC/experiment/denoising.py:compute_global_recovery``

Two metrics, both on per-token latents:

1. **Single-latent correlation** (``compute_global_recovery``):
   For each true feature i, find best-matching latent j by
   ``argmax_j |cos(d_j, f_i)|``. Compute Pearson correlation:
     - local: corr(z_j, s_i)  — noisy obs
     - global: corr(z_j, h_i) — hidden state
   Aggregate mean local, mean global, ratio.

2. **Linear probe** (``run_linear_probes``):
   Per-feature Ridge(α=1.0) probes on 80/20 train/test split:
     - local: z → s_i  → mean R²_local
     - global: z → h_i → mean R²_global

Both produce a (mean_local, mean_global) point per (model, k_pos, seed)
which we average across seeds for the wasteland-style scatter plot
(local x-axis, global y-axis).

Latent extraction follows wasteland recipe:
- TFA / TFA-pos: causal attention is length-agnostic → process full
  seq_len=64 in one forward pass.
- Stacked / TXC-base / TXC-pro: slide T-window stride 1 over seq;
  for window-level archs (TXC), broadcast same code to all T positions
  in the window; average across overlapping windows for each position.

Output:
- ``denoising_probe_results.json``: per-cell metrics (both methods).
- ``plots/c1_noisy_probe_scatter.png``: linear probe scatter.
- ``plots/c1_noisy_singlelatent_scatter.png``: single-latent corr scatter.
- ``plots/c1_noisy_denoising_panels.png``: 3-panel local/global/ratio vs k.

Usage:
    .venv/bin/python -m experiments.c1_noisy_filler.denoising_probes \\
        [--archs ...] [--max-cells N]
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from safetensors.torch import load_file

from temp_bench.cache import _read_jsonl, leaderboard_path
from temp_bench.config import (
    checkpoint_dir, instantiate_arch, load_arch, load_datasource,
)
from temp_bench.data.toy.markov import markov_chain_support


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPONENT = "c1_noisy"
DATASOURCE = "toy_markov_n20_d40_noisy"

# Wasteland setup (run_exp1c_linear_probe.py:55-60).
EVAL_N_SEQ = 2000
EVAL_SEED = 42
NUM_FEATURES = 20
DICT_WIDTH = 40
SEQ_LEN = 64

_DATA_CACHE: dict = {}


# ── Data ────────────────────────────────────────────────────────────────


def get_eval_data():
    if "data" not in _DATA_CACHE:
        spec = load_datasource(DATASOURCE)
        _DATA_CACHE["data"] = markov_chain_support(
            n_features=int(spec.n_features),
            d_in=int(spec.d_in),
            seq_len=int(spec.seq_len),
            rho_levels=list(spec.rho_levels),
            pi=float(spec.pi),
            n_seqs=EVAL_N_SEQ,
            p_A=float(getattr(spec, "p_A", 0.0)),
            p_B=float(getattr(spec, "p_B", 1.0)),
            seed=EVAL_SEED,
        )
    return _DATA_CACHE["data"]


# ── Latent extraction (wasteland-faithful) ──────────────────────────────


@torch.no_grad()
def extract_latents(model, eval_x: torch.Tensor, arch_name: str, T_win: int) -> np.ndarray:
    """Per-token latents, shape (n_eval * SEQ_LEN, d_sae).

    TFA: full-seq forward.
    Stacked / TXC: slide T-window stride 1, average overlaps. Window-level
    codes (TXC) are broadcast to all T positions before averaging.
    """
    n_eval, seq_len, d_in = eval_x.shape
    d_sae = DICT_WIDTH

    if arch_name in ("tfa", "tfa_pos"):
        all_z = []
        for s in range(0, n_eval, 256):
            x = eval_x[s:s + 256]
            z = model.encode(x)  # (B, T=seq_len, d_sae)
            all_z.append(z.cpu())
        z_full = torch.cat(all_z, dim=0)
        return z_full.reshape(-1, d_sae).numpy()

    z_sum = torch.zeros(n_eval, seq_len, d_sae)
    counts = torch.zeros(n_eval, seq_len)

    for t_start in range(seq_len - T_win + 1):
        windows = eval_x[:, t_start:t_start + T_win, :]
        for s in range(0, n_eval, 256):
            w = windows[s:s + 256]
            bs = w.shape[0]
            z = model.encode(w)
            if z.shape[1] == 1:  # window-level (TXC)
                z = z.expand(-1, T_win, -1)
            z = z.cpu()
            for t_off in range(T_win):
                z_sum[s:s + bs, t_start + t_off] += z[:, t_off]
                counts[s:s + bs, t_start + t_off] += 1

    z_avg = z_sum / counts.unsqueeze(-1).clamp(min=1)
    return z_avg.reshape(-1, d_sae).numpy()


# ── Metrics ─────────────────────────────────────────────────────────────


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    num = (xm * ym).sum()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    if den < 1e-12:
        return 0.0
    return float(num / den)


def run_single_latent_correlation(
    z: np.ndarray, decoder_directions: np.ndarray,
    true_features: np.ndarray, support: torch.Tensor, hidden: torch.Tensor,
) -> dict:
    """Per-feature single-latent correlation (wasteland recipe).

    Args:
        z: (n_tokens, d_sae) per-token latents.
        decoder_directions: (d_sae, d_in) decoder columns.
        true_features: (n_features, d_in) ground truth.
        support, hidden: (n_eval, n_features, T) binary tensors.
    """
    # Best-matching latent per feature via abs cosine.
    dd = decoder_directions / (np.linalg.norm(decoder_directions, axis=1, keepdims=True) + 1e-8)
    tf = true_features / (np.linalg.norm(true_features, axis=1, keepdims=True) + 1e-8)
    sims = np.abs(dd @ tf.T)  # (d_sae, n_features)
    best_latent = sims.argmax(axis=0)  # (n_features,)

    sup = support.permute(0, 2, 1).reshape(-1, NUM_FEATURES).cpu().numpy()
    hid = hidden.permute(0, 2, 1).reshape(-1, NUM_FEATURES).cpu().numpy()

    local_corrs, global_corrs = [], []
    for i in range(NUM_FEATURES):
        j = int(best_latent[i])
        z_j = z[:, j]
        local_corrs.append(_pearson(z_j, sup[:, i]))
        global_corrs.append(_pearson(z_j, hid[:, i]))

    local_arr = np.array(local_corrs)
    global_arr = np.array(global_corrs)
    mean_local = float(local_arr.mean())
    mean_global = float(global_arr.mean())
    return {
        "sl_local_corrs": local_arr.tolist(),
        "sl_global_corrs": global_arr.tolist(),
        "sl_mean_local": mean_local,
        "sl_mean_global": mean_global,
        "sl_ratio": mean_global / max(mean_local, 1e-12),
        "sl_denoising_frac": float((global_arr > local_arr).mean()),
    }


def run_linear_probes(
    z: np.ndarray, support: torch.Tensor, hidden: torch.Tensor,
) -> dict:
    """Ridge probes per feature on 80/20 split (wasteland convention)."""
    sup = support.permute(0, 2, 1).reshape(-1, NUM_FEATURES).cpu().numpy()
    hid = hidden.permute(0, 2, 1).reshape(-1, NUM_FEATURES).cpu().numpy()

    n = z.shape[0]
    split = int(0.8 * n)
    z_train, z_test = z[:split], z[split:]
    sup_train, sup_test = sup[:split], sup[split:]
    hid_train, hid_test = hid[:split], hid[split:]

    local_r2s, global_r2s = [], []
    for i in range(NUM_FEATURES):
        plocal = Ridge(alpha=1.0).fit(z_train, sup_train[:, i])
        local_r2s.append(float(r2_score(sup_test[:, i], plocal.predict(z_test))))
        pglobal = Ridge(alpha=1.0).fit(z_train, hid_train[:, i])
        global_r2s.append(float(r2_score(hid_test[:, i], pglobal.predict(z_test))))

    mean_local = float(np.mean(local_r2s))
    mean_global = float(np.mean(global_r2s))
    return {
        "lp_local_r2s": local_r2s,
        "lp_global_r2s": global_r2s,
        "lp_mean_local_r2": mean_local,
        "lp_mean_global_r2": mean_global,
        "lp_ratio": mean_global / max(mean_local, 1e-12),
    }


# ── Pipeline ────────────────────────────────────────────────────────────


def collect_cells(arch_filter=None, t_filter=None) -> list[dict]:
    cells = []
    seen = set()
    for r in _read_jsonl(leaderboard_path()):
        if r.get("component") != COMPONENT:
            continue
        if r.get("eval_cfg", {}).get("smoke"):
            continue
        if arch_filter and r["arch"] not in arch_filter:
            continue
        cfg = r["eval_cfg"]
        t_label = cfg.get("t_label", "default")
        if t_filter and t_label not in t_filter:
            continue
        # Dedup by train_key (each row has a unique train_key per cell).
        if r["train_key"] in seen:
            continue
        seen.add(r["train_key"])
        cells.append({
            "train_key": r["train_key"],
            "arch_name": r["arch"],
            "seed": int(r["seed"]),
            "k_pos": int(cfg["k_pos"]),
            "t_label": t_label,
            "arch_hparams_override": cfg.get("_arch_hparams_override", {}) or {},
        })
    return cells


def _get_T_win(arch_name: str, override: dict, spec_hparams: dict) -> int:
    if arch_name == "txc_pro":
        return int(override.get("T_max", spec_hparams.get("T_max", 10)))
    return int(override.get("T", spec_hparams.get("T", 5)))


def evaluate_cell(cell: dict, eval_x: torch.Tensor) -> dict | None:
    tk = cell["train_key"]
    ckpt_path = checkpoint_dir(tk) / "model.safetensors"
    if not ckpt_path.exists():
        return None

    spec = load_arch(cell["arch_name"], component="c1")
    if cell["arch_hparams_override"]:
        merged = {**spec.hparams, **cell["arch_hparams_override"]}
        spec = spec.model_copy(update={"hparams": merged})
    d_in = eval_x.shape[-1]
    model = instantiate_arch(spec, d_in=d_in).to(DEVICE).eval()
    state = load_file(str(ckpt_path), device=str(DEVICE))
    model.load_state_dict(state, strict=True)

    T_win = _get_T_win(cell["arch_name"], cell["arch_hparams_override"], spec.hparams)
    data = get_eval_data()

    z = extract_latents(model, eval_x, cell["arch_name"], T_win)
    decoder = model.decoder_directions().detach().cpu().numpy()
    features = data.features.cpu().numpy()

    sl = run_single_latent_correlation(z, decoder, features, data.support, data.hidden_support)
    lp = run_linear_probes(z, data.support, data.hidden_support)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "train_key": tk,
        "arch_name": cell["arch_name"],
        "t_label": cell["t_label"],
        "k_pos": cell["k_pos"],
        "seed": cell["seed"],
        "T_win": T_win,
        **{k: v for k, v in sl.items() if not k.startswith("sl_") or k.startswith("sl_mean") or k == "sl_ratio" or k == "sl_denoising_frac"},
        **{k: v for k, v in lp.items() if not k.startswith("lp_local") and not k.startswith("lp_global")},
        # Keep summary metrics; drop per-feature arrays to keep JSON small.
        "sl_mean_local": sl["sl_mean_local"],
        "sl_mean_global": sl["sl_mean_global"],
        "sl_ratio": sl["sl_ratio"],
        "lp_mean_local_r2": lp["lp_mean_local_r2"],
        "lp_mean_global_r2": lp["lp_mean_global_r2"],
        "lp_ratio": lp["lp_ratio"],
    }


# ── Plots ───────────────────────────────────────────────────────────────


# Wasteland-style: one color per arch, T encoded by colormap for TXC family.
import matplotlib.cm as _cm

PLOT_BASE = {
    ("topk_sae",    "default"): {"label": "TopK-SAE",        "color": "#000000", "marker": "P", "ls": "-"},
    ("tsae_paper",  "default"): {"label": "T-SAE",           "color": "#CC79A7", "marker": "h", "ls": "-"},
    ("tfa_pos",     "default"): {"label": "TFA-pos",         "color": "#2ca02c", "marker": "X", "ls": "-"},
    ("stacked_sae", "T=2"):     {"label": "Stacked T=2",     "color": "#9467bd", "marker": "o", "ls": "-"},
    ("stacked_sae", "default"): {"label": "Stacked T=5",     "color": "#9467bd", "marker": "^", "ls": "--"},
    ("txc_pro",     "default"): {"label": "TXC-pro T_max=10","color": "#1f77b4", "marker": "*", "ls": "-"},
}


def _build_style_map(unique_keys):
    """Build (arch, t_label) → style. TXC-base T values use RdPu cmap."""
    style = dict(PLOT_BASE)
    txc_t_labels = sorted(
        {t for (a, t) in unique_keys if a == "txc_base"},
        key=lambda t: int(t.split("=")[1]) if "=" in t else (5 if t == "default" else 0),
    )
    cmap = _cm.get_cmap("RdPu", max(len(txc_t_labels), 4) + 2)
    markers = ["o", "s", "D", "^", "v", "<", "p", ">", "h", "H", "*", "P"]
    for i, t in enumerate(txc_t_labels):
        # Map "default" to T=5 label
        nice = "TXC-base T=5" if t == "default" else f"TXC-base {t}"
        style[("txc_base", t)] = {
            "label": nice,
            "color": cmap(i + 2),
            "marker": markers[i % len(markers)],
            "ls": "-",
        }
    return style, txc_t_labels


def _aggregate_by_seeds(results: list[dict]) -> dict:
    grouped = defaultdict(list)
    for r in results:
        key = (r["arch_name"], r["t_label"], r["k_pos"])
        grouped[key].append(r)
    agg = {}
    for key, rs in grouped.items():
        # Clamp ratios to [-2, 2] before averaging — near-zero local
        # correlations produce spurious -1e10 ratios that ruin the mean.
        sl_ratios_clamped = [
            max(-2.0, min(2.0, r["sl_ratio"])) for r in rs
        ]
        lp_ratios_clamped = [
            max(-2.0, min(2.0, r["lp_ratio"])) for r in rs
        ]
        agg[key] = {
            "sl_mean_local":      float(np.mean([r["sl_mean_local"] for r in rs])),
            "sl_mean_global":     float(np.mean([r["sl_mean_global"] for r in rs])),
            "sl_ratio":           float(np.mean(sl_ratios_clamped)),
            "lp_mean_local_r2":   float(np.mean([r["lp_mean_local_r2"] for r in rs])),
            "lp_mean_global_r2":  float(np.mean([r["lp_mean_global_r2"] for r in rs])),
            "lp_ratio":           float(np.mean(lp_ratios_clamped)),
            "n":                  len(rs),
        }
    return agg


def plot_scatter(agg: dict, out_path: Path, *, mode: str) -> None:
    """Scatter local-vs-global. mode='sl' (single-latent corr) or 'lp' (linear probe)."""
    assert mode in ("sl", "lp")
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, lw=1, label="y = x  (no denoising)")

    unique = {(a, t) for (a, t, _) in agg.keys()}
    style_map, _ = _build_style_map(unique)
    # Order: topk_sae, tsae_paper, tfa_pos, stacked, txc_base (by T asc), txc_pro
    def _sort_key(at):
        a, t = at
        order = {"topk_sae": -2, "tsae_paper": -1,
                 "tfa_pos": 0, "stacked_sae": 1, "txc_base": 2, "txc_pro": 3}.get(a, 9)
        if a == "txc_base":
            try:
                t_int = int(t.split("=")[1]) if "=" in t else 5
            except (ValueError, IndexError):
                t_int = 5
            return (order, t_int)
        if a == "stacked_sae":
            return (order, int(t.split("=")[1]) if "T=" in t else 5)
        return (order, 0)
    ordered = sorted(unique, key=_sort_key)

    for arch, t_label in ordered:
        ks = sorted({k for (a, t, k) in agg.keys() if a == arch and t == t_label})
        if not ks:
            continue
        if mode == "sl":
            local = [agg[(arch, t_label, k)]["sl_mean_local"]   for k in ks]
            glob  = [agg[(arch, t_label, k)]["sl_mean_global"]  for k in ks]
        else:
            local = [agg[(arch, t_label, k)]["lp_mean_local_r2"]  for k in ks]
            glob  = [agg[(arch, t_label, k)]["lp_mean_global_r2"] for k in ks]
        s = style_map.get((arch, t_label), {"label": f"{arch} {t_label}", "color": "gray", "marker": "o"})
        ax.scatter(local, glob, color=s["color"], marker=s["marker"],
                   s=110, alpha=0.78, label=s["label"], zorder=5,
                   edgecolors="black", linewidth=0.5)

    if mode == "sl":
        ax.set_xlabel(r"Local correlation $\bar{r}_{\rm local}$  ($z_j \to s_i$, noisy obs)",
                      fontsize=12)
        ax.set_ylabel(r"Global correlation $\bar{r}_{\rm global}$  ($z_j \to h_i$, hidden state)",
                      fontsize=12)
        ax.set_title(r"Setup B: single-latent correlation  ($\gamma{=}0.25$)",
                     fontsize=12)
        # Auto-zoom: single-latent correlations are small (~0-0.15 here).
        all_local = [agg[(a, t, k)]["sl_mean_local"] for (a, t, k) in agg.keys()]
        all_global = [agg[(a, t, k)]["sl_mean_global"] for (a, t, k) in agg.keys()]
        hi = max(max(all_local), max(all_global)) + 0.02
        ax.set_xlim(-0.02, hi + 0.02)
        ax.set_ylim(-0.02, hi + 0.02)
    else:
        ax.set_xlabel(r"Local $R^2$  ($z \to s_i$, noisy obs)", fontsize=12)
        ax.set_ylabel(r"Global $R^2$  ($z \to h_i$, hidden state)", fontsize=12)
        ax.set_title(r"Setup B: linear probe  ($\gamma{=}0.25$)", fontsize=12)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

    ax.legend(fontsize=9, loc="lower right", framealpha=0.92)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".thumb.png"), bbox_inches="tight", dpi=72)
    plt.close(fig)


def plot_panels(agg: dict, out_path: Path, *, mode: str) -> None:
    """3-panel local/global/ratio vs k_pos."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    unique = {(a, t) for (a, t, _) in agg.keys()}
    style_map, _ = _build_style_map(unique)

    for arch, t_label in sorted(unique):
        ks = sorted({k for (a, t, k) in agg.keys() if a == arch and t == t_label})
        if not ks:
            continue
        if mode == "sl":
            local = [agg[(arch, t_label, k)]["sl_mean_local"]  for k in ks]
            glob  = [agg[(arch, t_label, k)]["sl_mean_global"] for k in ks]
            ratio = [agg[(arch, t_label, k)]["sl_ratio"]       for k in ks]
        else:
            local = [agg[(arch, t_label, k)]["lp_mean_local_r2"]  for k in ks]
            glob  = [agg[(arch, t_label, k)]["lp_mean_global_r2"] for k in ks]
            ratio = [agg[(arch, t_label, k)]["lp_ratio"]          for k in ks]
        s = style_map.get((arch, t_label), {"label": f"{arch} {t_label}", "color": "gray", "marker": "o", "ls": "-"})
        for ax, ys in zip(axes, [local, glob, ratio]):
            ax.plot(ks, ys, marker=s["marker"], color=s["color"], ls=s.get("ls", "-"),
                    lw=2, ms=7, label=s["label"], alpha=0.9)

    ylab = "$R^2$" if mode == "lp" else r"correlation $r$"
    axes[0].set(xlabel=r"$k_{\rm pos}$", ylabel=ylab,
                title=r"Local: $z \to s_i$")
    axes[1].set(xlabel=r"$k_{\rm pos}$", ylabel=ylab,
                title=r"Global: $z \to h_i$")
    axes[2].set(xlabel=r"$k_{\rm pos}$", ylabel="Global / Local ratio",
                title="Denoising ratio")
    axes[2].axhline(1.0, color="green", ls="--", alpha=0.5, lw=1, label="Perfect denoising")
    floor = 0.50 if mode == "sl" else 0.25
    axes[2].axhline(floor, color="red",   ls=":", alpha=0.5, lw=1, label="Per-token floor")

    for ax in axes:
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    title = "single-latent correlation" if mode == "sl" else "linear probe"
    fig.suptitle(rf"Setup B: denoising via {title}  ($\gamma{{=}}0.25$)", y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    fig.savefig(out_path.with_suffix(".thumb.png"), bbox_inches="tight", dpi=72)
    plt.close(fig)


# ── Entrypoint ──────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", nargs="+", default=None)
    ap.add_argument("--t-labels", nargs="+", default=None,
                    help="Filter by t_label (e.g. T=2 default).")
    ap.add_argument("--max-cells", type=int, default=None)
    ap.add_argument("--out-dir", default="experiments/c1_noisy_filler")
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}", flush=True)
    print("Loading eval data (2000 seqs × 64 tokens, seed=42)...", flush=True)
    data = get_eval_data()
    eval_x = data.x.to(DEVICE)
    print(f"  eval_x: {eval_x.shape}", flush=True)
    sh_corr = float(torch.corrcoef(torch.stack([
        data.support.flatten().float(), data.hidden_support.flatten().float()
    ]))[0, 1].item())
    print(f"  Corr(s, h) = {sh_corr:.3f}  (expected ~0.50 → ratio floor = "
          f"{sh_corr:.2f} for single-latent / {sh_corr**2:.2f} for linear probe)",
          flush=True)

    cells = collect_cells(
        arch_filter=set(args.archs) if args.archs else None,
        t_filter=set(args.t_labels) if args.t_labels else None,
    )
    if args.max_cells:
        cells = cells[: args.max_cells]
    print(f"Found {len(cells)} unique c1_noisy cells.", flush=True)

    results = []
    n_skip = 0
    t_start = time.time()
    for i, cell in enumerate(cells, start=1):
        try:
            r = evaluate_cell(cell, eval_x)
        except Exception as e:
            print(f"  [{i}/{len(cells)}] ERROR {cell['train_key'][:12]} "
                  f"({cell['arch_name']} {cell['t_label']} k={cell['k_pos']}): "
                  f"{type(e).__name__}: {str(e)[:80]}", flush=True)
            continue
        if r is None:
            n_skip += 1
            continue
        results.append(r)
        if i % 5 == 0 or i == len(cells):
            elapsed = time.time() - t_start
            eta = elapsed / i * (len(cells) - i) if i else 0
            print(f"  [{i:3d}/{len(cells)}] {cell['arch_name']:12s} {cell['t_label']:6s} "
                  f"k={cell['k_pos']:2d} seed={cell['seed']:2d} → "
                  f"sl_local={r['sl_mean_local']:+.3f} sl_global={r['sl_mean_global']:+.3f} "
                  f"sl_ratio={r['sl_ratio']:.2f} | "
                  f"lp_local={r['lp_mean_local_r2']:+.3f} lp_global={r['lp_mean_global_r2']:+.3f}"
                  f"  (eta {eta/60:.1f} min)",
                  flush=True)

    print(f"\nProcessed: {len(results)} cells. Skipped (no checkpoint): {n_skip}.", flush=True)

    out_json = out_dir / "denoising_probe_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=float))
    print(f"Wrote {out_json}", flush=True)

    if args.no_plots:
        return

    agg = _aggregate_by_seeds(results)
    plot_scatter(agg, plots_dir / "c2_noisy_singlelatent_scatter.png", mode="sl")
    plot_scatter(agg, plots_dir / "c2_noisy_probe_scatter.png",         mode="lp")
    plot_panels (agg, plots_dir / "c2_noisy_singlelatent_panels.png",   mode="sl")
    plot_panels (agg, plots_dir / "c2_noisy_denoising_panels.png",      mode="lp")
    print(f"Plots saved under {plots_dir}", flush=True)

    print("\n=== Summary (mean over seeds) ===")
    print(f"{'arch':14s} {'T':8s} {'k':>3s}  "
          f"{'sl_loc':>7s} {'sl_glo':>7s} {'sl_rat':>7s}  "
          f"{'lp_loc':>7s} {'lp_glo':>7s} {'lp_rat':>7s}")
    keys = sorted(agg.keys())
    for arch, t_label, k in keys:
        v = agg[(arch, t_label, k)]
        print(f"{arch:14s} {t_label:8s} {k:>3d}  "
              f"{v['sl_mean_local']:+7.3f} {v['sl_mean_global']:+7.3f} {v['sl_ratio']:>7.2f}  "
              f"{v['lp_mean_local_r2']:+7.3f} {v['lp_mean_global_r2']:+7.3f} {v['lp_ratio']:>7.2f}")


if __name__ == "__main__":
    main()
