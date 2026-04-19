"""Per-feature SVD spectrum of TXCDR decoders at different T.

Motivation: if TXCDR-T=20's per-feature W_dec[j] is substantially
"flatter" in its singular-value spectrum than TXCDR-T=5's, it suggests
the T=20 model is using the full per-position-per-feature rank
unnecessarily — i.e. under-regularization. A rank-K factorization of
the decoder would then be a principled fix.

For each trained TXCDR ckpt:
    - W_dec has shape (d_sae, T, d_in).
    - Per feature j, compute SVD of W_dec[j] (shape T × d_in).
    - Collect the top-min(T, 20) normalized singular values.
    - Plot mean + percentile band of the normalized spectrum
      (singular values divided by the largest for that feature, so
      each feature contributes a spectrum on [0, 1]).

Runs on CPU; no GPU needed (SVD of small T × d_in matrices is fast).

Output:
    results/plots/svd_spectrum_t5_vs_t20.png  (+ thumb)
    results/svd_spectrum.json                  (raw arrays)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from src.plotting.save_figure import save_figure


REPO = Path("/workspace/temp_xc")
RESULTS_DIR = REPO / "experiments/phase5_downstream_utility/results"
CKPT_DIR = RESULTS_DIR / "ckpts"
PLOTS_DIR = RESULTS_DIR / "plots"


def load_w_dec(ckpt_path: Path) -> tuple[np.ndarray, int]:
    """Return W_dec (d_sae, T, d_in) as float32 on CPU, and T."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["state_dict"]
    W = sd.get("W_dec", None)
    if W is None:
        raise KeyError(f"W_dec missing in {ckpt_path}")
    W = W.to(torch.float32).numpy()
    if W.ndim != 3:
        raise ValueError(f"Expected 3-D W_dec in {ckpt_path}, got {W.shape}")
    T = W.shape[1]
    return W, T


def per_feature_spectrum(
    W: np.ndarray, max_features: int | None = None,
) -> np.ndarray:
    """Return (n_features, T) normalized singular values per feature.

    Normalized: divide each feature's spectrum by its own top singular
    value so each row lies in [0, 1]. This isolates "shape" from "scale".
    """
    d_sae, T, d_in = W.shape
    if max_features is not None and max_features < d_sae:
        idx = np.random.RandomState(0).choice(d_sae, max_features, replace=False)
        W = W[idx]
        d_sae = W.shape[0]
    # Per-feature SVD: (T, d_in) -> T singular values each.
    # W[j] has shape (T, d_in); rank <= T since T < d_in typically.
    spectrum = np.zeros((d_sae, T), dtype=np.float32)
    for j in range(d_sae):
        s = np.linalg.svd(W[j], compute_uv=False)
        # s has length min(T, d_in) = T
        spectrum[j, : len(s)] = s
    # Normalize row-wise by the top singular value (first column).
    top = spectrum[:, 0].clip(min=1e-12)
    return spectrum / top[:, None]


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    archs = {
        "txcdr_t5": CKPT_DIR / "txcdr_t5__seed42.pt",
        "txcdr_t20": CKPT_DIR / "txcdr_t20__seed42.pt",
    }
    results: dict[str, dict] = {}
    for name, p in archs.items():
        if not p.exists():
            print(f"  {name}: ckpt missing, skip")
            continue
        print(f"Loading {name}...")
        t0 = time.time()
        W, T = load_w_dec(p)
        print(f"  W_dec.shape={W.shape} ({time.time() - t0:.1f}s)")
        # Subsample features for SVD speed — 2000 features gives stable stats.
        t0 = time.time()
        spec = per_feature_spectrum(W, max_features=2000)
        print(f"  spectrum computed in {time.time() - t0:.1f}s")
        results[name] = {
            "T": T,
            "mean_spectrum": spec.mean(axis=0).tolist(),
            "p10": np.percentile(spec, 10, axis=0).tolist(),
            "p50": np.percentile(spec, 50, axis=0).tolist(),
            "p90": np.percentile(spec, 90, axis=0).tolist(),
            "effective_rank_ratio": float(
                (spec.sum(axis=1) / (spec[:, 0] * T)).mean()
            ),
        }
        print(
            f"  {name}: effective_rank/T = "
            f"{results[name]['effective_rank_ratio']:.3f}"
        )

    # Plot: x-axis = singular-value index (0 = largest), y-axis = normalized σ.
    # One panel per T (shared overlay doesn't work because different T's have
    # different # of singular values).
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"txcdr_t5": "tab:red", "txcdr_t20": "tab:blue"}
    for name, r in results.items():
        T = r["T"]
        xs = np.arange(T) / max(T - 1, 1)  # normalized rank axis in [0, 1]
        ax.plot(
            xs, r["mean_spectrum"], "o-", color=colors.get(name),
            label=f"{name} (T={T}, eff-rank ratio={r['effective_rank_ratio']:.3f})",
            linewidth=2, markersize=5,
        )
        ax.fill_between(
            xs, r["p10"], r["p90"],
            alpha=0.15, color=colors.get(name),
        )
    ax.set_xlabel("normalized singular-value index (0 = largest, 1 = smallest)")
    ax.set_ylabel("σ / σ_max (per feature, averaged)")
    ax.set_title(
        "TXCDR decoder per-feature SVD: T=5 vs T=20\n"
        "Flatter curve → under-regularized (feature uses more of its rank budget)"
    )
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    fig.tight_layout()

    save_figure(fig, str(PLOTS_DIR / "svd_spectrum_t5_vs_t20.png"))
    plt.close(fig)

    (RESULTS_DIR / "svd_spectrum.json").write_text(json.dumps(results, indent=2))
    print(f"\nWrote plot + summary to {PLOTS_DIR} / {RESULTS_DIR}")


if __name__ == "__main__":
    main()
