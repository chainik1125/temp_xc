"""Sweep ridge-regularization strength for the W=60 window probe.

The W=60 dense-linear probe underperforms W=20 because the feature dim
(60·64=3840) is comparable to the effective training-sample count (~27k
80/20-split positions). The default ridge (1e-8) leaves the probe under-
regularized. This script refits the W=60 and W=30 probes across a range
of ridge λ values to see where they peak.

Writes `ridge_sweep.json` in cell dirs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT / "vendor"
for p in (
    REPO_ROOT / "src",
    REPO_ROOT / "experiments" / "standard_hmm",
    REPO_ROOT / "experiments" / "transformer_standard_hmm",
):
    sys.path.insert(0, str(p))

from sae_day.run_driver import build_generator, build_transformer  # noqa: E402
from run_transformer_standard_hmm_arch_sweep import extract_position_activations  # noqa: E402


def build_windowed(eval_acts: torch.Tensor, window: int):
    N, T, d = eval_acts.shape
    unfolded = eval_acts.unfold(1, window, 1).permute(0, 1, 3, 2).contiguous()
    return unfolded.reshape(-1, window * d).numpy(), T - window + 1


def fit_ridge_multi_lambda(
    x_all: np.ndarray, y_all: np.ndarray,
    n_sequences: int, samples_per_sequence: int,
    ridges: list[float], seed: int,
) -> dict[float, dict]:
    """Closed-form ridge sweep that recomputes XtX only once.

    For each λ, solve (X'X + λI) β = X'y on the training slice, evaluate
    per-component R² on the test slice. Returns {λ: {'mean_r2', 'per_component_r2'}}.
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    seq_is_train = np.zeros(n_sequences, dtype=bool)
    seq_is_train[perm[:split]] = True
    flat_train = np.repeat(seq_is_train, samples_per_sequence)
    flat_test = ~flat_train

    x_tr = x_all[flat_train].astype(np.float64, copy=False)
    y_tr = y_all[flat_train].astype(np.float64, copy=False)
    x_te = x_all[flat_test].astype(np.float64, copy=False)
    y_te = y_all[flat_test].astype(np.float64, copy=False)

    x_mean = x_tr.mean(axis=0, keepdims=True)
    y_mean = y_tr.mean(axis=0, keepdims=True)
    xc = x_tr - x_mean
    yc = y_tr - y_mean

    # Precompute once
    xtx = xc.T @ xc          # (d, d)
    xty = xc.T @ yc          # (d, n_comp)
    d = xtx.shape[0]

    # Test-set SST against y_train mean (matches fit_linear_probe_r2 convention)
    ss_tot = ((y_te - y_mean) ** 2).sum(axis=0) + 1e-12

    out = {}
    for lam in ridges:
        xtx_reg = xtx.copy()
        xtx_reg.flat[::d + 1] += float(lam)
        beta = np.linalg.solve(xtx_reg, xty)
        intercept = y_mean - x_mean @ beta
        y_pred = x_te @ beta + intercept
        ss_res = ((y_te - y_pred) ** 2).sum(axis=0)
        per_c = 1.0 - ss_res / ss_tot
        out[float(lam)] = {
            "mean_r2": float(per_c.mean()),
            "per_component_r2": [float(v) for v in per_c],
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cells-root", type=Path, required=True)
    p.add_argument("--windows", type=int, nargs="+", default=[20, 30, 60])
    p.add_argument("--ridges", type=float, nargs="+",
                   default=[1e-8, 1e-4, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0])
    p.add_argument("--eval-sequences", type=int, default=500)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  windows={args.windows}  ridges={args.ridges}")

    for cell_dir in sorted(args.cells_root.glob("cell_delta_*")):
        if not (cell_dir / "transformer.pt").exists():
            continue
        tp = json.loads((cell_dir / "training_params.json").read_text())
        seed = int(tp["seed"])
        sweep_value = float(tp["sweep_value"])
        tf_cfg = tp["transformer"]
        gen_cfg = {
            "type": tp["generator"]["type"],
            "sweep": {"parameter": tp["generator"]["sweep_parameter"]},
            "fixed_params": tp["generator"]["fixed_params"],
        }
        gen = build_generator(gen_cfg, sweep_value)

        transformer = build_transformer(tf_cfg, gen.vocab_size, seed, device)
        ckpt = torch.load(cell_dir / "transformer.pt", map_location=device, weights_only=False)
        transformer.load_state_dict(ckpt["state_dict"])
        transformer.eval()

        seq_len = int(tf_cfg["n_ctx"])
        eval_data = gen.sample(
            n_sequences=args.eval_sequences, seq_len=seq_len,
            seed=seed + 20_000, with_component_labels=True,
        )
        hook_name = tf_cfg.get("probe_hook_name", "blocks.1.hook_resid_post")
        eval_acts = extract_position_activations(
            transformer, eval_data.tokens, hook_name=hook_name, chunk_size=16,
        )
        eval_omega = eval_data.sequence_omegas
        N = eval_acts.shape[0]

        results = {}
        for W in args.windows:
            flat, spw = build_windowed(eval_acts, W)
            target = (
                eval_omega[:, None, :].expand(N, spw, eval_omega.shape[-1])
                .reshape(-1, eval_omega.shape[-1]).numpy()
            )
            per_ridge_dict = fit_ridge_multi_lambda(
                flat, target, n_sequences=N, samples_per_sequence=spw,
                ridges=args.ridges, seed=seed,
            )
            per_ridge = {f"{lam:g}": per_ridge_dict[lam] for lam in args.ridges}
            results[str(W)] = per_ridge
            best_lam = max(per_ridge.keys(), key=lambda k: per_ridge[k]["mean_r2"])
            best_val = per_ridge[best_lam]["mean_r2"]
            default_val = per_ridge[f"{args.ridges[0]:g}"]["mean_r2"]
            print(
                f"  δ={sweep_value:<5}  W={W:<3}  best λ={best_lam} → R²={best_val:.3f}  "
                f"(default λ={args.ridges[0]:g} → R²={default_val:.3f})"
            )

        out = cell_dir / "ridge_sweep.json"
        out.write_text(json.dumps({
            "sweep_value": sweep_value,
            "windows": args.windows,
            "ridges": args.ridges,
            "results": results,
        }, indent=2))
        print(f"  saved {out}")

        del transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
