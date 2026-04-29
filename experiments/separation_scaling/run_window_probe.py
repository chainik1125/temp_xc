"""Post-hoc: fit a window-W linear probe on cached transformers per cell.

For each cell under --cells-root, load training_params.json + transformer.pt,
rebuild eval activations with the same seed, then fit:
  - single-position linear probe on residual (baseline, matches existing)
  - window-W linear probe: input = concat of W consecutive residuals → R^{W·d_model}

Both probes use the same closed-form ridge regression (fit_linear_probe_r2),
same 80/20 sequence-level train/test split, so numbers are directly comparable.

Writes `window_probes.json` inside each cell dir.
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
from run_standard_hmm_arch_seed_sweep import fit_linear_probe_r2  # noqa: E402
from run_transformer_standard_hmm_arch_sweep import extract_position_activations  # noqa: E402


def fit_logistic_probe(
    x_all: np.ndarray, y_all: np.ndarray, n_sequences: int,
    samples_per_sequence: int, seed: int = 42, C: float = 1.0,
) -> dict:
    """Multinomial logistic regression on the same 80/20 sequence-level split.

    Returns mean log-loss, test accuracy, and per-class log-loss on the held-out
    sequences (labels are argmax of the one-hot target, so every position in a
    sequence shares the same label).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss, accuracy_score

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    seq_is_train = np.zeros(n_sequences, dtype=bool)
    seq_is_train[perm[:split]] = True
    flat_train = np.repeat(seq_is_train, samples_per_sequence)
    flat_test = ~flat_train

    y_labels = np.argmax(y_all, axis=1)
    x_tr, y_tr = x_all[flat_train], y_labels[flat_train]
    x_te, y_te = x_all[flat_test], y_labels[flat_test]

    # sklearn >=1.5 removed `multi_class`; multinomial is default for multi-class.
    clf = LogisticRegression(solver="lbfgs", C=C, max_iter=500, random_state=seed)
    clf.fit(x_tr, y_tr)
    prob_te = clf.predict_proba(x_te)
    pred_te = clf.predict(x_te)
    ll = log_loss(y_te, prob_te, labels=list(range(y_all.shape[1])))
    acc = accuracy_score(y_te, pred_te)
    # Per-class log-loss: for each true-class c, mean -log prob[true_class=c]
    per_class_ll = []
    for c in range(y_all.shape[1]):
        mask = y_te == c
        if mask.any():
            pc_prob = np.clip(prob_te[mask, c], 1e-12, 1.0)
            per_class_ll.append(float(-np.log(pc_prob).mean()))
        else:
            per_class_ll.append(float("nan"))
    return {
        "log_loss": float(ll),
        "accuracy": float(acc),
        "per_class_log_loss": per_class_ll,
        "n_train": int(x_tr.shape[0]),
        "n_test": int(x_te.shape[0]),
        "C": C,
    }


def build_windowed_features(
    eval_acts: torch.Tensor, window: int
) -> tuple[np.ndarray, int]:
    """Take (N, T, d) → (N*(T-W+1), W*d) numpy with each row = concat of W residuals."""
    N, T, d = eval_acts.shape
    if T < window:
        raise ValueError(f"T={T} smaller than window={window}")
    # unfold along T → (N, T-W+1, d, W); permute to put W adjacent to d before flatten
    unfolded = eval_acts.unfold(1, window, 1)               # (N, T-W+1, d, W)
    unfolded = unfolded.permute(0, 1, 3, 2).contiguous()    # (N, T-W+1, W, d)
    windowed = unfolded.reshape(N, T - window + 1, window * d)
    return windowed.reshape(-1, window * d).numpy(), T - window + 1


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--cells-root", type=Path, required=True,
                   help="Folder containing cell_delta_*/ subfolders with transformer.pt + training_params.json")
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--eval-sequences", type=int, default=500)
    p.add_argument("--probe-layer", type=int, default=1)
    p.add_argument("--seed-offset-eval", type=int, default=20_000)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  window={args.window}", flush=True)

    out_combined: list[dict] = []
    for cell_dir in sorted(args.cells_root.glob("cell_delta_*")):
        if not (cell_dir / "transformer.pt").exists():
            print(f"  [skip] {cell_dir.name} missing transformer.pt")
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

        # Build transformer shell, load weights
        transformer = build_transformer(tf_cfg, gen.vocab_size, seed, device)
        ckpt = torch.load(cell_dir / "transformer.pt", map_location=device, weights_only=False)
        transformer.load_state_dict(ckpt["state_dict"])
        transformer.eval()

        # Sample eval data with same seed as the sweep
        seq_len = int(tf_cfg["n_ctx"])
        eval_data = gen.sample(
            n_sequences=args.eval_sequences, seq_len=seq_len,
            seed=seed + args.seed_offset_eval, with_component_labels=True,
        )
        hook_name = tf_cfg.get("probe_hook_name", f"blocks.{args.probe_layer}.hook_resid_post")
        eval_acts = extract_position_activations(
            transformer, eval_data.tokens, hook_name=hook_name, chunk_size=16,
        )
        eval_omega = eval_data.sequence_omegas
        n_eval = eval_acts.shape[0]
        d_model = eval_acts.shape[-1]

        # (a) single-position baseline
        flat_single = eval_acts.reshape(-1, d_model).numpy()
        flat_omega_single = (
            eval_omega[:, None, :].expand(n_eval, seq_len, eval_omega.shape[-1])
            .reshape(-1, eval_omega.shape[-1]).numpy()
        )
        single = fit_linear_probe_r2(
            flat_single, flat_omega_single,
            n_sequences=n_eval, samples_per_sequence=seq_len, seed=seed,
        )
        single_logit = fit_logistic_probe(
            flat_single, flat_omega_single,
            n_sequences=n_eval, samples_per_sequence=seq_len, seed=seed,
        )

        # (b) window-W probe
        flat_window, samples_per_seq_w = build_windowed_features(eval_acts, args.window)
        flat_omega_window = (
            eval_omega[:, None, :].expand(n_eval, samples_per_seq_w, eval_omega.shape[-1])
            .reshape(-1, eval_omega.shape[-1]).numpy()
        )
        window = fit_linear_probe_r2(
            flat_window, flat_omega_window,
            n_sequences=n_eval, samples_per_sequence=samples_per_seq_w, seed=seed,
        )
        window_logit = fit_logistic_probe(
            flat_window, flat_omega_window,
            n_sequences=n_eval, samples_per_sequence=samples_per_seq_w, seed=seed,
        )

        payload = {
            "sweep_value": sweep_value,
            "window_size": args.window,
            "d_in_single": int(d_model),
            "d_in_window": int(args.window * d_model),
            "samples_per_seq_single": int(seq_len),
            "samples_per_seq_window": int(samples_per_seq_w),
            "single_position": {
                "mean_r2": float(single["mean_r2"]),
                "per_component_r2": [float(v) for v in single["per_component_r2"]],
                "logistic": single_logit,
            },
            "window": {
                "mean_r2": float(window["mean_r2"]),
                "per_component_r2": [float(v) for v in window["per_component_r2"]],
                "logistic": window_logit,
            },
        }
        out = cell_dir / f"window_probes_w{args.window}.json"
        out.write_text(json.dumps(payload, indent=2))
        out_combined.append(payload)
        print(
            f"  δ={sweep_value:<5}  "
            f"R² single={payload['single_position']['mean_r2']:+.3f}  "
            f"W{args.window}={payload['window']['mean_r2']:+.3f}  "
            f"||  logloss single={single_logit['log_loss']:.3f} (acc={single_logit['accuracy']:.3f})  "
            f"W{args.window}={window_logit['log_loss']:.3f} (acc={window_logit['accuracy']:.3f})",
            flush=True,
        )

        del transformer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    combined = args.cells_root / f"window_probes_w{args.window}_combined.json"
    combined.write_text(json.dumps(out_combined, indent=2))
    print(f"\nSaved {combined}", flush=True)


if __name__ == "__main__":
    main()
