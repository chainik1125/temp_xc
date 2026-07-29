"""Paper-faithful grouped Backtracking detection for a corrected 300K cell."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from train import (
    ACT_CACHE_KEY,
    EXPECTED_KEYS,
    HISTORICAL_COMMIT,
    PROTOCOL_VERSION,
)

DETECTION_PROTOCOL = "c7-detection-seeded-v1"
S_GRID = (1, 2, 4, 8, 16, 32)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
    temporary.replace(path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--sentence-acts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arch", choices=("txc_base", "tsae_paper"), required=True)
    parser.add_argument("--d-sae", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1_024)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    historical_root = args.historical_root.resolve()
    checkpoint_dir = args.checkpoint_dir.resolve()
    sentence_acts_path = args.sentence_acts.resolve()
    output_path = args.output.resolve()
    if (historical_root / "HISTORICAL_COMMIT").read_text().strip() != HISTORICAL_COMMIT:
        raise RuntimeError("historical source marker mismatch")
    if not sentence_acts_path.is_file():
        raise FileNotFoundError(sentence_acts_path)
    model_path = checkpoint_dir / "model.safetensors"
    config_path = checkpoint_dir / "config.json"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError(f"incomplete checkpoint: {checkpoint_dir}")

    checkpoint_config = json.loads(config_path.read_text())
    locked_key = EXPECTED_KEYS.get((args.arch, args.d_sae, args.seed))
    required = {
        "status": "complete",
        "protocol_version": PROTOCOL_VERSION,
        "historical_commit": HISTORICAL_COMMIT,
        "arch": args.arch,
        "d_sae": args.d_sae,
        "seed": args.seed,
        "train_key": locked_key,
        "act_cache_key": ACT_CACHE_KEY,
        "n_steps_completed": 300_000,
    }
    mismatches = {
        field: {"expected": expected, "actual": checkpoint_config.get(field)}
        for field, expected in required.items()
        if expected is None or checkpoint_config.get(field) != expected
    }
    if mismatches:
        raise RuntimeError(f"checkpoint provenance mismatch: {mismatches}")
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if (
            existing.get("status") == "complete"
            and existing.get("train_key") == locked_key
            and existing.get("detection_protocol") == DETECTION_PROTOCOL
        ):
            print(f"complete detection result already exists: {output_path}", flush=True)
            return 0
        raise RuntimeError(f"refusing to overwrite non-matching result: {output_path}")

    os.environ["TEMP_BENCH_ROOT"] = str(historical_root)
    sys.path.insert(0, str(historical_root / "src"))

    import torch
    from safetensors.torch import load_file

    from temp_bench.case_studies.backtracking import compute_probe_metrics_at_S
    from temp_bench.config import instantiate_arch, load_arch
    from temp_bench.utils.seed import set_seed

    set_seed(args.seed)
    spec = load_arch(args.arch, component="c7")
    spec = spec.model_copy(
        update={"hparams": {**spec.hparams, "d_sae": int(args.d_sae)}}
    )
    model = instantiate_arch(spec, d_in=4096)
    model.load_state_dict(load_file(str(model_path)))
    if torch.cuda.is_available():
        model = model.cuda()

    with np.load(sentence_acts_path, allow_pickle=True) as archive:
        sentence_acts = archive["X"]
        labels = archive["is_bt"].astype(np.int64)
        keys = archive["keys"]
    if sentence_acts.shape != (25_204, 6, 4_096):
        raise RuntimeError(f"unexpected sentence-activation shape: {sentence_acts.shape}")
    question_ids = np.asarray(
        [str(key).split("|", 1)[0] for key in keys], dtype=object
    )

    arch_window = int(getattr(model.config, "T", None) or 1)
    window_size = sentence_acts.shape[1]
    acts_for_arch = sentence_acts
    if arch_window < window_size:
        acts_for_arch = sentence_acts[:, -arch_window:, :]
    elif arch_window > window_size:
        padding = arch_window - window_size
        acts_for_arch = np.concatenate(
            [
                np.repeat(sentence_acts[:, :1, :], padding, axis=1),
                sentence_acts,
            ],
            axis=1,
        )

    device = next(model.parameters()).device
    parameter_dtype = next(model.parameters()).dtype
    chunks: list[np.ndarray] = []
    started = time.time()
    with torch.no_grad():
        for start in range(0, len(labels), args.batch_size):
            batch = torch.from_numpy(
                np.ascontiguousarray(acts_for_arch[start : start + args.batch_size])
            ).to(device=device, dtype=parameter_dtype)
            encoded = model.encode(batch).abs().float()
            if encoded.dim() == 3:
                encoded = encoded.amax(dim=1)
            chunks.append(encoded.cpu().numpy())
    feature_acts = np.concatenate(chunks, axis=0)
    metrics = compute_probe_metrics_at_S(
        feature_acts,
        labels,
        question_ids,
        S_grid=S_GRID,
        random_state=42,
    )
    payload = {
        "status": "complete",
        "detection_protocol": DETECTION_PROTOCOL,
        "training_protocol": PROTOCOL_VERSION,
        "historical_commit": HISTORICAL_COMMIT,
        "train_key": locked_key,
        "arch": args.arch,
        "d_sae": args.d_sae,
        "seed": args.seed,
        "n_sentences": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_fraction": float(labels.mean()),
        "sentence_window": int(sentence_acts.shape[1]),
        "arch_window": arch_window,
        "S_grid": list(S_GRID),
        "probe_random_state": 42,
        "pr_auc": {str(key): value for key, value in metrics["pr_auc"].items()},
        "roc_auc": {str(key): value for key, value in metrics["roc_auc"].items()},
        "elapsed_seconds": time.time() - started,
    }
    _atomic_json(output_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
