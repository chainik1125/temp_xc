"""Run the task-side spectral screen on the repository's synthetic panel.

The runner intentionally uses one leading tile per independently generated
sequence.  This prevents the common leakage bug where tiles from the same
sequence (and therefore the same hidden phase/level) appear in both probe
train and test folds.

Usage:

    uv run python -m experiments.power_spectrum.code.run_task_screen
    uv run python -m experiments.power_spectrum.code.run_task_screen --smoke
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from experiments.power_spectrum.code.task_spectrum import (
    classification_probe,
    dc_features,
    regression_probe,
    spectral_features,
    summarize_spectrum,
)

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = HERE / "configs" / "task_screen.json"


@dataclass(frozen=True)
class TargetSpec:
    datasource: str
    label_key: str
    kind: Literal["classification", "regression"]
    label_mode: Literal["sequence", "token"] = "token"
    transform: Literal[
        "none", "phasepair_pair", "phasepair_sign", "multilane_lane0"
    ] = "none"
    negative_is_invalid: bool = False


TARGETS: dict[str, TargetSpec] = {
    "frequency_velocity": TargetSpec(
        "toy_cyclic_circle_M101_d128", "velocity_labels", "classification", "sequence"
    ),
    "phasepair_pair": TargetSpec(
        "toy_phasepair_M101_d24", "velocity_labels", "classification", "sequence",
        "phasepair_pair",
    ),
    "phasepair_sign": TargetSpec(
        "toy_phasepair_M101_d24", "velocity_labels", "classification", "sequence",
        "phasepair_sign",
    ),
    "permuted_schedule": TargetSpec(
        "toy_permuted_circle_M101_d128", "schedule_labels", "classification", "sequence"
    ),
    "signed_motion_sign": TargetSpec(
        "toy_signed_motion_M19_d40", "sign_labels", "classification", "sequence"
    ),
    "multilane_lane0": TargetSpec(
        "toy_multilane_circle_M101_d24",
        "lane_velocity_labels",
        "classification",
        transform="multilane_lane0",
    ),
    "backtracking_lambda": TargetSpec(
        "toy_backtracking_selfexcite_d64", "lambda_labels", "regression"
    ),
    "changepoint_mode": TargetSpec(
        "toy_changepoint_modes_d64", "mode_labels", "classification"
    ),
    "changepoint_tss": TargetSpec(
        "toy_changepoint_modes_d64", "time_since_switch", "regression"
    ),
    "hedging_confidence": TargetSpec(
        "toy_hedging_drift_d64", "conf_labels", "regression"
    ),
    "recipe_equality": TargetSpec(
        "toy_recipe_instruction_d64", "equality_labels", "classification"
    ),
    "assumption_next_state": TargetSpec(
        "toy_assumption_consequence_d64", "next_state_labels", "classification",
        negative_is_invalid=True,
    ),
}


def _generate(datasource: str, *, n_sequences: int, seed: int):
    """Materialize a reduced dataset without populating the training cache."""
    from temp_bench.core.config import load_datasource
    from temp_bench.data.synthetic import _GENERATORS

    spec = load_datasource(datasource)
    if spec.generator not in _GENERATORS:
        raise ValueError(f"{datasource}: unsupported generator {spec.generator!r}")
    params = dict(spec.params or {})
    params.update(n_seqs=int(n_sequences), seed=int(seed))
    return _GENERATORS[spec.generator](**params)


def _labels(data, spec: TargetSpec) -> np.ndarray:
    labels = data.extra[spec.label_key].detach().cpu().numpy()
    if spec.label_mode == "sequence" and labels.ndim == 2:
        labels = labels[:, 0]
    if spec.transform == "multilane_lane0":
        labels = labels[..., 0]
    elif spec.transform != "none":
        omega = np.asarray(data.extra["omega"], dtype=np.int64)
        M = int(data.extra["M"])
        velocity = omega[labels.astype(np.int64)]
        if spec.transform == "phasepair_pair":
            labels = np.minimum(velocity, M - velocity)
        elif spec.transform == "phasepair_sign":
            labels = (velocity < (M / 2)).astype(np.int64)
    return labels


def _leading_tiles(
    x: np.ndarray, labels: np.ndarray, spec: TargetSpec, tile_size: int
) -> tuple[np.ndarray, np.ndarray]:
    """One tile per sequence, preserving sequence-level independence."""
    tiles = x[:, :tile_size, :]
    if spec.label_mode == "sequence":
        target = labels
    else:
        target = labels[:, tile_size - 1]
    return tiles, np.asarray(target)


def _run_probe(
    x: np.ndarray,
    calibration_x: np.ndarray,
    labels: np.ndarray,
    spec: TargetSpec,
    *,
    n_components: int,
    n_splits: int,
    seed: int,
) -> dict:
    out = {}
    for feature_kind, remove_dc in (
        ("power_full", False),
        ("power_ac", True),
        ("cross_full", False),
        ("cross_ac", True),
    ):
        kind = "power" if feature_kind.startswith("power") else "cross"
        features = spectral_features(
            x,
            kind=kind,
            n_components=n_components,
            remove_dc=remove_dc,
            fit_x=calibration_x,
        )
        if spec.kind == "classification":
            score = classification_probe(features, labels, n_splits=n_splits, seed=seed)
        else:
            score = regression_probe(features, labels, n_splits=n_splits, seed=seed)
        out[feature_kind] = score.to_dict()
    dc = dc_features(x, n_components=n_components, fit_x=calibration_x)
    if spec.kind == "classification":
        dc_score = classification_probe(dc, labels, n_splits=n_splits, seed=seed)
    else:
        dc_score = regression_probe(dc, labels, n_splits=n_splits, seed=seed)
    out["dc_vector"] = dc_score.to_dict()
    # Backward-friendly short aliases point to the explicit AC screen.
    out["power"] = out["power_ac"]
    out["cross"] = out["cross_ac"]
    out["dc_gain"] = (
        out["dc_vector"]["score_mean"] - out["power_ac"]["score_mean"]
    )
    out["phase_gain"] = (
        out["cross_ac"]["score_mean"] - out["power_ac"]["score_mean"]
    )
    return out


def run(config: dict, *, smoke: bool = False) -> dict:
    seeds = [int(v) for v in config["seeds"]]
    tile_sizes = [int(v) for v in config["tile_sizes"]]
    n_sequences = int(config["n_sequences"])
    n_calibration = int(config.get("n_calibration_sequences", 128))
    if smoke:
        seeds = seeds[:1]
        tile_sizes = [tile_sizes[0], tile_sizes[-1]]
        n_sequences = min(n_sequences, 96)
        n_calibration = min(n_calibration, 32)

    result = {
        "config": {
            **config,
            "seeds": seeds,
            "tile_sizes": tile_sizes,
            "n_sequences": n_sequences,
            "n_calibration_sequences": n_calibration,
            "smoke": smoke,
        },
        "summaries": [],
        "probes": [],
    }
    cache: dict[tuple[str, int], object] = {}

    def data_for(datasource: str, seed: int):
        key = (datasource, seed)
        if key not in cache:
            total = n_calibration + n_sequences
            print(
                f"[data] {datasource} seed={seed} "
                f"n_probe={n_sequences} n_cal={n_calibration}",
                flush=True,
            )
            cache[key] = _generate(datasource, n_sequences=total, seed=seed)
        return cache[key]

    for seed in seeds:
        for datasource in config["summary_datasources"]:
            data = data_for(datasource, seed)
            summary = summarize_spectrum(
                data.x.detach().cpu().numpy()[n_calibration:],
                low_cutoff=float(config["low_cutoff"]),
            )
            result["summaries"].append({
                "datasource": datasource,
                "seed": seed,
                **summary.to_dict(),
            })
            print(
                f"[spectrum] {datasource} s{seed} "
                f"DC={summary.dc_fraction:.3f} lowAC={summary.ac_low_fraction:.3f} "
                f"centroid={summary.ac_centroid:.3f} dir={summary.max_directionality:.3f}",
                flush=True,
            )

        for target_name in config["targets"]:
            spec = TARGETS[target_name]
            data = data_for(spec.datasource, seed)
            all_x = data.x.detach().cpu().numpy()
            x = all_x[n_calibration:]
            calibration_x = all_x[:n_calibration]
            labels = _labels(data, spec)[n_calibration:]
            for tile_size in tile_sizes:
                tiles, target = _leading_tiles(x, labels, spec, tile_size)
                calibration_tiles = calibration_x[:, :tile_size, :]
                valid = np.isfinite(target)
                if spec.negative_is_invalid:
                    valid &= target >= 0
                probe = _run_probe(
                    tiles[valid],
                    calibration_tiles,
                    target[valid],
                    spec,
                    n_components=int(config["n_components"]),
                    n_splits=min(int(config["n_splits"]), 3 if smoke else 5),
                    seed=seed,
                )
                row = {
                    "target": target_name,
                    "datasource": spec.datasource,
                    "target_kind": spec.kind,
                    "seed": seed,
                    "tile_size": tile_size,
                    **probe,
                }
                result["probes"].append(row)
                print(
                    f"[probe] {target_name} s{seed} T={tile_size} "
                    f"full={probe['power_full']['score_mean']:.3f} "
                    f"DCvec={probe['dc_vector']['score_mean']:.3f} "
                    f"AC={probe['power_ac']['score_mean']:.3f} "
                    f"crossAC={probe['cross_ac']['score_mean']:.3f} "
                    f"dc={probe['dc_gain']:+.3f} phase={probe['phase_gain']:+.3f}",
                    flush=True,
                )
    return result


def _write_outputs(result: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            lineterminator="\n",
            fieldnames=[
                "target", "datasource", "target_kind", "seed", "tile_size",
                "power_full_score", "dc_vector_score", "power_ac_score", "power_ac_std",
                "power_ac_null", "cross_ac_score", "cross_ac_std",
                "cross_ac_null", "dc_gain", "phase_gain",
            ],
        )
        writer.writeheader()
        for row in result["probes"]:
            writer.writerow({
                "target": row["target"],
                "datasource": row["datasource"],
                "target_kind": row["target_kind"],
                "seed": row["seed"],
                "tile_size": row["tile_size"],
                "power_full_score": row["power_full"]["score_mean"],
                "dc_vector_score": row["dc_vector"]["score_mean"],
                "power_ac_score": row["power_ac"]["score_mean"],
                "power_ac_std": row["power_ac"]["score_std"],
                "power_ac_null": row["power_ac"]["shuffled_mean"],
                "cross_ac_score": row["cross_ac"]["score_mean"],
                "cross_ac_std": row["cross_ac"]["score_std"],
                "cross_ac_null": row["cross_ac"]["shuffled_mean"],
                "dc_gain": row["dc_gain"],
                "phase_gain": row["phase_gain"],
            })
    print(f"wrote {output} and {csv_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=HERE / "results" / "task_screen.json")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output
    if args.smoke and output == HERE / "results" / "task_screen.json":
        output = HERE / "results" / "task_screen_smoke.json"
    _write_outputs(run(config, smoke=args.smoke), output)


if __name__ == "__main__":
    main()
