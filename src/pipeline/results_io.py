"""Results I/O — consistent JSON serialization for experiment results."""

import json
from dataclasses import asdict
from datetime import datetime

import numpy as np

from src.eval.toy_unified import EvalResult


class NumpySafeEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def save_results(
    path: str,
    experiment_type: str,
    config: dict,
    model_results: dict[str, list[EvalResult]],
) -> None:
    """Save experiment results to JSON with consistent schema."""
    data = {
        "experiment_type": experiment_type,
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "model_results": {
            name: [r.to_dict() for r in results]
            for name, results in model_results.items()
        },
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NumpySafeEncoder)


def load_results(path: str) -> dict:
    """Load experiment results from JSON."""
    with open(path) as f:
        return json.load(f)
