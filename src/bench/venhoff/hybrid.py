"""Phase 3: hybrid-model inference on MATH500 via Venhoff's scripts.

Drives `vendor/thinking-llms-interp/hybrid/hybrid_token.py` via
subprocess once per arch. Their script sweeps the 10 × 5 hyperparam
grid (coefficients × token_windows) in one invocation and writes a
JSONL with accuracy per (coefficient, token_window) cell, which our
`grade.py` consumes to compute Gap Recovery.

Per their `hybrid/run_llama_8b.sh` MATH500 line:
  hybrid_token.py
    --dataset math500
    --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    --base_model meta-llama/Llama-3.1-8B
    --steering_layer 12
    --sae_layer 6
    --n_clusters 15
    --max_new_tokens 2000
    --max_thinking_tokens 2000
    --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
    --token_windows 0 -1 -15 -50 -100

Resume: the output JSONL path is conventional to Venhoff's layout
(`hybrid/results/...`). We check for the file + sidecar meta before
invoking. `--force` re-runs the whole grid.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.bench.venhoff.paths import ArtifactPaths, can_resume, write_with_metadata
from src.bench.venhoff.steering import _venhoff_python
from src.bench.venhoff.vendor_patches import ensure_hybrid_judge_patched

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.hybrid")


DEFAULT_COEFFICIENTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
DEFAULT_TOKEN_WINDOWS = (0, -1, -15, -50, -100)


@dataclass(frozen=True)
class HybridConfig:
    dataset: str = "math500"
    base_model: str = "meta-llama/Llama-3.1-8B"
    thinking_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    steering_layer: int = 12
    sae_layer: int = 6
    n_clusters: int = 15
    max_new_tokens: int = 2000
    max_thinking_tokens: int = 2000
    coefficients: tuple[float, ...] = field(default_factory=lambda: DEFAULT_COEFFICIENTS)
    token_windows: tuple[int, ...] = field(default_factory=lambda: DEFAULT_TOKEN_WINDOWS)
    seed: int = 42
    # Venhoff's hybrid_token.py has a `--n_tasks` arg (default 500) with
    # built-in resume. Set to a subset size to cap problem count.
    n_tasks: int = 500


def _venhoff_script_dir(venhoff_root: Path) -> Path:
    return venhoff_root / "hybrid"


def _expected_results_dir(venhoff_root: Path, cfg: HybridConfig) -> Path:
    """Venhoff's hybrid_token.py writes under
    `hybrid/results/{dataset}/{base_model}_{thinking_model}_L{steering_layer}_SAEL{sae_layer}_n{n_clusters}/`.

    We don't know the exact filename convention without reading the
    126k-line script, so we check that the results dir exists + has
    JSONL files as a resume heuristic. The hybrid_token.py CLI respects
    its own resume behavior anyway (it skips problems it's already
    solved).
    """
    base_short = cfg.base_model.split("/")[-1]
    think_short = cfg.thinking_model.split("/")[-1]
    return (
        venhoff_root / "hybrid" / "results" / cfg.dataset
        / f"{base_short}_{think_short}_L{cfg.steering_layer}_SAEL{cfg.sae_layer}_n{cfg.n_clusters}"
    )


def run_hybrid(
    venhoff_root: Path,
    cfg: HybridConfig,
    paths: ArtifactPaths,
    force: bool = False,
) -> Path:
    """Invoke Venhoff's hybrid_token.py. Returns path to the results directory."""
    results_dir = _expected_results_dir(venhoff_root, cfg)
    sentinel = paths.run_dir / "hybrid" / f"{cfg.dataset}_complete.meta.json"
    meta = {
        "stage": "hybrid_eval",
        "dataset": cfg.dataset,
        "base_model": cfg.base_model,
        "thinking_model": cfg.thinking_model,
        "steering_layer": cfg.steering_layer,
        "sae_layer": cfg.sae_layer,
        "n_clusters": cfg.n_clusters,
        "max_new_tokens": cfg.max_new_tokens,
        "coefficients": list(cfg.coefficients),
        "token_windows": list(cfg.token_windows),
        "seed": cfg.seed,
    }
    if not force and can_resume(sentinel, meta) and results_dir.exists():
        log.info("[info] resume | stage=hybrid | cache=%s", results_dir)
        return results_dir

    cmd = [
        _venhoff_python(venhoff_root), "hybrid_token.py",
        "--dataset", cfg.dataset,
        "--thinking_model", cfg.thinking_model,
        "--base_model", cfg.base_model,
        "--steering_layer", str(cfg.steering_layer),
        "--sae_layer", str(cfg.sae_layer),
        "--n_clusters", str(cfg.n_clusters),
        "--n_tasks", str(cfg.n_tasks),
        "--max_new_tokens", str(cfg.max_new_tokens),
        "--max_thinking_tokens", str(cfg.max_thinking_tokens),
        "--coefficients", *[str(c) for c in cfg.coefficients],
        "--token_windows", *[str(w) for w in cfg.token_windows],
    ]
    # Patch hybrid_token.py's hardcoded gpt-5.2 judge to claude-haiku-4.5
    # (idempotent — no-op if already applied). Deliberate deviation from
    # Venhoff; documented in VENHOFF_PROVENANCE.md.
    ensure_hybrid_judge_patched(venhoff_root)

    log.info("[info] hybrid | dataset=%s | cmd=%s", cfg.dataset, " ".join(cmd))
    # Same PYTHONPATH fix as steering.py — hybrid_token.py does
    # `from utils.sae import load_sae` without a sys.path prefix.
    env = {**os.environ, "PYTHONPATH": str(venhoff_root.absolute())}
    result = subprocess.run(cmd, cwd=_venhoff_script_dir(venhoff_root), env=env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"hybrid_token.py failed on dataset={cfg.dataset} (rc={result.returncode})")

    if not results_dir.exists():
        raise FileNotFoundError(
            f"expected hybrid results dir not produced at {results_dir}. "
            "Venhoff's hybrid_token.py may use a different naming convention than we assumed; "
            "inspect vendor/thinking-llms-interp/hybrid/results/ and adjust _expected_results_dir()."
        )

    # Sentinel records the completion of this hybrid run.
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    write_with_metadata(sentinel, json.dumps({"results_dir": str(results_dir)}), meta)
    log.info("[done] hybrid | dataset=%s | results=%s", cfg.dataset, results_dir)
    return results_dir
