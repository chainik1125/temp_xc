"""Phase 2: train per-cluster steering vectors via Venhoff's scripts.

We don't re-implement their 53k-line `optimize_steering_vectors.py`.
We vendor their repo (`vendor/thinking-llms-interp/`), drop our trained
ckpts into the Venhoff-format path via `venhoff_format.export()`, and
invoke their script via subprocess once per cluster + once for the
general `bias` vector.

For the Llama-8B cell that's **16 optimize runs per arch** (15 clusters
+ bias), ~15 min each, ~4 H100-hours per arch. Skips if the output
vector file already exists and has a sidecar meta hash matching the
current config.

Per their `train-vectors/run_llama_8b.sh`:
  - `--model meta-llama/Llama-3.1-8B`
  - `--max_iters 50`
  - `--n_training_examples 2048 --n_eval_examples 512`
  - `--optim_minibatch_size 4`
  - `--layer 12`       (steering layer on Llama-8B base model)
  - `--steering_vector_idx -1`  (bias) or cluster_id (0..n_clusters-1)
  - `--lr 1e-2`
  - `--use_activation_perplexity_selection`  (for cluster vectors only,
    not the bias)
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.bench.venhoff.paths import ArtifactPaths, can_resume, write_with_metadata

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.steering")


def _venhoff_python(venhoff_root: Path) -> str:
    """Resolve the python interpreter to run Venhoff's scripts with.

    Priority:
      1. `VENHOFF_PYTHON` env var (explicit override)
      2. `<venhoff_root>/.venv/bin/python` (from `uv sync` inside vendor)
      3. `sys.executable` — our venv, with a warning (will fail on
         chat-limiter + numpy<2.0 deps; only used when the user
         explicitly installed Venhoff's deps into the main venv).
    """
    override = os.environ.get("VENHOFF_PYTHON", "").strip()
    if override:
        if not Path(override).exists():
            raise FileNotFoundError(f"VENHOFF_PYTHON points to missing file: {override}")
        return override
    candidate = venhoff_root / ".venv" / "bin" / "python"
    if candidate.exists():
        return str(candidate)
    log.warning(
        "[warn] venhoff_venv_missing | expected=%s | falling_back=sys.executable | hint=run 'uv sync' inside %s",
        candidate, venhoff_root,
    )
    return sys.executable


@dataclass(frozen=True)
class SteeringConfig:
    base_model: str = "meta-llama/Llama-3.1-8B"
    thinking_model: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    steering_layer: int = 12
    sae_layer: int = 6
    n_clusters: int = 15
    max_iters: int = 50
    n_training_examples: int = 2048
    n_eval_examples: int = 512
    optim_minibatch_size: int = 4
    lr: str = "1e-2"
    seed: int = 42


def _venhoff_expected_vector_path(
    venhoff_root: Path, base_model: str, cluster_idx: int
) -> Path:
    """Venhoff saves vectors at `train-vectors/results/vars/optimized_vectors/{model}_idx{idx}.pt`.

    `cluster_idx = -1` → bias vector; otherwise the cluster id.
    """
    model_short = base_model.split("/")[-1]
    name = f"{model_short}_idx{cluster_idx}.pt"
    return venhoff_root / "train-vectors" / "results" / "vars" / "optimized_vectors" / name


def _venhoff_script_dir(venhoff_root: Path) -> Path:
    return venhoff_root / "train-vectors"


def train_one_vector(
    venhoff_root: Path,
    cfg: SteeringConfig,
    cluster_idx: int,
    paths: ArtifactPaths,
    force: bool = False,
) -> Path:
    """Invoke Venhoff's optimize_steering_vectors.py for one cluster/bias."""
    expected_out = _venhoff_expected_vector_path(venhoff_root, cfg.base_model, cluster_idx)

    meta_path = paths.run_dir / "steering" / f"vector_{cluster_idx}.meta.json"
    meta = {
        "stage": "train_steering_vector",
        "cluster_idx": cluster_idx,
        "base_model": cfg.base_model,
        "steering_layer": cfg.steering_layer,
        "sae_layer": cfg.sae_layer,
        "n_clusters": cfg.n_clusters,
        "max_iters": cfg.max_iters,
        "lr": cfg.lr,
        "seed": cfg.seed,
    }
    if not force and expected_out.exists() and can_resume(meta_path, meta):
        log.info("[info] resume | stage=steering | cluster_idx=%d | cache=%s", cluster_idx, expected_out)
        return expected_out

    cmd = [
        _venhoff_python(venhoff_root), "optimize_steering_vectors.py",
        "--model", cfg.base_model,
        "--max_iters", str(cfg.max_iters),
        "--n_training_examples", str(cfg.n_training_examples),
        "--n_eval_examples", str(cfg.n_eval_examples),
        "--optim_minibatch_size", str(cfg.optim_minibatch_size),
        "--layer", str(cfg.steering_layer),
        "--steering_vector_idx", str(cluster_idx),
        "--lr", cfg.lr,
        "--seed", str(cfg.seed),
    ]
    # Per Venhoff's run_llama_8b.sh: cluster vectors get
    # `--use_activation_perplexity_selection`; the bias vector (idx=-1) does not.
    if cluster_idx != -1:
        cmd.append("--use_activation_perplexity_selection")

    log.info("[info] steering | cluster_idx=%d | cmd=%s", cluster_idx, " ".join(cmd))
    result = subprocess.run(cmd, cwd=_venhoff_script_dir(venhoff_root), check=False)
    if result.returncode != 0:
        raise RuntimeError(f"optimize_steering_vectors.py failed for cluster_idx={cluster_idx} (rc={result.returncode})")

    if not expected_out.exists():
        raise FileNotFoundError(f"expected steering vector not produced at {expected_out}")

    # Record our own resume sidecar next to the vector (under our run_dir).
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_with_metadata(meta_path, json.dumps({"vector_path": str(expected_out)}), meta)
    log.info("[done] steering | cluster_idx=%d | vector=%s", cluster_idx, expected_out)
    return expected_out


def train_all_vectors(
    venhoff_root: Path,
    cfg: SteeringConfig,
    paths: ArtifactPaths,
    force: bool = False,
) -> list[Path]:
    """Train bias (idx=-1) + n_clusters vectors. Returns ordered list of paths."""
    out = []
    # Venhoff trains the bias first, then each cluster idx 0..n_clusters-1.
    out.append(train_one_vector(venhoff_root, cfg, -1, paths, force=force))
    for cluster_idx in range(cfg.n_clusters):
        out.append(train_one_vector(venhoff_root, cfg, cluster_idx, paths, force=force))
    return out
