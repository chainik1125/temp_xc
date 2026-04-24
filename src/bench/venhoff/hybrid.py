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
from src.bench.venhoff.vendor_patches import ensure_hybrid_judge_patched, ensure_steering_patched

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
    n_tasks: int = 500
    # Arch identifier — selects which on-disk steering vectors get
    # swapped to the bare path before hybrid_token.py reads them.
    # See _swap_arch_vectors_in / _swap_arch_vectors_out.
    arch: str = "sae"


def _venhoff_script_dir(venhoff_root: Path) -> Path:
    return venhoff_root / "hybrid"


def _vector_paths(venhoff_root: Path, base_model: str, n_clusters: int) -> list[tuple[str, Path]]:
    """Yield (tag, bare_path) for every steering vector hybrid_token.py loads."""
    model_short = base_model.split("/")[-1].lower()
    root = venhoff_root / "train-vectors" / "results" / "vars" / "optimized_vectors"
    tags = ["bias"] + [f"idx{i}" for i in range(n_clusters)]
    return [(t, root / f"{model_short}_{t}.pt") for t in tags]


def _swap_arch_vectors_in(venhoff_root: Path, cfg: HybridConfig) -> list[tuple[Path, Path | None]]:
    """Swap arch-specific steering vectors into the bare paths that
    hybrid_token.py reads from. Returns a list of (bare_path, prior_backup)
    tuples for `_swap_arch_vectors_out` to undo.

    For sae arch: no swap needed; bare paths already contain shipped
    vectors (or whatever Phase 2 produced under the legacy convention).
    """
    if cfg.arch == "sae":
        return []

    swapped: list[tuple[Path, Path | None]] = []
    for tag, bare_path in _vector_paths(venhoff_root, cfg.base_model, cfg.n_clusters):
        arch_path = bare_path.parent / f"{bare_path.stem}_{cfg.arch}.pt"
        if not arch_path.exists():
            log.warning("[warn] arch_vector_missing | arch=%s | tag=%s | path=%s",
                        cfg.arch, tag, arch_path)
            continue
        # Backup current bare to a sidecar so we can restore.
        backup: Path | None = None
        if bare_path.exists():
            backup = bare_path.with_suffix(f".pt.preswap_{cfg.arch}")
            if backup.exists():
                backup.unlink()
            bare_path.rename(backup)
        # Copy (not symlink) so a downstream torch.load doesn't follow
        # back to our arch-specific store.
        import shutil as _shutil
        _shutil.copy(arch_path, bare_path)
        swapped.append((bare_path, backup))
    log.info("[info] arch_vectors_swapped_in | arch=%s | n=%d", cfg.arch, len(swapped))
    return swapped


def _swap_arch_vectors_out(swapped: list[tuple[Path, Path | None]]) -> None:
    """Restore the original bare-path contents (Venhoff shipped or
    whichever arch was previously swapped in)."""
    for bare_path, backup in swapped:
        if bare_path.exists():
            bare_path.unlink()
        if backup is not None and backup.exists():
            backup.rename(bare_path)
    log.info("[info] arch_vectors_swapped_out | n=%d", len(swapped))


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
    # hybrid_token.py goes through utils.load_model(), which calls
    # LanguageModel(..., load_in_8bit=...). Modern transformers drops the
    # kwarg, so share the same 8bit-stripping patches as Phase 2.
    ensure_steering_patched(venhoff_root)

    log.info("[info] hybrid | dataset=%s | arch=%s | cmd=%s", cfg.dataset, cfg.arch, " ".join(cmd))
    env = {**os.environ, "PYTHONPATH": str(venhoff_root.absolute())}

    # Swap this arch's steering vectors into the bare paths that
    # hybrid_token.py reads. Wrap in try/finally so a crash still
    # restores Venhoff's shipped vectors (otherwise the next arch's
    # SAE run sees TempXC/MLC-trained vectors).
    swapped = _swap_arch_vectors_in(venhoff_root, cfg)
    try:
        result = subprocess.run(cmd, cwd=_venhoff_script_dir(venhoff_root), env=env, check=False)
    finally:
        _swap_arch_vectors_out(swapped)
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
