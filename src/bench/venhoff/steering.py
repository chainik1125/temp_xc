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
from src.bench.venhoff.vendor_patches import ensure_steering_patched

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("venhoff.steering")


def _venhoff_python(venhoff_root: Path) -> str:
    """Resolve the python interpreter to run Venhoff's scripts with.

    Returns an absolute path that PRESERVES any symlink to the venv's
    shim binary. A uv venv's `.venv/bin/python` is a symlink to the
    base Python interpreter; what makes it a *venv* Python is the
    `pyvenv.cfg` sitting next to the symlink, which Python detects
    via the invocation path. If we `.resolve()` the symlink we hand
    subprocess the raw base interpreter path, Python doesn't find
    pyvenv.cfg, and the venv's site-packages is invisible — that's
    what caused the `No module named 'dotenv'` error on 2026-04-20.

    Use `.absolute()` (no symlink resolution) instead. subprocess.run
    uses cwd=<venhoff_root>/train-vectors|hybrid, so the path must be
    absolute or the relative prefix is wrong — but NOT symlink-resolved.

    Priority:
      1. `VENHOFF_PYTHON` env var (explicit override)
      2. `<venhoff_root>/.venv/bin/python`
      3. `sys.executable` — warn-level fallback
    """
    override = os.environ.get("VENHOFF_PYTHON", "").strip()
    if override:
        p = Path(override).absolute()
        if not p.exists():
            raise FileNotFoundError(f"VENHOFF_PYTHON points to missing file: {p}")
        return str(p)
    candidate = (venhoff_root / ".venv" / "bin" / "python").absolute()
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
    # Paper defaults (Venhoff et al. 2025 Appendix C.1):
    #   max_iters=50, n_training_examples=2048, optim_minibatch_size=6.
    max_iters: int = 50
    n_training_examples: int = 2048
    n_eval_examples: int = 64
    optim_minibatch_size: int = 6
    lr: str = "1e-2"
    seed: int = 42
    cluster_indices: tuple[int, ...] | None = None
    bias_only: bool = False
    # Arch identifier. Determines steering-vector output filename so
    # multiple arches don't share the same `.pt` files on disk
    # (the bug surfaced by the 2026-04-23 paper-budget run where SAE,
    # TempXC, and MLC all read the same vectors → identical Gap Recovery).
    #   - sae   → bare `{model}_{tag}.pt` (Venhoff's shipped path)
    #   - tempxc → `{model}_{tag}_tempxc.pt`
    #   - mlc    → `{model}_{tag}_mlc.pt`
    arch: str = "sae"


def _venhoff_expected_vector_path(
    venhoff_root: Path,
    base_model: str,
    cluster_idx: int,
    arch: str = "sae",
    steering_type: str = "linear",
) -> Path:
    """Resolve the canonical steering-vector path for an arch.

    Venhoff saves vectors at `train-vectors/results/vars/optimized_vectors/`.
    We arch-key the filename so SAE / TempXC / MLC don't share the same
    `.pt` file on disk (whoever trains last would otherwise overwrite the
    shipped vectors and contaminate the other arches' Phase 3 runs):

      sae    → bare `{model}_{tag}.pt`        (Venhoff's shipped path)
      tempxc → `{model}_{tag}_tempxc.pt`      (our fresh-trained path)
      mlc    → `{model}_{tag}_mlc.pt`         (our fresh-trained path)

    Within `arch=sae`, we still honor the legacy `_linear.pt` fresh-train
    output if present (so older runs that didn't have arch-keying remain
    loadable), but the canonical resolution order is:
      1. arch-specific suffix (for non-sae)
      2. bare shipped (for sae)
      3. legacy `_linear.pt` (only for sae backcompat)
    """
    model_short = base_model.split("/")[-1].lower()
    tag = "bias" if cluster_idx == -1 else f"idx{cluster_idx}"
    root = venhoff_root / "train-vectors" / "results" / "vars" / "optimized_vectors"

    if arch != "sae":
        return root / f"{model_short}_{tag}_{arch}.pt"

    # SAE path: prefer Venhoff's bare shipped file; fall back to legacy
    # `_linear.pt` if some old run left one around.
    bare = root / f"{model_short}_{tag}.pt"
    if bare.exists():
        return bare
    legacy_linear = root / f"{model_short}_{tag}_{steering_type}.pt"
    if legacy_linear.exists():
        return legacy_linear
    return bare


def _venhoff_subprocess_output_path(
    venhoff_root: Path,
    base_model: str,
    cluster_idx: int,
    steering_type: str = "linear",
) -> Path:
    """Where `optimize_steering_vectors.py` actually writes its trained
    output (always `{model}_{tag}_{steering_type}.pt`, regardless of our
    arch convention). We rename this to the arch-specific path after
    training succeeds so the disk layout matches `_venhoff_expected_vector_path`.
    """
    model_short = base_model.split("/")[-1].lower()
    tag = "bias" if cluster_idx == -1 else f"idx{cluster_idx}"
    root = venhoff_root / "train-vectors" / "results" / "vars" / "optimized_vectors"
    return root / f"{model_short}_{tag}_{steering_type}.pt"


def _venhoff_script_dir(venhoff_root: Path) -> Path:
    return venhoff_root / "train-vectors"


def train_one_vector(
    venhoff_root: Path,
    cfg: SteeringConfig,
    cluster_idx: int,
    paths: ArtifactPaths,
    force: bool = False,
    gpu_id: int | None = None,
) -> Path:
    """Invoke Venhoff's optimize_steering_vectors.py for one cluster/bias.

    If `gpu_id` is set, pins the subprocess to that GPU via
    CUDA_VISIBLE_DEVICES. Used by train_all_vectors when multi-GPU
    parallel training is requested.
    """
    expected_out = _venhoff_expected_vector_path(venhoff_root, cfg.base_model, cluster_idx, arch=cfg.arch)

    # Arch in meta path + meta hash so a later arch's resume check can't
    # match an earlier arch's training output (the contamination bug
    # behind the 2026-04-23 paper-budget run's identical-across-arches
    # Gap Recovery numbers).
    meta_path = paths.run_dir / "steering" / f"vector_{cluster_idx}_{cfg.arch}.meta.json"
    meta = {
        "stage": "train_steering_vector",
        "arch": cfg.arch,
        "cluster_idx": cluster_idx,
        "base_model": cfg.base_model,
        "steering_layer": cfg.steering_layer,
        "sae_layer": cfg.sae_layer,
        "n_clusters": cfg.n_clusters,
        "max_iters": cfg.max_iters,
        "lr": cfg.lr,
        "seed": cfg.seed,
    }
    # Resume rules:
    #   1. expected_out exists AND our meta sidecar matches → we trained
    #      this before, skip.
    #   2. expected_out exists AND NO sidecar (i.e., Venhoff's shipped
    #      pre-trained vector) → skip training. Write a sidecar so the
    #      skip reason is greppable.
    if not force and expected_out.exists():
        if meta_path.exists() and can_resume(meta_path, meta):
            log.info("[info] resume | stage=steering | cluster_idx=%d | cache=%s", cluster_idx, expected_out)
            return expected_out
        if not meta_path.exists():
            log.info("[info] reuse_shipped | stage=steering | cluster_idx=%d | cache=%s",
                     cluster_idx, expected_out)
            meta_path.parent.mkdir(parents=True, exist_ok=True)
            write_with_metadata(
                meta_path,
                json.dumps({"vector_path": str(expected_out), "source": "venhoff_shipped"}),
                {**meta, "source": "venhoff_shipped"},
            )
            return expected_out

    cmd = [
        _venhoff_python(venhoff_root), "optimize_steering_vectors.py",
        "--model", cfg.base_model,
        "--max_iters", str(cfg.max_iters),
        "--n_training_examples", str(cfg.n_training_examples),
        "--n_eval_examples", str(cfg.n_eval_examples),
        "--optim_minibatch_size", str(cfg.optim_minibatch_size),
        "--base_gen_minibatch_size", str(cfg.optim_minibatch_size),
        "--layer", str(cfg.steering_layer),
        "--steering_vector_idx", str(cluster_idx),
        "--lr", cfg.lr,
        "--seed", str(cfg.seed),
    ]
    # Per Venhoff's run_llama_8b.sh: cluster vectors get
    # `--use_activation_perplexity_selection`; the bias vector (idx=-1) does not.
    if cluster_idx != -1:
        cmd.append("--use_activation_perplexity_selection")

    log.info("[info] steering | cluster_idx=%d | gpu=%s | cmd=%s",
             cluster_idx, gpu_id if gpu_id is not None else "default", " ".join(cmd))
    # Inject PYTHONPATH=<venhoff_root> so `from utils.X import ...` resolves
    # regardless of cwd. Some Venhoff scripts sys.path.append('..') themselves;
    # hybrid_token.py doesn't, so we make it uniform here.
    env = {**os.environ, "PYTHONPATH": str(venhoff_root.absolute())}
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Capture subprocess stdout+stderr to a per-vector log. Parallel workers'
    # output in the parent tmux is otherwise interleaved and a failure is
    # opaque. Tee to both the log file and the parent (so existing tmux view
    # stays informative for the happy path) via a pipe + `tee`.
    per_vec_log = paths.run_dir / "steering" / f"vector_{cluster_idx}_{cfg.base_model.split('/')[-1]}.log"
    per_vec_log.parent.mkdir(parents=True, exist_ok=True)
    with per_vec_log.open("w") as lf:
        result = subprocess.run(
            cmd,
            cwd=_venhoff_script_dir(venhoff_root),
            env=env,
            check=False,
            stdout=lf,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        # Surface the last ~30 lines of the per-vector log so the parent
        # exception message is actionable without going hunting.
        tail_lines: list[str] = []
        try:
            tail_lines = per_vec_log.read_text().splitlines()[-30:]
        except Exception:
            pass
        tail_str = "\n    ".join(tail_lines) if tail_lines else "(no log captured)"
        raise RuntimeError(
            f"optimize_steering_vectors.py failed for cluster_idx={cluster_idx} "
            f"(rc={result.returncode}). Tail of {per_vec_log}:\n    {tail_str}"
        )

    # optimize_steering_vectors.py writes to its own canonical path
    # (`{model}_{tag}_{steering_type}.pt`, e.g. `_linear.pt`). For
    # non-sae arches we rename to the arch-keyed path so disk layout
    # matches our resume / hybrid-swap conventions. SAE writes through
    # to the bare path (which would clobber Venhoff's shipped vectors)
    # — so for SAE we always reuse_shipped above and never reach here.
    if cfg.arch != "sae":
        venhoff_default = _venhoff_subprocess_output_path(venhoff_root, cfg.base_model, cluster_idx)
        if venhoff_default.exists() and venhoff_default != expected_out:
            if expected_out.exists():
                expected_out.unlink()
            venhoff_default.rename(expected_out)
            log.info("[info] steering | renamed | %s -> %s",
                     venhoff_default.name, expected_out.name)

    if not expected_out.exists():
        raise FileNotFoundError(f"expected steering vector not produced at {expected_out}")

    # Record our own resume sidecar next to the vector (under our run_dir).
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    write_with_metadata(meta_path, json.dumps({"vector_path": str(expected_out)}), meta)
    log.info("[done] steering | cluster_idx=%d | arch=%s | vector=%s",
             cluster_idx, cfg.arch, expected_out)
    return expected_out


def train_all_vectors(
    venhoff_root: Path,
    cfg: SteeringConfig,
    paths: ArtifactPaths,
    force: bool = False,
    num_gpus: int = 1,
) -> list[Path]:
    """Train bias (idx=-1) + requested cluster vectors.

    Respects cfg.bias_only (only bias) and cfg.cluster_indices (subset
    of cluster idx to train).

    If `num_gpus > 1`, vectors are trained in parallel — each
    subprocess is pinned to a different GPU via CUDA_VISIBLE_DEVICES.
    Linear speedup since every vector is independent. Returns ordered
    list of output paths (insertion order matches scheduling order,
    which may differ from completion order).
    """
    # Apply vendor patches (drop deprecated load_in_8bit=) before any
    # subprocess fires. Idempotent; no-op on second call.
    ensure_steering_patched(venhoff_root)

    cluster_indices = (
        tuple(range(cfg.n_clusters))
        if cfg.cluster_indices is None
        else cfg.cluster_indices
    )
    # Bias always comes first. Cluster vectors only if not bias-only.
    all_idxs: list[int] = [-1]
    if not cfg.bias_only:
        all_idxs.extend(cluster_indices)
    else:
        log.info("[info] steering | bias_only=True | skipping cluster vectors")

    if num_gpus <= 1 or len(all_idxs) == 1:
        # Serial path — one GPU or just the bias.
        out = []
        for idx in all_idxs:
            out.append(train_one_vector(venhoff_root, cfg, idx, paths, force=force))
        return out

    # Parallel path — fan N vectors across num_gpus processes.
    log.info(
        "[info] steering | multi_gpu | num_gpus=%d | n_vectors=%d",
        num_gpus, len(all_idxs),
    )
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Snapshot config for pickling to worker processes.
    worker_args = [
        (venhoff_root, cfg, idx, paths, force, idx_position % num_gpus)
        for idx_position, idx in enumerate(all_idxs)
    ]
    out: list[Path | None] = [None] * len(all_idxs)
    with ProcessPoolExecutor(max_workers=num_gpus) as pool:
        futures = {
            pool.submit(train_one_vector, vr, c, ci, p, f, g): pos
            for pos, (vr, c, ci, p, f, g) in enumerate(worker_args)
        }
        for fut in as_completed(futures):
            pos = futures[fut]
            path = fut.result()  # re-raises on worker error
            out[pos] = path
    assert all(p is not None for p in out)
    return [p for p in out if p is not None]
