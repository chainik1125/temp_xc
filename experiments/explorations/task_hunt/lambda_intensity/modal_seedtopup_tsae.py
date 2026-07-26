"""Modal orchestration for the tsae-arm seed top-up (mac-a overnight, 2026-07-26).

Runs `run_stage2_seedtopup_tsae.py` (frozen at PIN, commit-then-run) on
Modal per `briefings/overnight-mac-modal.md`: the image clones the repo
at the PINNED commit and installs via the repo's own `uv sync`;
containers NEVER push git — every scientific step executes through the
repo's venv inside the clone, results are returned/persisted to a
`modal.Volume`, and mac-a merges locally (dup-eval-key check) before
committing.

Stages (local entrypoint `--stage`):
  bringup  — CPU container: in-container `run.py validate` (queue item 1).
  caches   — A10G: re-port traces.json (sha256-pinned), rebuild the Ward
             stream + λ̂ labels + base/hs13 activation cache from the
             COMMITTED builders; hard-fail unless the committed
             byte-identity receipts (`ward_stream_stats.json`,
             `lambda_labels_stats.json`) reproduce git-clean; assert
             `lam_hist_dense` all-finite (the fff7877c NaN-guard no-op
             condition); persist stream+labels+hs13 to the Volume with a
             fresh fingerprint for future audits.
  cells    — 3 parallel A10G/high-CPU containers, one frozen cell each
             (`--only-seed {3,4,5}`); each returns its result JSON + the
             leaderboard rows it appended + persists its checkpoint to
             the Volume (force-majeure lesson: keep weights).

Dirty-tree note: the tree at PIN is clean; the only in-repo write before
cells run is the re-ported `traces.json` (UNTRACKED data, recovered
byte-exactly per its committed ATTRIBUTION.md recipe and verified
against TRACES_SHA256). It is added to `.git/info/exclude` in-container
so `code_version.dirty` keeps meaning "code differs from PIN" — the
receipts (tracked files) are still byte-compared via `git diff`.

Launch (from the mac-a clone root; modal client in the scratchpad venv):
  modal run experiments/explorations/task_hunt/lambda_intensity/modal_seedtopup_tsae.py --stage bringup
"""

import json
import subprocess
import time
from pathlib import Path

import modal

PIN = "c93473ad3482de441f3c13bea2def5c90de3f5cd"  # FREEZE commit of run_stage2_seedtopup_tsae.py
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
# sha256 of origin/aniket-ward-stage-b:results/ward_backtracking/traces.json,
# computed locally from the same blob the ATTRIBUTION.md recipe names.
TRACES_SHA256 = "dc6513e7d3d104de096bb46f52245f5794406ecf745ad5477804e3a2e4e0f9cd"
REPO = "/repo"
VENV_PY = f"{REPO}/.venv/bin/python"

app = modal.App("temp-xc-tsae-seedtopup")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git", "curl", "build-essential")
    .pip_install("uv")
    .run_commands(
        f"git clone {REPO_URL} {REPO}",
        f"cd {REPO} && git checkout {PIN}",
        f"cd {REPO} && uv sync --frozen",
    )
)

vol = modal.Volume.from_name("temp-xc-ward-caches", create_if_missing=True)

ENV = {
    "AGENT_NAME": "mac-a",
    "TOKENIZERS_PARALLELISM": "false",
    "HF_HOME": "/workspace/hf",
}


def _sh(cmd: str, timeout: int | None = None) -> str:
    """Run a shell command in the repo, streaming output; raise on failure."""
    print(f"[sh] {cmd}", flush=True)
    p = subprocess.run(cmd, shell=True, cwd=REPO, text=True, timeout=timeout,
                       capture_output=True, env=_env())
    if p.stdout:
        print(p.stdout[-8000:], flush=True)
    if p.stderr:
        print("[stderr]", p.stderr[-4000:], flush=True)
    if p.returncode != 0:
        raise RuntimeError(f"command failed ({p.returncode}): {cmd}")
    return p.stdout


def _env():
    import os
    e = dict(os.environ)
    e.update(ENV)
    return e


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _report_traces():
    """Re-port traces.json per its committed ATTRIBUTION.md recipe; verify sha."""
    dst = f"{REPO}/results/c7_backtracking/stage_a/traces.json"
    _sh("git show origin/aniket-ward-stage-b:results/ward_backtracking/traces.json "
        f"> {dst}")
    got = _sha256(dst)
    if got != TRACES_SHA256:
        raise RuntimeError(f"traces.json sha256 mismatch: {got}")
    # Untracked data file, recovered byte-exactly — exclude from dirty
    # detection so code_version keeps meaning "code differs from PIN".
    excl = Path(f"{REPO}/.git/info/exclude")
    excl.write_text(excl.read_text() + "\nresults/c7_backtracking/stage_a/traces.json\n")
    print("[traces] re-ported + sha256 verified", flush=True)


def _receipt_clean(tracked_path: str, label: str):
    """Hard-fail unless the builder reproduced the committed receipt byte-identically."""
    p = subprocess.run(f"git diff --exit-code -- {tracked_path}", shell=True,
                       cwd=REPO, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout[-4000:], flush=True)
        raise RuntimeError(f"BYTE-IDENTITY RECEIPT FAILED: {label} ({tracked_path})")
    print(f"[receipt] {label}: byte-identical to committed — PASS", flush=True)


@app.function(image=image, timeout=1800, cpu=4)
def bringup() -> str:
    out = _sh(f"{VENV_PY} run.py validate", timeout=1500)
    sha = _sh("git rev-parse HEAD").strip()
    assert sha == PIN, f"container not at PIN: {sha}"
    return f"PIN={sha}\n{out[-2000:]}"


@app.function(image=image, gpu="A10G", cpu=8, memory=49152, timeout=7200,
              ephemeral_disk=150_000, volumes={"/vol": vol})
def build_caches() -> dict:
    t0 = time.time()
    Path("/workspace").mkdir(exist_ok=True)
    _report_traces()

    _sh(f"{VENV_PY} -m experiments.explorations.conversion_depth.build_ward_stream",
        timeout=3600)
    _receipt_clean("experiments/explorations/conversion_depth/results/ward_stream_stats.json",
                   "ward stream stats")

    _sh(f"{VENV_PY} -m experiments.explorations.task_hunt.lambda_intensity.build_labels",
        timeout=1800)
    _receipt_clean("experiments/explorations/task_hunt/lambda_intensity/results/lambda_labels_stats.json",
                   "lambda labels stats")
    _sh(f"{VENV_PY} -c \"import numpy as np; a = np.load('/workspace/task_hunt_labels/"
        "lambda_intensity/lam_hist_dense.npy'); assert np.isfinite(a).all(), "
        "'lam_hist_dense has non-finite entries'; print('[labels] lam_hist_dense "
        "all-finite:', a.shape, float(a.mean()), float(a.std()))\"")

    _sh(f"{VENV_PY} -m experiments.explorations.conversion_depth.cache_depth base",
        timeout=5400)

    gpu_info = _sh(f"{VENV_PY} -c \"import torch; print(torch.cuda.get_device_name(0), "
                   "torch.__version__, torch.version.cuda)\"").strip()
    hs13 = "/workspace/conv_depth_caches/base/hs13.npy"
    fp = {
        "hs13_sha256": _sha256(hs13),
        "hs13_bytes": Path(hs13).stat().st_size,
        "gpu": gpu_info,
        "stats": _sh(f"{VENV_PY} -c \"import numpy as np; a = np.load('{hs13}', "
                     "mmap_mode='r'); import json; print(json.dumps({'shape': list(a.shape), "
                     "'dtype': str(a.dtype), 'slice_mean': float(np.asarray(a[:64], "
                     "dtype=np.float64).mean()), 'slice_std': float(np.asarray(a[:64], "
                     "dtype=np.float64).std())}))\"").strip(),
    }

    import shutil
    for src, dst in [
        ("/workspace/conv_depth_caches/ward_stream", "/vol/conv_depth_caches/ward_stream"),
        ("/workspace/task_hunt_labels/lambda_intensity", "/vol/task_hunt_labels/lambda_intensity"),
    ]:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    Path("/vol/conv_depth_caches/base").mkdir(parents=True, exist_ok=True)
    shutil.copy2(hs13, "/vol/conv_depth_caches/base/hs13.npy")
    shutil.copy2("/workspace/conv_depth_caches/base/meta.json",
                 "/vol/conv_depth_caches/base/meta.json")
    (Path("/vol") / "cache_fingerprint.json").write_text(json.dumps(fp, indent=2))
    vol.commit()
    fp["elapsed_s"] = round(time.time() - t0)
    print(f"[caches] DONE in {fp['elapsed_s']}s", flush=True)
    return fp


@app.function(image=image, gpu="A10G", cpu=8, memory=65536, timeout=19800,
              volumes={"/vol": vol})
def train_cell(seed: int) -> dict:
    t0 = time.time()
    import shutil
    vol.reload()
    # Local copies (the datasource mmaps + full-reads; keep it off FUSE).
    for src, dst in [
        ("/vol/conv_depth_caches/base", "/workspace/conv_depth_caches/base"),
        ("/vol/task_hunt_labels/lambda_intensity", "/workspace/task_hunt_labels/lambda_intensity"),
    ]:
        shutil.copytree(src, dst, dirs_exist_ok=True)
    _report_traces()  # not read by training; keeps the repo state uniform across stages

    lb = Path(f"{REPO}/results/leaderboard.jsonl")
    n0 = len(lb.read_text().splitlines())
    ckpt_before = {p.name for p in Path(f"{REPO}/checkpoints").glob("*") if p.is_dir()}

    _sh(f"{VENV_PY} -m experiments.explorations.task_hunt.lambda_intensity."
        f"run_stage2_seedtopup_tsae 1 --only-seed {seed}", timeout=19000)

    new_rows = lb.read_text().splitlines()[n0:]
    out_json = json.loads(Path(
        f"{REPO}/experiments/explorations/task_hunt/lambda_intensity/results/"
        f"stage2_seedtopup_tsae_ward_real_lambda_base_l12_s{seed}.json").read_text())

    saved = []
    for p in Path(f"{REPO}/checkpoints").glob("*"):
        if p.is_dir() and p.name not in ckpt_before:
            shutil.copytree(p, f"/vol/checkpoints_topup/{p.name}", dirs_exist_ok=True)
            saved.append(p.name)
    vol.commit()

    return {"seed": seed, "results": out_json, "leaderboard_rows": new_rows,
            "checkpoints_saved": saved, "elapsed_s": round(time.time() - t0)}


@app.local_entrypoint()
def main(stage: str = "bringup", out: str = ""):
    t0 = time.time()
    if stage == "bringup":
        print(bringup.remote())
    elif stage == "caches":
        print(json.dumps(build_caches.remote(), indent=2))
    elif stage == "cells":
        payloads = list(train_cell.map([3, 4, 5], return_exceptions=True))
        results = []
        for p in payloads:
            if isinstance(p, Exception):
                print(f"[cells] FAILED: {p!r}", flush=True)
                results.append({"error": repr(p)})
            else:
                print(f"[cells] seed {p['seed']} ok in {p['elapsed_s']}s "
                      f"(+{len(p['leaderboard_rows'])} lb rows, "
                      f"ckpts {p['checkpoints_saved']})", flush=True)
                results.append(p)
        dst = Path(out or "modal_cells_payload.json")
        dst.write_text(json.dumps(results, indent=2))
        print(f"[cells] payload -> {dst}", flush=True)
    else:
        raise SystemExit(f"unknown stage {stage}")
    print(f"[{stage}] wall {time.time()-t0:.0f}s", flush=True)
