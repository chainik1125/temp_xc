"""Modal fallback for the PSC Bridges-2 SAE training jobs.

Runs `psc_train_sae.py` unchanged (as a subprocess, same CLI args as the
sbatch files) so the Modal and PSC arms are the same program. Artifacts go
to the `diffusion-txc` volume under /vol/psc_fallback/<name>/ and are pushed
to HF by the training script itself.

    uvx modal run --detach experiments/diffusion_txc/psc/modal_fallback.py::smoke
    uvx modal run --detach experiments/diffusion_txc/psc/modal_fallback.py::gemma6

The volume is committed every 5 min from a background thread so a container
loss costs at most one commit interval of progress.
"""

import json
import pathlib
import shlex

import modal

try:
    HERE = pathlib.Path(__file__).resolve().parent
except NameError:
    HERE = pathlib.Path("/work/psc")

app = modal.App("dtxc-psc-fallback")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
hf_write = modal.Secret.from_name("hf-write-dmc")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "datasets==3.1.0", "zstandard", "sentencepiece", "accelerate",
                 "huggingface_hub")
    .add_local_dir(str(HERE), "/work/psc")
)

# HF cache stays on the container's own disk. Pointing HF_HOME at the shared
# volume would have six containers writing the same snapshot concurrently,
# and volume writes aren't visible across containers until commit anyway.
ENV = {
    "HF_HOME": "/root/.cache/huggingface",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
}

HF_REPO = "dmanningcoe/diffusion-topk-saes"

# Mirrors experiments/diffusion_txc/psc/jobs/{gemma,llama}_*.sbatch exactly.
GEMMA_COMMON = ["--model", "google/gemma-2-2b", "--hook", "resid12",
                "--steps", "24000", "--k", "40", "--dataset", "pile"]
GEMMA_SUBDIR = "gemma2-2b-l12-100M"
GEMMA_ARMS = (("dsm", "dsm"), ("dsmann", "dsm_anneal"), ("recon", "recon"))

LLAMA_COMMON = ["--model", "NousResearch/Meta-Llama-3.1-8B", "--hook", "ln110",
                "--steps", "20000", "--k", "64", "--dataset", "fineweb"]
LLAMA_SUBDIR = "llama31-8b-ln1L10-20k"
LLAMA_ARMS = (("dsm", "dsm"), ("recon", "recon"))


def _jobs(family: str, seeds: tuple[int, ...]):
    common, arms = ((GEMMA_COMMON, GEMMA_ARMS) if family == "gemma"
                    else (LLAMA_COMMON, LLAMA_ARMS))
    return [(f"{family}_{tag}_s{seed}",
             common + ["--arm", arm, "--seed", str(seed)])
            for tag, arm in arms for seed in seeds]


def _committer(stop):
    """Commit the volume every 5 min so a lost container costs <=5 min."""
    import threading
    import time

    def loop():
        while not stop.wait(300):
            try:
                vol.commit()
            except Exception as e:                            # noqa: BLE001
                print(f"[commit] failed: {e}", flush=True)

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    return t


def _run(name: str, extra: list[str], subdir: str, push: bool = True) -> dict:
    import os
    import subprocess
    import threading
    import time

    out = pathlib.Path("/vol/psc_fallback") / name
    out.mkdir(parents=True, exist_ok=True)
    
    cmd = ["python", "/work/psc/psc_train_sae.py", *extra, "--out", str(out),
           "--hf-repo", HF_REPO if push else "", "--hf-subdir", subdir]
    print(f"[{name}] {shlex.join(cmd)}", flush=True)

    stop = threading.Event()
    _committer(stop)
    t0 = time.time()
    proc = subprocess.run(cmd, env={**os.environ, **ENV})
    stop.set()
    wall = round(time.time() - t0, 1)

    final = out / f"{extra[extra.index('--arm') + 1]}_s{extra[extra.index('--seed') + 1]}_final.json"
    res = {"name": name, "returncode": proc.returncode, "wall_s": wall,
           "final_json": json.loads(final.read_text()) if final.exists() else None}
    (out / "modal_result.json").write_text(json.dumps(res, indent=2))
    vol.commit()
    print(f"[{name}] rc={proc.returncode} wall={wall}s", flush=True)
    return res


@app.function(image=image, gpu="A10G", timeout=43200, volumes={"/vol": vol},
              secrets=[hf_secret, hf_write], memory=16384)
def train_a10g(name: str, extra: list[str], subdir: str,
               push: bool = True) -> dict:
    return _run(name, extra, subdir, push)


@app.function(image=image, gpu="L40S", timeout=43200, volumes={"/vol": vol},
              secrets=[hf_secret, hf_write], memory=16384)
def train_l40s(name: str, extra: list[str], subdir: str,
               push: bool = True) -> dict:
    return _run(name, extra, subdir, push)


@app.function(image=image, gpu="A100-40GB", timeout=43200,
              volumes={"/vol": vol}, secrets=[hf_secret, hf_write],
              memory=32768)
def train_a100(name: str, extra: list[str], subdir: str,
               push: bool = True) -> dict:
    return _run(name, extra, subdir, push)


FNS = {"A10G": train_a10g, "L40S": train_l40s, "A100-40GB": train_a100}


@app.function(image=image, timeout=3600, volumes={"/vol": vol},
              secrets=[hf_secret, hf_write], memory=8192)
def stage(ckpt_dir: str = "logs_100M", subdir: str = GEMMA_SUBDIR,
          arms: list[str] | None = None, run_prefix: str = "gemma_") -> dict:
    """Collect trained checkpoints into <ckpt_dir> under the names run_evals
    expects ({arm}_s{seed}_final.pt), preferring the Modal fallback's own
    output and falling back to the HF repo for anything PSC produced.

    Never writes into `logs/` itself: that holds the earlier 10M-token
    signs-of-life checkpoints and their eval JSONs.

    `run_prefix` is load-bearing: short smoke runs write the same
    {arm}_s{seed}.pt basename into their own /vol/psc_fallback/smoke_* dir,
    so an unqualified glob can silently stage a 500-step SAE as the 24k-step
    result. Only full-run directories are eligible, and an ambiguous match is
    reported rather than resolved arbitrarily.
    """
    import shutil

    arms = arms or [f"{a}_s{s}" for a in ("recon", "dsm", "dsm_anneal")
                    for s in (0, 1)]
    dest = pathlib.Path("/vol") / ckpt_dir
    dest.mkdir(parents=True, exist_ok=True)
    got = {}
    for tag in arms:
        target = dest / f"{tag}_final.pt"
        if target.exists():
            got[tag] = "already-staged"
            continue
        local = sorted(pathlib.Path("/vol/psc_fallback").glob(
            f"{run_prefix}*/{tag}.pt"))
        if len(local) > 1:
            got[tag] = f"AMBIGUOUS: {[str(p) for p in local]}"
            continue
        if local:
            shutil.copy(local[0], target)
            got[tag] = f"modal:{local[0]}"
            continue
        try:
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(repo_id=HF_REPO, filename=f"{subdir}/{tag}.pt")
            shutil.copy(p, target)
            got[tag] = "hf"
        except Exception as e:                                # noqa: BLE001
            got[tag] = f"MISSING ({type(e).__name__})"
    vol.commit()
    print(json.dumps(got, indent=2), flush=True)
    return got


@app.function(image=image, timeout=900, volumes={"/vol": vol}, memory=4096)
def collect() -> dict:
    """Every training run's final metrics, straight off the volume."""
    root = pathlib.Path("/vol/psc_fallback")
    out = {}
    for d in sorted(root.glob("*")):
        if not d.is_dir() or d.name.startswith("smoke"):
            continue
        rec = {}
        for f in d.glob("*_final.json"):
            rec["final"] = json.loads(f.read_text())
        f = d / "modal_result.json"
        if f.exists():
            r = json.loads(f.read_text())
            rec["returncode"], rec["wall_s"] = r.get("returncode"), r.get("wall_s")
        # last logged row, so a still-running job still reports progress
        for f in d.glob("*.jsonl"):
            lines = [x for x in f.read_text().splitlines() if x.strip()]
            if lines:
                rec["last_row"] = json.loads(lines[-1])
        rec["ckpt_bytes"] = {p.name: p.stat().st_size for p in d.glob("*.pt")}
        out[d.name] = rec
    return out


@app.local_entrypoint()
def collect_only():
    print("COLLECT:", json.dumps(collect.remote(), indent=2), flush=True)


@app.local_entrypoint()
def stage_only(ckpt_dir: str = "logs_100M"):
    print("STAGED:", json.dumps(stage.remote(ckpt_dir), indent=2), flush=True)


@app.local_entrypoint()
def smoke(gpu: str = "A10G,L40S", steps: int = 500, family: str = "gemma"):
    """Short runs on candidate GPUs to measure steps/s before committing."""
    common = GEMMA_COMMON if family == "gemma" else LLAMA_COMMON
    calls = {}
    for g in gpu.split(","):
        extra = list(common)
        extra[extra.index("--steps") + 1] = str(steps)
        extra += ["--arm", "recon", "--seed", "0"]
        calls[g] = FNS[g].spawn(f"smoke_{family}_{g.lower()}", extra, "", False)
    print("SPAWNED:", json.dumps({g: c.object_id for g, c in calls.items()}),
          flush=True)
    for g, c in calls.items():
        try:
            print(f"SMOKE {g}:", json.dumps(c.get()), flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"SMOKE {g} FAILED: {type(e).__name__}: {e}", flush=True)


def _launch(family: str, gpu: str, seeds: tuple[int, ...], subdir: str):
    fn = FNS[gpu]
    jobs = _jobs(family, seeds)
    calls = {n: fn.spawn(n, extra, subdir) for n, extra in jobs}
    ids = {n: c.object_id for n, c in calls.items()}
    print("SPAWNED:", json.dumps(ids), flush=True)
    (HERE / f"modal_calls_{family}.json").write_text(json.dumps(
        {"gpu": gpu, "seeds": list(seeds), "backend": "modal", "calls": ids},
        indent=2))
    for n, c in calls.items():
        try:
            print(f"DONE {n}:", json.dumps(c.get()), flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"FAILED {n}: {type(e).__name__}: {e}", flush=True)
    print("ALL DONE", flush=True)


@app.local_entrypoint()
def gemma6(gpu: str = "L40S", seeds: str = "0,1"):
    _launch("gemma", gpu, tuple(int(s) for s in seeds.split(",")), GEMMA_SUBDIR)


@app.local_entrypoint()
def llama4(gpu: str = "A100-40GB", seeds: str = "2,3"):
    _launch("llama", gpu, tuple(int(s) for s in seeds.split(",")), LLAMA_SUBDIR)
