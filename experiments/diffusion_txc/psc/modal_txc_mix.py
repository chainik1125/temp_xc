"""TXC (T=6 window) on a 50/50 trace+FineWeb document mix: dsm + recon arms.

Same recipe as the w6 trio (modal_txc.py: Llama-3.1-8B resid_L10, window 6,
k=96, H=16384) but --dataset mixed_traces interleaves ward backtracking trace
documents (prompt + full_response) 50/50 at the document level with the
fineweb stream. One seed per arm, 8000 steps.

LAUNCH DISCIPLINE: one spawned function per detached invocation —

    uvx modal run --detach experiments/diffusion_txc/psc/modal_txc_mix.py::main --job w6mix_dsm
    uvx modal run --detach experiments/diffusion_txc/psc/modal_txc_mix.py::main --job w6mix_recon

Smoke first (foreground, 250 steps, separate out dir, no HF push):

    uvx modal run experiments/diffusion_txc/psc/modal_txc_mix.py::main --job w6mix_dsm --smoke
"""

import json
import pathlib
import subprocess

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-txc-w6-mix")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "datasets==3.1.0", "zstandard", "sentencepiece", "accelerate",
                 "huggingface_hub")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "psc"),
                   "/work/psc")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "traces.json"),
                    "/work/traces.json")
)

LLAMA = "NousResearch/Meta-Llama-3.1-8B"
DISTILL = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

# job -> (out subdir, arm args, capture model, volume prefix, hf subdir)
# w6dist_*: identical recipe/stream to w6mix_*, ONLY the capture model
# changes — the theory note's pre-registered discriminating cell (P1/P2).
JOBS = {
    "w6mix_dsm": ("txc_dsm", ["--arm", "dsm"], LLAMA,
                  "txc_w6_mix", "llama31-8b-residL10-txc6-mix"),
    "w6mix_recon": ("txc_recon", ["--arm", "recon"], LLAMA,
                    "txc_w6_mix", "llama31-8b-residL10-txc6-mix"),
    "w6dist_dsm": ("txc_dsm", ["--arm", "dsm"], DISTILL,
                   "txc_w6_distill", "r1-distill-8b-residL10-txc6-mix"),
    "w6dist_recon": ("txc_recon", ["--arm", "recon"], DISTILL,
                     "txc_w6_distill", "r1-distill-8b-residL10-txc6-mix"),
}


@app.function(image=image, gpu="A100-80GB", timeout=21600, volumes={"/vol": vol},
              secrets=[hf_secret], memory=16384)
def train(job: str, steps: int = 8000, out_sub: str = "") -> dict:
    import os
    import threading
    import time

    def committer():
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception:                                 # noqa: BLE001
                pass

    threading.Thread(target=committer, daemon=True).start()
    sub, arm_args, model, prefix, hf_subdir = JOBS[job]
    out_sub = out_sub or sub
    cmd = ["python3", "/work/psc/psc_train_sae.py",
           "--model", model, "--hook", "resid10",
           "--window", "6", "--seed", "0", "--steps", str(steps),
           "--k", "96", "--dataset", "mixed_traces",
           "--traces-path", "/work/traces.json",
           "--out", f"/vol/{prefix}/{out_sub}", *arm_args]
    if not out_sub.startswith("smoke"):
        cmd += ["--hf-repo", "dmanningcoe/diffusion-topk-saes",
                "--hf-subdir", hf_subdir]
    t0 = time.time()
    rc = subprocess.run(cmd, env={
        **os.environ, "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
    vol.commit()
    print(f"[{job}] rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    return {"job": job, "out_sub": out_sub, "steps": steps, "rc": rc,
            "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main(job: str = "", smoke: bool = False):
    if job not in JOBS:
        raise SystemExit(f"--job must be one of {sorted(JOBS)}")
    if smoke:
        call = train.spawn(job, 250, f"smoke_{JOBS[job][0]}")
    else:
        call = train.spawn(job)
    print("SPAWNED:", json.dumps({job: call.object_id}), flush=True)
    print("DONE", json.dumps(call.get()), flush=True)
