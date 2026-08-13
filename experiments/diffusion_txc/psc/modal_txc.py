"""TXC (T=6 window) trio on Llama-3.1-8B resid_L10: recon / dsm / bayes_gate.

    uvx modal run --detach experiments/diffusion_txc/psc/modal_txc.py

Matches the stage-B TXC site and window budget (resid_L10, T=6, H=16384,
window L0=96; flat TopK-96 over the window rather than per-position
BatchTopK — noted as an architectural delta). Window layout is time-major
(T, d) flattened. One seed per arm.
"""

import json
import pathlib
import subprocess

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-txc-w6")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "datasets==3.1.0", "zstandard", "sentencepiece", "accelerate",
                 "huggingface_hub")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "psc"),
                   "/work/psc")
)

ARMS = {
    "txc_recon": ["--arm", "recon"],
    "txc_dsm": ["--arm", "dsm"],
    "txc_bayes": ["--arm", "bayes_gate", "--target-l0", "96",
                  "--sparsity-coef", "0.05"],
}


@app.function(image=image, gpu="L40S", timeout=21600, volumes={"/vol": vol},
              secrets=[hf_secret], memory=16384)
def train(name: str) -> dict:
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
    cmd = ["python3", "/work/psc/psc_train_sae.py",
           "--model", "NousResearch/Meta-Llama-3.1-8B", "--hook", "resid10",
           "--window", "6", "--seed", "0", "--steps", "20000",
           "--k", "96", "--dataset", "fineweb",
           "--out", f"/vol/txc_w6/{name}",
           "--hf-repo", "dmanningcoe/diffusion-topk-saes",
           "--hf-subdir", "llama31-8b-residL10-txc6", *ARMS[name]]
    t0 = time.time()
    rc = subprocess.run(cmd, env={
        **os.environ, "PYTHONUNBUFFERED": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}).returncode
    vol.commit()
    print(f"[{name}] rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    return {"name": name, "rc": rc, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    calls = {n: train.spawn(n) for n in ARMS}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    for n, c in calls.items():
        print("DONE", json.dumps(c.get()), flush=True)
    print("ALL DONE", flush=True)
