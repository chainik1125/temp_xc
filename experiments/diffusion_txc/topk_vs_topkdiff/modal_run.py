"""Modal harness: cache once, then 4 detached training jobs.

    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_run.py            # cache + train
    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_run.py::train_only # cache exists

Volume `diffusion-txc` holds the cache, JSONL logs, weights, autointerp
dumps. Jobs commit before returning (dropout-proof).
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("topk-vs-topkdiff")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "transformers==4.46.2",
                 "datasets==3.1.0", "accelerate", "sentencepiece", "zstandard")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc"),
                   "/work/experiments/diffusion_txc")
)

ARMS = [("recon", 0), ("recon", 1), ("dsm", 0), ("dsm", 1)]


def _setup():
    import sys

    sys.path.insert(0, "/work")


@app.function(image=image, gpu="A10G", timeout=7200, volumes={"/vol": vol},
              secrets=[hf_secret])
def cache() -> dict:
    _setup()
    from experiments.diffusion_txc.topk_vs_topkdiff.cache_activations import run

    res = run(out_dir="/vol")
    vol.commit()
    return res


@app.function(image=image, gpu="A10G", timeout=10800, volumes={"/vol": vol},
              secrets=[hf_secret], memory=32768,
              env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
def train(arm: str, seed: int) -> dict:
    import time

    _setup()
    from experiments.diffusion_txc.topk_vs_topkdiff.train_arms import run

    t0 = time.time()
    res = run(device="cuda", vol="/vol", arm=arm, seed=seed)
    res["_runtime_s"] = round(time.time() - t0, 1)
    (pathlib.Path("/vol") / "logs" / f"{arm}_s{seed}_final.json").write_text(
        json.dumps(res)
    )
    vol.commit()
    print(f"[{arm} s{seed}] done in {res['_runtime_s']}s", flush=True)
    return res


def _train_all():
    calls = {f"{a}_s{s}": train.spawn(a, s) for a, s in ARMS}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    outdir = ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "call_ids.json").write_text(
        json.dumps({n: c.object_id for n, c in calls.items()}, indent=2)
    )
    for n, c in calls.items():
        res = c.get()
        (outdir / f"{n}_final.json").write_text(json.dumps(res, indent=2))
        print(f"DONE {n}", flush=True)


@app.local_entrypoint()
def main():
    print("CACHE:", json.dumps(cache.remote()), flush=True)
    _train_all()
    print("ALL DONE", flush=True)


@app.local_entrypoint()
def train_only():
    _train_all()
    print("ALL DONE", flush=True)
