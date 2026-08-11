"""Post-hoc gate-swap evals on the six 100M checkpoints.

    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_posthoc.py
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("topkdiff-posthoc")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "scikit-learn", "transformers==4.46.2",
                 "datasets==3.1.0", "accelerate", "sentencepiece", "zstandard")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc"),
                   "/work/experiments/diffusion_txc")
)

ARMS = [(a, s) for a in ("recon", "dsm", "dsm_anneal") for s in (0, 1)]


@app.function(image=image, gpu="L4", timeout=5400, volumes={"/vol": vol},
              secrets=[hf_secret], memory=24576)
def posthoc(arm: str, seed: int) -> dict:
    import sys
    import time

    sys.path.insert(0, "/work")
    from experiments.diffusion_txc.topk_vs_topkdiff.posthoc_gate_evals import run

    t0 = time.time()
    res = run(device="cuda", vol="/vol", arm=arm, seed=seed)
    res["_runtime_s"] = round(time.time() - t0, 1)
    (pathlib.Path("/vol") / "logs_100M" / f"posthoc_{arm}_s{seed}.json").write_text(
        json.dumps(res))
    vol.commit()
    print(f"[posthoc {arm} s{seed}] done in {res['_runtime_s']}s", flush=True)
    return res


@app.local_entrypoint()
def main():
    calls = {f"posthoc_{a}_s{s}": posthoc.spawn(a, s) for a, s in ARMS}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    outdir = ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for n, c in calls.items():
        try:
            res = c.get()
            (outdir / f"{n}.json").write_text(json.dumps(res, indent=2))
            print(f"DONE {n}", flush=True)
        except Exception as e:                                # noqa: BLE001
            print(f"FAILED {n}: {e}", flush=True)
    print("ALL DONE", flush=True)
