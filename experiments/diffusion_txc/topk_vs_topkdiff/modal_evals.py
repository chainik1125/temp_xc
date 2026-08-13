"""Modal harness for the detection-eval round.

    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_evals.py::main

Prep job caches concept-dataset activations, then 4 eval jobs (one per
checkpoint) run sparse probing / absorption / fragility / judged
autointerp. Results JSONs committed to the diffusion-txc volume.
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("topkdiff-evals")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
judge_secret = modal.Secret.from_name("em-sprint-judges")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy", "scikit-learn", "transformers==4.46.2",
                 "datasets==3.1.0", "accelerate", "sentencepiece", "zstandard")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc"),
                   "/work/experiments/diffusion_txc")
)

ARMS = [("recon", 0), ("recon", 1), ("dsm", 0), ("dsm", 1)]

# The 100M-token scale-up adds the annealed-sigma arm. Note the arm string is
# "dsm_anneal" (not "dsmann") because psc_train_sae.py tags artifacts by --arm.
SCALEUP_ARMS = ARMS + [("dsm_anneal", 0), ("dsm_anneal", 1)]
SCALEUP_CKPT_DIR = "logs_100M"

# Seeds are de-collided across backends so PSC and Modal both completing gives
# extra seeds rather than duplicates. Backend is therefore confounded with seed
# group; every results JSON records which trained it so the morning table can
# never mix a 10M checkpoint with a 100M one, or attribute a seed to the wrong
# machine. Same program and recipe on both sides.
SEED_BACKEND = {0: "modal", 1: "modal", 2: "psc", 3: "psc"}


def _setup():
    import sys

    sys.path.insert(0, "/work")


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/vol": vol},
              secrets=[hf_secret])
def prep() -> dict:
    _setup()
    from experiments.diffusion_txc.topk_vs_topkdiff.evals_cache import run

    res = run(out_dir="/vol")
    vol.commit()
    return res


@app.function(image=image, gpu="L4", timeout=5400, volumes={"/vol": vol},
              secrets=[hf_secret, judge_secret], memory=24576)
def evaluate(arm: str, seed: int, ckpt_dir: str = "logs",
             budget: str = "10M", base_model: str = "gemma-2-2b") -> dict:
    import time

    _setup()
    from experiments.diffusion_txc.topk_vs_topkdiff.run_evals import run

    t0 = time.time()
    res = run(device="cuda", vol="/vol", arm=arm, seed=seed, ckpt_dir=ckpt_dir)
    res["_runtime_s"] = round(time.time() - t0, 1)
    res["provenance"] = {"backend": SEED_BACKEND.get(seed, "unknown"),
                         "budget": budget, "base_model": base_model,
                         "ckpt_dir": ckpt_dir}
    dest = pathlib.Path("/vol") / ckpt_dir / f"evals_{arm}_s{seed}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(res))
    vol.commit()
    print(f"[evals {arm} s{seed}] done in {res['_runtime_s']}s", flush=True)
    return res


@app.local_entrypoint()
def one(arm: str = "recon", seed: int = 0, ckpt_dir: str = "logs"):
    """Single checkpoint — used to validate the eval path before a full round."""
    print("RESULT:", json.dumps(evaluate.remote(arm, seed, ckpt_dir))[:2000],
          flush=True)


@app.local_entrypoint()
def evals_only(ckpt_dir: str = "logs", scaleup: bool = False,
               seeds: str = "", budget: str = ""):
    """`seeds` overrides the seed list — don't assume 0/1, since which seeds
    exist depends on which backend actually finished."""
    arms = SCALEUP_ARMS if scaleup else ARMS
    if scaleup and ckpt_dir == "logs":
        ckpt_dir = SCALEUP_CKPT_DIR
    if seeds:
        want = {int(s) for s in seeds.split(",")}
        arms = [(a, s) for a, s in arms if s in want]
    budget = budget or ("100M" if scaleup else "10M")
    calls = {f"evals_{a}_s{s}": evaluate.spawn(a, s, ckpt_dir, budget)
             for a, s in arms}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    outdir = ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for n, c in calls.items():
        try:
            res = c.get()
        except Exception as e:                                # noqa: BLE001
            print(f"FAILED {n}: {e}", flush=True)
            continue
        (outdir / f"{n}_{ckpt_dir.replace('/', '_')}.json").write_text(
            json.dumps(res, indent=2))
        print(f"DONE {n}", flush=True)
    print("ALL DONE", flush=True)


@app.local_entrypoint()
def main():
    print("PREP:", json.dumps(prep.remote()), flush=True)
    calls = {f"evals_{a}_s{s}": evaluate.spawn(a, s) for a, s in ARMS}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    outdir = ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    for n, c in calls.items():
        res = c.get()
        (outdir / f"{n}.json").write_text(json.dumps(res, indent=2))
        print(f"DONE {n}", flush=True)
    print("ALL DONE", flush=True)
