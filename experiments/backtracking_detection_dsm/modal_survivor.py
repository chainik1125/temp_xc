"""Survivor-latent decomposition of the w6mix DSM dictionary (survivor_interp).

LAUNCH (entrypoint MUST be named -- bare `modal run file.py` with >=2
entrypoints prints the list and exits 1, which looks like a launch in a
detached log; and each entrypoint .spawn()s exactly ONE remote function and
exits in seconds, so the local client's fate never holds the app's):

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_survivor.py::smoke
    # volume artifact backtracking_eval/survivor_interp_smoke.json is the
    # launch evidence; check it (judge scores present, dumps rendered), THEN
    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_survivor.py::main

Writes on the `diffusion-txc` volume (committed after every block):
    backtracking_eval/survivor_interp{_smoke}.json
    backtracking_eval/survivor_interp/dumps_{group}{_smoke}.json
"""

import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:                                   # inside the container
    ROOT = pathlib.Path("/work")

app = modal.App("survivor-interp-dsm")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache",
                                  create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")
judge_secret = modal.Secret.from_name("em-sprint-judges")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # scikit-learn pinned for the reference probe API (see modal_detect.py)
    .pip_install("torch==2.5.1", "numpy>=2.0", "scipy", "scikit-learn==1.5.2",
                 "transformers==4.46.3", "accelerate", "sentencepiece",
                 "huggingface_hub", "hf_transfer", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "traces.json"),
                    "/work/data/traces.json")
    .add_local_file(
        str(ROOT / "results" / "ward_backtracking" / "sentence_labels.json"),
        "/work/data/sentence_labels.json")
)


@app.function(image=image, gpu="L4", timeout=10800, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[hf_secret, judge_secret])
def survivor_job() -> dict:
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.survivor_interp import run

    res = run(device="cuda", vol="/vol", commit_cb=vol.commit)
    vol.commit()
    hf_cache.commit()
    return {"runtime_s": res.get("_runtime_s")}


@app.function(image=image, gpu="L4", timeout=3600, memory=49152,
              volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[hf_secret, judge_secret])
def survivor_smoke(n_traces: int = 20) -> dict:
    """20-trace smoke through every code path: capture, gate, fire counts,
    mass slices, single-latent probes (capped 40/group), capacity draws,
    corr, dump rendering, and 3-latent-per-group judging (verifies the
    em-sprint-judges secret end to end)."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.survivor_interp import run

    res = run(device="cuda", vol="/vol", limit_traces=n_traces,
              tag_suffix="_smoke", commit_cb=vol.commit,
              n_auto=3, min_fires=5, max_single=40)
    vol.commit()
    hf_cache.commit()
    return {"runtime_s": res.get("_runtime_s")}


@app.local_entrypoint()
def smoke():
    call = survivor_smoke.spawn()
    print(f"SPAWNED survivor_smoke call_id={call.object_id}", flush=True)
    print("evidence = volume artifact "
          "backtracking_eval/survivor_interp_smoke.json", flush=True)


@app.local_entrypoint()
def main():
    call = survivor_job.spawn()
    print(f"SPAWNED survivor_job call_id={call.object_id}", flush=True)
    print("evidence = volume artifact "
          "backtracking_eval/survivor_interp.json", flush=True)
