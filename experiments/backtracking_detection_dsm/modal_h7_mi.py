"""H7 -- l0-matching vs MI-matching (run_h7_mi.py) on Modal.

Writes /vol/backtracking_eval/h7_mi_matching.json on the `diffusion-txc`
volume (flushed + committed after every arm; the volume artifact is the
launch evidence).

Modal launch traps (all five bit this project before): spawn-style
entrypoints still need --detach; name the entrypoint explicitly (::main);
ONE spawned function per detached invocation; the volume artifact -- not the
local log -- is the launch evidence; pollers do not notify a stopped
session, so the caller drives with sleeps in its own shell.

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_h7_mi.py::main
"""

import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("h7-mi-matching")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache",
                                  create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # scikit-learn pinned for the reference probe API (see modal_detect.py)
    .pip_install("torch==2.5.1", "numpy>=2.0", "scipy", "scikit-learn==1.5.2",
                 "transformers==4.46.3", "accelerate", "sentencepiece",
                 "huggingface_hub", "hf_transfer", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" /
                       "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" /
                        "traces.json"),
                    "/work/data/traces.json")
    .add_local_file(
        str(ROOT / "results" / "ward_backtracking" / "sentence_labels.json"),
        "/work/data/sentence_labels.json")
)


@app.function(image=image, gpu="L4", timeout=10800, memory=65536, cpu=16.0,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def h7() -> dict:
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_h7_mi import run

    res = run(device="cuda", vol="/vol", commit_cb=vol.commit,
              w6_dir="/vol/txc_w6_mix")
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=3600, memory=49152, cpu=8.0,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def h7_smoke(n_traces: int = 20) -> dict:
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_h7_mi import run

    res = run(device="cuda", vol="/vol", limit_traces=n_traces,
              tag_suffix="_smoke", commit_cb=vol.commit,
              w6_dir="/vol/txc_w6_mix")
    vol.commit()
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(smoke: bool = False):
    call = (h7_smoke if smoke else h7).spawn()
    print("SPAWNED h7:", call.object_id, flush=True)
