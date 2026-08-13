"""c7 backtracking detection on the diffusion-arm Llama dictionaries.

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_detect.py

Writes /vol/backtracking_eval/detection_results.json on the `diffusion-txc`
volume (committed after every arm, so a timeout still leaves partial results).

w6 trio (window-6 flat dicts from /vol/txc_w6, arms w6_recon/w6_dsm/w6_bayes,
results in detection_results_w6.json — the original file is never touched):

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_detect.py::smoke_w6
    # read logs, check the flatten-order NMSE gate + per-arm L0s, THEN
    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_detect.py::detect_w6

Instrument-sensitivity controls on the w6mix pair (run_controls.py: live-pool-
matched recon subsample, random-dict floor, label-shuffle floor; writes
backtracking_eval/detection_controls.json, smoke writes _smoke variant):

    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_detect.py::smoke_controls
    # volume artifact detection_controls_smoke.json is the evidence, THEN
    uvx modal run --detach \
        experiments/backtracking_detection_dsm/modal_detect.py::detect_controls
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:                                   # inside the container
    ROOT = pathlib.Path("/work")

app = modal.App("backtracking-detection-dsm")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)
hf_secret = modal.Secret.from_name("hf-token")

image = (
    modal.Image.debian_slim(python_version="3.12")
    # scikit-learn pinned: 1.7+ deprecates LogisticRegression(penalty=...) in
    # favour of l1_ratio and warns "inconsistent values" for penalty="l1".
    # The l1 fit is still applied there, but pinning keeps the probe in the
    # reference implementation's unambiguous API.
    .pip_install("torch==2.5.1", "numpy>=2.0", "scipy", "scikit-learn==1.5.2",
                 "transformers==4.46.3", "accelerate", "sentencepiece",
                 "huggingface_hub", "hf_transfer", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm")
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff")
    .add_local_file(
        str(ROOT / "experiments" / "diffusion_txc" / "psc" / "psc_train_sae.py"),
        "/work/experiments/diffusion_txc/psc/psc_train_sae.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "architectures.py"),
        "/work/experiments/ward_backtracking_txc/architectures.py")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "traces.json"),
                    "/work/data/traces.json")
    .add_local_file(
        str(ROOT / "results" / "ward_backtracking" / "sentence_labels.json"),
        "/work/data/sentence_labels.json")
)


@app.function(image=image, gpu="L4", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def detect() -> dict:
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", commit_cb=vol.commit)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def detect_gate() -> dict:
    """Threshold-gate-swap arms only. Split from `detect` so the native-TopK
    table does not have to be recomputed."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", modes=("gate",),
              tag_suffix="_gate", commit_cb=vol.commit)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def detect_w6() -> dict:
    """The w6 trio only (w6_recon / w6_dsm / w6_bayes from /vol/txc_w6).
    Existing arms are not recomputed; writes detection_results_w6.json."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", modes=("topk",), tag_suffix="_w6",
              commit_cb=vol.commit, w6_dir="/vol/txc_w6", w6_only=True)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def detect_w6_mix() -> dict:
    """The mixed-corpus pair (w6_recon / w6_dsm from /vol/txc_w6_mix; no
    bayes arm was trained there). Writes detection_results_w6mix.json."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", modes=("topk",), tag_suffix="_w6mix",
              commit_cb=vol.commit, w6_dir="/vol/txc_w6_mix", w6_only=True)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=5400, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def detect_controls() -> dict:
    """Instrument-sensitivity controls for the w6mix pair (run_controls.py):
    live-pool-matched recon subsample x3, random-dict floor, label-shuffle
    floor x3, plus re-scored w6mix references. Writes
    backtracking_eval/detection_controls.json."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_controls import run

    res = run(device="cuda", vol="/vol", commit_cb=vol.commit,
              w6_dir="/vol/txc_w6_mix")
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=2400, memory=49152,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def smoke_controls(n_traces: int = 20) -> dict:
    """Controls smoke: 20 traces through every new code path (fire counts,
    subsample, random init, shuffle). Writes detection_controls_smoke.json."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_controls import run

    res = run(device="cuda", vol="/vol", limit_traces=n_traces,
              tag_suffix="_smoke", commit_cb=vol.commit,
              w6_dir="/vol/txc_w6_mix", n_draws=2)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=2400, memory=49152,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def smoke_w6(n_traces: int = 20) -> dict:
    """w6-trio smoke: verifies the flatten-order NMSE gate on w6_recon, all
    three checkpoint loads and encode paths, and prints per-arm L0s."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", limit_traces=n_traces,
              modes=("topk",), tag_suffix="_w6_smoke", commit_cb=vol.commit,
              w6_dir="/vol/txc_w6", w6_only=True)
    vol.commit()
    hf_cache.commit()
    return res


@app.function(image=image, gpu="L4", timeout=2400, memory=49152,
              volumes={"/vol": vol, "/hf": hf_cache}, secrets=[hf_secret])
def smoke(n_traces: int = 20, modes: str = "topk,gate") -> dict:
    """End-to-end on n_traces: exercises every checkpoint download, state-dict
    load, hook and encode path before the full run commits an hour of GPU."""
    import sys

    sys.path.insert(0, "/work")
    from experiments.backtracking_detection_dsm.run_detection import run

    res = run(device="cuda", vol="/vol", limit_traces=n_traces,
              tag_suffix="_smoke", commit_cb=vol.commit,
              modes=tuple(modes.split(",")), calib_rows=8000)
    vol.commit()
    hf_cache.commit()
    return res


@app.local_entrypoint()
def main(smoke_first: bool = True):
    if smoke_first:
        s = smoke.remote(20)
        print("SMOKE OK:", json.dumps(
            {k: v for k, v in s.get("example_sets", {}).items()}), flush=True)
        print("SMOKE arms:", list(s.get("arms", {}).keys()), flush=True)
    res = detect.remote()
    outdir = ROOT / "experiments" / "backtracking_detection_dsm" / "results"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "detection_results.json").write_text(json.dumps(res, indent=2,
                                                             default=str))
    print("WROTE", outdir / "detection_results.json", flush=True)
