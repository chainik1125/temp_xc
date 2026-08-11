"""Mine the best backtracking feature per wave-1 dictionary.

    uvx modal run --detach experiments/backtracking_steering_dsm/modal_mine.py

Writes backtracking_eval/steering/features/<arm>.npz (top-8 features with their
decoder rows) plus features/mining_summary.json on the `diffusion-txc` volume.
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[2]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("backtracking-steering-mine")
vol = modal.Volume.from_name("diffusion-txc", create_if_missing=True)
hf_cache = modal.Volume.from_name("sae-deadlatent-hf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.5.1", "numpy>=2.0", "scipy", "scikit-learn==1.5.2",
                 "transformers==4.46.3", "accelerate", "sentencepiece",
                 "huggingface_hub", "hf_transfer", "pyyaml")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_dir(str(ROOT / "experiments" / "backtracking_steering_dsm"),
                   "/work/experiments/backtracking_steering_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "backtracking_detection_dsm"),
                   "/work/experiments/backtracking_detection_dsm",
                   ignore=["__pycache__"])
    .add_local_dir(str(ROOT / "experiments" / "diffusion_txc" / "topk_vs_topkdiff"),
                   "/work/experiments/diffusion_txc/topk_vs_topkdiff",
                   ignore=["__pycache__"])
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "architectures.py"),
        "/work/experiments/ward_backtracking_txc/architectures.py")
    .add_local_file(
        str(ROOT / "experiments" / "ward_backtracking_txc" / "b1_steer_eval.py"),
        "/work/experiments/ward_backtracking_txc/b1_steer_eval.py")
    .add_local_dir(str(ROOT / "temporal_crosscoders"), "/work/temporal_crosscoders",
                   ignore=["__pycache__"])
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "traces.json"),
                    "/work/data/traces.json")
    .add_local_file(
        str(ROOT / "results" / "ward_backtracking" / "sentence_labels.json"),
        "/work/data/sentence_labels.json")
    .add_local_file(str(ROOT / "results" / "ward_backtracking" / "prompts.json"),
                    "/work/data/prompts.json")
)

OUT = "/vol/backtracking_eval/steering/features"


@app.function(image=image, gpu="L4", timeout=7200, memory=65536,
              volumes={"/vol": vol, "/hf": hf_cache},
              secrets=[modal.Secret.from_name("hf-token")])
def mine() -> dict:
    import sys
    import time

    sys.path.insert(0, "/work")
    import numpy as np
    import torch
    from experiments.backtracking_detection_dsm.detect_core import gather_windows
    from experiments.backtracking_steering_dsm import steer_core

    t0 = time.time()
    from experiments.backtracking_detection_dsm.detect_core import capture_traces
    cache, offsets, trace_meta, cap_meta = capture_traces(
        "/work/data/traces.json", "/work/data/sentence_labels.json")
    print(f"[capture] {cap_meta}", flush=True)

    prompts = json.loads(pathlib.Path("/work/data/prompts.json").read_text())
    dom_qids = {p["id"] for p in prompts if p.get("split", "dom") == "dom"}
    examples = steer_core.dom_example_set(trace_meta, dom_qids)
    is_bt = np.asarray([e[2] for e in examples], dtype=bool)
    print(f"[examples] {len(examples)} dom sentences, {int(is_bt.sum())} D+",
          flush=True)

    windows = {h: gather_windows(cache[h], offsets, examples)
               for h in ("ln1", "resid")}
    for h, w in windows.items():
        print(f"[windows] {h}: {tuple(w.shape)}", flush=True)

    feats, summary = [], []
    for arm in steer_core.WAVE1_ARMS:
        model, meta = steer_core.load_arm(arm)
        f = steer_core.mine_arm(model, meta, windows[arm["hook"]], is_bt)
        feats.append(f)
        summary.append({
            "arm": f["name"], "arch": f["arch"], "hook": f["hook"],
            "d_sae": f["d_sae"], "T": f["T"],
            "top_feature": int(f["top_features"][0]),
            "top_score": float(f["scores"][0]),
            "top_tstat": float(f["tstat"][0]),
            "top8_features": [int(x) for x in f["top_features"]],
            "top8_scores": [float(x) for x in f["scores"]],
            "final_eval_fvu": meta.get("final_eval_fvu"),
        })
        print(f"[mined] {f['name']:18s} arch={f['arch']:14s} "
              f"top_f={int(f['top_features'][0]):6d} "
              f"score={float(f['scores'][0]):+.5f} "
              f"t={float(f['tstat'][0]):+.2f}", flush=True)
        del model
        torch.cuda.empty_cache()

    steer_core.save_features(feats, OUT)
    out = {"arms": summary, "capture": cap_meta,
           "n_examples": len(examples), "n_pos": int(is_bt.sum()),
           "ranking": "meandiff on window latent (mine_features.py)",
           "dom_split_only": True,
           "_runtime_s": round(time.time() - t0, 1)}
    pathlib.Path(OUT + "/mining_summary.json").write_text(json.dumps(out, indent=1))
    vol.commit()
    hf_cache.commit()
    print("ALL DONE", flush=True)
    return out


@app.local_entrypoint()
def main():
    print(json.dumps(mine.remote(), indent=2))
