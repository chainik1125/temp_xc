"""Detection-eval round for the bayes_gate signs-of-life checkpoints.

    uvx modal run experiments/diffusion_txc/topk_vs_topkdiff/modal_evals_bayes.py::sanity
    uvx modal run --detach experiments/diffusion_txc/topk_vs_topkdiff/modal_evals_bayes.py::main

Same eval data/config as the signs-of-life TopK round (modal_evals.py):
cache/meta.json + concepts/ + cache/eval_shard.pt on the diffusion-txc
volume, rms taken from meta.json. Pinned eval semantics (sigma=0 gate,
hard threshold 0.5, shrunk-linear magnitudes) live in bayes_eval_adapter.py.
Results JSONs are committed to the volume under logs_bayes_evals/.
Autointerp skips gracefully: the bayes trainings emit no top-context dump.

Both arms run sequentially in ONE container: Modal's detached mode only
keeps the LAST triggered function alive after the local client dies, so a
launcher that spawns two functions loses one (that killed the first round,
app ap-ZILfDixovAMkzhfruU0I4q, mid-eval with no error in the logs).
Progress is made durable by writing each arm's JSON (with "partial": true
until the arm completes) and vol.commit()ing after every eval section.
"""

import json
import pathlib

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("topkdiff-evals-bayes")
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

CKPTS = {
    "bg6_sol": "bayes_gate/bg6_sol/bayes_gate_s0.pt",
    "bg7_sol": "bayes_gate/bg7_sol/bayes_gate_s0.pt",
}
# u=0 gate>0.5 L0 that psc_train_sae.py logged at its final step (same
# criterion as the pinned eval semantics, so these should nearly match).
TRAIN_L0 = {"bg6_sol": 67.64, "bg7_sol": 48.65}
OUT_DIR = "logs_bayes_evals"


def _setup():
    import sys

    sys.path.insert(0, "/work")


def _sanity(name: str, sae, X) -> dict:
    z = sae.encode(X)
    l0_z = float((z > 0).float().sum(1).mean())
    l0_gate = sae.gate_l0(X)
    ratio = l0_gate / TRAIN_L0[name]
    return {"name": name,
            "d": int(sae.sae.W_enc.shape[0]), "H": int(sae.sae.W_enc.shape[1]),
            "n_rows": int(X.shape[0]),
            "l0_z": l0_z, "l0_gate_half": l0_gate,
            "train_l0_gate_half": TRAIN_L0[name],
            "ratio_vs_train": ratio,
            "flagged_gt2x": bool(ratio > 2.0 or ratio < 0.5)}


@app.function(image=image, timeout=1800, volumes={"/vol": vol},
              cpu=8, memory=32768)
def l0_sanity() -> dict:
    """CPU-only pre-launch check: shapes + mean L0 at sigma=0 for both ckpts."""
    _setup()
    import torch

    from experiments.diffusion_txc.topk_vs_topkdiff.bayes_eval_adapter import (
        load_bayes_sae)

    volp = pathlib.Path("/vol")
    meta = json.loads((volp / "cache" / "meta.json").read_text())
    ev = torch.load(volp / "cache" / "eval_shard.pt", weights_only=True)
    X = ev["acts"][:4096].to(torch.float32)
    out = {}
    for name, rel in CKPTS.items():
        sae = load_bayes_sae(volp / rel, dev="cpu", d=meta["d"])
        out[name] = _sanity(name, sae, X)
        print("SANITY", json.dumps(out[name]), flush=True)
    return out


def _eval_one(name: str, volp: pathlib.Path, meta: dict, tok, dev) -> dict:
    """One arm's full eval round, checkpointing the JSON to the volume after
    every section ("partial": true until the arm is done)."""
    import time

    import torch

    from experiments.diffusion_txc.topk_vs_topkdiff import run_evals
    from experiments.diffusion_txc.topk_vs_topkdiff.bayes_eval_adapter import (
        load_bayes_sae)

    t0 = time.time()
    sae = load_bayes_sae(volp / CKPTS[name], dev=dev, d=meta["d"])
    res = {"experiment": "topkdiff_evals", "arm": "bayes_gate", "name": name,
           "seed": 0, "ckpt": CKPTS[name], "partial": True,
           "eval_semantics": {"u": 0.0, "gate_threshold": 0.5,
                              "z": "shrunk_linear_magnitude * 1[gate>0.5]"}}
    dest = volp / OUT_DIR / f"evals_bayes_{name}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)

    def save(section: str):
        dest.write_text(json.dumps(res))
        vol.commit()
        print(f"[{name}] {section} done ({time.time() - t0:.0f}s), committed",
              flush=True)

    ev = torch.load(volp / "cache" / "eval_shard.pt", weights_only=True)
    res["sanity"] = _sanity(name, sae, ev["acts"][:4096].to(dev, torch.float32))
    del ev
    print("SANITY", json.dumps(res["sanity"]), flush=True)
    save("sanity")

    res["sparse_probing"] = run_evals.sparse_probing(sae, volp / "concepts", dev)
    save("sparse_probing")

    res["absorption"], res["fragility"] = run_evals.absorption_and_fragility(
        sae, volp, tok, dev, rms=meta["rms"])
    save("absorption_fragility")

    res["autointerp"] = run_evals.autointerp_judge(
        volp, "bayes_gate", 0, ckpt_dir=f"bayes_gate/{name}")
    res["_runtime_s"] = round(time.time() - t0, 1)
    res["provenance"] = {
        "backend": "modal", "base_model": "gemma-2-2b",
        "eval_cache": "sol round (cache/ + concepts/, rms from meta.json)",
        "train_final": json.loads(
            (volp / "bayes_gate" / name / "bayes_gate_s0_final.json").read_text()),
    }
    res["partial"] = False
    save("all")
    return res


@app.function(image=image, gpu="L4", timeout=14400, volumes={"/vol": vol},
              secrets=[hf_secret, judge_secret], memory=24576)
def eval_all(names: str = "") -> dict:
    """Both arms sequentially in one container (single-function detach rule)."""
    _setup()
    import torch
    from transformers import AutoTokenizer

    dev = torch.device("cuda")
    volp = pathlib.Path("/vol")
    meta = json.loads((volp / "cache" / "meta.json").read_text())
    tok = AutoTokenizer.from_pretrained(meta["model"])
    todo = [n for n in names.split(",") if n] or list(CKPTS)
    out = {}
    for name in todo:
        out[name] = _eval_one(name, volp, meta, tok, dev)
    print("ALL ARMS DONE:", ",".join(todo), flush=True)
    return out


@app.local_entrypoint()
def sanity():
    print("SANITY RESULT:", json.dumps(l0_sanity.remote(), indent=2),
          flush=True)


@app.local_entrypoint()
def main(names: str = ""):
    call = eval_all.spawn(names)
    print("SPAWNED:", call.object_id, flush=True)
