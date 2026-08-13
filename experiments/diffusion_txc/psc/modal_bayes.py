"""Launch the two bayes_gate trainings (1 seed, 2 scales) on Modal L40S.

    uvx modal run --detach experiments/diffusion_txc/psc/modal_bayes.py
"""

import json
import pathlib
import subprocess

import modal

try:
    ROOT = pathlib.Path(__file__).resolve().parents[3]
except IndexError:
    ROOT = pathlib.Path("/work")

app = modal.App("dtxc-bayes-gate")
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

JOBS = {
    "bg_100M": {"steps": 24000, "subdir": "gemma2-2b-l12-100M", "extra": []},
    "bg_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bayesgate",
               "extra": []},
    # v2: target-L0 controller (the lambda=0.02 v1 runs went dense, ~2200 L0)
    "bg2_100M": {"steps": 24000, "subdir": "gemma2-2b-l12-100M-bg2",
                 "extra": ["--target-l0", "40", "--sparsity-coef", "0.05"]},
    "bg2_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg2",
                "extra": ["--target-l0", "40", "--sparsity-coef", "0.05"]},
    # v3: bg2 railed at coef clamp 30 with L0~185; raise clamp + exponent
    "bg3_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg3",
                "extra": ["--target-l0", "40", "--sparsity-coef", "1.0",
                          "--ctrl-exp", "0.5", "--coef-max", "300"]},
    "bg3_100M": {"steps": 24000, "subdir": "gemma2-2b-l12-100M-bg3",
                 "extra": ["--target-l0", "40", "--sparsity-coef", "1.0",
                           "--ctrl-exp", "0.5", "--coef-max", "300"]},
    # v4: replace mean-gate controller with per-latent rate-KL (anti-collapse)
    "bg4_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg4",
                "extra": ["--target-l0", "40", "--sparsity-coef", "0",
                          "--rate-kl-coef", "0.01"]},
    # v5: bg4's rate-KL never reached the loss (silent edit no-op) and the
    # controller ran instead; v5 decouples them and actually wires the KL
    "bg5_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg5",
                "extra": ["--target-l0", "40", "--sparsity-coef", "0",
                          "--rate-kl-coef", "0.01"]},
    # v6: bg5 proved rate-KL keeps the pool 100% alive but L0 landed 219;
    # raise coefficient toward the L0=40 equilibrium
    "bg6_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg6",
                "extra": ["--target-l0", "40", "--sparsity-coef", "0",
                          "--rate-kl-coef", "0.06"]},
    # v7: coef->L0 map {0.01: 219, 0.06: 67.6} gives L0 ~ coef^-0.66;
    # 0.13 targets L0=40. bg6 NMSE 0.291 at L0 68 (vs TopK-40 recon 0.299)
    "bg7_sol": {"steps": 6000, "subdir": "gemma2-2b-l12-sol-bg7",
                "extra": ["--target-l0", "40", "--sparsity-coef", "0",
                          "--rate-kl-coef", "0.13"]},
}
LAUNCH = ["bg7_sol"]


@app.function(image=image, gpu="L40S", timeout=14400, volumes={"/vol": vol},
              secrets=[hf_secret], memory=16384)
def train(name: str) -> dict:
    import os
    import threading
    import time

    cfg = JOBS[name]
    out = f"/vol/bayes_gate/{name}"

    def committer():
        while True:
            time.sleep(300)
            try:
                vol.commit()
            except Exception:                                 # noqa: BLE001
                pass

    threading.Thread(target=committer, daemon=True).start()
    cmd = ["python3", "/work/psc/psc_train_sae.py",
           "--model", "google/gemma-2-2b", "--hook", "resid12",
           "--arm", "bayes_gate", "--seed", "0",
           "--steps", str(cfg["steps"]), "--dataset", "pile",
           "--out", out, "--hf-repo", "dmanningcoe/diffusion-topk-saes",
           "--hf-subdir", cfg["subdir"], *cfg.get("extra", [])]
    t0 = time.time()
    rc = subprocess.run(cmd, env={**os.environ,
                                  "PYTHONUNBUFFERED": "1"}).returncode
    vol.commit()
    print(f"[{name}] rc={rc} wall={time.time() - t0:.0f}s", flush=True)
    return {"name": name, "rc": rc, "wall_s": round(time.time() - t0, 1)}


@app.local_entrypoint()
def main():
    calls = {n: train.spawn(n) for n in LAUNCH}
    print("SPAWNED:", json.dumps({n: c.object_id for n, c in calls.items()}),
          flush=True)
    for n, c in calls.items():
        print("DONE", json.dumps(c.get()), flush=True)
    print("ALL DONE", flush=True)
