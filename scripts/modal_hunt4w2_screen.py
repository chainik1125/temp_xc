"""Modal driver — gen-4 WAVE-2 screens (HUNT4W2_SCREEN_CARD.md;
directive 4dbb57e54 § 3).

  modal run --detach scripts/modal_hunt4w2_screen.py
  modal run --detach scripts/modal_hunt4w2_screen.py --jobs pycode:llama31_8b

One (corpus, model) per L40S container — 4 first-wave containers.
COLD caches: each container builds its own activation cache from the
committed gen4c stream (hunt4w2/cache_acts.py, mapping-verified)
before probing. Result JSONs persist to Volume /workspace/
hunt4w2_screen after every cell + repatriate; containers never push.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "22b38d65efb2aba707c603ce5ad5c696974f90bf"  # hunt4w2 freeze, ORIGIN-history rev-parse post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
DEFAULT_JOBS = ("wikitext103:gpt2", "wikitext103:gemma2_2b",
                "pycode:gpt2", "pycode:gemma2_2b")
PY = "/repo/.venv/bin/python"
RES_DIR = "/repo/experiments/explorations/task_hunt/hunt4w2/results"

app = modal.App("mac-a-hunt4w2-screen")
vol = modal.Volume.from_name("temp-xc-replag-caches", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /repo && git -C /repo checkout {PINNED_COMMIT}",
        "pip install uv",
        "cd /repo && uv sync --frozen",
    )
    .env({"HF_HOME": "/workspace/hf_cache"})
)


def _sh(cmd: str):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/repo")


@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=4 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_screen(corpus: str, model: str) -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at hunt4w2 freeze {head[:10]}", flush=True)
    _sh(f"{PY} -m experiments.explorations.task_hunt.hunt4w2.cache_acts "
        f"{corpus} {model}")
    vol_dir = "/workspace/hunt4w2_screen"
    _sh(f"mkdir -p {vol_dir}")
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.hunt4w2.screen "
            f"{corpus} {model}")
    finally:
        src = Path(RES_DIR) / f"screen_{corpus}_{model}.json"
        if src.exists():
            (Path(vol_dir) / src.name).write_text(src.read_text())
            vol.commit()
            print(f"[payload] hunt4w2 {src.name} persisted", flush=True)
    return (Path(RES_DIR) / f"screen_{corpus}_{model}.json").read_text()


@app.local_entrypoint()
def main(jobs: str = ""):
    todo = tuple(jobs.split(",")) if jobs else DEFAULT_JOBS
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/hunt4w2/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    calls = []
    for job in todo:
        corpus, model = job.split(":")
        calls.append((f"{corpus}_{model}",
                      run_screen.spawn(corpus, model)))
    for name, call in calls:
        try:
            payload = call.get()
            p = out_dir / f"screen_{name}.json"
            p.write_text(payload)
            print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
        except Exception as e:                    # noqa: BLE001
            print(f"[FAILED] hunt4w2/{name}: {e!r} — result persists on "
                  f"Volume /workspace/hunt4w2_screen", flush=True)
    print("HUNT4W2 SCREEN PIPELINE COMPLETE", flush=True)
