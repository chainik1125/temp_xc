"""Modal driver — B8 slen Stage-1 screen (mac-b overnight, 2026-07-26).

Ops per `briefings/overnight-mac-modal.md`: image built from the repo's
own `uv sync` at the PINNED freeze commit (clean tree in-container);
caches rebuilt by the COMMITTED builders onto a Volume; the canonical
screen module runs in-container; containers never push — results JSON
is returned and merged locally by mac-b. GPU: A10G (23 GB A10 — the
llama-8b bf16 forward and every probe fit; the frozen card § 7).

Stages (local entrypoint `main`):
  modal run scripts/modal_slen_screen.py --stage smoke    # bring-up receipt
  modal run scripts/modal_slen_screen.py --stage caches   # tokens + hs caches
  modal run scripts/modal_slen_screen.py --stage screen   # both models
  modal run scripts/modal_slen_screen.py                  # all of the above
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "b7121a2084f74ee88e8b314ae8e9507145820725"  # mac-b FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
KEYS = ["gpt2", "llama31_8b"]          # card § 1: gemma pending (no HF secret)
PY = "/repo/.venv/bin/python"
RESULTS_VOL_DIR = "/workspace/slen_results"
REPO_RES = "/repo/experiments/explorations/task_hunt/slen/results"

app = modal.App("mac-b-slen-screen")
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


def _assert_pinned():
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at freeze commit {head[:10]}", flush=True)


@app.function(image=image, gpu="A10G", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              timeout=15 * 60)
def smoke() -> str:
    _assert_pinned()
    _sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    _sh(f"{PY} run.py validate")
    return "SMOKE PASS"


@app.function(image=image, gpu="A10G", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              timeout=2 * 60 * 60)
def build_caches() -> str:
    _assert_pinned()
    for key in KEYS:
        _sh(f"{PY} -m experiments.explorations.task_hunt.replag.build_labels "
            f"{key}")
        vol.commit()
        _sh(f"{PY} -m experiments.explorations.task_hunt.replag.cache_acts "
            f"{key}")
        vol.commit()
    return "CACHES DONE"


# L40S: the llama T32 flatten-MLP standardization peak OOMed the A10
# (22 GiB; "tried to allocate 5.86 GiB" with 17.9 in use) — ops-doc rule:
# >20 GB ⇒ L40S. GPU choice does not touch cells/seeds; resume from the
# Volume partials is the card § 7 pre-authorized adaptation.
@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              timeout=4 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_screen(key: str) -> str:
    _assert_pinned()
    # gemma fill: build THIS key's caches here (idempotent via
    # acts_meta.json; in-container per the 34GB shutdown-grace lesson)
    _sh(f"{PY} -m experiments.explorations.task_hunt.replag.build_labels "
        f"{key}")
    _sh(f"{PY} -m experiments.explorations.task_hunt.replag.cache_acts "
        f"{key}")
    vol.commit()
    _sh(f"mkdir -p {RESULTS_VOL_DIR} {REPO_RES}")
    part = Path(RESULTS_VOL_DIR) / f"screen_{key}.json"
    if part.exists():                      # resume partial cells (card § 7)
        _sh(f"cp {part} {REPO_RES}/")
        print(f"[resume] restored partial {part.name}", flush=True)
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.slen.screen {key}")
    finally:
        out = Path(REPO_RES) / f"screen_{key}.json"
        if out.exists():
            _sh(f"cp {out} {RESULTS_VOL_DIR}/")
            vol.commit()
    return (Path(REPO_RES) / f"screen_{key}.json").read_text()


@app.local_entrypoint()
def main(stage: str = "all", models: str = ""):
    keys = [k for k in models.split(",") if k] or KEYS
    if stage in ("smoke", "all"):
        print(smoke.remote(), flush=True)
    if stage in ("caches", "all"):
        print(build_caches.remote(), flush=True)
    if stage in ("screen", "all"):
        local_res = Path(__file__).resolve().parents[1] / \
            "experiments/explorations/task_hunt/slen/results"
        local_res.mkdir(parents=True, exist_ok=True)
        # sequential .remote() per model: one model's infra failure can
        # never cancel another's in-flight call (the 19:06 PT lesson)
        for key in keys:
            try:
                text = run_screen.remote(key)
            except Exception as e:          # noqa: BLE001 — report, go on
                print(f"[FAILED] {key}: {e!r}", flush=True)
                continue
            p = local_res / f"screen_{key}.json"
            p.write_text(text)
            print(f"[repatriated] {p} ({len(text)} bytes)", flush=True)
    print("PIPELINE COMPLETE", flush=True)
