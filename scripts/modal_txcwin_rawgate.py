"""Modal driver — txcwin raw-gate gap-fill (mac-b, salvage W2, GAP-B of
experiments/explorations/txcwin/crossratify/MINI_CARD.md).

Ops per briefings/salvage-shared.md: L40S, sequential .remote + retries,
caches built IN the work container by Andrii's own `build_cache`
(stream-SHA-keyed, idempotent, persisted on the Volume), launch with
`uvx modal run --detach`, containers never push. PIN taken verbatim
from `git rev-parse HEAD` at the crossratify freeze.

  uvx modal run --detach scripts/modal_txcwin_rawgate.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "fedf75aa96811419fc35a84adf46e0eb7f2c6eda"  # crossratify FREEZE
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
REPO_RES = "/repo/experiments/explorations/txcwin/crossratify/results"
RESULTS_VOL_DIR = "/workspace/txcwin_crossratify_results"

# (model, layer, t_ladder, tag) — gpt2 fills the missing T=8; the 8B
# never had the gate at any T. Function memory is sized for the largest
# raw_window ridge (8B T=16: 65536 features -> ~17 GB Gram).
JOBS = [
    ("gpt2", 6, "8", "gpt2_L6"),
    ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 12, "4,8,16", "8b_L12"),
]

app = modal.App("mac-b-txcwin-rawgate")
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


@app.function(image=image, gpu="L40S", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              memory=131072, cpu=8, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_rawgate(model: str, layer: int, t_ladder: str, tag: str) -> str:
    _assert_pinned()
    # Results dir -> SYMLINK onto the Volume: rawgate_fill writes its
    # JSON incrementally after every cell, so each write lands on the
    # Volume directly and survives a mid-flight kill (the first gpt2
    # run was cancelled during the finally-copy and the JSON died with
    # the container while the logs kept the numbers).
    _sh(f"mkdir -p {RESULTS_VOL_DIR}")
    _sh(f"rm -rf {REPO_RES} && ln -s {RESULTS_VOL_DIR} {REPO_RES}")
    try:
        _sh(f"{PY} -m experiments.explorations.txcwin.crossratify.rawgate_fill"
            f" --model '{model}' --layer {layer} --t-ladder {t_ladder}"
            f" --tag {tag}")
    finally:
        vol.commit()          # cache (from build_cache) + results
    return (Path(REPO_RES) / f"rawgate_fill_{tag}.json").read_text()


@app.local_entrypoint()
def main(only: str = ""):
    jobs = [j for j in JOBS if not only or j[3] == only]
    local_res = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/txcwin/crossratify/results"
    local_res.mkdir(parents=True, exist_ok=True)
    for model, layer, t_ladder, tag in jobs:
        try:
            text = run_rawgate.remote(model, layer, t_ladder, tag)
        except Exception as e:          # noqa: BLE001
            print(f"[FAILED] {tag}: {e!r}", flush=True)
            continue
        p = local_res / f"rawgate_fill_{tag}.json"
        p.write_text(text)
        print(f"[repatriated] {p} ({len(text)} bytes)", flush=True)
    print("TXCWIN RAWGATE GAP-FILL COMPLETE", flush=True)
