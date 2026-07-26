"""Modal driver — thin-pool diagnostic cell (run_calib_diag.py; af2247d43 § 2).

  modal run --detach scripts/modal_diafaces_calib_diag.py

ONE H100 cell (post_btkonly k256@T32 s3). Payload → Volume
/workspace/diafaces_calib_diag; containers never push.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "63ac1b208443b94879a7471271a3507322c0114c"  # diag freeze, from `git rev-parse HEAD` post-push
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
LB = "/repo/results/leaderboard.jsonl"
RES_DIR = "/repo/experiments/explorations/task_hunt/diafaces/results"
DS = "dial_real_ttrend_gpt2_l7"

app = modal.App("mac-a-diafaces-calib-diag")
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


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=65536, timeout=90 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def run_diag() -> str:
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at diag freeze {head[:10]}", flush=True)
    _sh(f"{PY} -m experiments.explorations.task_hunt.dialevel.cache_acts gpt2")
    n0 = sum(1 for _ in open(LB)) if Path(LB).exists() else 0
    vol_dir = "/workspace/diafaces_calib_diag"
    _sh(f"mkdir -p {vol_dir}")
    try:
        _sh(f"{PY} -m experiments.explorations.task_hunt.diafaces.run_calib_diag 1")
    finally:
        lb_delta = (open(LB).readlines()[n0:] if Path(LB).exists() else [])
        out = Path(RES_DIR) / f"calib_diag_run_{DS}.json"
        payload = {
            "results": json.loads(out.read_text()) if out.exists() else [],
            "leaderboard_delta": lb_delta,
        }
        dst = Path(vol_dir) / "payload_diag.json"
        dst.write_text(json.dumps(payload))
        vol.commit()
        print(f"[payload] diag: {len(payload['results'])} results, "
              f"{len(lb_delta)} leaderboard rows", flush=True)
    return json.dumps(payload)


@app.local_entrypoint()
def main():
    out_dir = Path(__file__).resolve().parents[1] / \
        "experiments/explorations/task_hunt/diafaces/results/calib_payloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    call = run_diag.spawn()
    try:
        payload = call.get()
        p = out_dir / "payload_diag.json"
        p.write_text(payload)
        print(f"[repatriated] {p} ({len(payload)} bytes)", flush=True)
    except Exception as e:                        # noqa: BLE001
        print(f"[FAILED] diag: {e!r} — payload persists on Volume "
              f"/workspace/diafaces_calib_diag", flush=True)
    print("DIAFACES CALIB-DIAG PIPELINE COMPLETE", flush=True)
