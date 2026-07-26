"""Modal driver — BTK re-run: paper-arch composite vs btk-only T-sweep.

Dmitry's ACTMIX re-run gate (isolated sprint lane, branch
``dmitry-btk-txc-sprint``). One shard = (arch, datasource, T) —> the
in-repo driver ``experiments.explorations.btk_rerun.driver`` runs
9 cells (3 k_pos x 3 seeds). 24 shards total. A10G, detached,
containers never push; shard rows persist to the Volume AND repatriate
via return values (house discipline).

  modal run --detach scripts/modal_btk_rerun.py
  modal run scripts/modal_btk_rerun.py --collect-only   # pull from Volume
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "2ce33ac5ccffbf90fe2e6f1086da0b1fa2bd51b4"  # btk_rerun FREEZE v2
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
VOL_DIR = "/workspace/btk_rerun_v2"

ARMS = ["txc_base", "txc_base_btk"]
DATASOURCES = ["toy_markov_n20_d40_noisy", "toy_coupled_K10_M20_d256"]
T_GRID = [1, 2, 4, 5, 8, 10]

app = modal.App("dmitry-btk-rerun")
vol = modal.Volume.from_name("temp-xc-btk-rerun", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /repo && git -C /repo checkout {PINNED_COMMIT}",
        "pip install uv",
        "cd /repo && uv sync --frozen",
    )
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
              memory=32768, cpu=8, timeout=2 * 60 * 60,
              max_containers=8,
              retries=modal.Retries(max_retries=2, initial_delay=10.0))
def run_shard(shard: tuple[str, str, int]) -> str:
    arch, datasource, T = shard
    _assert_pinned()
    tag = f"{arch}__{datasource}__T{T}"
    out_vol = Path(VOL_DIR) / f"{tag}.json"
    _sh(f"mkdir -p {VOL_DIR}")
    if out_vol.exists():
        print(f"[resume] shard {tag} already on Volume — skipping", flush=True)
        return out_vol.read_text()
    out_repo = f"/repo/results/btk_rerun_{tag}.json"
    _sh(f"{PY} -m experiments.explorations.btk_rerun.driver "
        f"--arch {arch} --datasource {datasource} --T {T} "
        f"--allow-dirty --out {out_repo}")
    text = Path(out_repo).read_text()
    out_vol.write_text(text)
    vol.commit()
    return text


def _merge_rows(all_rows: list[dict], dest: Path) -> tuple[int, int]:
    """Append rows to dest leaderboard, dup-key checked. Returns (new, dup)."""
    seen = set()
    if dest.exists():
        for line in dest.open():
            try:
                seen.add(json.loads(line)["eval_key"])
            except Exception:  # noqa: BLE001
                continue
    new = dup = 0
    with dest.open("a") as f:
        for row in all_rows:
            if row["eval_key"] in seen:
                dup += 1
                continue
            f.write(json.dumps(row) + "\n")
            seen.add(row["eval_key"])
            new += 1
    return new, dup


@app.local_entrypoint()
def main(collect_only: bool = False, single: str = ""):
    shards = [(a, d, t) for a in ARMS for d in DATASOURCES for t in T_GRID]
    if single:
        a, d, t = single.split(":")
        shards = [(a, d, int(t))]
    all_rows: list[dict] = []
    if collect_only:
        print("[collect] pulling shard files from Volume", flush=True)
        for entry in vol.listdir(VOL_DIR.removeprefix("/workspace/")):
            data = b"".join(vol.read_file(entry.path))
            all_rows += json.loads(data)
    else:
        print(f"[launch] {len(shards)} shards, max 8 containers", flush=True)
        for i, text in enumerate(run_shard.map(shards, order_outputs=False)):
            rows = json.loads(text)
            all_rows += rows
            print(f"[shard done {i + 1}/{len(shards)}] +{len(rows)} rows",
                  flush=True)
    dest = Path(__file__).resolve().parents[1] / "results" / "leaderboard.jsonl"
    new, dup = _merge_rows(all_rows, dest)
    print(f"BTK RERUN COMPLETE: {new} new rows merged, {dup} dups skipped "
          f"-> {dest}", flush=True)
