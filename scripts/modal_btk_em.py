"""Modal driver — EM (§ 5.3) window scaling for the PAPER arch, both arms.

Dmitry's re-run gate extended to Emergent Misalignment: txc_base
(paper composite) vs txc_base_btkonly (canonical btk-only) × T, on the
exact task/conventions runpod-2 pinned for ACTMIX P2 (CARD.md in
experiments/explorations/actmix_em/): datasource
qwen_2_5_7b_instruct_medical_l15 (BASE-forward train), cohort detect on
the merged medical organism at L15 (hs16), evaluator em 3.0.0
(pr_auc_S16 primary; within-window shuffle + realized-l0 inside the
eval). COMPLEMENTARY to runpod-2's v2-arch sweep: different arch family
(paper txc_base lineage at its registry em budget k_pos=25), same task,
same eval — tables mergeable side-by-side.

Stage once (caches to the Volume), then cells endpoint-first.

  modal run --detach scripts/modal_btk_em.py --stage-only
  modal run --detach scripts/modal_btk_em.py                    # wave 1
  modal run --detach scripts/modal_btk_em.py --cells "txc_base:2:42,txc_base_btkonly:2:42"
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import modal

PINNED_COMMIT = "0d208fa6a11ddf775c09ccd4f89f52e6c8eea515"  # em FREEZE v10 (ckpt persistence)
REPO_URL = "https://github.com/chainik1125/temp_xc.git"
PY = "/repo/.venv/bin/python"
DATASOURCE = "qwen_2_5_7b_instruct_medical_l15"

# Endpoint-first wave (runpod-2 convention): T16 + T1 first, then T4,
# then fill {2, 8}. Seed 42 first (paper em convention).
WAVE1 = [("txc_base", 16, 42), ("txc_base_btkonly", 16, 42),
         ("txc_base", 1, 42), ("txc_base_btkonly", 1, 42),
         ("txc_base", 4, 42), ("txc_base_btkonly", 4, 42)]

app = modal.App("dmitry-btk-em")
vol = modal.Volume.from_name("temp-xc-btk-rerun", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .run_commands(
        f"git clone {REPO_URL} /repo && git -C /repo checkout {PINNED_COMMIT}",
        "pip install uv",
        "cd /repo && uv sync --frozen && uv pip install peft",
    )
    .env({"HF_HOME": "/workspace/hf_cache",
          "TEMP_BENCH_ALLOW_DIRTY": "1"})
)


def _sh(cmd: str):
    print(f"+ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd="/repo")


def _assert_pinned():
    head = subprocess.run("git -C /repo rev-parse HEAD", shell=True,
                          capture_output=True, text=True).stdout.strip()
    assert head == PINNED_COMMIT, f"container tree at {head}, not the freeze"
    print(f"[pin] container at freeze {head[:10]}", flush=True)


def _link_data_cache():
    # Training caches persist on the Volume; the repo path is a symlink.
    _sh("mkdir -p /workspace/em_data_cache /repo/results")
    _sh("rm -rf /repo/results/data_cache && "
        "ln -s /workspace/em_data_cache /repo/results/data_cache")


def _link_checkpoints():
    # Trained EM dictionaries are the expensive artifact — persist them
    # (and the manifest) on the Volume; cache-hits resume across
    # containers. Seed the Volume copy from the repo's committed
    # config.json skeleton once.
    _sh("mkdir -p /workspace/em_checkpoints")
    _sh("cp -rn /repo/checkpoints/. /workspace/em_checkpoints/ 2>/dev/null "
        "|| true")
    _sh("rm -rf /repo/checkpoints && "
        "ln -s /workspace/em_checkpoints /repo/checkpoints")


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              memory=65536, cpu=8, timeout=3 * 60 * 60,
              retries=modal.Retries(max_retries=1, initial_delay=10.0))
def stage() -> str:
    """Build both caches, idempotently, onto the Volume."""
    _assert_pinned()
    _link_data_cache()
    # 1) BASE-forward training cache (runpod-2's committed builder).
    _sh(f"{PY} -m experiments.explorations.actmix_em.build_train_cache_base")
    # 2) Cohort hs caches: judge_outputs.jsonl was pre-staged at
    #    /workspace/em_medical/; phase4 expects it under
    #    /workspace/conv_depth_caches/em_medical/.
    _sh("mkdir -p /workspace/conv_depth_caches/em_medical")
    _sh("cp -n /workspace/em_medical/judge_outputs.jsonl "
        "/workspace/conv_depth_caches/em_medical/ || true")
    _sh(f"{PY} -m experiments.explorations.conversion_depth.phase4_em_depth "
        "cache")
    vol.commit()
    return "staged"


CELL_RUNNER = r"""
import json, sys
from temp_bench.core.runner import run_experiment
from temp_bench.core.schemas import TrainingConfig
arch, T, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
res = run_experiment(
    experiment="em", arch_name=arch, seed=seed,
    datasource_name="qwen_2_5_7b_instruct_medical_l15",
    training_cfg=TrainingConfig(
        n_steps=25_000, batch_size=1024,
        bricken_enabled=True, ema_auxk_alpha=0.125,
        dead_threshold_tokens=128_000,
        arch_hparams_override={"T": T},
    ),
    eval_cfg={}, agent="dmitry-btk-sprint", allow_dirty=True,
)
print("[cell done]", res.eval_key,
      json.dumps(res.row.metrics), flush=True)
open("/tmp/cell_row.json", "w").write(res.row.model_dump_json())
"""


@app.function(image=image, gpu="H100", volumes={"/workspace": vol},
              memory=65536, cpu=8, timeout=6 * 60 * 60,
              max_containers=3,
              retries=modal.Retries(max_retries=1, initial_delay=30.0))
def run_cell(cell: tuple[str, int, int]) -> str:
    arch, T, seed = cell
    _assert_pinned()
    _link_data_cache()
    _link_checkpoints()
    tag = f"em__{arch}__T{T}__s{seed}"
    out_vol = Path(f"/workspace/btk_rerun_v2/{tag}.json")
    if out_vol.exists():
        print(f"[resume] {tag} already on Volume", flush=True)
        return out_vol.read_text()
    try:
        _sh(f"{PY} - {arch} {T} {seed} <<'EOF'\n{CELL_RUNNER}\nEOF")
    finally:
        vol.commit()          # checkpoints/manifest persist even on failure
    text = Path("/tmp/cell_row.json").read_text()
    out_vol.parent.mkdir(parents=True, exist_ok=True)
    out_vol.write_text(json.dumps([json.loads(text)]))
    vol.commit()
    return out_vol.read_text()


@app.function(image=image, volumes={"/workspace": vol},
              secrets=[modal.Secret.from_name("hf-token")],
              cpu=8, memory=16384, timeout=2 * 60 * 60)
def hf_mirror(dry_run: bool = True) -> str:
    """Mirror EM artifacts (checkpoints + caches) from the Volume to HF.

    Uses the house `hf-token` Modal secret. dry_run reports the token
    identity and what would upload, without pushing.
    """
    import os
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
    who = api.whoami()
    name = who.get("name")
    orgs = [o.get("name") for o in who.get("orgs", [])]
    print(f"[hf] token identity: {name}; orgs: {orgs}", flush=True)
    targets = {
        "/workspace/em_checkpoints": ("temp-bench-models", "model", ""),
        "/workspace/em_data_cache": ("temp-bench-data", "dataset",
                                     "em_data_cache"),
        "/workspace/conv_depth_caches/em_medical":
            ("temp-bench-data", "dataset", "conv_depth_caches/em_medical"),
    }
    report = [f"identity={name} orgs={orgs}"]
    for src, (repo, rtype, sub) in targets.items():
        p = Path(src)
        n_files = sum(1 for f in p.rglob("*") if f.is_file()) if p.exists() else 0
        size = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())             if p.exists() else 0
        line = (f"{src}: {n_files} files, {size/1e9:.2f} GB -> "
                f"{name}/{repo}[{rtype}]/{sub or '.'}")
        print(("[dry] " if dry_run else "[push] ") + line, flush=True)
        report.append(line)
        if not dry_run and n_files:
            rid = f"{name}/{repo}"
            api.create_repo(rid, repo_type=rtype, private=True, exist_ok=True)
            api.upload_folder(folder_path=src, repo_id=rid, repo_type=rtype,
                              path_in_repo=sub or None,
                              commit_message="btk-sprint EM artifacts "
                                             "(dmitry-btk-txc-sprint lane)")
    return "\n".join(report)


def _merge_rows(all_rows, dest: Path):
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
def main(stage_only: bool = False, cells: str = "", skip_stage: bool = False,
         mirror: str = ""):
    if mirror:
        print(hf_mirror.remote(dry_run=(mirror != "push")), flush=True)
        return
    if not skip_stage:
        print("[stage]", stage.remote(), flush=True)
    if stage_only:
        return
    wave = ([tuple([a, int(t), int(s)]) for a, t, s in
             (c.split(":") for c in cells.split(",") if c)]
            or WAVE1)
    all_rows = []
    for i, text in enumerate(run_cell.map(wave, order_outputs=False)):
        rows = json.loads(text)
        all_rows += rows
        for r in rows:
            m = r.get("metrics", {})
            print(f"[cell {i+1}/{len(wave)}] {r.get('arch')} "
                  f"T={((r.get('training_cfg') or {}).get('arch_hparams_override') or {}).get('T')} "
                  f"pr_auc_S16={m.get('pr_auc_S16')}", flush=True)
    dest = Path(__file__).resolve().parents[1] / "results" / "leaderboard.jsonl"
    new, dup = _merge_rows(all_rows, dest)
    print(f"EM WAVE COMPLETE: {new} new rows, {dup} dups -> {dest}", flush=True)
