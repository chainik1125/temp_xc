#!/usr/bin/env python3
"""Materialize small, pinned source artifacts from existing Git objects.

No network, dataset, checkpoint, training, or source-tree checkout is involved.
Run with --collect to create/update the archive, or --verify to check its hashes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "purified/artifacts/camera_ready_2026"
DOCS = ROOT / "purified/docs/aniket/camera-ready-2026"
HF_REVISION = "3e935fd2fa5feff053da90517907011687bdcb4b"
TEXT_EXTENSIONS = {".py", ".sh", ".json", ".yaml", ".yml", ".toml", ".md", ".csv", ".tsv", ".txt", ".tex", ".bib"}
MAX_FILE_BYTES = 8_000_000
MAX_TOTAL_BYTES = 40_000_000


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", *args], cwd=ROOT)


def tree(ref: str):
    for line in git("ls-tree", "-r", "--long", "-z", ref).split(b"\0"):
        if not line:
            continue
        meta, path = line.split(b"\t", 1)
        mode, kind, oid, size = meta.decode().split()
        if kind == "blob" and mode != "120000":
            yield path.decode(), oid, int(size)


def collect(hf_dir: Path):
    if not hf_dir.is_dir():
        raise SystemExit(
            f"HF source directory is unavailable: {hf_dir}. "
            "Use --verify for the existing archive, or pass --hf-dir "
            "purified/artifacts/camera_ready_2026/hf_reviewer_results "
            "to reuse its already archived small files."
        )
    selections = json.loads((DOCS / "figure-recovery-selection.json").read_text())
    groups = [
        ("origin/dmitry-txcwins-10h", ["src/v6_colored_sources/", "src/temp_bench/", "src/bench/architectures/", "temporal_crosscoders/han_arch/", "temporal_crosscoders/han_tsae/", "configs/", "results/v6_colored_sources/", "results/reviewer_multiseed/", "docs/dmitry/reviewer_responses/"],
         ["src/__init__.py", "src/bench/__init__.py", "temporal_crosscoders/__init__.py", "temporal_crosscoders/models.py", "tests/test_polynomial_clock.py", "tests/test_v6_colored_sources.py", "pyproject.toml"],
         "Historical synthetic code/results and rebuttal drafts. Final episode-disjoint W10 Shamir run is NOT established by this archive."),
        ("origin/codex/em-paper-window-sweep-s42-20260727", ["src/temp_bench/", "configs/", "experiments/c6_em/"], ["pyproject.toml"],
         "Pinned medical EM seed-2/window runner, architecture implementations and configurations; data and checkpoints excluded."),
        ("origin/arxiv", ["experiments/explorations/task_hunt/sycgen/", "experiments/explorations/txcwin/", "experiments/explorations/actmix_rlhf/", "experiments/probing/", "src/", "configs/"], ["pyproject.toml", "REBUTTAL_HANDOFF.md", "REBUTTAL_CELL_CENSUS.md", "REBUTTAL_CODE_GUIDE.md", "scripts/gen_handoff_tables.py", "scripts/gen_sycgen_budget_table.py", "figs_writeup/tab_sycgen_budget_matched.md", "figs_writeup/tab_sycgen_shuffle_matched.md", "figs_writeup/tab_sycgen_shuffle_tsweep.md"],
         "Rebuttal probing/RLHF sweeps and supplemental sycgen code/results. Read class/substrate/budget caveats; not all rows were used in posted responses."),
        ("origin/dmitry-stacked-arxiv", ["src/", "configs/", "docs/dmitry/sprints/2026-07-27_stacked_sae_10h/"], ["pyproject.toml", "run.py"],
         "Completed Stacked SAE sprint source and numeric summary/log. Exact raw results are referenced in a private HF archive inaccessible in this session."),
        ("origin/dmitry-stacked-c7-300k", ["src/temp_bench/", "configs/", "experiments/c7_backtracking/"], ["pyproject.toml", "PROTOCOL.md"],
         "Pinned Stacked C7 runner/architectures. Legacy results in this tree are not the sprint's 300K Stacked result; use sprint log with raw-artifact limitation."),
        ("origin/dmitry-stacked-em-steer", ["purified/src/temp_bench/", "purified/configs/", "purified/experiments/c6_em/"], ["purified/pyproject.toml"],
         "Stacked EM steering source snapshot; same-grid aggregate results live in the Stacked sprint summary."),
    ]
    for entry in json.loads((DOCS / "rebuttal-code-selection.json").read_text())["selections"]:
        groups.append((entry["ref"], entry["prefixes"], entry["files"], entry["status"]))
    for ref, prefixes, files, reason in groups:
        for path, oid, size in tree(ref):
            if (path in files or any(path.startswith(p) for p in prefixes)) and Path(path).suffix in TEXT_EXTENSIONS:
                selections.append({"ref": ref, "path": path, "reason": reason})

    # Preserve the exact source and small plot inputs from the independent paper repo.
    paper_ref = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT / "paper").decode().strip()
    paper_paths = [p for p in (ROOT / "paper").rglob("*") if p.is_file() and (p.relative_to(ROOT / "paper").parts[0] in {"scripts", "notes", "images"}) and p.suffix in TEXT_EXTENSIONS | {".jsonl"}]
    pending = []
    # The complete arxiv leaderboard also has 9,547 unrelated synthetic rows.
    # Keep only complete, verbatim real-task rows, with an explicit filter receipt.
    ref = "origin/arxiv"
    commit = git("rev-parse", ref).decode().strip()
    source_path = "results/leaderboard.jsonl"
    raw = git("show", f"{commit}:{source_path}")
    selected = [line for line in raw.splitlines() if line.strip() and json.loads(line).get("experiment") in {"probing", "rlhf", "em"}]
    data = b"\n".join(selected) + b"\n"
    pending.append((OUT / "derived/arxiv_real_task_leaderboard.jsonl", data, {"source_kind": "git_filtered_rows", "source_ref": ref, "source_commit": commit, "source_path": source_path, "source_sha256": hashlib.sha256(raw).hexdigest(), "filter": "experiment in {'probing', 'rlhf', 'em'}; original JSON line bytes retained in original order", "selected_rows": len(selected), "reason": "Full numeric real-task rebuttal rows, including per-task probing values and protocol identity; not a merged headline table."}))
    for entry in selections:
        ref, path = entry["ref"], entry["path"]
        commit = git("rev-parse", ref).decode().strip()
        oid = git("rev-parse", f"{commit}:{path}").decode().strip()
        data = git("cat-file", "blob", oid)
        destination = OUT / "sources" / ref.removeprefix("origin/").replace("/", "__") / path
        pending.append((destination, data, {"source_kind": "git", "source_ref": ref, "source_commit": commit, "source_path": path, "git_blob": oid, "reason": entry["reason"]}))
    for p in paper_paths:
        path = p.relative_to(ROOT / "paper").as_posix()
        try:
            data = subprocess.check_output(["git", "show", f"{paper_ref}:{path}"], cwd=ROOT / "paper", stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError:
            continue
        pending.append((OUT / "sources/paper" / path, data, {"source_kind": "overleaf_git", "source_commit": paper_ref, "source_path": path, "reason": "Current paper plotting scripts and numeric sidecars. Source-specific limitations are documented in the figure audit."}))
    for p in sorted(hf_dir.rglob("*")):
        if p.is_file() and ".cache" not in p.parts and p.suffix in {".json", ".md"}:
            path = p.relative_to(hf_dir).as_posix()
            pending.append((OUT / "hf_reviewer_results" / path, p.read_bytes(), {"source_kind": "huggingface", "repo_id": "dmanningcoe/temp-xc-reviewer-results", "source_commit": HF_REVISION, "source_path": path, "reason": "Numeric results and provenance only; judge text JSONL and all models/caches excluded."}))
    dedup = {str(p): (p, data, meta) for p, data, meta in pending}
    total = sum(len(data) for _, data, _ in dedup.values())
    if total > MAX_TOTAL_BYTES or any(len(data) > MAX_FILE_BYTES for _, data, _ in dedup.values()):
        raise SystemExit("Archive exceeds explicit size limits; inspect the selection before proceeding.")
    records = []
    for p, data, meta in sorted(dedup.values(), key=lambda x: str(x[0])):
        if p.exists() and p.read_bytes() != data:
            raise SystemExit(f"Refusing to overwrite changed archive file: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        records.append({"path": p.relative_to(ROOT).as_posix(), "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(), **meta})
    manifest = {"schema_version": 1, "date": "2026-09-27", "scope": "Small source/metric archive; not a complete dataset/checkpoint release or blanket validation of historical claims", "network_payload_bytes": sum(r["bytes"] for r in records if r["source_kind"] == "huggingface"), "total_bytes": total, "files": records}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Archived {len(records)} files, {total:,} bytes; HF payload {manifest['network_payload_bytes']:,} bytes.")


def verify():
    manifest = json.loads((OUT / "manifest.json").read_text())
    for record in manifest["files"]:
        p = ROOT / record["path"]
        data = p.read_bytes()
        assert len(data) == record["bytes"], p
        assert hashlib.sha256(data).hexdigest() == record["sha256"], p
    print(f"Verified {len(manifest['files'])} pinned source artifacts ({manifest['total_bytes']:,} bytes).")
    native = OUT / "native_artifact_index.json"
    if native.exists():
        records = json.loads(native.read_text())["files"]
        for record in records:
            data = (ROOT / record["path"]).read_bytes()
            assert len(data) == record["bytes"], record["path"]
            assert hashlib.sha256(data).hexdigest() == record["sha256"], record["path"]
        print(f"Verified {len(records)} existing native artifacts referenced in place.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--hf-dir", type=Path, default=Path("/tmp/txc-hf-reviewer-results"))
    args = parser.parse_args()
    if args.collect:
        collect(args.hf_dir)
    if args.verify or not args.collect:
        verify()
