"""Aggregate per-seed test ASR results from the two seed_average logs.

Walks both `seed_average_tiny.log` (a40_tiny_1, SEEDS=0,1, accidentally
also partial seed=2) and `seed_average_txc.log` (a40_txc_1, SEEDS=2)
and pulls out (tag, chosen_f, chosen_alpha, val_asr, test_asr, test_dCE)
tuples per pair. Then computes mean ± std across the 3 seeds for each
(arch_family, hookpoint) cell and prints a markdown table.

Pair structure per pair in either log:
    [sweep] === <tag> ===
    [sweep]   ranked features in ...
    [sweep]   chosen: f=<int> α=<float> val_asr=<float> Δlogp=<float> ΔCE=<float>
    [sweep]   running test eval…
    [sweep]   test: asr_16=<float> (base 0.990) Δlogp=<float>  ΔCE=<float>

Usage: uv run python aggregate_seeded.py
"""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

ROOT = Path(__file__).parent
LOGS = [
    ROOT / "outputs/seeded_logs/seed_average_tiny.log",
    ROOT / "outputs/seeded_logs/seed_average_txc.log",
]

OUT_JSON = ROOT / "outputs/seeded_logs/aggregated.json"
OUT_TABLE = ROOT / "outputs/seeded_logs/aggregated_table.md"


TAG_RE = re.compile(r"\[sweep\] === (?P<tag>\S+) ===")
CHOSEN_RE = re.compile(
    r"chosen: f=(?P<f>\d+) α=(?P<alpha>[-\d.]+) "
    r"val_asr=(?P<val_asr>[-\d.]+) "
    r"Δlogp=(?P<val_dlogp>[+\-\d.]+) "
    r"ΔCE=(?P<val_dce>[+\-\d.e]+)"
)
TEST_RE = re.compile(
    r"test: asr_16=(?P<test_asr>[-\d.]+) "
    r"\(base (?P<base>[-\d.]+)\) "
    r"Δlogp=(?P<test_dlogp>[+\-\d.]+) +ΔCE=(?P<test_dce>[+\-\d.e]+)"
)


def parse_log(path: Path) -> dict[str, dict]:
    """Return {tag: {f, alpha, val_asr, test_asr, test_dce, ...}}."""
    text = path.read_text()
    lines = text.splitlines()
    results: dict[str, dict] = {}
    cur_tag: str | None = None
    cur: dict | None = None
    for line in lines:
        m = TAG_RE.search(line)
        if m:
            # If we already had a partial pending pair, drop it (no chosen line).
            cur_tag = m.group("tag")
            cur = {"tag": cur_tag, "log": str(path.name)}
            continue
        if cur is None:
            continue
        m = CHOSEN_RE.search(line)
        if m:
            cur.update({
                "f": int(m.group("f")),
                "alpha": float(m.group("alpha")),
                "val_asr": float(m.group("val_asr")),
                "val_dlogp": float(m.group("val_dlogp")),
                "val_dce": float(m.group("val_dce")),
            })
            continue
        m = TEST_RE.search(line)
        if m:
            cur.update({
                "test_asr": float(m.group("test_asr")),
                "test_base": float(m.group("base")),
                "test_dlogp": float(m.group("test_dlogp")),
                "test_dce": float(m.group("test_dce")),
            })
            # Pair complete. Save and reset.
            if cur_tag is not None:
                # If the same tag appears twice (host duplicate), keep the
                # first entry — it's the canonical one for that host's seed
                # assignment. Skip subsequent duplicates.
                if cur_tag not in results:
                    results[cur_tag] = cur
            cur = None
            cur_tag = None
    return results


def parse_basetag(tag: str) -> tuple[str, str, int]:
    """`sae_l0_pre_s0` → ('sae', 'l0_pre', 0)."""
    m = re.match(r"^(?P<arch>sae|tsae|txc)_(?P<hook>.+)_s(?P<seed>\d+)$", tag)
    if not m:
        raise ValueError(f"unparseable tag: {tag}")
    return m.group("arch"), m.group("hook"), int(m.group("seed"))


def main() -> None:
    all_pairs: dict[str, dict] = {}
    for log in LOGS:
        if not log.exists():
            print(f"[warn] log not found: {log}")
            continue
        parsed = parse_log(log)
        for tag, entry in parsed.items():
            if tag in all_pairs:
                # Cross-host duplicate (e.g. seed 2 on both hosts).
                # Keep both for sanity check.
                all_pairs[tag]["dup"] = entry
            else:
                all_pairs[tag] = entry
    print(f"[aggregate] {len(all_pairs)} unique tags parsed")

    # Group by (arch, hookpoint) → list of test_asr per seed.
    cells: dict[tuple[str, str], dict[int, dict]] = {}
    for tag, entry in all_pairs.items():
        try:
            arch, hook, seed = parse_basetag(tag)
        except ValueError:
            print(f"[skip] {tag} not a seeded tag")
            continue
        if "test_asr" not in entry:
            print(f"[skip] {tag}: no test_asr in log")
            continue
        cells.setdefault((arch, hook), {})[seed] = entry

    # Output: per-cell mean ± std.
    cell_rows = []
    for (arch, hook), seed_map in sorted(cells.items()):
        seeds = sorted(seed_map.keys())
        if not seeds:
            continue
        test_asrs = [seed_map[s]["test_asr"] for s in seeds]
        test_dces = [seed_map[s]["test_dce"] for s in seeds]
        mean = statistics.mean(test_asrs)
        std = statistics.stdev(test_asrs) if len(test_asrs) > 1 else 0.0
        ce_mean = statistics.mean(test_dces)
        cell_rows.append({
            "arch": arch,
            "hook": hook,
            "n_seeds": len(seeds),
            "seeds_done": seeds,
            "test_asr_mean": mean,
            "test_asr_std": std,
            "test_asr_per_seed": dict(zip(seeds, test_asrs)),
            "test_dce_mean": ce_mean,
            "test_dce_per_seed": dict(zip(seeds, test_dces)),
            "f_per_seed": {s: seed_map[s]["f"] for s in seeds},
            "alpha_per_seed": {s: seed_map[s]["alpha"] for s in seeds},
        })

    # Save JSON.
    OUT_JSON.write_text(json.dumps({
        "n_unique_tags": len(all_pairs),
        "n_cells": len(cell_rows),
        "cells": cell_rows,
    }, indent=2))
    print(f"[aggregate] wrote {OUT_JSON}")

    # Save markdown table.
    table = [
        "## Phase 8 — 3-seed aggregate",
        "",
        "Per-cell test ASR₁₆ across 3 seeds (n_seeds shown; if <3, was capped by stop-early).",
        "Format: mean (std) [s0/s1/s2 individual values].",
        "",
        "| arch | hookpoint | n | mean ± std | per-seed | mean ΔCE |",
        "|------|-----------|--:|-----------:|---------|---------:|",
    ]
    HOOK_ORDER = ["l0_ln1", "l0_pre", "l0_mid", "l0_post", "l1_ln1"]
    ARCH_ORDER = ["sae", "tsae", "txc"]
    for hook in HOOK_ORDER:
        for arch in ARCH_ORDER:
            row = next(
                (r for r in cell_rows if r["arch"] == arch and r["hook"] == hook),
                None,
            )
            if row is None:
                continue
            per_seed_strs = " / ".join(
                f"{s}:{row['test_asr_per_seed'][s]:.2f}" for s in sorted(row["seeds_done"])
            )
            table.append(
                f"| {arch.upper():<5}| {hook} | {row['n_seeds']} | "
                f"{row['test_asr_mean']:.3f} ± {row['test_asr_std']:.3f} | "
                f"{per_seed_strs} | {row['test_dce_mean']:+.4f} |"
            )
    OUT_TABLE.write_text("\n".join(table) + "\n")
    print(f"[aggregate] wrote {OUT_TABLE}")
    print("\n".join(table))


if __name__ == "__main__":
    main()
