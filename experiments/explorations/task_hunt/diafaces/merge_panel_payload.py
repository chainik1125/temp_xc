"""Local merge of repatriated diafaces-panel payloads (containers
never push — the overnight repatriate-merge-locally rule).

Reads `results/panel_payloads/payload_*.json` (written by the driver
client or fetched from the Volume), then:

1. asserts every leaderboard row belongs to THIS panel (ds, arch set,
   seeds {1,2,42}, buffer 524288, commit stamp == the panel freeze,
   clean tree) — the seed-topup merge discipline;
2. appends rows to the canonical leaderboard, skipping eval_keys
   already present (dedup receipt printed);
3. merges result cells into `results/stage2_<DS>.json` by cell
   identity via `run_panel._merge_into_panel`.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.merge_panel_payload
"""

from __future__ import annotations

import json
from pathlib import Path

import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LB = ROOT / "results" / "leaderboard.jsonl"
# Two panels, two freezes (the tt/gpt2 race resolution + panel 2):
PANELS = {
    "tt": {"ds": "dial_real_ttrend_gpt2_l7",
           "freeze": "7ba2e10fd2c822d8dac820a307ec4f9f3c4f0005",
           "payloads": HERE / "results" / "panel_payloads"},
    "dq": {"ds": "dial_real_dqgap_llama31_8b_l14",
           # filled by mac-b (merge support) via `git rev-parse cfa341c34`
           # — the panel-2 freeze commit the containers check out
           "freeze": "cfa341c34094f993904bae9b8e01a32d672a74d0",
           "payloads": HERE / "results" / "panel2_payloads"},
}
ARCHS = {"batchtopk_sae", "tsae", "txc_batchtopk_pre",
         "txc_batchtopk_post", "stacked_batchtopk"}
SEEDS = {1, 2, 42}


def _merge_into_panel(new_results, ds: str):
    panel_file = HERE / "results" / f"stage2_{ds}.json"
    existing = (json.loads(panel_file.read_text())
                if panel_file.exists() else [])

    def _key(c):
        return (c["arch"], c["T"], c["d_sae"], c["k_pos"], c["seed"],
                c["n_steps"], c.get("kind"))

    by_key = {_key(r): r for r in existing}
    added = 0
    for r in new_results:
        if not r.get("ok"):
            continue
        if _key(r) not in by_key:
            added += 1
        by_key[_key(r)] = r
    merged = list(by_key.values())
    tmp = panel_file.with_name(panel_file.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=2))
    tmp.replace(panel_file)
    print(f"[merge] {panel_file.name}: {len(merged)} cells (+{added} new)",
          flush=True)


def main():
    panel = PANELS[sys.argv[1] if len(sys.argv) > 1 else "tt"]
    DS, FREEZE = panel["ds"], panel["freeze"]
    assert "FILL" not in FREEZE, "panel 2 freeze SHA not filled yet"
    payloads = sorted(panel["payloads"].glob("payload_*.json"))
    assert payloads, f"no payloads under {panel['payloads']}"
    existing_keys = set()
    with LB.open() as fh:
        for line in fh:
            existing_keys.add(json.loads(line).get("eval_key"))

    new_rows, results = [], []
    for p in payloads:
        pay = json.loads(p.read_text())
        results.extend(pay["results"])
        for line in pay["leaderboard_delta"]:
            r = json.loads(line)
            assert r["datasource"] == DS, r["datasource"]
            assert r["arch"] in ARCHS, r["arch"]
            assert r["seed"] in SEEDS, r["seed"]
            assert r["training_cfg"]["buffer_tokens"] == 524288
            cv = r["code_version"]
            assert cv["commit_sha"] == FREEZE, cv["commit_sha"]
            assert not cv["dirty"], "dirty stamp in container row"
            new_rows.append((r["eval_key"], line))

    dup = [k for k, _ in new_rows if k in existing_keys]
    seen = set()
    appended = 0
    with LB.open("a") as fh:
        for k, line in new_rows:
            if k in existing_keys or k in seen:
                continue
            fh.write(line if line.endswith("\n") else line + "\n")
            seen.add(k)
            appended += 1
    print(f"[leaderboard] +{appended} rows ({len(dup)} dups skipped — "
          f"idempotent re-merge)")
    _merge_into_panel(results, DS)


if __name__ == "__main__":
    main()
