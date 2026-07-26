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

from experiments.explorations.task_hunt.diafaces.run_panel import (
    DS,
    _merge_into_panel,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LB = ROOT / "results" / "leaderboard.jsonl"
PAYLOADS = HERE / "results" / "panel_payloads"
FREEZE = "7ba2e10fd2c822d8dac820a307ec4f9f3c4f0005"
ARCHS = {"batchtopk_sae", "tsae", "txc_batchtopk_pre",
         "txc_batchtopk_post", "stacked_batchtopk"}
SEEDS = {1, 2, 42}


def main():
    payloads = sorted(PAYLOADS.glob("payload_*.json"))
    assert payloads, f"no payloads under {PAYLOADS}"
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
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
