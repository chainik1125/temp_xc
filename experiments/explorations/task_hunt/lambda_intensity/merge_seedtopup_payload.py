"""Merge a repatriated Modal cells-payload into the canonical artifacts.

The Modal containers NEVER push git (`briefings/overnight-mac-modal.md`
recipe step 4): each `train_cell` returns its per-cell results JSON plus
the exact leaderboard line(s) it appended in-container. This script is
the LOCAL half of that contract:

1. every incoming leaderboard row is sanity-checked (datasource / arch /
   frozen seed set / buffer_tokens / code_version at the PIN, clean);
2. dup-eval-key check against `results/leaderboard.jsonl` — a row whose
   eval_key already exists is SKIPPED loudly (idempotent re-merge);
3. surviving rows are appended to the canonical leaderboard;
4. cell results are merged into the round-1 panel file via the frozen
   runner's `_merge_into_panel` (dedup by cell identity).

Run:  .venv/bin/python -m \
  experiments.explorations.task_hunt.lambda_intensity.merge_seedtopup_payload <payload.json>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from experiments.explorations.task_hunt.lambda_intensity.run_stage2_seedtopup_tsae import (
    BUFFER_TOKENS, DS, EXTRA_SEEDS, _merge_into_panel,
)

PIN = "c93473ad3482de441f3c13bea2def5c90de3f5cd"
ROOT = Path(__file__).resolve().parents[4]
LEADERBOARD = ROOT / "results" / "leaderboard.jsonl"


def main(payload_path: str):
    payload = json.loads(Path(payload_path).read_text())
    existing_keys = set()
    with LEADERBOARD.open() as fh:
        for line in fh:
            existing_keys.add(json.loads(line)["eval_key"])
    print(f"[merge] canonical leaderboard: {len(existing_keys)} eval_keys")

    to_append, results = [], []
    for cell_payload in payload:
        if "error" in cell_payload:
            print(f"[merge] SKIP failed cell payload: {cell_payload['error']}")
            continue
        results.extend(cell_payload["results"])
        for line in cell_payload["leaderboard_rows"]:
            r = json.loads(line)
            assert r["datasource"] == DS, r["datasource"]
            assert r["arch"] == "tsae", r["arch"]
            assert r["seed"] in EXTRA_SEEDS, r["seed"]
            assert r["training_cfg"]["buffer_tokens"] == BUFFER_TOKENS
            cv = r["code_version"]
            assert cv["commit_sha"] == PIN, f"row not at PIN: {cv['commit_sha']}"
            assert not cv["dirty"], "row stamped dirty — investigate before merging"
            if r["eval_key"] in existing_keys:
                print(f"[merge] DUP eval_key {r['eval_key']} "
                      f"(tsae seed {r['seed']}) — skipped")
                continue
            existing_keys.add(r["eval_key"])
            to_append.append(line.rstrip("\n"))
            print(f"[merge] + tsae/T1 seed {r['seed']} "
                  f"lambda_recovery={r['metrics']['lambda_recovery']:.5f} "
                  f"eval_key={r['eval_key']}")

    if to_append:
        with LEADERBOARD.open("a") as fh:
            for line in to_append:
                fh.write(line + "\n")
    print(f"[merge] appended {len(to_append)} leaderboard rows")

    ok_results = [r for r in results if r.get("ok")]
    _merge_into_panel(ok_results)
    print(f"[merge] panel merge done ({len(ok_results)} ok cells)")


if __name__ == "__main__":
    main(sys.argv[1])
