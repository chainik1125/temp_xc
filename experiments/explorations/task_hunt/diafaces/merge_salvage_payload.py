"""Local merge of repatriated salvage payloads (containers never push).

Reads `results/salvage_payloads/payload_*.json`, then:
1. asserts every leaderboard row belongs to THIS salvage panel (ds,
   arch subset, seeds {3,4,5}, buffer 524288, k_pos legal for its arm,
   commit stamp in the freeze set, paired v2 columns present — the
   defect receipt);
2. appends to the canonical leaderboard, skipping existing eval_keys;
3. merges cells into `results/salvage_stage2_<DS>.json`.

Dirty stamps are disclosed, not fatal (pool leaderboard-growth
convention — see merge_panel_payload.py for the full note).

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.merge_salvage_payload
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.explorations.task_hunt.diafaces.run_salvage import (
    _merge_into_panel,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LB = ROOT / "results" / "leaderboard.jsonl"
DS = "dial_real_ttrend_gpt2_l7"
PAYLOADS = HERE / "results" / "salvage_payloads"
# Filled at freeze time from `git rev-parse` — never hand-typed.
FREEZES = {
    "FILL_AT_FREEZE": "salvage freeze (SALVAGE_CARD.md)",
}
ARCHS = {"batchtopk_sae", "tsae", "txc_batchtopk_post"}
SEEDS = {3, 4, 5}


def main():
    payloads = sorted(PAYLOADS.glob("payload_*.json"))
    assert payloads, f"no payloads under {PAYLOADS}"
    existing_keys = set()
    with LB.open() as fh:
        for line in fh:
            existing_keys.add(json.loads(line).get("eval_key"))

    new_rows, results, stamp_counts = [], [], {}
    for p in payloads:
        pay = json.loads(p.read_text())
        results.extend(pay["results"])
        for line in pay["leaderboard_delta"]:
            r = json.loads(line)
            assert r["datasource"] == DS, r["datasource"]
            assert r["arch"] in ARCHS, r["arch"]
            assert r["seed"] in SEEDS, r["seed"]
            assert r["training_cfg"]["buffer_tokens"] == 524288
            k, T = r["training_cfg"]["k_pos"], r["training_cfg"]["T"]
            if r["arch"] == "txc_batchtopk_post":
                assert k in (8, 8 * T), (k, T)
            else:
                assert (k, T) == (8, 1), (k, T)
            assert "lambda_recovery_v2" in r["metrics"], \
                "row missing paired v2 columns — the defect assert"
            cv = r["code_version"]
            assert cv["commit_sha"] in FREEZES, cv["commit_sha"]
            stamp_counts[cv["commit_sha"]] = \
                stamp_counts.get(cv["commit_sha"], 0) + 1
            new_rows.append((r["eval_key"], line, bool(cv["dirty"])))

    dup = [k for k, _, _ in new_rows if k in existing_keys]
    n_dirty = sum(1 for _, _, d in new_rows if d)
    seen, appended = set(), 0
    with LB.open("a") as fh:
        for k, line, _ in new_rows:
            if k in existing_keys or k in seen:
                continue
            fh.write(line if line.endswith("\n") else line + "\n")
            seen.add(k)
            appended += 1
    stamps = "; ".join(f"{n}× {sha[:9]} [{FREEZES[sha]}]"
                       for sha, n in sorted(stamp_counts.items()))
    print(f"[leaderboard] +{appended} rows ({len(dup)} dups skipped); "
          f"{n_dirty}/{len(new_rows)} dirty-stamped (pool convention); "
          f"pins verified: {stamps}")
    _merge_into_panel(results)


if __name__ == "__main__":
    main()
