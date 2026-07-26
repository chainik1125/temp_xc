"""Local merge of repatriated top-up payloads (TOPUP_CARD.md § 3;
containers never push). Same asserts as merge_salvage_payload with
the top-up population: seeds {6,7,8}, k_pos = 8 only, claiming Ts.

Run: .venv/bin/python -m experiments.explorations.task_hunt.diafaces.merge_topup_payload
"""

from __future__ import annotations

import json
from pathlib import Path

from experiments.explorations.task_hunt.diafaces.run_topup import (
    _merge_into_panel,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
LB = ROOT / "results" / "leaderboard.jsonl"
DS = "dial_real_ttrend_gpt2_l7"
PAYLOADS = HERE / "results" / "topup_payloads"
# Filled at freeze time from `git rev-parse` — never hand-typed.
FREEZES = {
    "85c87fd7602fb36dd2e63488b8d33ad3311789e5":
        "topup freeze (TOPUP_CARD.md)",
}
ARCHS = {"batchtopk_sae", "tsae", "txc_batchtopk_post"}
SEEDS = {6, 7, 8}


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
            hp = r["training_cfg"]["arch_hparams_override"]
            k, T = hp["k_pos"], hp["T"]
            assert k == 8, (k, T)
            assert T in ((16, 32) if r["arch"] == "txc_batchtopk_post"
                         else (1,)), (r["arch"], T)
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
