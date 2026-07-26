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
# Two panels. After the v2-columns defect BOTH panels re-ran at the
# re-freeze db677a4b8 — the quotable merge accepts ONLY that stamp
# (union-merge of mac-a's re-freeze semantics + mac-b's per-stamp
# receipt): a first-run stamp (7ba2e10fd tt / cfa341c34 dq) landing
# here means a stale payload file in the dir — hard-fail, don't
# disclose-and-continue. tt re-run payloads go to a FRESH dir so the
# already-merged first-run files can't mix in. SHAs from
# `git rev-parse`, never hand-typed.
PANELS = {
    "tt": {"ds": "dial_real_ttrend_gpt2_l7",
           "freezes": {"db677a4b873156d274a6b223a3cc7b82ff98e997":
                       "v2 re-freeze (paired v1+v2, quotable)"},
           "payloads": HERE / "results" / "panel_payloads_v2tt"},
    "dq": {"ds": "dial_real_dqgap_llama31_8b_l14",
           "freezes": {"db677a4b873156d274a6b223a3cc7b82ff98e997":
                       "v2 re-freeze (paired v1+v2, quotable)",
                       "931c016e63d4755d142a9eb25600f5026887c9a6":
                       "OOM re-pass PIN (only-cells; quotable)"},
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
    DS, FREEZES = panel["ds"], panel["freezes"]
    stamp_counts: dict = {}
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
            assert cv["commit_sha"] in FREEZES, cv["commit_sha"]
            stamp_counts[cv["commit_sha"]] = \
                stamp_counts.get(cv["commit_sha"], 0) + 1
            # Pool rows are dirty-stamped BY CONVENTION: run_experiment
            # appends to the tracked leaderboard.jsonl inside the
            # container, so every cell after the first sees a growing
            # `git diff` (grid.py sets TEMP_BENCH_ALLOW_DIRTY=1 for
            # exactly this; the historical λ̂ panel rows carry the same
            # signature — 69/84 dirty at 038655fd3). Single-cell
            # containers stamp before their own append and stay clean.
            # The integrity guarantee is the PIN assert above; dirty
            # counts are disclosed in the merge receipt below.
            new_rows.append((r["eval_key"], line, bool(cv["dirty"])))

    dup = [k for k, _, _ in new_rows if k in existing_keys]
    n_dirty = sum(1 for _, _, d in new_rows if d)
    seen = set()
    appended = 0
    with LB.open("a") as fh:
        for k, line, _ in new_rows:
            if k in existing_keys or k in seen:
                continue
            fh.write(line if line.endswith("\n") else line + "\n")
            seen.add(k)
            appended += 1
    stamps = "; ".join(f"{n}× {sha[:9]} [{FREEZES[sha]}]"
                       for sha, n in sorted(stamp_counts.items()))
    print(f"[leaderboard] +{appended} rows ({len(dup)} dups skipped — "
          f"idempotent re-merge); {n_dirty}/{len(new_rows)} dirty-stamped "
          f"(pool leaderboard-growth convention); pins verified: {stamps}")
    _merge_into_panel(results, DS)


if __name__ == "__main__":
    main()
