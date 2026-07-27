# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`; read-only archaeology lane, $0 compute)
**Briefings:** `briefings/actmix-shared.md` → `briefings/actmix-mac-c.md` (ACTMIX W3)
**Last update:** 2026-07-27 ~14:00 London (mirrors COMPLETE + ratified; gen-4 corpus scout delivered)

## Supersession note

This workstream **supersedes mac-local's ~20:40 subagent dispatch** (recalled cleanly at Han's direction before anything was frozen/launched/pushed — see be755651a).

## Current posture (Han's 2026-07-27 dispatch)

- **HF mirror: COMPLETE (~12:45)** — was the one open item; confirmation in LOG. Detail: to `han1823123123/temp_xc_a40_checkpoints` — λ̂ tsae trio DONE (`hunt_lambda_tsae_topup_checkpoints/`, manifest+sha256); remaining = dialogue-panel cells, salvage/topup/calib payloads, hunt3 screen bundles + seed payloads + provenance → `hunt_payload_bundles/` prefix (same manifest pattern). **Gates the post-weekend token rotations**; completion confirmation will be pushed to LOG.
- **"part 3" is DEFINED and ARCHIVED**: post-deadline archaeology continuation, no cycles now (Han, this dispatch — closes the "part 3 undefined" flag from my earlier LOG entry).
- **Gen-4 corpus scout (beat review ~12:15 item 3): DELIVERED** — see `experiments/explorations/task_hunt/GEN4_CORPUS_SCOUT.md` (NOT a freeze; facts + evidence lines feeding mac-a's slate). 5 faces on 2 untouched substrates: wikitext103 (tret/tretd transplants + sage section-age) and permissive-licence Python code (tret + drev identifier-revival). Pullers pinned w/ receipts, streams committed for the gpt2+gemma2 first-wave pair (per-doc 1024-token cap so gpt2 can screen), `labels/gen4c_stats.json` = artifact of record, `gen4c_lib.py` under 9 green tests. I scout labels; mac-a owns screens; no Modal spend from mac-c.
- Support the one-pager/meeting on request.
- Listener RE-ARMED for 10h (sleep-first loop, briefings/ ex-ledger + LOG lines matching mac-c/audit/vwin/G6/mirror, 150s poll); re-arm after every wake.

## Mirror state: COMPLETE

`hunt_payload_bundles/` = 455 files + manifest.jsonl (sha256 + source volume:path + volume mtime per file) + README, remote-verified 457/457. Incidents on record (LOG ~12:45): plain `modal volume ls` truncation (10/299 on btk_rerun_v2 — use `--json`); one 3h16m wedged `get` (fixed w/ 120s watchdog). λ̂ trio was already at `hunt_lambda_tsae_topup_checkpoints/`. **Rotations unblocked from my side.** Local staging copy in session scratchpad `mirror_bundles2/`.

## Delivered (on record — receipts in LOG + the files themselves)

- **COMPOSITION_AUDIT.md** — per-arm paper compositions for all 5 tasks; A1/A2/A4/A5/A9/A12 resolved; **A3 FULLY RESOLVED** incl. the train/serve mismatch finding (fra_proj wrapper serves Nura's SAE uncentered TopK→ReLU, threshold skipped, vs its ReLU-first+threshold training family — evidence to Dmitry); A6 public-search exhausted (→ Dmitry); three paper-integrity flags (c6 caption-vs-figure, c7 double declaration, c3 T5-replica phantoms).
- `tbm_census.jsonl` (1,283 ckpt configs), Phase-B recipe (runpod-1 in use), A2 k-budget paragraph (ratified), ONEPAGER_SKELETON.md (consumed by mac-local's noon one-pager), V-win figure + G-6 patch proposal (pending Andrii), WRITEUP §9 R30 staging (applied `bef389a47`).

## Git position

Branch `arxiv`, rebased on origin at resume. My pushes touch only: task_hunt/{COMPOSITION_AUDIT.md, LOG.md, tbm_census.jsonl, ONEPAGER_SKELETON.md}, agents/mac-c/, txcwin §6c files. Identity `mac-c-agent`.

## If resuming from compact

Read this file, then LOG tail. Open item = the mirror pipeline above; check background tasks (driver/listener) with `ps`, check `mirror_bundles2/` counts vs 166, rerun driver v2 (skip-existing) if short, then manifest → upload → LOG + push. Token path only, never the value; rotations happen after mirrors confirm.
