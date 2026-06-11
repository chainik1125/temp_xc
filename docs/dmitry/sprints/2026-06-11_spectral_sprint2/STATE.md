---
author: Claude (sprint 2)
date: 2026-06-11
tags:
  - reference
---

## Sprint-2 live state (for takeover agents)

Hard stop: 2026-06-11 15:32 UTC. Budget guard: terminate all RunPod pods
named s2-* / freqbench-* by 15:20 UTC at the latest (RunPod GraphQL, key in
env RP_API_KEY_MATS on the Mac; from cloud use the API over HTTPS).

### Pipeline state (update on every change)

- pod s2-wscan `0j22cflrjods8e` (A40, 194.68.245.66:22180; http proxy port
  19123): running `bt_wscan.py --out ws_out` (window-timescale scan, Task A).
  Results: /workspace/ws/ws_out/*.json; logWS.log. HTTP server on :19123
  serves /workspace/ws (started — see log).
- pod s2-hh `15uqvobknr31rr` (A40, 69.30.85.129:22040; proxy
  https://15uqvobknr31rr-19123.proxy.runpod.net/): running hh_screen.py
  (HH-RLHF chosen-vs-rejected screening) -> /workspace/hh/hh_out/hh_screen.json.
  Dead-man timer armed.
- pod s2-wscan also queued bt_relabel.py after the W-scan ->
  ws_out/relabel_screen.json (verification/conclusion/uncertainty rows).
- Task C workflow running locally (run wf_ebf30e77-675): brainstorm->rank->
  eval-design->red-team for spectral-XC behaviour candidates; results will be
  written into summary.md when it returns.
- Raw timescale curve (Question A headline) already in log.md: peak T~48.

- pod s2-em `dtgd24ap8nz0j8` (L40S, 103.196.86.40:56418): em_screen.py
  running -> /workspace/em/em_out/em_screen.json (dead-man 5h from ~09:30).
- pods s2-wscan and s2-hh TERMINATED (results synced to repo).

### Takeover instructions

1. If last branch commit < 45 min old: main agent alive → do nothing.
2. Else: fetch ws_out JSONs (https://0j22cflrjods8e-19123.proxy.runpod.net/
   ws_out/), append an analysis + table to
   docs/dmitry/sprints/2026-06-11_spectral_sprint2/summary.md (create if
   missing; raw_mean_scan.json AUC-vs-T curve is the headline: peak T = the
   backtracking-state timescale), commit+push to dmitry-spectral-sprint2.
3. At/after 15:20 UTC: terminate ALL pods via RunPod API; final commit.
