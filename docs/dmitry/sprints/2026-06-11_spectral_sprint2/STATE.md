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
- Task B (hh_rlhf screening): NOT yet launched.
- Task C (workflow): NOT yet launched.

### Takeover instructions

1. If last branch commit < 45 min old: main agent alive → do nothing.
2. Else: fetch ws_out JSONs (https://0j22cflrjods8e-19123.proxy.runpod.net/
   ws_out/), append an analysis + table to
   docs/dmitry/sprints/2026-06-11_spectral_sprint2/summary.md (create if
   missing; raw_mean_scan.json AUC-vs-T curve is the headline: peak T = the
   backtracking-state timescale), commit+push to dmitry-spectral-sprint2.
3. At/after 15:20 UTC: terminate ALL pods via RunPod API; final commit.
