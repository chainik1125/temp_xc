# MODAL SPEND LEDGER — shared, append-only (union-merge like the LOG)

**$500 HARD CEILING (Dmitry). $400 soft stop TOTAL. Read the total
before EVERY launch; append after. Caps: mac-a $150, mac-b $100
(raises only by mac-local amendment). Modal dashboard is authority.**

| when (PT) | agent | what | GPU/CPU | est. cost | running total |
|---|---|---|---|---|---|
| 07-26 02:10 | mac-local | smoke attempts + A10 hello | A10 | ~$1 | ~$1 |
| 07-25 18:40 | mac-b | slen screen pipeline: image build + smoke + fineweb-400 caches (gpt2+llama31) + 2× screen (est ≤ 2 A10G-h each, parallel) | A10G | ~$5 | ~$6 |
| 07-25 18:55 | mac-a | tsae top-up bring-up: image build (uv sync) + in-container run.py validate | CPU 4-core | ~$0.5 | ~$6.5 |
| 07-25 19:20 | mac-a | Ward cache rebuild (stream + labels + base/hs13, receipts hard-fail) → Volume | A10G + 8cpu + 48G + 512GiB disk, ~1h | ~$3 | ~$9.5 |
| 07-25 19:35 | mac-a | (correction: cache rebuild took 6.5 min not 1h — actual ≲$0.5; receipts ALL PASS) 3× tsae/T1 cells, one per container, timebox 5.5h | 3× (A10G + 8cpu + 64G), est ~3h each | ~$35 | ~$42 |
| 07-25 19:00 | mac-b | slen screens RELAUNCH on L40S (A10 OOM at llama T32 flatten-MLP; both partials on Volume, resuming; gpt2 ~2 cells from done) | L40S ×2 | ~$4 | ~$46 |
| 07-25 20:15 | mac-a | attempt 1 CANCELLED at ~24 min (local client disconnect, non-detached run — Modal cancels in-flight inputs; ~$4.5 burned on top of the $35 still-planned rerun). RELAUNCH detached, payloads persist to Volume | 3× (A10G + 8cpu + 64G) | +$4.5 net | ~$50.5 |
| 07-25 19:12 | mac-b | gpt2 screen DONE + repatriated; llama input cancelled ~4 min in (same non-detached-client mode mac-a hit) — llama-only relaunch DETACHED, sequential .remote + retries | L40S ×1 | ~$2 | ~$52.5 |
| 07-25 19:55 | mac-b | slen COMPLETE (verdicts pushed; actual mac-b burn ≈ $5–6 vs ~$11 est). STRETCH: refmark image build + caches (2.5M tok, 2 models) + screen ×2, detached (gate cleared; mac-local approval in LOG) | L40S | ~$8 | ~$60.5 |
