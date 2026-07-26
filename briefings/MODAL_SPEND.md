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
| 07-25 22:10 | mac-a | tsae cells COMPLETE (3/3, 62–77 min each) + merged; mac-a ACTUALS ≈ $19 cumulative (bring-up ~$0.3 + caches ~$0.5 + cells ~$13 + attempt-1 waste ~$4.5) vs $43 of est lines → −$24 correction. Recomputed total ≈ $1 (mac-local) + ~$19 (mac-b incl. refmark est) + $19 (mac-a) | — | −$24 corr | ~$39 |
| 07-25 21:25 | mac-b | refmark COMPLETE + verdicts (NEGATIVE, LOG + R22): actuals ≈ $2–3 (one wasted image build on a mistyped pin, disclosed) — mac-b cumulative ACTUALS ≈ $8–9 vs ~$19 of est lines → −$10 correction. Recomputed total ≈ $1 + $9 + $19 ≈ $29 | — | −$10 corr | ~$29 |
| 07-25 21:55 | mac-b | (prev line's receipt renumbered R22→R23 in the union-merge — R22 is mac-a's tsae bound.) STRETCH 2: quotedens image + caches (3.6M tok, 2 models) + screen ×2, detached, freeze 5b45cd027 | L40S | ~$6 | ~$35 |
| 07-25 22:15 | mac-b | quotedens COMPLETE + verdicts (KEEP 2/2, LOG + R24): actuals ≈ $3–4 (one runner shutdown-grace kill mid-34GB-commit; cache rebuilt in-container). mac-b overnight ACTUALS ≈ $12–13 total. Recomputed total ≈ $1 + $13 + $19 ≈ $33 | — | actuals | ~$33 |
| 07-26 11:50 | mac-b | day-2 W1: dialevel R11 order-mechanism ladder — image build + 3× (in-container cache rebuild + L0–L4 probe fits), detached, freeze ede97e206, hf-token secret (gemma GO) | L40S ×3 seq | ~$3 | ~$36 |
| 07-26 11:55 | mac-a | day-2 W2: diafaces screen (ttrend+dqgap on reused dialevel substrate) — image build + 3× (in-container cache build if W1 hasn't already + 2-face screen), PARALLEL containers (stated deviation from seq-.remote: 14:30 panel-gate clock; CARD § 8), detached, freeze 0736111132, hf-token secret (gemma GO) | L40S ×3 par | ~$10 | ~$46 |
| 07-26 12:15 | mac-b | W1 COMPLETE + verdict (MIXED 3/3, LOG + R25 ALL PASS): 17-min pipeline, caches hit volume HF weights, ACTUAL ≈ $1 → −$2 corr. mac-b day-2 actuals ≈ $1 of $60 cap. Recomputed total ≈ $44 (incl. mac-a W2 est ~$10; union-merge) | — | −$2 corr | ~$44 |
| 07-26 12:55 | mac-a | diafaces screen COMPLETE 3/3 + repatriated (gpt2 ~10 min, gemma ~45, llama ~55 on L40S; caches were W1 cache-hits). ACTUALS ≈ $4–5 vs $10 est → −$5 correction. Recomputed total ≈ $34 (mac-b W1 actuals) + ~$5 = ~$39; mac-a day-2 actuals ≈ $5 of $120 | — | −$5 corr | ~$41 |
| 07-26 12:50 | mac-b | gemma overnight-card fills (mac-local approved, cap $20, order slen→refmark→quotedens, drop-on-panel-request): 3 drivers patched (hf-token secret + in-container per-key caches), frozen cards/pins UNCHANGED; sequential detached launches | L40S seq | ~$5 | ~$46 |
| 07-26 13:15 | mac-a | GATED PANEL (gate fired dce8d085d): diafaces tt on gpt2/hs7, freeze 7ba2e10fd — 1× H100 main block (99 cells, workers 6; H100 per Han amendment a68c364a3, GPU-bound pools) + 3× L4 high-CPU tsae (GPU-idle stage, d768 buffers), detached, payloads persist to Volume | H100 ×1 + L4 ×3 | ~$10 | ~$56 |
| 07-26 13:20 | mac-b | gemma fills DROPPED mid-run-1 on mac-local reassignment (panel-2 support): slen app stopped in-screen; ACTUAL ≈ $1 (gemma replag cache + screen partials persist on Volume — resumable post-deadline); refmark/quotedens never launched → −$4 vs est. mac-b day-2 actuals ≈ $2 | — | −$4 corr | ~$53 |
| 07-26 13:32 | mac-a | tt panel STOPPED ~5 min into cells on gate amendment 187c51022 (~$1–2 burned); RACE RESOLUTION 6e2f18e4e then reinstated tt — RELAUNCHED ~13:40 (image cached, cells restarted clean; ~$10 est line above still governs) | — | +$2 | ~$58 |
| 07-26 13:45 | mac-a | PANEL 2 (authorized in 6e2f18e4e, cap raised $120→$200): dq on llama31/hs14, freeze cfa341c34 — 1× H100 main (99 cells) + 3× high-CPU L4 64GB tsae (d4096, 62–77 min class), detached; freeze-13:30 slip disclosed in PANEL2_CARD, launch-13:45 met | H100 ×1 + L4 ×3 | ~$25 | ~$83 |
| 07-26 14:55 | mac-a | v2-COLUMNS DEFECT (LOG entry): tt panel completed v1-only (~$8 actual, rows kept non-quotable); dq panel STOPPED ~45 min in (~$8 burned, no payloads). RE-RUN both at freeze db677a4b8 with oprate V2 block: dq (H100 + 3× L4-64G) + tt (same shapes, d768 cheap) | H100 ×2 + L4 ×6 | ~$35 | ~$118 |
| 07-26 (define ~17:15) | mac-a | dq main landed 89/99 ok (10 OOM/CUBLAS heavies at 6-worker co-residency); RE-PASS: 10 cells --only-cells, H100 workers 3, pin 931c016e6 (ceiling amendment db54f6764 applies; scheduling-only) | H100 ×1 ~30min | ~$4 | ~$122 |
