# mac-local STATUS — PRE-COMPACT SNAPSHOT #3 (13:13 London 07-28, date-checked)

**I am mac-local: hub/orchestrator/reviewer (no compute).** Stamp
from `date` at write (interpolate, never pre-write — many
corrigenda). **Post-compact: read THIS, then the LOG tail from
~12:30 forward** (the pace-anomaly → substrate-verdict arc).
Deliverable surfaces: `REBUTTAL_HANDOFF.md` + `REBUTTAL_CODE_GUIDE.md`
+ `REBUTTAL_CELL_CENSUS.md` (regen: `scripts/cell_census.py --write`).

## ⚑⚑⚑ HAN'S POST-COMPACT ORDER #1 (13:5x, binding)
**"Orchestrate RLHF missing cells, MAX THROUGHPUT."** = (a) the
pf grid: 21 cells `agentic_txc_02` port on **base-l12** (A/B
SETTLED 13:5x: FVU 0.0036 vs 0.0367 — card §8's l13-IT premise
FALSE; l12 caches present, registry tag `l12base_phase7`; the 9h
pilot VOID; mismatched T5 anchors RETRACTED); (b) deferred btk
T6/s2 + T10/{s1,s2}. Sequence: runpod-2 relaunches pilot on l12 →
convergence vs upstream T5 log (upstream converged 5.8k; if ours
tracks, adopt upstream stopping ⇒ ~4× fewer steps) → G1 → wave-1
s42 (7 cells) → waves 2-3. **Map (31930ad8c): T16 = old-pod GPU 0
SINGLE-TENANT (≥72 GiB); old-pod GPU 1 = 1 lane; pod B = 2×2
lanes (runpod-c; 1.82× measured; needs l12 cache install —
hardlink installer, minutes).** Launcher = `agents/runpod-a/
run_pf_grid_lanes.sh` (fleet property: per-T GPU_FRACTION from
measured peaks 3.6→39 GiB + T16≥72; packing guard; thread budget
OMP=quota/lanes — the 2.3× oversubscription trap is an IMAGE
property). Cost model MEASURED: feed-bound 818 MiB/s = 95% wall,
T-independent, ~213 GPU-h pre-levers; fp32→fp16 feed = ~2× prize
(post-G1, equivalence receipt). 12-lane ~1.0×/lane measured (no
bandwidth ceiling; memory+cores bound only).

## DELIVERABLES SCOREBOARD (Han's 7 + extras; submission delayed past 13:00 — NeurIPS rule change, no hard deadline; amendable to Aug 3)
1+2. Probing k5/k20: **COMPLETE BOTH ARMS** — btk FINAL figs
  (+38task twin) + pf FINAL 7-pt figs (`_pf_k{5,20}`) embedded;
  E1-E3 gates passed (E3 = archived anchors interpolate). Tables =
  actmix RESULTS files (HANDOFF pointers fixed).
3. RLHF: btk = 10:15 checkpoint render SHIPPED (T{1,2,4,5,8,16}×3s
  + T6 2/3; deferral caption). pf = THE order-#1 lane above.
  Both-arms = RM_CERTIFICATE v1.0 (rmx_b 6/6; T10/s42
  tensor-grade). **T=5 SANITY: PASS** (published 0.899-0.902 =
  38-task trapezoid; ours 0.9007; RLHF papermatch = 16-digit).
4. λ̂ COMPLETE. 5. dq COMPLETE (toy-class).
6. sycgen: DELIVERED (KEEP 3/3 → 15/18 FINAL, plot+table embedded,
  twin-control quote-form v2 binding — level story, NOT order).
7. OPEN: struqpos = sound KILL 3/3 (C1 proximity; salvage
  amendment); evalage WEAK; retryesc_gen = mac-c SILENT ~9h (Han
  may reassign; mac-e recommended, not yet spawned).
Extras: T-SAE width DONE. **⚑ CAMERA-READY ERRATUM found: paper
caption/prose say 36-task but plotted fig = 38-task (receipts:
trapezoid 0.9007∈[0.899,0.902] on 38; 0.9334 on 36; appendix §c3
admits 38). One-word amendment fix recommended — Dmitry's call.
PANEL PIN: rebuttal figs = 36; never cross-quote (offset ~+0.03
T-invariant ⇒ verdicts panel-invariant).**

## FLEET (burn $14.95/h)
**OLD POD** ($8.97, Han's, 3×H100; runpod-1 + runpod-2 aboard):
GPU 0 = reserved T16 single-tenant; GPU 1 = wave-1 lane; GPU 2 =
pilot relaunch venue. **POD B** ($5.98, "t_scaling_hillclimb" =
its account name; runpod-c aboard): 2×2 wave lanes at 1.82×;
l13-IT substrate saga MOOT (l12 is the cache). **POD A TERMINATED**
(204→404 receipts; runpod-a/b closed clean, no objection, 20/20
durable; launcher preserved). mac-d = alive (renderer
wave-1-verified; L40S + pod-D terminated). mac-c = SILENT.
G2 INCIDENT re-review assigned runpod-1 (why did an
ordering-gate pass on the wrong substrate) — amendment window.

## HOUSE RULES ADDED TODAY (post-04:00)
Liveness = /proc receipts, never GPU-samples+log-size (pilot
misdiagnosis owned); venue-readiness = ARTIFACT receipts
(bytes/manifest), never process state (my stage-B HF_TOKEN
omission owned — launcher owns the completion check); explicit-
path commits while canonical jsonl live; no blind add -A rebases;
stamps interpolated at write; watcher = own run_in_background
call, NEVER inline & (two strays tonight); single instance,
liveness-check before arming; gold-visibility (KEEP → HANDOFF
same-beat); artifact-vs-subject-line: artifacts win.

## OPS
Watcher `scratchpad/watch_origin.sh` run_in_background, re-arm
after each fire. Push recipe: explicit-path commit + retry loop w/
LOG keep-both python + rebase-aware healing (dedupe check: entry
appears once). Pods: old=j42plcul70a2es, B=l2bp61kg82epel, both
@ssh.runpod.io -i ~/.ssh/id_ed25519 (piped-PTY + grep -av 2004;
single-line cmds; base64-ship scripts; scp works direct-IP only).
RunPod API = `dmitrys-runpod-api-key` (mac-only): DELETE
/v1/pods/<id> = terminate. Ledger: scope ruling 13:0x (agent-spun
vs Han-provisioned); $300 gen cap standing; tokens rotate
post-weekend.

## POST-COMPACT CHECKLIST
1. **ORDER #1: RLHF missing cells, max throughput** (above; watch
   runpod-2's l12 pilot relaunch + convergence verdict + wave-1
   launches; fill every idle lane; levers on receipts).
2. Review-on-push beats (watcher); ratify/bounce; census regen at
   landings; HANDOFF/GUIDE re-stamps at milestones; gold-visibility.
3. Item-3 HANDOFF: add pf-RLHF fig embed slot at first render
   (mac-d flagged none exists); T5-anchor-inline = runpod-2 call.
4. mac-c decision w/ Han (retryesc_gen reassignment / mac-e).
5. G2 re-review (runpod-1) lands amendment-window.
6. Final full-surface pass + readiness message when Han sets the
   submission time.

*Rewrite before any compact.*
