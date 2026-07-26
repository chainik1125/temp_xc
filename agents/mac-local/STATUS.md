# Working state — agent `mac-local`

**Last rewrite:** 2026-07-26 ~10:50 BST (OVERNIGHT COMPLETE; pivot
executed). **Resume by reading `briefings/overnight-mac-modal.md`
FIRST.** Read with `private/rebuttal_plan.md` (untracked).

## DONE tonight (see LOG 2026-07-26 mac-local entry, commit f07a1f3a5)
- Queue item 1 COMPLETE: expedited gate-reviews of BOTH panels.
  oprate NEGATIVE **APPROVED** (artifact-exact). fineweb cross-model
  verdict **APPROVED with 2 corrections filed** (gpt2 last-digit
  seed-means; gemma within-doc T2 = 0.067 not 0.047 — monotone-from-T2
  withdrawn, T4-onward growth + window-over-token contrast stand).
  RECEIPTS rows all clean — corrections were LOG-narrative only.
  Flush commit c8ab5fa0 owned by me in the LOG (preservation-flush
  labeling lesson adopted). Both stage2 briefings retired. Suite
  green (332+1skip). **Distillation DRAFT already written:**
  `private/sunday_distillation_2026-07-26.md` — update with overnight
  landings, deliver by 09:30 PT.
- Origin watcher running in background (scratchpad/watch_origin.sh):
  re-invokes me on any push to arxiv or at the 07:00 PT pivot.
- mac-a / mac-b: NOT YET LAUNCHED at last check (ledger ~$1/$500).

## STRUCTURE TONIGHT (Han, 2026-07-26 — supersedes the single-executor plan)
I stay ORCHESTRATOR. Two fresh local agents execute autonomous loops
in their own clones (~/research/projects/agents/<id>/temp_xc):
**mac-a** = Modal bring-up + tsae/T1 seed top-up {3,4,5} (bounds R5;
cap $150; briefings/overnight-mac-a.md). **mac-b** = B8 slen screen →
refmark → quotedens (cap $100; overnight-mac-b.md). Shared ops doc:
overnight-mac-modal.md; **shared spend ledger briefings/MODAL_SPEND.md
($500 hard / $400 soft total)**. Modal smoke PASSED (A10). Roster
entries mac-a/mac-b added (set_agent_env.sh + agents/README).

## MY OVERNIGHT QUEUE (orchestrator; item 1 DONE, distillation drafted early)
1. Expedited gate-reviews of the two completed panels (oprate
   NEGATIVE; fineweb gemma/gpt2/llama) — no compute; mark "expedited
   — full depth at team review". Do FIRST.
2. Rolling review loop: pull → review a/b pushes → amend their
   briefings if needed → watch the ledger. (Ask Han to run
   `/loop 45m` on me if unattended review cadence is wanted.)
3. Ratify RECEIPTS updates the agents PROPOSE (they never edit
   RECEIPTS themselves tonight).
4. Panel-approval gate: a B8 panel tonight needs MY written LOG
   approval (conditions in the shared doc).
5. **HARD PIVOT 07:00 PT: Sunday distillation** (contents in the
   shared doc §; quote ONLY via RECEIPTS.md; draft in private/;
   deliver by 09:30 PT).

## STATE OF THE PROGRAM (for the distillation)
- **ONE confirmed case study**: λ̂ backtracking (v1 numbers; rise
  p=0.0093; pre/T8 n=6 CI [0.179,0.235]; **pre-vs-T-SAE NOT bounded**
  — R5; item 2 tonight tries to fix exactly this).
- **oprate: COMPLETE 84/84, NEGATIVE** (pre-registered branch;
  RECORD §3d) — case study #2 dead on sound grounds. UNREVIEWED.
- **fineweb punctint-q: COMPLETE** — gemma NO-RULE-FIRES (K2 fails on
  v1), gpt2 WEAK (T4 bounded), llama NEGATIVE; paired-v2
  bounded-positive story = 3rd receipted v1-conservatism instance,
  2nd corpus. §10 re-quote reconciled the nonlinear-vs-linear tension
  (win_mean dilution artifact). UNREVIEWED.
- METHODS DECISION taken: **v1 canonical**; v2 reported never claimed;
  PROBE_V2_SPEC (with §0 lower-bound limitation) = post-deadline
  freeze candidate.
- **RECEIPTS.md = quote source of record** (24 claims ALL PASS;
  R5/R10 negative-space; extend + `receipts_check` for any new claim).
- Checkpoints of both panels mirrored:
  `han1823123123/temp_xc_a40_checkpoints` (private) — see
  checkpoints/HF_MIRROR.md; eval-only plans verify weights FIRST.
- Kill/negative table + amended order finding (never "anywhere";
  dialevel counterexample; recency hypothesis → B8 tests it).
- PAUSED (post-deadline): em-redo review, factory, v2 adoption,
  estimator-attenuation check.

## Discipline reminders for the executor window
Commit-then-run (freeze + push BEFORE first cell; pin containers to
the freeze commit). Containers never push git — repatriate results,
merge locally with dup-key check. LOG lines as `mac-local (executor)`;
every overnight verdict = PENDING TEAM REVIEW. Self-review hazard is
named in the briefing — pre-registration is the compensation.
Briefings live: overnight-mac-modal (mine), em-redo (paused),
a40-bootstrap (historical, pod gone).

## Git: clean, pushed after this rewrite. Suite 333 green local.


## ACTMIX PHASE (Han, ~20:15 London 2026-07-26) — CURRENT
Post-meeting pivot. Activation-mixing finding on record (LOG
~20:25): txc_base TopK→ReLU k=8T (paper d(perf)/dT biased down);
v2 backbone ReLU→BatchTopK (sae 4.4/8 most handicapped, hunt
margins flattered — guarded by our l0 notes). Allocation:
mac-a = relu_mode impl + ttrend calibration (Dmitry's re-run gate,
want before 9am PT) + KEEP survives-the-fix; mac-b = forensics +
salvage shortlist (ACTMIX_FORENSICS.md); mac-c NEW (clone created,
~/research/projects/agents/mac-c/temp_xc) = branch archaeology
(paper = han-phase7-unification / dmitry-em-repl /
aniket-runpod-ward-stage-a) + HF inventory han1823123123 (3
paper-era datasets; token ~/.tokens/hf_token_datasets, 600,
verified; NEVER in git; rotate post-weekend) →
COMPOSITION_AUDIT.md; runpod-1 = probing, runpod-2 = EM (H100s,
Han spinning up — briefings pushed, WAITING on pod access from
Han; Phase A btk-only immediately, Phase B paper-match blocked on
mac-c). Backtracking = Aniket 100%, hands off. Arm labels
relu-mix/btk-only/paper-match mandatory. $150/day/person cap.
MY queue: dispatch mac-c; confirm mac-a/b pickup (liveness rule);
freeze-review calib + forensics + audit; pod driving once access
arrives; 9am PT one-pager (composition audit + calibration read).

## (superseded) SALVAGE CLOSED ~19:15 London 2026-07-26 — BOTH DELIVERED
**ttrend TXC-post = KEEP T{16,32}** (R28+R29 ratified by me, team
pending; pooling-free L1 lane carries; fig4 in WRITEUP §4).
**Novelty cross-ratified** (CROSSRATIFY.md FINAL, doubly pending;
gate gap closed favorably; 3 flags for Andrii: T-pin, pooled-audit
defect, position-triage contradiction). dq = Task 3
passed-then-demoted. WRITEUP restructured 15907d434 (9 sections).
Sprint actuals ≈$10; program ≈$96 actuals of $500. REMAINING QUEUE:
team ratifications (R22 caveats, R28/R29, quote licences, novelty
handoff to Andrii); HF checkpoint mirrors BEFORE token rotation;
token rotations post-weekend; post-deadline items (em-redo,
PROBE_V2_SPEC freeze, factory r4, rate_ver, gemma slen-fill).
Salvage briefings deletable per briefings/README convention once
team ratifies (kept until then as the sprint's terms of reference).

## (superseded) SALVAGE PHASE (Han, ~16:30 London 2026-07-26)
dq DEMOTED (surface-reading fragility at T≥16) to order-mechanism
support; salvage briefed: mac-a = ttrend POST fresh-seed panel
(salvage-mac-a.md), mac-b = txcwin novelty cross-ratification
(salvage-mac-b.md), shared = salvage-shared.md. Budget fine < $500
total (spent ≈ $87).

**EXECUTION MODE (corrected ~16:55): the PRIOR STANDALONE worker
sessions are LIVE and own their lanes.** My ~16:36 dispatch of
fresh executor subagents was a near-miss on a false dead-session
premise — both stood down $0/untouched (BOTH confirmed; mac-b's
second-eyes pre-flight yielded the report.py pooled-pseudo-cell
audit finding, now in LOG). Adopted rule in LOG: a quiet clone is
UNKNOWN, not dead — positively check for a live session before
re-dispatching. W1 in flight (freeze 50af78f12, Modal app launched
~16:38, est ~$10); W2 GAP-A landed (2e163e126), GAP-B rawgate
launched (ledger 789b4f10d, est ~$4). Program est ≈ $101 of $500.

DONE 16:35–16:55: WRITEUP demotion edit pushed 9b6199be5 (title,
§1, §4 'passed-then-demoted' + objection, §5 note); distillation
§9 EVENING ADDENDUM (demotion, salvage state, revised decision
items incl. bless-the-salvage-bars-tonight); BOTH freeze-reviews
APPROVED in LOG (W1 k-resolution RATIFIED — my briefing's k=8·T
was the error, primary claiming = k_pos 8 panel-identical; W2
approved with the audit's 8B/T-pin flag); GAP-A RULING in LOG
(8B is band 2 by the card's letter via oracle-position V-all;
surface-quiet survives only in decomposed form + position-residual
instrument caveat; V-win joint fit + T16 nuance required in memo).

MY queue: (1) freeze-review both salvage cards PRE-RESULTS the
moment freeze commits land (watcher fires on push); (2) rolling
review + ratification of salvage verdicts/receipts; (3) WRITEUP +
distillation salvage-outcome updates (possible ttrend/novelty
replacement §4 + fig4 integration); (4) 18:00 London check-in
decisions: R22 caveats first, verdict ratifications, quote
licences (dq licence now order-mechanism-scoped), salvage-bar
blessing, HF checkpoint mirrors before token rotation.
Watcher: scratchpad/watch_origin.sh RUNNING; re-arm after each
firing. /loop cron 44d0aa83 active (13,43 * * * *).

## PIVOT EXECUTED 2026-07-26 ~11:00 London — DISTILLATION FINAL
Overnight wave COMPLETE and fully reviewed: R22 (tsae bound, 2
caveats for team), R20/R21 (slen KEEPs + ladder collapse), R23
(refmark kill), R24 (quotedens KEEP, T≤32). RECEIPTS 28 claims ALL
PASS at close; suite 332+1skip green; ledger ACTUALS ~$33/$500;
both agents idle, queue-complete. **DELIVERABLE:
`private/sunday_distillation_2026-07-26.md` (FINAL)** — agenda item
1 = R22 caveat ratification; item 6 = mirror the 3 Volume
checkpoints to HF BEFORE token rotation. Check-in 10:00 PT
(18:00 London).
