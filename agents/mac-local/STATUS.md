# Working state — agent `mac-local`

**Last rewrite:** 2026-07-26 ~02:20 PT (PRE-COMPACT; next window =
OVERNIGHT EXECUTION). **Resume by reading
`briefings/overnight-mac-modal.md` FIRST — it is the complete
operating plan for tonight and this file only summarizes it.** Read
with `private/rebuttal_plan.md` (untracked).

## STRUCTURE TONIGHT (Han, 2026-07-26 — supersedes the single-executor plan)
I stay ORCHESTRATOR. Two fresh local agents execute autonomous loops
in their own clones (~/research/projects/agents/<id>/temp_xc):
**mac-a** = Modal bring-up + tsae/T1 seed top-up {3,4,5} (bounds R5;
cap $150; briefings/overnight-mac-a.md). **mac-b** = B8 slen screen →
refmark → quotedens (cap $100; overnight-mac-b.md). Shared ops doc:
overnight-mac-modal.md; **shared spend ledger briefings/MODAL_SPEND.md
($500 hard / $400 soft total)**. Modal smoke PASSED (A10). Roster
entries mac-a/mac-b added (set_agent_env.sh + agents/README).

## MY OVERNIGHT QUEUE (orchestrator)
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
