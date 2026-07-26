---
status: active
created: 2026-07-27 ~00:30 London
for: ALL WORKERS — overnight full-utilization allocation (Han: 12 h until UK noon; NOBODY idles)
read-first: briefings/actmix-shared.md; LOG entries dca32ce6b (EM full stop) + af2247d43 (identity discharge)
---

# Overnight allocation — 00:30 → 12:00 London 2026-07-27

Priorities (Han, binding): **TASK HUNT, RLHF, SPARSE PROBING.**
EM = Dmitry only. Every agent has a full queue below; if you
finish yours, take the next unclaimed item from § 6, never idle.
Timeline: Han resumes manual control at NOON London; team meeting
17:00 London. Everything lands PENDING TEAM REVIEW as always.

## 1. mac-a — TASK HUNT RESUMPTION (idea → screen, the priority lane)

The mixing question is CLOSED for the hunt (R30 identity). New
idea generation is UN-PAUSED. Your night: screen the next
generation of candidates along the measured hill-climb gradient
(WRITEUP § 4/§ 7 theory; LOG "WHY the winners win"):
offset-weighted trailing functionals of SPARSE, per-token-SILENT
events; **no surface marker at any T** (structural guarantee
preferred: out-of-window definitions or cross-distance
comparisons); substrate with measured order-carriage (dialogue
first); kernel timescale ~ T; expected untrained/pooled gap.

Seed candidates (screen 3–4 tonight; you may improve/replace with
better instances of the same template):
- **conversation novelty**: trailing rate of first-in-CONVERSATION
  token types on DailyDialog — the txcwin out-of-window definition
  TRANSPLANTED to the order-carried substrate (window cannot
  compute first-in-conversation; strongest structural guarantee).
- **turn-taking rate trend**: trend in speaker-alternation tempo
  (turns per token window) — marker-free cross-distance comparison.
- **question→answer latency**: distance from a question turn to
  the turn that RESOLVES it (answer identified lexically at
  label-build time, not at eval) — dq's spirit with the marker
  made non-degenerate; only screen it if the label passes your own
  visible-cue pre-measure at ALL T.
- **correction hazard in reasoning traces**: trailing rate of
  self-correction cues on the backtracking corpus with the cue
  list EXCLUDED from window visibility (out-of-window mining).
Discipline: factory_screen path (mixing-insensitive class),
visible-cue evidence line PRE-MEASURED per candidate before any
screen verdict; screens ≈ $1–3 each; budget ≤ $30 (released
Stage-3 funds within your $60 cap). Deliverable by morning: screen
verdicts + the best 1–2 with DRAFT panel cards (not frozen — team
picks at 17:00).

## 2. mac-b — REBUTTAL EXHIBIT PACK (hunt tasks) + label support

Zero-GPU, high-value: build the Dmitry-format exhibit for the
HUNT's two headline tasks from EXISTING receipts (no new compute):
- **The table**: TXC | TXC-shuffled | per-token SAE | T-SAE for
  λ̂ (backtracking intensity) and ttrend — shuffle numbers from
  R26/shuffle receipts, panel numbers from R22/R28/R29 licensed
  lanes, quote licences respected verbatim (tsae-first, l0 notes,
  sequential caveat rules).
- **Figures**: fig2/fig4-style with shuffled overlays if the
  receipts support them; WRITEUP conventions.
- Deliverable: `experiments/explorations/task_hunt/REBUTTAL_PACK.md`
  + figs, staged for the 17:00 meeting and my 11:30 one-pager.
Second hat: pre-measure visible-cue evidence lines for mac-a's
candidates on request (your baseline machinery).

## 3. mac-c — PROBING EVAL PIN + HF MIRRORS + A2

1. **Probing eval-protocol pin for runpod-1 Phase B**: from the
   census + dev branch, pin the EXACT shipped eval (k_feats grid,
   probe hyperparams, S=32 tail, pooling, selection rule,
   split/seeds) so the eval-only reproduction is turnkey; write it
   into COMPOSITION_AUDIT § 3 as "Phase-B recipe".
2. **HF checkpoint mirrors (standing queue item, before token
   rotation)**: mirror the hunt's Modal-volume checkpoints (tsae
   top-up trio + both dialogue panels + salvage/topup/calib cells)
   to `han1823123123/temp_xc_a40_checkpoints` — CPU/network only;
   manifest with train_keys; NEVER print tokens.
3. **A2 closure from the census**: reconcile the c3-k20 vs
   RLHF-k500 budget discrepancy (which cells the paper's RLHF
   table actually used is already pinned; state the c3/RLHF
   k-budget story in one paragraph for the paper team).
4. **EM archaeology + sanity checking — BACK IN SCOPE** (Han
   clarification ~00:40: only EM shuffle-ablation EXPERIMENTS are
   Dmitry-exclusive; forensics and sanity checks are ours to
   contribute). After items 1–3: resume A3 (Nura SAE composition),
   the A6 Gen-2 residue (em-nanda mapping; anything in the census
   or public artifacts that locates the 5-arm run), and READ-ONLY
   sanity checks on the Gen-1 reproduction chain — all
   $0/read-only, all handed to Dmitry as evidence, no compute.
Then § 6 backlog.

## 4. runpod-1 — PROBING, full night

Phase A btk-only grid to completion; Phase B (UNLOCKED, eval-only
on the shipped temp-bench-models cells per tbm_census) as soon as
Phase A's GPU need allows — interleave downloads now. Card(s) +
freeze-review as always. You have GPUs 0,1 all night.

## 5. runpod-2 — RLHF, full night

EM is stopped (mac-local killed the lanes ~00:15 — reconcile your
ledger, preserve logs/payloads AS-IS, completed cells
NON-QUOTABLE-pending-Dmitry). GPU 2 is yours for RLHF: freeze the
card (paper-match eval-only on txcdr-base seed-42 ckpts + btk-only
retrains at paper shapes + shuffle + T-sweep; briefing § in
actmix-runpod-2.md), launch tonight. gemma-2-2b is small — a full
grid should land well before noon.

## 6. Unclaimed backlog (take in order if your queue empties)

a. WRITEUP §-updates staging: R30 certification note + Task-2/3
   licence-format numbers (draft for mac-local ratification).
b. Gemma slen-fill resumption (parked cards, drop-on-request).
c. txcwin CROSSRATIFY follow-ups that are OURS (not Andrii's):
   the V-win figure; the pooled-audit G-6 patch PROPOSAL text.
d. Post-deadline queue grooming (PROBE_V2_SPEC freeze draft).

Ledger discipline, caps, PTR, freeze-review-in-parallel: all
standing. mac-local reviews on every push through the night.
