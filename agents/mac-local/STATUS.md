# mac-local STATUS — COMPREHENSIVE COMPACT SNAPSHOT (2026-07-27 ~21:45 London, date-checked)

**I am mac-local: hub/orchestrator/reviewer.** Wall-clock discipline:
stamp LOG entries from `date` (two drift corrigenda on record — commit
order is always authoritative). Post-compact: read THIS file, then LOG
tail from `c1c5c949e` (budget raise) forward for the day's arc.

## THE BIG PICTURE

NeurIPS rebuttal deadline 13:00 BST TOMORROW (07-28); responses
amendable until Aug 3 (meeting transcript). Dmitry drafts responses
midday Chicago; 3pm PT check-in (23:00 London). All numbers PTR
(team-ratification pending) unless noted. The paper story: probing +
RLHF land honestly order-free/no-T-win with the controls the paper
lacked; λ̂/backtracking + dialogue carry the positive side; "the claim
narrows to where it is true."

## CRITICAL OPEN THREAD — RM identity investigation (Han challenged)

- runpod-1 claimed relu-mix ≡ btk-only TRAINING-IDENTICAL at probing
  sweep configs (bit-identical ckpts, 4/4 equivalence pairs) → I
  approved HALT of remaining RM cells as information-free.
- Han challenged (theoretical prior: BatchTopK should differ —
  Dmitry's dead-latent mechanism). MY INDEPENDENT CHECK (local, over
  results/leaderboard.jsonl): 30/30 landed relu-mix rows vs btk-only
  twins, **1,230 shared float fields, worst |Δ| = 0.000e+00** —
  identity REAL on landed rows.
- Theory reconciliation (in LOG): compositions differ ONLY when TopK
  selects negative pre-activations (rectify-after-select wastes
  slots). Positive-rich pools (+3σ margin) ⇒ ReLU-after-TopK is a
  no-op ⇒ exact identity. Same pattern as hunt's R30 (which measured
  its own thin-pool divergence boundary). Dmitry's mechanism lives
  BEYOND the boundary (paper-era k_win=8·T thin pools) — no
  contradiction.
- **GAP: landed rows are low-T; high-T identity is extrapolated
  (T-invariance claim). TWO HARDENING TESTS ORDERED (runpod-1,
  before any certificate entry): (a) POSITIVE CONTROL — checker must
  report DIVERGENCE on a thin-pool config (else instrument broken,
  halt void); (b) HIGH-T TWIN — one relu-mix TXC-pre cell at T16 +
  twin the new T10 cell, diff weights+metrics.** Certificate posts
  only after (a) diverges AND (b) is identical. Halt stands
  meanwhile. runpod-2's RLHF equivalence gate inherits the same
  standard (checker + own positive control BEFORE cancel-or-train).

## DELIVERABLES STATE (rebuttal figures)

- **Probing**: FINAL figs committed — per-k family (k5, k20),
  SAEBench-36 headline (CT pair winogrande/wsc excluded per
  camera-ready convention e77574ffd) + 38-task robustness twins.
  Dual-rendered (me + runpod-1) digit-identical. k-INVERSION
  finding ratified: k=20 declines with T; k=5 U-shaped, T16 ties
  SAE band (+0.001), l0 confound RULED OUT (cross-k identity of
  serving). Quote licence: "probe-budget-dependent, no monotone
  window win at any k" + framing guard (12:25: level story leads;
  order-gap = cross-task comparison or decline-mitigation ONLY).
- **RLHF**: btk-only 3-seed set: T{1,2,5} n=3, T8 n=3 (s2≡s42 to 4
  decimals), **s2_T16 the last cell, ETA ~21:00-21:15 London → FINAL
  3-seed fig render by runpod-2**. Interim 2-seed fig committed.
- **T6+T10 EXTENSIONS DIRECTED (Han)**: probing T{6,10}×3 seeds on
  freed GPUs 0/1 overnight (k_win=20·T; REAL T10 vs the paper's
  PHANTOM T10 label — A12 rebuttal note); RLHF T{6,10}×3 on GPU 2
  after FINAL + equivalence gate (k_win=100·T). 7-point fig
  re-renders on landing.
- **λ̂ (backtracking intensity)**: shuffle-overlay retrain PASSED
  anchor gates cells-so-far in-band; tsae anchor cells CPU-bound
  (venue note), drain imminent, **verdict + fig TONIGHT** (chased,
  answered). Identity tol 2e-3 (A1/A2 amendments approved,
  mechanism-quantified) + my 5e-3 quoted-gap floor.
- **ttrend**: overlay UNLICENSED (anchor gate FIRED: retrained T32
  +0.022 > 1σ_quoted — different objects; STOP honored, no
  re-rolls). Deliverable = committed two-instrument fallback fig
  (quoted trained T-sweep + screen shuffle curve, labels on-figure).
  Venue-effect datum recorded. **Dmitry ruling: ttrend = appendix,
  OUT of rebuttal.**
- **dq**: fig2 exists (3 seeds); passed-then-demoted framing; Dmitry
  ok'd for within-SAE rebuttal use.
- **Meeting deck**: private/meeting_tsweep_plots_2026-07-27.pdf
  (7 pages, served the 17:00 meeting; superseded for next meeting by
  the per-k + 7-point renders when they land).
- **Sparsity convention AUDITED program-wide** (Han's oversight
  flag): ONE convention everywhere incl. Aniket's sweep — constant
  per-token budget, k_win = k_pos·T. Receipts in LOG (18:22-entry).
  Rebuttal sentence ready if asked.

## FLEET ROSTER + LANE STATE

- **runpod-1** (3×H100 pod, GPUs 0/1): P1 grid CLOSED (zero fails);
  RM HALTED (identity); owes: hardening tests (a)+(b) → identity
  CERTIFICATE entry; T{6,10} probing extension; per-k fig re-render
  at 7 points. eval_consumes seam owner-review CLOSED (T=1 test).
- **runpod-2** (same pod, GPU 2): s2_T16 draining → RLHF FINAL fig;
  then RLHF equivalence check (with positive control) → relu-mix
  card CANCELLED if identical, else train; then T{6,10} RLHF
  extension overnight.
- **runpod-a** (2×H100 pod A, GPU 0): released from RM hold →
  **reask screen card** (the wave-3 survivor: 548 events, floors
  ≤.57, anti-dup clean) — freeze+run when GPU frees; then re-entry
  packets optional. Wave-1/2 hunt CLOSED (tret/xtrend/sage/tret_py
  breadth w/ replication receipts; cnov panel DEFERRED to Aug-3
  window, prep durable).
- **runpod-b** (pod A, GPU 1): λ̂ overlay drain → gate verdict + fig
  TONIGHT (sole open lane); then W2 replication receipts row
  updates staged.
- **runpod-c** (2×H100 pod B, dedicated): T-SCALING HILL-CLIMB.
  CARD 0 split frozen (dev-8/holdout-28, pyramid gates); candidate
  1 = RECOVERED txc_pro (docs/recovered/txc_pro_phase5b_subseq_h8.py
  — NOT from yaml: h_size=d_sae//5, k_pos=20, k_train100/k_inf200
  asymmetry pre-registered, revive as NEW id txc_pro_r1);
  **ZERO real T-scaling evidence exists for txc_pro** (A12-aware
  dig) — no prior. No-idle directive: baseline twin on GPU 0 NOW,
  txc_pro_r1 L1 on GPU 1 when plugin lands; owes utilization
  report. Their eval-dispatch seam CLOSED (25/25 tests).
- **mac-c** (mac, CPU + Dmitry's RunPod key): LANE CLOSED $0 —
  dharm KILLED (155.6 tok/chain, 3 position strata — corpus fails
  the clock bar; predictions confirmed; schema leak disclosed
  pre-run); msdose_r1 verdict instrument frozen (realized-vs-
  realized, pre-registered kill bands); sycgen frozen (age face
  single-carry after clock-bar self-demotion of the rate face;
  "geometry can kill but not clear" = standing rule for generated
  corpora). SAFETY_TASK_MENU (16 + §10's 7 entries) = wave-3
  source; emoinst WRITEUP row corrected (ran-and-KILLED 07-24).
  Available for next assignment.
- Retired: mac-a/mac-b (→ runpod-a/b, dirs removed).

## SECURITY / KEYS / GOVERNANCE

- HF tokens: ~/.tokens/{hf_token,hf_token_datasets}; pods have
  /workspace/.tokens/ (gh, hf×2; NO Modal, NO Anthropic-API — Max
  /login per pod session).
- **Dmitry's RunPod key: keychain `dmitrys-runpod-api-key`**
  (plain `runpod-api-key` RESERVED for Han's future key). Verified
  live (balance ≈ $544, acct limit $80/h). GOVERNANCE (binding, in
  actmix-shared): mac agents only; $10/h/agent; terminate-when-done
  + API-verify; NO writes to pods you didn't spin up (incl. Han's 3
  hand-provisioned); name `<agent>-<purpose>-<mmdd>`; ledger both
  ends. S2 key: keychain `s2-api-key`, hygiene per briefing (mac-c
  never used it — clew sufficed). ALL tokens rotate post-weekend.
- dharm HF gate: mac-c's request packet went to Han; access may
  have landed (their kill ran) — moot now the corpus is dead.

## HOUSE RULES (living)

Pull-rebase before push; LOG conflicts keep BOTH blocks
(python re.subn recipe), stray-marker grep after EVERY resolution
(baseline count 1 = the rule quoting itself); scorer-committed-
before-deciding-result; freeze→pin→ledger; venue amendments = one
disclosed line; eval_extra namespacing for non-canonical cells;
PTR everything, mac-local ratifies on push; binding wave-3 bars:
out-of-window-by-construction + clock-stated-first; generated
corpora: per-token baseline binding, geometry can kill not clear;
identity-claim standard: independent verification + positive
control + far-end spot-check (this window's addition).

## WATCHER / OPS

Hub watcher: SESSION scratchpad `scratchpad/watch_origin.sh`
(NOT repo scratchpad); exit 0=arxiv push, 3=neurips-aniket push,
2=deadline (now set 2026-07-28 12:59). Re-arm via run_in_background
after EVERY beat (never inline `&`). Multiple stale instances can
pile up — each fires once on the next push; check
`git log HEAD..origin/arxiv`, empty = own-push echo. Pod ssh =
PTY-only piped-stdin (`-tt`, stty -echo for secrets, grep -av 2004
filter). Pods: old=j42plcul70a2es-64410eb7, A=0lmrs9lk8apyhm-644121b8,
B(runpod-c)=l2bp61kg82epel-64411fb1, all @ssh.runpod.io -i
~/.ssh/id_ed25519.

## IMMEDIATE NEXT ACTIONS (post-compact checklist)

1. Watch for: runpod-1 hardening tests (a)/(b) results → rule on
   the certificate; runpod-2 RLHF FINAL fig + equivalence-gate
   result; runpod-b λ̂ verdict + fig; runpod-c utilization report +
   L1 first signals; runpod-a reask card freeze.
2. Review-on-push everything; ratify or bounce.
3. Rolling: T6/T10 rows land overnight → 7-point fig re-renders →
   next-meeting deck refresh.
4. Ledger sweep + pod-idle checks each beat; hunt envelope ≈$60
   spent of $200; pods within caps.
5. 3pm PT (23:00 London) check-in support if Han asks; Dmitry's
   draft may need the quote licences (all in LOG: k-inversion
   sentence, identity certificate pending, framing guards).
