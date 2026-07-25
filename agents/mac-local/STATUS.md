# Working state — agent `mac-local`

**Last rewrite:** 2026-07-25 (pre-compact; endgame allocation LOCKED).
Read with `private/rebuttal_plan.md` (untracked) and the transcript.

## Who / where
Local CC on the Mac at `~/research/projects/temp_xc`, branch `arxiv`.
Role: orchestration + review. NEVER commit/quote `private/` content.
Box warning: case-insensitive checkout — on phantom dirt after a pull,
`git ls-files | tr A-Z a-z | sort | uniq -d`.

## THE SITUATION (endgame)
Reviews 5/4/1, R3 swing. **Deadline 2026-07-27; check-in Sunday
2026-07-26 10:00 PT.** Everything through the 2026-07-25 overnight
wave is REVIEWED & APPROVED (LOG is the binding record).

**Scoreboard — say this precisely, it matters:**
- **ONE confirmed TXC case study**: λ̂ backtracking (Ward, R1-Distill
  L12). TXC-pre 0.13→0.19→0.21 over T=2/4/8 at matched budget vs
  T-SAE 0.154, per-token 0.113; rise exact p=0.0093; pre/T8 CI now
  [0.179,0.235] at n=6. **pre-vs-T-SAE STILL formally unbounded**
  (tsae arm stuck at n=3 — buffer-path cost, fix refused because it
  breaks train_key comparability). Say so in the rebuttal.
- **ONE partial**: hedging panel NEGATIVE overall, but a bounded
  single-point win at T4 (CIs exclude 0).
- **FIVE Stage-1 KEEPs** (λ̂_sc, oprate ver, oprate case, qrate,
  vslope) + tss/novelty KEEP-PENDING-REVIEW. **KEEPs license panels,
  NOT case studies — never let "5 KEEPs" become "5 case studies".**
- Kill table with receipts (forbidden-word, emotional, replag,
  hedging, eqdens, vlevel, redundancy, refusal-as-posed D7…).

**Program findings adopted (2026-07-25):** (1) **order does not matter
anywhere** — g_order −0.004…+0.008 across 5 targets; the order leg is
a reported NEGATIVE, not an open search item; every window advantage
here is regime-2 order-free aggregation. (2) **No card may score
against a max-over-arms "best window"** (runpod-e) — fix probe class,
control width, foreign-context nulls. (3) `doc_mean_only_auc` =
disclosure statistic that TRIGGERS A CONTROL, never a kill bar.

## ⏭ ALLOCATION LOCKED (Han, 2026-07-25: "breadth — two panels" + factory)
- **runpod-d → `briefings/stage2-oprate.md`** — Stage-2 on oprate
  (`rate_case` primary), the shot at CASE STUDY #2. Independent
  candidate; linear pool carries the gain on Ward.
- **runpod-e → `briefings/stage2-fineweb.md`** — Stage-2 on punctint-q
  (primary) / tss (secondary): CASE STUDY #3 on a different corpus ×
  3 models = the breadth axis. Doc-identity control BINDING (q is
  0.901). Tests whether a sparse window code captures the NONLINEAR
  gain e found on fineweb (Ward is linear; fineweb is not).
- **runpod → `briefings/candidate-factory-broad-3.md`** — ledger
  re-vet under the order-NEGATIVE + estimator findings, 2–3 new
  bundles. Nothing built now can reach a panel before the deadline;
  that is accepted (pipeline for next round).
- **runpod-b → `briefings/mirror-probe-truth.md`** (AMENDED) — in
  flight, nothing pushed since 21:47 on 07-24. Its plan item 1
  (known-truth probe swept to p/n = 1.0) is the priority.
- **runpod-c → `briefings/em-redo.md`** — em Phase A/L9 arm
  progressing; unreviewed.

**Both panels MUST carry `lambda_probe_v2` in `eval_extra`** so every
row has paired v1+v2 columns ⇒ the methods decision never forces a
re-run. Claim on v1 (leaderboard-canonical) until the rule fires.

## ⏭ MY QUEUE
1. Review the two panels as they land (they are the rebuttal's
   marginal content).
2. **Methods rule fires on runpod-b's mirror receipt.** AMENDED: the
   four branches are evaluated at **MATCHED p/n swept to 1.0**, never
   at canonical mirror budget — b caught that my original briefing
   would have forced a false DECLINE (p/n 0.01 vs the panel's 1.0).
   A mirror result at p/n ≪ 0.1 fires NO branch. Branch 4 (v1 stays
   canonical, diagnostic ships as a caveat) is the Saturday-midday
   default and costs no headline — ordering survives under all four.
3. **Open threat, no owner:** if per-token probes attenuate faster
   than window probes at small corpora, every Stage-1 screen gap is
   overstated (would touch all 5 KEEPs). Does NOT threaten Stage-2
   panels (trained dictionaries at matched budget). Until checked,
   quote screen gaps with the training size stated.
4. runpod-c em-redo review when it stops.
5. **Sunday check-in distillation** (mine): headline + receipts, the
   two new panels, the kill table, the order-NEGATIVE finding, the
   probe-capacity story + decision state.
6. Keep `private/rebuttal_plan.md` current.

## Standing context
- Rebuttal-quotable: λ̂ rise (p=0.0093, v1 numbers); shuffle receipt;
  dissection § 7; T-SAE fairness receipt; split-integrity receipt
  (zero leakage); dip = cause-not-established (never "dilution");
  order-NEGATIVE as an honest scope statement. **Do not type new
  absolute panel numbers until the methods rule resolves.**
- Key science: ambience → regimes → subtype rule → T-taxonomy; FIVE
  g(ℓ) shapes; conversion = the recurring killer; screen↔panel
  convention mismatch; Ward-linear vs fineweb-nonlinear window gain.
- Platform note: "byte-identical" claims are per-platform (x86↔ARM
  last-ulp drift seen, harmless).
- Git: clean, pushed.
