# ACTMIX forensics (W2) — did the ReLU→BatchTopK mixing move any hunt verdict?

**Status: PENDING TEAM REVIEW** (all classifications; nothing here re-scores
any frozen card). Author mac-b, 2026-07-26, per `briefings/actmix-mac-b.md`
(read after `briefings/actmix-shared.md`). Spend: **$0** (leaderboard-only
analysis; no runs launched, none proposed for launch here — Stage-3 re-runs
are mac-a's lane, gated by mac-local). Data: `results/leaderboard.jsonl`
(canonical), 819 hunt rows across all 8 hunt datasources, at the branch state
of this commit.

**TL;DR.** The mixing's bias direction is **pro-TXC**: at every T where a
verdict's deciding bar was measured, the per-token comparators (sae, tsae)
are handicapped MORE than the window arms, so TXC-vs-baseline margins were
**flattered**, never depressed. Consequences: **(1) no kill on the record is
attributable to the mixing** — every margin-based kill was made against a
flattered TXC and can only harden under `btk-only`; the remaining kills are
screen-stage (mixing-insensitive **by construction**: the screens fit probes
on raw model activations, no SAE in the loop) or evidence-line kills that
bind at T ≥ 8 where window-arm realization is ≥ 0.98. **(2) The exposure is
concentrated in the KEEPs' comparator legs** (λ, ttrend-salvage, dq, and the
tt round-1 quote licence), ranked in § 6 for mac-a's KEEP-recheck lane.
**(3) The ranked salvage shortlist from the kills/parks table is empty of
HIGH items** — the honest Stage-2 result. One conditional MEDIUM (punct/gemma
bound top-up) is listed with cells + cost; it is a power fix, not a mixing fix.

---

## 1. Method

Scan: group the 819 hunt rows by `(datasource, arch, T, k_pos, trained)`
(`trained` = `training_cfg.n_steps > 0`; nominal k from
`training_cfg.arch_hparams_override`; realized from `metrics.l0_per_window`).
**Normalization anchor = the matched untrained cell** (same datasource, arch,
T, k), not the analytic nominal — because untrained cells realize the analytic
nominal almost exactly (the on-record falsifier: "Every untrained matched cell
realizes l0_per_token = 8.000 (±<0.01) at every T", LOG 2026-07-25) **except
untrained `txc_batchtopk_pre`**, which itself under-realizes at T ≥ 4 (see
§ 2c; also already on record: "Untrained pre realizes 7.54–7.93, declining
with T", qrate panel entry). Analytic fallback where no untrained twin exists:
per-window nominal = k_pos (sae, tsae, post) or k_pos·T (stacked, pre).
Oracle check: 107 untrained cells, 18 deviate > 1% from analytic — all 18 are
`txc_batchtopk_pre` at T ≥ 4.

**Screens vs panels — the structural split.** The hunt had two instrument
classes. *Screens* (`factory_screen.py` + per-task screen scripts) fit probes
(`fit_probe`, ridge/logistic) directly on model activations — grep confirms
**zero** references to `temp_bench` archs or BatchTopK in the screen path. No
SAE ⇒ no ReLU→BatchTopK composition ⇒ every screen-stage verdict is
mixing-insensitive by construction. *Panels* train the 5-arch grid through
`temp_bench` and are the only place the composition can bite. Receipts
R10, R11, R17, R20, R21, R23, R24, R25 are screen-stage; R5/R13–R16, R18,
R19, R22, R27, R28, R29 and the tt round-1 / oprate-case panels are
panel-stage.

Reproduction: the scan script is deterministic pandas-free stdlib; rerun =
`.venv/bin/python` over `results/leaderboard.jsonl` grouping as above
(script preserved in the LOG PTR entry's commit).

## 2. Fingerprint tables

### 2a. Realized/nominal by (arch, T) — trained cells, range across substrates

| arch | T | k | cells | min | med | max |
|---|---|---|---|---|---|---|
| batchtopk_sae | 1 | 8 | 8 | 0.548 | 0.567 | 0.762 |
| tsae | 1 | 8 | 8 | 0.711 | 0.829 | 0.951 |
| stacked_batchtopk | 2 | 8 | 6 | 0.725 | 0.748 | 0.827 |
| stacked_batchtopk | 4 | 8 | 6 | 0.845 | 0.895 | 0.913 |
| stacked_batchtopk | 8 | 8 | 6 | 0.927 | 0.957 | 0.971 |
| stacked_batchtopk | 16 | 8 | 6 | 0.967 | 0.987 | 0.990 |
| stacked_batchtopk | 32 | 8 | 2 | 0.979 | 1.001 | 1.001 |
| txc_batchtopk_pre | 2 | 8 | 6 | 0.727 | 0.744 | 0.876 |
| txc_batchtopk_pre | 4 | 8 | 8 | 0.862 | 0.896 | 0.989 |
| txc_batchtopk_pre | 8 | 8 | 8 | 0.956 | 0.988 | 1.042 |
| txc_batchtopk_pre | 16 | 8 | 6 | 1.008 | 1.012 | 1.039 |
| txc_batchtopk_pre | 32 | 8 | 2 | 1.022 | 1.050 | 1.050 |
| txc_batchtopk_post | 2 | 8 | 3 | 0.687 | 0.709 | 0.712 |
| txc_batchtopk_post | 4 | 8 | 3 | 0.791 | 0.799 | 0.802 |
| txc_batchtopk_post | 8 | 8 | 3 | 0.864 | 0.876 | 0.877 |
| txc_batchtopk_post | 16 | 8 | 3 | 0.945 | 0.969 | 0.976 |
| txc_batchtopk_post | 32 | 8 | 2 | 0.992 | 1.002 | 1.002 |
| txc_batchtopk_post (k=8·T arm) | 2–16 | 16–128 | 5 ea | 0.750 | — | 1.014 |
| txc_batchtopk_post (k=8·T arm) | 32 | 256 | 1 | **0.647** | — | — |

(pre/post ratios are vs the untrained anchor; "cells" = (substrate, k)
groups, 3+ rows each. Full per-substrate table: § 8 appendix.)

### 2b. The three structural facts

1. **Per-token comparators are permanently handicapped, T-independent.**
   sae realizes 0.548–0.762 of nominal (by substrate: ttrend 0.548,
   punct/llama 0.548, dq 0.562, λ 0.566, oprate 0.567, slope8 0.568,
   punct/gemma 0.670, punct/gpt2 0.762). tsae realizes 0.711–0.951
   (λ 0.711 — dragged by two under-band seeds, see R22 CAVEAT 2; dq 0.830;
   ttrend 0.844; punct/gemma 0.951). Every TXC-vs-token margin at ANY T
   carries this comparator deficit.
2. **Window arms are depressed only at small T and are clean at the Ts that
   decided verdicts.** T=2: 0.69–0.88; T=4: 0.79–0.94; T=8: 0.86–1.04;
   T ≥ 16: 0.95–1.05. Every evidence-line comparison and every claiming cell
   at T ≥ 8 sits on a ≥ 0.86-realized window arm (post@T8) or ≥ 0.95
   (everything else).
3. **Untrained cells realize full nominal** (the trained−untrained
   comparisons are therefore biased AGAINST trained arms at small T, by the
   §2a deficit) — with the one exception that untrained pre under-realizes at
   T ≥ 4 (0.93–0.99), which makes trained pre ratios exceed 1.0 at T ≥ 16.
   At T = 32 (where tt round-1's P4 was decided) trained stacked/pre/post
   realize 0.98–1.05 vs untrained ≈ 1.0 — **realization-matched**, so the P4
   untrained-control failure is NOT a mixing artifact (§ 4).

The lone deep-selection cell on the hunt boards: **post k=256@T32 realized
0.647** (ttrend secondary budget-parity arm, non-claiming) — the same
starve-with-depth regime as the paper's `txc_base` composition, and the
secondary arm already failed its untrained control (R28: 0.74×), consistent
with the ratified k-resolution.

### 2c. Direction of the bias, stated once

At every deciding bar in the record, comparator deficit ≥ window-arm deficit.
So under `btk-only` (pre-registered expectations, `actmix-shared.md`: "the
per-token sae baseline improves MOST ⇒ hunt TXC-vs-sae margins likely
shrink; tsae margins move least"): **TXC-vs-token margins shrink or hold;
they do not grow.** Therefore: a kill recorded as "TXC fails to beat token"
was made with TXC flattered — the fix hardens the kill. A KEEP whose deciding
leg is a TXC-vs-token margin is exposed in proportion to its comparator's
deficit and its margin's slack. T-trend receipts computed across T = 2→8/16
partially conflate label signal with the § 2a capacity-recovery gradient
(pre 0.73→0.99, post 0.71→0.97, stacked 0.75→0.99 over T2→T8/16).

## 3. Per-verdict sensitivity table

Class (a) = deciding bar is a TXC-vs-batchtopk_sae (or tsae) margin
(flattered); (b) = deciding bar rests on small-T window cells in the
ReLU→BatchTopK regime (depressed); (c) = mixing-INSENSITIVE grounds
(screen-stage/probe-only, evidence-line at T ≥ 8, identity, order-freeness,
per-token readability, same-model shuffle arms). "Realized" quotes the § 2
scan at the deciding cells. Verdicts of record, KEEPs AND kills:

| verdict (receipt) | deciding bar | realized l0 at that bar | class | read |
|---|---|---|---|---|
| **λ KEEP (R22)** pre/T8 − tsae, paired LB +0.0200 | TXC-vs-tsae margin | pre@T8 0.983; tsae pooled 0.711 (seeds s3=3.59, s4=3.12 UNDER band; round-1 6.52–7.20) | **(a) — most exposed KEEP on the board** | Existing guard already states the direction and the damage: CAVEAT 2 ("an under-spent tsae comparator plausibly INFLATES the pre−tsae margin") and the POST-HOC exclusion goes to **paired n=4 LB −0.0088 NOT bounded**. btk-only re-run of the comparator settles it cleanly. |
| **dq KEEP (R27)** pre−tsae@T8 +0.155 [+0.126, +0.184]; 2→8 trend p=0.0046; T32 −0.017 | TXC-vs-tsae margin + low-T trend | pre@T8 0.989; tsae 0.830 (6.64); sae 0.562 (4.50) | **(a)** margin, **(b)** trend | Margin leg: tsae 17% under-spent; LB +0.126 has real slack but "tsae moves least" cuts both ways — exposure moderate. Trend leg: pre realization climbs 0.736→0.872→0.989 over T2→4→8, so part of the 2→8 slope is capacity recovery, not label signal. Existing guard (binding quote licence): "pre − sae +0.176 is quotable ONLY with the note that sae realized l0 = 4.50 — the known llama-d4096 under-spend signature (fineweb precedent 4.27–4.57) — which inflates that margin's face value." The licence calls tsae 6.64 "well-spent" — under this forensics that is 83%-spent, and the margin leg inherits the residual. |
| **ttrend salvage KEEP (R28/R29)** four S1 legs at T16/T32 vs sae AND tsae; S2 untrained ratios; S3 trend T16→32 p=0.0156; S4 evidence line 0.282 > 0.114 | margins vs sae/tsae; trend; evidence line | post@T16 0.945 / T32 1.002; sae 0.548 (4.38 — R29's "4.12–4.69 UNIFORM across all 6 seeds"); tsae 0.844 | **(a)** on S1 (esp. the sae legs and T16), minor **(b)** on S3 (post 0.945→1.002 gradient), **(c)** on S4 (visible-boundary bar; post cell fully realized) | The T32 margins (+0.246/+0.248, LB ≥ +0.204) have ~10× the slack of any plausible comparator recovery effect at these d768 widths — likely robust. The T16 legs (L1 sae +0.117 LB +0.110; tsae +0.104 LB +0.094) are the exposed ones. Existing guards: R29's realized-l0 disclosure ("arch property … drop-s7 sensitivity passes both sae legs") and WRITEUP § 9 ("that baseline landed 4.1–4.7 active features per token against a nominal 8 … The temporal-SAE comparison is the clean one, and Task 2 passes both"). The guards disclosed the fingerprint; they could not know it was a fixable composition artifact — that is what Stage 3 tests. |
| **tt round-1 KEEP-with-licence** P1 stacked−sae@T32 +0.186; P2 margins grow with T; P4 FAILED (untrained stacked 0.81×; untrained pre beats trained pre); P5 v2 grouped; P6 evidence line | P1: margin vs sae; P2: T-slope; P4: trained-vs-untrained at T32 | stacked@T32 0.979; sae 0.548 (4.34 on record); tsae 0.844 (6.50 on record); untrained ≈ 1.0 | **(a)** P1, **(b)** P2, **(c)** P4/P5/P6 | P1's +0.186 margin over sae rides the largest comparator deficit on the board; the licensed "per-token-quiet (0.032–0.041)" profile is partly the sae handicap speaking — under btk-only the per-token floor RISES. P2 (margins grow 2→32) partially rides the stacked 0.725→0.979 recovery gradient. **P4 is mixing-ROBUST**: at T32 trained and untrained are realization-matched (§ 2b fact 3), so "recovery is mostly architecture prior" survives the fix — round 1's failure-to-claim was NOT the mixing's fault. Existing guard: "l0 bands: pooled 5.80–7.90, tsae 6.50, sae 4.34 — in-band, no under-band cells". |
| **qrate/punctint gemma NO-RULE-FIRES (R13)** K1 +0.0541@T8 direction-only, K2 (CI bound) ✗ | margin vs better-token, bound at n=3 | pre@T8 1.019; tsae 0.951 (7.61); sae 0.670 | (a) nominally, **effectively (c)** | Both sides of the deciding margin are ≥ 0.95 realized — gemma is the one substrate where the comparator is nearly clean. The kill is power-limited, not mixing-limited. btk-only would not buy the bound; seeds would. |
| **qrate/punctint gpt2 WEAK (R18)** pre−tsae +0.028@T4 under the +0.05 bar | margin vs tsae at small T | pre@T4 0.984; tsae 0.749 (5.99) | **(a) — kill-conservative** | The comparator is 25% under-spent, the window arm ~clean: the +0.028 was FLATTERED and still missed the bar. Under btk-only tsae recovers more than pre ⇒ margin shrinks ⇒ **kill hardens**. |
| **qrate/punctint llama NEGATIVE (R19)** pre−tsae −0.018 (T4) / −0.014 (T8) | margin vs tsae at small T | pre@T4 0.938, T8 0.988; tsae 0.781 (6.25); sae 0.548 (4.39) | **(a) — kill-conservative** | Same direction, stronger: pre LOST to a comparator that was itself 22% handicapped. btk-only makes the loss bigger. "The per-token code simply wins" (§ 8) is UNDERSTATED by the mixing, not manufactured by it. |
| **qrate evidence-line + identity (R16)** count bar 0.345/0.461 beats every window cell at T ≥ 8 (best pre/T16 v2 0.321 < 0.462); doc floor 0.575–0.587 | visible-count regression + identity floor vs window cells | failing window cells at T ≥ 8 realize 0.98–1.04 (pre@T16 gemma 1.039); "Untrained pre realizes 7.54–7.93" on record | **(c) — empirically, not just by class** | The bars are computed on visible counts/doc means (no SAE); the window cells that fail them are fully realized. A 0.14 deficit at T16 with a 1.04-realized arm is not budget. |
| oprate/case panel kill | every window cell below the visible event-sentence count baseline (§ 8) | window arms 0.75–1.01 across T2–16, same family fingerprint | **(c)** | Evidence-line grounds; binds at T ≥ 8 where realization ≥ 0.98. Class-prior kill stands. |
| refmark kill (R23) | window arms below visible marker-count floor (gpt2 best −0.008, worst −0.069); wd control flat | screen-stage — no SAE in the instrument | **(c)** | By construction + visible-cue + identity (r_doc ≈ 0.97) grounds. |
| slen/lat, slen/lev (R20, R21) | order-free (shuffle cost ≤ 0.019 vs content +0.020…+0.147); identity-bounded | screen-stage | **(c)** | Shuffle arms share one probe instrument on raw activations. |
| slen/disp kill | sub-threshold everywhere | screen-stage | **(c)** | — |
| quotedens KEEP-deferred (R24) | order-free + quote-char counting above T≈32 | screen-stage | **(c)** | Deferral grounds (class prior) untouched by mixing. |
| dialevel kill + R11/R25 order ladder | identity 0.98; ladder costs L0/L1/L2 additive | screen-stage; perm arms share one trained model | **(c)** | The R11/R25 order receipts are the program's most mixing-robust numbers: every arm of a shuffle comparison shares the same fitted instrument. |
| interleave/tss kill | converted single-position | screen-stage | **(c)** | — |
| Ward order receipts (R10) | g_order −0.004…+0.008 at T32 | screen-stage | **(c)** | — |
| R15 within-doc (pre 0.086 vs tsae 0.039, gemma) | doc-demeaned margin | tsae 0.951 on gemma | (a) minor | Comparator nearly clean on this substrate; the +0.047 ordering is unlikely to be realization. |
| R14 gemma v2 trend 2→16 p=0.0009 (non-canonical) | full-ladder trend | pre 0.876→1.039 over T2→16 | **(b)** | Already quarantined as "ordering robust, widens under an adequate probe" — add: low-T cells depressed, trend partially capacity recovery. |
| R17 requote (window−token at T64) | probe-class grid | screen corpus, probe-only | **(c)** | — |
| novelty park → txcwin cross-ratification | calibrate_k matches MEASURED realized nnz per arm | mixing-robust **by construction** | **(c)** | The txcwin harness budget-matches on realized sparsity, not nominal — the one thread that pre-solved this problem. |
| triage deads; refusal-direction; sc_lambda / oprate-ver / qrate / vslope / emotional_instability parks | per-token/position readability; priority/class-prior parks | no mixing-touched bar anywhere in the deciding chain | **(c)** / n.a. | Parks were priority decisions, not measurements; btk-only does not change their queue position. |

## 4. Stage-2 ranked salvage shortlist (kills/parks → btk-only re-runs)

**Result: NO HIGH-priority salvage candidates exist.** Applying the
briefing's mechanism-fit rule honestly, with directions: every kill that was
decided on a TXC-vs-token margin (punctint-q ×3) had the margin FLATTERED by
the mixing and still killed — `btk-only` re-runs would harden, not reverse,
those verdicts (§ 3 rows R18/R19). Every other kill sits on
mixing-insensitive grounds: screen-stage instruments (slen ×3, refmark,
quotedens deferral, dialevel, tss, R10), evidence-lines binding at T ≥ 8
with fully-realized window arms (oprate/case, qrate count bar), identity
floors, or per-token readability (triage class). ttrend round-1's
untrained-control failure — the one kill I pre-registered as potentially
mixing-sensitive before running the scan — is realization-matched at its
deciding T=32 and therefore mixing-robust; and its salvageable content was
already salvaged by the fresh-seed post thread (R28/R29), whose OWN exposure
is a KEEP-side issue (§ 6).

| rank | item | cells to re-run | est | verdict |
|---|---|---|---|---|
| — (no HIGH) | — | — | — | — |
| MEDIUM-conditional | punct/gemma K2 bound (R13's unbounded +0.0541) | +3 seeds × {pre@T8, tsae@T1} btk-only, d2304 = 6 cells L40S | ~$4 | **A power fix, not a mixing fix** (both arms ≥ 0.95 realized). Worth doing ONLY as a rider on mac-a's btk-only calibration (it doubles as a mid-width calibration point); zero claim to a salvage slot on mixing grounds. |
| NIL | punctint-q llama, gpt2 | — | — | Kills harden under the fix (§ 3). Re-running to CONFIRM hardening is Stage-3 optional garnish, not salvage. |
| NIL | oprate/case, refmark, qrate-evidence, slen/{lat,lev,disp}, quotedens-deferral, dialevel, tss, triage deads, refusal-direction | — | — | Evidence-line / visible-cue / identity / order-free / per-token-readability / screen-stage grounds — the briefing's NIL classes, all confirmed by instrument (screens have no SAE) or by realization at the binding T. |
| n.a. | sc_lambda, oprate/ver, qrate, vslope, emotional_instability parks | — | — | Parked on priority/class prior, not on any measured bar; btk-only re-runs would not change their status. They re-enter (or not) on ordinary priority at team discretion. |

**No re-runs are launched from this document** (briefing rule). The
MEDIUM-conditional item goes to mac-local for gating as a calibration rider,
not a salvage lane.

## 5. What the mixing DID contaminate — but Stage 1 must not overclaim

For symmetric honesty: the pro-TXC bias direction means the record's
false-positive risk lives in the KEEPs, and its false-negative risk is ~nil.
But "exposed" ≠ "wrong": the pre-registered expectation is that margins
SHRINK, not that they vanish; T32-scale margins (+0.19…+0.25) dwarf any
plausible realization effect, while margins in the +0.03…+0.12 band with
comparator deficits of 17–45% are genuinely at risk. The point of § 6's
ranking is to spend re-run dollars in risk order.

## 6. KEEP-exposure ranking (input to mac-a's Stage-3 lane — NOT this
briefing's shortlist; costs are planning estimates only)

1. **λ (R22)** — smallest margin (+0.0569, LB +0.0200) over the most
   under-spent comparator instance (tsae pooled 0.711; s3/s4 at 3.12–3.59),
   and the existing post-hoc exclusion already goes unbounded. Re-run: 6
   btk-only tsae comparator seeds on ward base (reuse pre rows) ≈ 6 × ~70 min
   L40S ≈ **$12–15** (minimal 2-seed s3/s4-replacement probe ≈ $5).
2. **tt round-1 quote licence (P1 leg)** — stacked−sae +0.186@T32 over the
   board's deepest comparator deficit (sae 0.548); the "per-token-quiet"
   phrasing needs a btk-only per-token floor. Re-run: sae + tsae btk-only
   baselines × 3 seeds on ttrend (gpt2 d768) ≈ 6 cells ≈ **$2–3**.
3. **ttrend salvage T16 S1 legs (R29)** — LBs +0.065…+0.110 vs sae@0.548 /
   tsae@0.844. Re-run: btk-only sae+tsae at T16 (and T32 for symmetry) × 3
   seeds ≈ 12 cells ≈ **$3–4** (post btk-only arm +12 cells ≈ $3 if mac-a
   wants the two-sided fix).
4. **dq (R27)** — margin leg LB +0.126 vs tsae@0.830; trend leg 2→8 rides
   the capacity gradient. Re-run: btk-only tsae × 3 seeds + pre T{2,4,8} × 3
   on 8B ≈ 12 cells H100 ≈ **$8–12**.
5. **R14/R15 gemma v2 legs** — non-canonical already; lowest priority;
   covered free if item MEDIUM-conditional (§ 4) runs.

Full ranking ≈ **$30–37** if taken end-to-end — inside one day-cap, but
sequencing and any launch belong to mac-a under mac-local's gate.

## 7. Existing-guard index (the record already said much of this — quotes)

- R22 CAVEAT 2: "round-1 tsae realized l0/token 6.52–7.20; new s5 = 7.08
  in-band, but s3 = 3.59 and s4 = 3.12 UNDER band — residual mismatches,
  disclosed not smoothed. Direction matters: an under-spent tsae comparator
  plausibly INFLATES the pre−tsae margin".
- dq binding quote licence: "pre − sae +0.176 is quotable ONLY with the note
  that sae realized l0 = 4.50 — the known llama-d4096 under-spend signature
  (fineweb precedent 4.27–4.57) — which inflates that margin's face value."
- R29: "the trained sae baseline runs at 4.12–4.69 of nominal 8 UNIFORMLY
  across all 6 seeds (arch property, outside the card's post-arm band clause;
  drop-s7 sensitivity passes both sae legs)".
- WRITEUP § 9: "that baseline landed 4.1–4.7 active features per token
  against a nominal 8 (an architecture property at this width; a sensitivity
  check passes). The temporal-SAE comparison is the clean one".
- tt round-1: "l0 bands: pooled 5.80–7.90, tsae 6.50, sae 4.34 — in-band, no
  under-band cells."
- LOG 2026-07-25 falsifier: "Every untrained matched cell realizes
  l0_per_token = 8.000 (±<0.01) at every T"; qrate panel: "Untrained pre
  realizes 7.54–7.93, declining with T".
- Card discipline (standing): "numeric l0 bands with an out-of-band ⇒
  non-claiming".

What the guards did NOT know: that the under-spend was a fixable composition
artifact with a pre-registrable direction, rather than an immutable "arch
property at this width". That reframing — and only that — is new here.

## 8. Appendix — full per-substrate table

Scan output (trained cells, untrained-anchored; substrate short names:
dq/8b = dial_real_dqgap_llama31_8b_l14, lambda/base = ward_real_lambda_base_l12,
ttrend/gpt2 = dial_real_ttrend_gpt2_l7, slope8/dist = ward_real_slope8_distill_l14,
oprate/base = ward_real_oprate_case_base_l12, punct/* = fineweb_punctint_q_*):

| substrate | arch | T | k | rows | realized/win | nominal/win | ratio |
|---|---|---|---|---|---|---|---|
| dq/8b | batchtopk_sae | 1 | 8 | 3 | 4.50 | 8.00 | 0.562 |
| dq/8b | tsae | 1 | 8 | 3 | 6.64 | 8.00 | 0.830 |
| dq/8b | stacked | 2/4/8/16/32 | 8 | 3 ea | 11.94/28.55/61.27/126.78/256.32 | 16/32/64/128/256 | 0.746/0.892/0.957/0.990/1.001 |
| dq/8b | pre | 2/4/8/16/32 | 8 | 3 ea | 11.75/27.71/62.40/126.40/246.24 | 15.96/31.79/63.07/124.13/240.90 | 0.736/0.872/0.989/1.018/1.022 |
| dq/8b | post | 2/4/8/16/32 | 8 | 3 ea | 5.67/6.39/6.91/7.81/7.93 | 8.00 | 0.709/0.799/0.864/0.976/0.992 |
| lambda/base | batchtopk_sae | 1 | 8 | 3 | 4.53 | 8.00 | 0.566 |
| lambda/base | tsae | 1 | 8 | 6 | 5.69 | 8.00 | 0.711 |
| lambda/base | stacked | 2/4/8/16 | 8 | 3 ea | 11.97/28.52/61.09/125.83 | 16/32/64/128 | 0.748/0.891/0.955/0.983 |
| lambda/base | pre | 2/4/8/16 | 8 | 3–6 | 11.61/27.45/61.96/125.51 | 15.96/31.79/63.05/124.03 | 0.727/0.864/0.983/1.012 |
| lambda/base | post | 2/4/8/16 | 8 | 3 ea | 5.70/6.41/7.02/7.75 | 8.00 | 0.712/0.802/0.877/0.969 |
| lambda/base | post (8·T arm) | 2/4/8/16 | 16–128 | 3 ea | 12.08/29.99/64.70/127.88 | 16/32/64/128 | 0.755/0.937/1.011/0.999 |
| oprate/base | batchtopk_sae | 1 | 8 | 3 | 4.53 | 8.00 | 0.567 |
| oprate/base | tsae | 1 | 8 | 3 | 6.63 | 8.00 | 0.829 |
| oprate/base | stacked | 2/4/8/16 | 8 | 3 ea | 11.90/28.71/61.23/125.88 | 16/32/64/128 | 0.744/0.897/0.957/0.983 |
| oprate/base | pre | 2/4/8/16 | 8 | 3 ea | 11.61/27.39/61.94/125.27 | 15.96/31.79/63.04/124.02 | 0.728/0.862/0.982/1.010 |
| oprate/base | post (8·T arm) | 2/4/8/16 | 16–128 | 3 ea | 11.99/29.49/64.85/128.47 | 16/32/64/128 | 0.750/0.922/1.013/1.004 |
| slope8/dist | batchtopk_sae | 1 | 8 | 3 | 4.55 | 8.00 | 0.568 |
| slope8/dist | tsae | 1 | 8 | 3 | 6.38 | 8.00 | 0.797 |
| slope8/dist | stacked | 2/4/8/16 | 8 | 3 ea | 11.97/28.65/62.14/126.36 | 16/32/64/128 | 0.748/0.895/0.971/0.987 |
| slope8/dist | pre | 2/4/8/16 | 8 | 3 ea | 11.88/27.62/61.87/125.19 | 15.96/31.78/63.05/124.08 | 0.744/0.869/0.981/1.009 |
| slope8/dist | post (8·T arm) | 2/4/8/16 | 16–128 | 3 ea | 12.16/29.82/64.92/127.49 | 16/32/64/128 | 0.760/0.932/1.014/0.996 |
| punct/gemma | batchtopk_sae | 1 | 8 | 3 | 5.36 | 8.00 | 0.670 |
| punct/gemma | tsae | 1 | 8 | 3 | 7.61 | 8.00 | 0.951 |
| punct/gemma | stacked | 2/4/8/16 | 8 | 3 ea | 13.22/29.22/62.13/126.32 | 16/32/64/128 | 0.827/0.913/0.971/0.987 |
| punct/gemma | pre | 2/4/8/16 | 8 | 3 ea | 13.90/31.15/63.29/125.39 | 15.86/31.51/62.10/120.63 | 0.876/0.989/1.019/1.039 |
| punct/gemma | post (8·T arm) | 2/4/8/16 | 16–128 | 3 ea | 14.45/31.73/64.10/128.28 | 16/32/64/128 | 0.903/0.992/1.002/1.002 |
| punct/gpt2 | batchtopk_sae | 1 | 8 | 3 | 6.10 | 8.00 | 0.762 |
| punct/gpt2 | tsae | 1 | 8 | 3 | 5.99 | 8.00 | 0.749 |
| punct/gpt2 | pre | 4/8 | 8 | 3 ea | 30.60/62.12 | 31.08/59.59 | 0.984/1.042 |
| punct/llama | batchtopk_sae | 1 | 8 | 3 | 4.39 | 8.00 | 0.548 |
| punct/llama | tsae | 1 | 8 | 3 | 6.25 | 8.00 | 0.781 |
| punct/llama | pre | 4/8 | 8 | 3 ea | 29.82/62.31 | 31.79/63.06 | 0.938/0.988 |
| ttrend/gpt2 | batchtopk_sae | 1 | 8 | 12 | 4.38 | 8.00 | 0.548 |
| ttrend/gpt2 | tsae | 1 | 8 | 12 | 6.75 | 8.00 | 0.844 |
| ttrend/gpt2 | stacked | 2/4/8/16/32 | 8 | 6 ea | 11.60/27.05/59.33/123.73/250.63 | 16/32/64/128/256 | 0.725/0.845/0.927/0.967/0.979 |
| ttrend/gpt2 | pre | 2/4/8/16/32 | 8 | 6 ea | 12.10/28.51/60.31/125.13/252.93 | 15.97/31.81/63.11/124.19/240.91 | 0.758/0.896/0.956/1.008/1.050 |
| ttrend/gpt2 | post | 2/4/8/16/32 | 8 | 9–12 | 5.50/6.33/7.01/7.56/8.01 | 8.00 | 0.687/0.791/0.876/0.945/1.002 |
| ttrend/gpt2 | post (8·T arm) | 2/4/8/16/32 | 16–256 | 3 ea | 12.08/29.22/62.96/128.97/**165.74** | 16/32/64/128/256 | 0.755/0.913/0.984/1.008/**0.647** |

Untrained cells (107): all realize analytic nominal to < 1% except untrained
pre at T ≥ 4 (18 cells, 0.931–0.986 of k·T; worst punct/gpt2 T8 59.59/64).

---
*mac-b, ACTMIX W2. Everything above PENDING TEAM REVIEW; shortlist and
KEEP-exposure ranking gate through mac-local; re-run lane is mac-a's.*
