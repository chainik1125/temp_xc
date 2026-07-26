# FROZEN panel-2 card — diafaces: `dqgap` on llama31_8b/hs14 (`dial_real_dqgap_llama31_8b_l14`)

**Status: FROZEN at commit (commit-then-run; NO panel-2 cell has
been executed when this card is committed).** Agent: **mac-a
(executor)**; **mac-b = merge + variance-harness support** (the
race-resolution work split). Authorization chain, in order:
gate fired for tt (`dce8d085d`) → amendment to dq (`187c51022`) →
**RACE RESOLUTION (`6e2f18e4e`): the tt/gpt2 freeze `7ba2e10fd`
GOVERNS and runs (relaunched ~13:40 after my amendment-triggered
stop — ≈ $1–2 burned, disclosed in the ledger); PANEL 2 = dq on
llama31/l14 authorized with hard conditions.** Timing disclosure:
the resolution's freeze-by-13:30 line was already past when the
resolution landed on this clone (~13:38); this freeze commits
~13:43 with the **launch-by-13:45 and repatriate-by-16:15 lines
MET**; the slip is stated here and in the ledger — if mac-local
reads it as gate-failing, stopping the app in writing costs
minutes. Screen provenance: `CARD.md` freeze `073611113` (dq KEEP
3/3, order-carried T32 3/3, **Q1 violated 3/3 — the disclosed hard
opponent**).

## 1. Question

Same Stage-2 question as the tt card § 1, on the face that is ABOVE
its visible floor at every T on 3/3 models: do position-mixing
architectures recover the turns-since-last-question distance state
better than order-free ones — with the "?"-register conversion
standing as the alternative explanation the KILL clause below is
built to catch?

## 2. Grid (identical shape to the tt panel; only DS differs)

`dial_real_dqgap_llama31_8b_l14` (YAML in this commit): 5 archs, T ∈
{2,4,8,16,32}, k_pos 8 (post 8·T/window), d_sae 2048, seeds {1,2,42},
trained+untrained, eval_window_L 32 — **102 cells** via
`run_panel.py` (DS switched in this commit; enumeration verified at
the tt freeze). **buffer_tokens 524288 UNCHANGED with the llama
disclosure: corpus 3653 × 128 = 467,584 < 524,288 ⇒ each refill
oversamples ≈ 1.12×** (buffer kept for cross-panel train_key
comparability — the seedtopup refusal precedent; the λ̂ panel itself
ran 1.3 % over its corpus). Labels NaN before the first question
turn (labeled frac 0.853), at BOS and boundary tokens.

## 3. Receipts + the KILL clause (the resolution's addition)

Paired v1+v2 (claim on v1; v2 = conversation-grouped split, identity
receipt; label-side doc_mean_only 0.854 quoted beside); variance
receipts via mac-b's harness (`--row-layout auto --post-k-rule
times-T`); realized-l0 band, under-band disclosed; **evidence line
per T, measured label-side BEFORE this freeze**
(`results/panel_evidence_line_dq.json`, "?"-count vs dqgap):
|r| = 0.106 / 0.199 / 0.310 / 0.423 / 0.499 at T = 2/4/8/16/32.
**KILL clause (binding, oprate § 3d style): no KEEP unless the
claiming arm's v1 recovery |r| BEATS the evidence line's |r| at its
claiming T.** The § 3d quoting note from the tt freeze review
applies verbatim: latent-state language only at Ts where the
claiming arm beats the evidence line; elsewhere arch-ordering only.

## 4. Predictions + KEEP/KILL (bars frozen)

P1–P4 as the tt card § 5 verbatim (P1: best trained pooled arm beats
trained sae by Δr ≥ +0.02 at some T ∈ {8,16,32}, CI clear; P2 margin
larger at T ∈ {16,32} than {2,4}; P3 tsae between; P4 untrained
≤ 0.5× trained). **P5**: winning arm's v2 r > 0 at its T. **P6 (this
panel only)**: the winning arm beats the evidence line at its
claiming T (else KILL regardless of P1–P5 — the clause above).
**KEEP iff P1 ∧ P5 ∧ P6.** KILL if best trained pooled ≤ sae + CI
noise everywhere, or P5 fails, or P6 fails. Else WEAK. No
max-over-arms; every quoted number carries v2 pair, l0 band, and
the evidence line at its T.

## 5. Venue, economics, discipline

1× **H100** `--block main` (99 cells, workers 6) + 3× **high-CPU
L4** `--block tsae --only-seed {1,2,42}` (d 4096 ⇒ the 62–77 min
class; **memory 64 GB** per the overnight recipe — the 32 GB L4
default is NOT enough at d 4096). Driver
`scripts/modal_diafaces_panel2.py`, PIN via rev-parse of THIS
commit, `--detach`, payloads to Volume
(`/workspace/diafaces_panel2/`), containers never push; local merge
`merge_panel_payload.py dq` (freeze-stamp + clean + dedup asserts).
Est **≤ $25**; ledger read-before/append-after; mac-a cap **$200**
(raised in the resolution). **Repatriation 16:15, everything pushed
16:30, check-in 18:00.** Deliverable: LOG verdict scoring P1–P6
(`mac-a (executor)`, PENDING TEAM REVIEW), leaderboard rows (0
dups), panel JSON, receipts proposal with mac-b's variance outputs,
checkpoints to Volume (HF mirror = Han follow-up rule).

## AMENDMENT (2026-07-26, mac-a) — v2-columns defect: stopped mid-flight and re-run

Same defect as the tt card's amendment (λ̂-clone enumeration,
v1-only). The dq run was STOPPED mid-flight on discovery (~45 min
in; no payload had been repatriated; partial spend disclosed in the
ledger) because its § 3–4 receipts (P5 grouped-split, P6 KILL
clause) are UNSCOREABLE without v2 columns — every cell it would
have produced was already non-compliant with this card. Re-run at
the amendment freeze with the oprate § 2 V2 block on every cell;
all § 2–5 terms unchanged.
