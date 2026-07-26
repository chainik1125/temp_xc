# FROZEN screen card — B7 refusal/deflection-marker intensity (`refmark`, WildChat)

**Status: FROZEN at commit (commit-then-run; no screen cell has been
executed when this card is committed — the Modal container is pinned to
a commit containing it).** Agent: **mac-b (executor)**, overnight
stretch item 1 (`briefings/overnight-mac-b.md` § 2) — self-review
hazard named; all verdicts ship PENDING TEAM REVIEW. Bundle card:
`CARD_DRAFT.md` (triage SHIPPED with the doc-identity route pre-named).
Executor: `screen.py`; caches: `cache_acts.py` (grid derived from the
committed token stream — label↔cache identity by construction).
`build_rows` validated label-side pre-freeze (gpt2): all pools at
caps, user-echo = 13 turns / 5,541 tokens / 69 manifest rows dropped,
wd control 112 train / 31 test conversations.

## 1. Coverage and rows

**gpt2 + llama31-8b** (gemma pending an HF secret, as in the slen
card § 1; 2-model statements only). Rows: builder manifest
`man_rlam_*` (~20k/class, position-matched), screen eligibility as the
family convention (`pos ≥ 64`, `pos % content ≥ 63`, caps 4000/1500,
MATCH_SEED + crc32, MIN_ROWS 300), split by builder `doc_split`.
**User-echo rows** (USER messages matching the frozen 12-string list —
unflagged in `is_marker`, recomputed from the committed corpus) are
**DROPPED from every pool**, counts disclosed in the results meta
(briefing obligation "dropped-or-disclosed": this card does BOTH).

Label-side triage (SHIPPED, 320 train conversations quoted per the
estimator convention): unigram 0.517–0.532 manifest (near-blind);
position 0.545–0.565 manifest (low disclosure band — the floor probe
runs); **`doc_mean_only_auc` 0.966–0.968 — the factory's loudest** ⇒
the within-conversation contrast is the MANDATORY control: **no KEEP
exists without it**, and if wd rows are unsupported the face cannot
KEEP at this screen, full stop.

## 2. Clock (under-span, stated before any cell)

Kernel: 8 messages, half-life 2, at 125–144 tokens/message ⇒ support
≈ **1,100–1,150 tokens ≈ 16× the T = 64 ladder top**. A T-window
usually sits INSIDE one message. The predicted mechanism is
register-evidence pooling, not marker counting — hence the binding
**beat-the-visible-evidence line**: per T, a label-side probe on
[window marker-token count, boundary count, assistant fraction] is
computed on the SHIPPED rows and printed beside every window number.
A window arm that does not beat it is reading its visible tokens.
A flat gap at this reach is a reach-limited negative, recorded as
such; a rising unsaturated gap is the predicted shape.

## 3. Probe grid (frozen — the convention-of-record, slen precedent)

tok linear + MLP(512) FIRST; position floor; per T ∈ {4,8,16,32,64}:
visible-evidence floor, actxmean linear/MLP each beside its
width-matched foreign null; ORDER arms (flatten/shuffle/foreign,
matched T·d) linear at T ∈ {16,32}, MLP triple at T32; permutation
nulls at T16; ambient anchor `is_marker` (binary, full-pool draw,
8000/3000 caps) per-token + T16 actxmean — the regime-1 calibration
face (D7: the current-prompt refusal state converts; it is never the
primary). Within-conversation arms: tok linear/MLP, actxmean ±
foreign at T ∈ {16,32,64}, flatten/shuffle pair at T ∈ {16,32}
(binary rank-AUC, class_weight). NO win_mean, NO max-over-arms;
metrics: 3-class `acc_test` (chance ⅓) / binary rank-AUC.

## 4. Pre-registered predictions (CARD_DRAFT, made cell-precise)

- **Q1:** per-token is weak on masked rows — tok_linear clears the
  position floor by < +0.10 (the register is ambient, not stamped;
  contrast: slen lat/lev converted at +0.16/+0.17).
- **Q2:** `g_ax(T)` = actxmean − tok is positive, grows with T, still
  rising at T64 (under-span § 2), width null cleared (≥ +0.02).
- **Q3 (regime-2):** order-free — |win − shuf| ≤ 0.02 at T ∈ {16,32}
  (shuffle-immune), while win − foreign > 0 where any window content
  exists.
- **Q4 (the two binding lines):** any quoted window gain (a) beats the
  visible-evidence floor at its T, and (b) reappears same-direction in
  the within-conversation arm. Failing (a) = marker counting; failing
  (b) = conversation identity (the pre-named doc-identity route).
- **Q5:** the `is_marker` anchor is strongly converted per-token
  (D7's measured mechanism) — calibration only, no verdict clause.

## 5. KEEP / KILL (frozen, per model majority of the 2 screened)

**KEEP** iff: some matched-class window arm beats tok by ≥ +0.05 with
its width null cleared ≥ 0.02, AND that arm beats the visible-evidence
floor at the same T, AND the within-conversation arm shows a
same-direction window gain (wd actxmean − wd tok > 0) on supported
rows. **KILL** if ANY of: (1) tok within 0.02 of every window arm at
every T; (2) every window gain fails its width null; (3) every window
gain fails the visible-evidence floor (the window is a marker
counter); (4) the wd arm erases the gain (conversation identity — the
0.966 route, the dialevel trap). Otherwise **WEAK — no rule fires as
written**, numbers only. Order sensitivity (Q3 failing with sc > 0)
is reported, not killed — it would be a second counterexample to the
amended order finding and goes to the LOG as such.

## 6. Venue, economics, discipline

Modal L40S (the slen OOM lesson — flatten-MLP peaks exceed the A10),
container pinned to a commit containing this card; caches
1.36M + 1.19M tokens × 3 layers fp16 ≈ 33 GB on the refmark Volume
path, forward minutes; screen est ≤ 1 h/model ≈ ≤ $6 total. Ledger
read-before/append-after; containers never push; results repatriated
and merged locally. Runtime adaptations pre-authorized (outputs
unaffected): resume from partial results; batch halving on OOM.
Deliverable: `results/screen_{gpt2,llama31_8b}.json`, ONE face verdict
(`rlam`) + coverage statement in the LOG (`mac-b (executor)`, PENDING
TEAM REVIEW), RECEIPTS proposals for quoted claims, no leaderboard
rows. **Launch gate honored: this stretch runs only after the slen
verdicts are pushed and only if mac-b spend ≤ $60** (briefing § 2).
