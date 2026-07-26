# FROZEN screen card — B9 quoted-speech intensity on PG19 fiction (`quotedens`)

**Status: FROZEN at commit (commit-then-run; no screen cell has been
executed when this card is committed — the Modal container is pinned
to a commit containing it).** Agent: **mac-b (executor)**, overnight
stretch item 2 (`briefings/overnight-mac-b.md` § 2) — self-review
hazard named; verdicts ship PENDING TEAM REVIEW. Bundle card:
`CARD_DRAFT.md` (SHIPPED, factory r3, mac-local-ratified). Executor:
`screen.py`; caches `cache_acts.py` (grid from the committed stream,
the refmark recipe). `build_rows` validated label-side pre-freeze
(gpt2): all pools at caps; within-book control **345 train / 81 test
books** — the deepest within-doc substrate in the factory, as the
bundle promised.

## 1. Coverage and rows

**gpt2 + llama31-8b** (gemma pending an HF secret; 2-model
statements only — the overnight standing scope). Builder manifest
`man_qd_*` (~100k/class label-side), screen eligibility as the family
convention (pos ≥ 64, pos % content ≥ 63, caps 4000/1500, MATCH_SEED
+ crc32, MIN_ROWS 300), builder `doc_split` (800 train / 200 test
books). Label-side triage (SHIPPED, **800 train books** quoted per
the estimator convention): unigram **0.588–0.600 (disclosure band —
the attribution-register leak, named with its lower-bound caveat)**;
position 0.511–0.515 clean; `doc_mean_only_auc` **0.890–0.896** ⇒
the within-book contrast is BINDING: no KEEP without it.
`zero_split` fired (66–68 % exact zeros — quiet narration); event =
double-quote-family sentence, event-sentence tokens masked from
probe rows (visible in window context — hence the floor below).

## 2. Clock — the best-spanned ladder of the overnight screens

Fiction sentences are short: ≈ **12.9 tokens/sentence** (1.94M
tokens / 150k sentences, gpt2). Kernel HL 2 / support 8 sentences ≈
103 tokens ⇒ **T64 spans ≈ 5 of 8 support sentences ≈ 0.87 kernel
mass** — unlike slen (0.7) and refmark (0.06), this ladder nearly
reaches its label's timescale. Consequence, stated pre-run: a real
trailing-intensity state should APPEAR AND BEGIN SATURATING inside
this ladder; "the window can't reach it" is not an available excuse
here, in either direction.

## 3. Probe grid (frozen — identical shape to the refmark screen)

tok linear + MLP(512) FIRST; position floor; per T ∈ {4,8,16,32,64}:
**visible-evidence floor** (label-side probe on [window `is_qd` token
count, window `in_span` fraction] — the second feature is
near-constant on eligible rows, disclosed, standardization zeroes
it; the count is the loaded feature, mean ≈ 13 visible event tokens
at T64), actxmean linear/MLP ± width-matched foreign nulls; order
arms (flatten/shuffle/foreign) linear at T ∈ {16,32}, MLP triple at
T32; permutation nulls at T16; ambient anchor `is_qd` (binary,
8000/3000, regime-1 bracket face, calibration only); within-book
arms (tok linear/MLP, actxmean ± foreign at T ∈ {16,32,64},
flatten/shuffle pair at T ∈ {16,32}; binary rank-AUC). NO win_mean;
NO max-over-arms; metrics 3-class `acc_test` / binary rank-AUC.

## 4. Pre-registered predictions

- **Q1:** per-token weak-to-moderate above its position floor (the
  register is ambient; the unigram leak is real but sub-bar) —
  tok − floor < +0.10.
- **Q2:** g_ax = actxmean − tok positive, grows with T, and — the
  reach point — **flattens or saturates by T64** (≈ 0.87 mass); width
  nulls cleared (≥ +0.02).
- **Q3 (regime-2):** order-free: |win − shuf| ≤ 0.02 at T ∈ {16,32}.
- **Q4 (the binding lines):** a quoted window gain must (a) beat the
  visible-evidence floor at its T (else it is quote-counting) and
  (b) reappear same-direction in the within-book arm (else it is
  book identity — the 0.89 route).
- **Q5:** the `is_qd` anchor is strongly converted per-token
  (bracket-family; calibration only, no verdict clause).

## 5. KEEP / KILL (frozen; majority of the 2 screened models)

**KEEP** iff: some matched-class window arm beats tok by ≥ +0.05
with width null cleared ≥ 0.02, AND beats the visible-evidence floor
at the same T, AND the within-book arm shows a same-direction window
gain on supported rows. **KILL** if ANY of: (1) tok within 0.02 of
every window arm at every T; (2) every window gain fails its width
null; (3) every window gain fails the visible-evidence floor
(quote-counting); (4) the within-book arm erases the gain (book
identity). Else **WEAK — no rule fires as written**, numbers only.
Order sensitivity (Q3 failing with sc > 0) is reported to the LOG as
a potential counterexample to the amended order finding, not killed.

## 6. Venue, economics, discipline

Modal L40S, container pinned to a commit containing this card;
caches 1.94M + 1.68M tokens × 3 layers fp16 ≈ 45 GB on the
quotedens Volume path (minutes); screen ≈ 30–60 min/model, est
≤ $6. Ledger read-before/append-after; containers never push;
results repatriated + merged locally; resume + batch-halving
pre-authorized. Deliverable: `results/screen_{gpt2,llama31_8b}.json`,
ONE face verdict + coverage in the LOG (`mac-b (executor)`, PENDING
TEAM REVIEW), RECEIPTS proposal, no leaderboard rows. Launch gate:
starting now (~21:50 PT) finishes hours before the 06:30 PT
no-new-starts line; mac-b spend ≈ $8–9 actual of the $100 cap.
