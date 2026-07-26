# FROZEN screen card — day-2 W2 dialogue-native faces: `ttrend` + `dqgap` (`diafaces`, DailyDialog)

**Status: FROZEN at commit (commit-then-run; NO screen cell has been
executed when this card is committed — the Modal container is pinned
to a commit containing it).** Agent: **mac-a (executor)**, day-2 W2
(`briefings/day2-dialogue-mac-a.md`; shared ops
`briefings/day2-dialogue-shared.md`) — self-review hazard named;
verdicts ship PENDING TEAM REVIEW; mac-local reviews the freeze in
parallel with the cache build (shared doc, work split).

**The bet (one bundle, one card, corr disclosed):** dialogue is the
only substrate with measured order-carriage (R11: anchor-fixed
context-shuffle cost +0.057/+0.063/+0.035 at T = 32, 3/3 models; slen
killed generic recency, R20). A second order-carried case study needs
a trailing STATE whose value requires comparing positions — not a
level readable from any bag of tokens. `ttrend` is the Δ/slope face
of dialevel's LEVEL; `dqgap` is distance-to-anchor in its purest
form. The two faces share the substrate and windows, so their errors
are correlated — one bundle verdict, faces reported side by side,
never pooled.

Corpus licence: DailyDialog **CC BY-NC-SA 4.0** (research use); the
note travels with any figure that graduates.

## 1. Substrate — REUSED, not rebuilt

Token stream, caches, screen layers are **dialevel's verbatim**
(`labels/build_dialevel.py` corpus + `dialevel/cache_acts.py`
builders; flat↔windowed mapping re-verified row-by-row in-container
before any forward pass; screen layers gpt2 hs7 / gemma2-2b hs14 /
llama31-8b hs14). Zero new forward-pass designs; the only new
artifacts are labels (`labels/build_diafaces.py`, committed at
`3a8f331a8` BEFORE its outputs, the family rule) derived entirely
from committed arrays — the stream is never re-tokenized. Question
detection is token-level ("?" in decoded vocab string), so the label
evidence and the visible floor count the SAME tokens.

**Coverage: gpt2 + llama31-8b + gemma2-2b** — 3-model, per the
shared-doc HF-secret amendment (secret `hf-token` live in Modal;
gemma carries the LARGEST R11 cost +0.063).

## 2. Faces (frozen; pure logic `labels/diafaces_lib.py`, tests green)

1. **`ttrend`** — kernel-weighted WLS slope (tokens/turn) of the
   previous **5** turn lengths, kernel half-life **2 turns**, current
   turn never in its own label; NaN under 5 previous turns. 3-class
   plain terciles (train-edge; realized edges ≈ ±1.05 tokens/turn on
   all 3 tokenizers — falling / flat / rising).
2. **`dqgap`** — turns since the most recent PREVIOUS turn containing
   a "?" token (≥ 1; NaN before the first question turn). **Measured
   per-turn "?" rate: 0.363** on all 3 tokenizers (dense — the fineweb
   qgap that parked P7 sat at 0.038/sentence). Quantile terciles are
   unusable on a small-integer face (ties at gap = 1 can empty a
   class), so 3-class via DETERMINISTIC balanced integer edges on
   train rows: realized **[1, 2] → gap = 1 / gap = 2 / gap ≥ 3**,
   eligible class balance 41 / 28 / 31 % (disclosed; the zero_split
   check is moot — values start at 1 by the current-turn-excluded
   family rule).

## 3. Design numbers (label-side, measured pre-freeze; screen elig pos ≥ 64)

| quantity | tt (gpt2/gemma2/llama31) | dq (gpt2/gemma2/llama31) |
|---|---|---|
| unigram AUC | 0.548 / 0.550 / 0.548 | 0.577 / 0.580 / 0.578 |
| position AUC | 0.455 / 0.454 / 0.453 | 0.565 / 0.566 / 0.561 |
| doc_mean_only AUC | **0.761 / 0.764 / 0.768** | **0.848 / 0.848 / 0.854** |
| labeled frac | 0.567–0.571 | 0.851–0.853 |

The doc-identity route is real on BOTH faces ⇒ **within-dialogue
arms are BINDING** (shared-doc ops rule 7; dialevel's own 0.98 trap
is the substrate precedent). No KEEP without a same-direction
within-dialogue window gain.

Realized rows (gpt2, manifest build only, no activation read): tt
4000/4000/4000 train + 1500/1470/1500 test per class; dq at caps
both splits. Within-dialogue pools: tt 8000/class train over **1737
dialogues**, 3000/class test over **430**; dq 8000 over 1704, 3000
over 431. MIN_ROWS 300; seeds MATCH_SEED 1013 + crc32; family
eligibility (pos ≥ 64, pos % content ≥ 63, boundary tokens excluded,
caps 4000/1500).

## 4. Clock bridge (measured)

Turn ≈ 14.5–15.7 tokens, ≈ 11.2 turns/dialogue. T16 ≈ 1 turn,
T32 ≈ 2, T64 ≈ 4. `ttrend`'s 5-turn support ≈ 75 tokens: **T64 spans
most of it; complete previous turns visible in-window: ≈ 1.3 at T32,
≈ 3.0 at T64** (measured label-side) — the slope's raw material only
becomes window-visible near the ladder top, so an activation-side
trend state that appears EARLIER than its visible floor is the
interesting outcome. `dqgap`: a "?" token is visible in **85 % of
T32 windows** — the visible floor is a serious opponent by
construction, exactly as intended.

## 5. Probe grid (frozen — the slen/refmark/quotedens shape, verbatim)

Per face: tok linear + MLP(512) FIRST; position floor; per
T ∈ {4,8,16,32,64}: **visible-evidence floor** (dq: ["?" count in
window, any-flag]; tt: [kernel-WLS slope over previous turns COMPLETE
in the window, their count, their mean length] — same kernel as the
label, i.e. exactly what boundary-counting affords), actxmean
linear/MLP ± width-matched foreign nulls; order arms
(flatten/shuffle/foreign) linear at T ∈ {16,32}, MLP triple at T32;
permutation nulls at T16; **within-dialogue arms** (tok linear/MLP,
actxmean ± foreign at T ∈ {16,32,64}, flatten/shuffle/foreign at
T ∈ {16,32}; per-dialogue lo/hi tercile classes, ≥ 30 eligible
rows/dialogue, binary rank-AUC). NO win_mean as primary; NO
max-over-arms; 3-class acc / binary rank-AUC; probe stack
`conversion_depth/problib.fit_probe`, never retuned.

## 6. Pre-registered predictions (scored either way)

- **Q1 (per-token first):** both faces read weak per token — tok −
  position_floor < +0.10 (the label is never a property of the
  anchor token; dq's unigram 0.58 is the "?"-adjacency register).
- **Q2 (window gain):** actxmean − tok grows with T; for tt the gain
  is small below T32 and largest at T ∈ {32,64} (slope material
  enters the window late); for dq present from T16. Width nulls
  cleared (≥ +0.02).
- **Q3 (THE ORDER PREDICTION — the thread's point):** if these faces
  are real TXC-candidates on the one order-carrying substrate,
  **sc > 0 where wc > 0**: win_linear − win_shuf_linear ≥ +0.03 at
  T ∈ {16,32} on the within-dialogue arms wherever the window gain
  itself is positive. An order-FREE KEEP goes to the breadth table,
  not to a panel (gate clause ii).
- **Q4 (the binding lines):** any window gain must (a) beat the
  visible-evidence floor at its T (else tt is boundary-counting / dq
  is question-counting — KILL clause) and (b) reappear
  same-direction in the within-dialogue arm (else it is dialogue
  identity — the 0.76/0.85 route).
- **Q5 (face contrast):** dq behaves regime-1-like (distance to a
  visible anchor: floor-dominated at small T), tt regime-3-like
  (comparison of levels at different distances). If BOTH collapse to
  their floors, the bundle verdict is that dialogue's order signal
  (R11) lives in none of the candidate state variables tested —
  W1's ladder decides what it IS; that is a sound day (falsifier
  honesty, briefing § 4).

## 7. KEEP / KILL (frozen; majority of the 3 screened models)

**KEEP** iff: some matched-class window arm beats tok by ≥ +0.05
with width null cleared ≥ +0.02, AND beats the visible-evidence
floor at the same T, AND the within-dialogue arm shows a
same-direction window gain on supported rows. **KILL** if ANY of:
(1) tok within 0.02 of every window arm at every T; (2) every window
gain fails its width null; (3) every window gain fails the visible
floor; (4) the within-dialogue arm erases the gain. Else **WEAK — no
rule fires as written**, numbers only. Order sensitivity is scored
by Q3 but KEEPs/KILLs nothing by itself; it decides which TABLE the
face goes to (panel gate clause ii vs breadth).

## 8. Venue, economics, discipline

Modal **L40S**, container pinned via `git rev-parse` to the freeze
commit (`_assert_pinned()` in-container); Volume
`temp-xc-replag-caches`, dialevel caches (re)built in the work
container by the COMMITTED builder (idempotent; W1/mac-b shares the
same path — first build wins). Est: caches ≈ 6–10 min/model, screen
≈ 60–90 min/model (2 faces), **3 models in parallel containers, one
model per container** — a stated deviation from shared-ops § 3
"sequential .remote per model", reason: the 14:30 panel-gate clock;
robustness is kept by per-model containers, per-model result files,
Volume persistence after every cell, `--detach`, retries. Est ≤ $12
of the ≤ $15 briefing envelope, mac-a cap $120. Ledger
read-before/append-after; containers never push; repatriate +
merge locally. Deliverable: `results/screen_{gpt2,llama31_8b,
gemma2_2b}.json`, ONE bundle verdict scoring Q1–Q5 in the LOG
(`mac-a (executor)`, PENDING TEAM REVIEW), receipts proposal for
quotable claims, no leaderboard rows (Stage-1 screens are
raw-activation probes). Panel gate: the shared doc's five clauses,
decided by mac-local in writing — nothing here launches a panel.
