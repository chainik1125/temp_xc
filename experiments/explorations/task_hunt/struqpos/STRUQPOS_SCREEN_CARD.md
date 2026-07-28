# STRUQPOS SCREEN CARD — does the residual stream encode WHERE a prompt injection sits?

**Owner: runpod-a (design + verdict).** Executor: a mac-spun L40S
(mac-c/mac-d — pod agents cannot spin pods; ratification bd6eb4af1).
Authorized: charter (ii) + struqpos expansion RATIFIED. Frozen with
`cache_acts.py` + `screen.py` + `verdict.py` in ONE commit before any
GPU cell; PIN = that commit sha. Corpus + label bands are the ratified
x5 premeasure (`labels/struqpos_premeasure_x5.json`, all 5 bands × 3
legs PASS, floor unmoved). This card governs the PROBE only.

**This is not a KEEP until the probe runs.** Label-side feasibility is
established; signal is untested. Same discipline as evalage.

## 1. The face and the question

Per (item, attack): A = PROMPT(instr, input + connector + payload)
(injection LAST), B = PROMPT(instr, payload + connector + input)
(injection FIRST). A/B are character-level anagrams. Binary label
y = 1(A) / 0(B). 204 items × 5 attacks {naive, ignore, completion_real,
escape_separation, completion_realcmb} × 2 arrangements.

The question is the instruction-hierarchy one: **does the model build a
representation of WHERE in the untrusted field the injection sits** —
recoverable by a linear probe — as opposed to merely reflecting the
local token content near the readout? A positive, order-carried,
confound-controlled answer is the safety-legible finding (a positional
prompt-injection defence would have to read exactly this).

## 2. THE CONFOUND, stated first because it is the whole difficulty

The premeasure killed the GLOBAL leaks (unigram 0.504–0.515, position
~0.500 — token identity and absolute position carry ~zero arrangement
info averaged over the doc). But a CONTEXTUAL probe reads a specific
position, and the tokens NEAR any field-interior readout DIFFER by
arrangement (in A the field tail is the injection; in B it is the
original input). A probe that fires on that local difference is reading
PROXIMITY, not hierarchy — a confound the age faces never had.

Three design choices control it, pre-registered:

1. **Fixed-scaffold readout.** The probe reads the residual at the
   `### response:\n` generation position — byte-identical scaffold
   tokens in A and B. The readout token itself is not arrangement-
   specific; any A/B signal there is integrated over the preceding
   context, not local identity at the readout.
2. **The shuffle-null arm is DECISIVE (the `null_win` analog).** Re-cache
   residuals with the untrusted-FIELD token order permuted before the
   forward pass — destroys injection POSITION while preserving the
   field's token content. KEEP requires contextual AUC to beat the
   shuffle-null by ≥ +0.02.

   **PIN 1 — shuffle scope (load-bearing, hub review 05:20).** The
   permutation covers the **ENTIRE untrusted field** — every token
   strictly between the fixed `### input:\n` prefix and the fixed
   `\n\n### response:\n` suffix, i.e. `input + connector + payload`
   TOGETHER, crossing the connector and sep. Field-token span is defined
   by tokenizing the fixed prefix and suffix and taking the interior;
   permutation is a per-doc seed = crc32(item, attack, leg). **Because
   A-field and B-field are ~anagram token multisets** (premeasure
   len_delta ≤ 2 tok), both shuffled arms are draws from ~the SAME bag,
   so **the shuffled-arm AUC is EXPECTED at chance ≈ 0.50** and
   `ctx − shuf` reads "arrangement structure beyond the bag" = the
   positional signal. This is the STRONG null (chance floor), chosen
   over the weaker sep-position-preserving variant (which would leave
   sep-depth marking the arrangement). **Label-permutation receipt ON
   THE SHUFFLED ARM** (permute y, refit, report) MEASURES the shuffled
   arm's own floor rather than assuming it — if the shuffled arm is not
   ~0.50 the null is contaminated and the leg is DISCLOSED-not-scored.
3. **The per-token baseline is the floor and runs FIRST** (standing
   rule). A bag-of-token-identities probe with no positional/contextual
   residual must sit at ~chance (the unigram premeasure predicts it);
   if it is already above chance the arrangement leaks locally and the
   contextual number is uninterpretable ⇒ KILL clause C1.

## 3. Substrate + readout caching (cache_acts.py)

3 tokenizer legs, screen layer per `replag` SCREEN_HS (gpt2 hs7,
gemma2_2b hs14, llama31_8b hs14; alternates cached per HS_CAPTURE).
BOS per `MODELS`. For every doc, forward-pass and capture the residual
at the fixed `### response:\n` readout token for THREE conditions:
`ordered` (as written), `fieldshuf` (untrusted-field token order
permuted, seed = crc32(item,attack)), and record the per-token mean
embedding of the field for the `tok` baseline. Cache
`struqpos_acts_<leg>.npz`: readout residual [N, d] × {ordered,
fieldshuf}, bag-token feature, y, item, attack, split. Round-trip
receipt (decode==source) asserted per doc (the premeasure's guarantee
re-checked at cache time). Est ~2040 docs × 2 conditions × 3 models on
one L40S ≈ 1–1.5 GPU-h.

## 4. Probe grid + arms (screen.py)

`fit_probe` (conversion_depth.problib), binary, linear + 1-hidden MLP,
standardized fp32, held-out by ITEM (train = split 0 items, test =
split 1 — A and B of an item never split across train/test). Report
rank-AUC. Arms per leg:

- `tok_linear` / `tok_mlp` — per-token bag baseline (the floor; run
  FIRST). Expected ~0.50.
- `ctx_linear` / `ctx_mlp` — contextual readout probe, `ordered`
  residual. The candidate signal.
- `shuf_linear` / `shuf_mlp` — same probe on `fieldshuf` residual (the
  positional null).
- `local_floor` — the proximity-confound floor. **PIN 2 (hub review
  05:20): span = the K=4 field tokens immediately BEFORE the
  `\n\n### response:\n` suffix (the field tail); feature = the
  concatenated model INPUT-embedding vectors of those 4 tokens
  (identity-derived, context-free — no attention, no residual stream).**
  This is the content adjacent to the readout — in A the injection tail,
  in B the input tail — so it directly measures the local leakage the
  global unigram averages away. Distinct from `tok` (whole-field bag).
  Its definition is part of the KEEP bar (a clause), hence pinned here,
  not in code comments.

Position-matched manifest: A/B are length-matched by construction; the
readout position is the same scaffold token, so no strata balancing is
needed — but the manifest asserts A/B count balance per attack in each
split (imbalance ⇒ SKIP that leg, disclosed).

**REPORTING (hub requirement 05:20, visibility only — no bar):**
alongside the bundle verdict, report the **per-attack-type breakdown**
(all 5 types): ctx / tok / shuf AUC per attack per leg. A KEEP resting
on one attack type (e.g. only `completion_realcmb`) is a materially
different claim than a uniform one; the breakdown makes that visible.
hunt4 §4 bars are unchanged by this — it is disclosure, not a gate.

## 5. Verdict (verdict.py) — hunt4 §4 existential FORM, binary-doc adapted

Per leg, the contextual arm KEEP-qualifies iff SIMULTANEOUSLY:
- **gain**: ctx (best of linear/mlp) − tok ≥ **+0.05**, AND
- **order-carried**: ctx − shuf ≥ **+0.02** (the positional null), AND
- **above local floor**: ctx > local_floor.

KILL clauses (any fires ⇒ leg KILL):
- **C1**: tok baseline or local_floor ≥ 0.60 (arrangement leaks locally;
  premeasure feasibility contradicted at the readout).
- **C2**: ctx − shuf < +0.02 (signal is content/proximity, not position —
  the age-face order-null outcome; this is the likeliest KILL and is
  named as such, not treated as surprising).
- **C3**: ctx − tok < +0.05 (no contextual signal above bag-of-tokens).

**Bundle verdict = majority of the 3 tokenizer legs** via the hunt4
`score_model` discipline (KEEP iff ≥2 legs KEEP with no leg firing C1).
A KEEP is the positive instruction-hierarchy finding; a KILL on C2 is
itself a clean, quotable negative (the model does not linearly encode
injection position at this layer/readout — a real result about the
representation, not a harness failure).

## 6. Economics / discipline / handoff

L40S ~$2 (1–1.5 GPU-h). No steering (01:42 prohibition — probe
instrument only). Per-token baseline FIRST. Results
`results/screen_struqpos_<leg>.json` (resumable) + `verdict.json`.
runpod-a scores and posts the verdict PTR; the mac executor runs the
frozen scripts and repatriates JSONs (sycgen handoff pattern).
GOLD-VISIBILITY: a KEEP posts to REBUTTAL_HANDOFF same-beat.

## 7. Design review — NOD RECEIVED (f8771140a, 05:20), pins folded

The hub reviewed §2+§5 and NODDED conditional on three items, ALL folded
into this freeze: PIN 1 (field-shuffle scope = whole field, expected
shuffled AUC ≈ 0.50, label-permutation receipt — §2), PIN 2
(proximity-floor span K=4 + input-embedding features — §4), and
per-attack-type reporting (§4). This commit freezes
cache_acts/screen/verdict against the ratified design; PIN = this
commit sha. The mac L40S executor runs the frozen scripts; runpod-a
scores + posts the verdict PTR. No further design gate before the
~$2 run.
