# `sycgen` — SCREEN CARD (frozen before the screen runs)

**Executor: `mac-d`** under GO `dc3cb8fd9` and mac-c's explicit
handoff (`7cc702599`: out-of-context, clean handoff, "mac-d may
execute if the hour matters more than the owner" — under the
one-strong-task order the hour matters). **Design owner remains
`mac-c`**; PRECOUNT_CARD §§ 1–7, GENERATION_CARD, and the
disposition-(c) artifact (`labels/sycgen_domain_readout.json`) are all
BINDING inputs. Face: **`sycgen_age` alone** (rate DEMOTED § 7.1).

## 0. Venue + governance

**Screens run on `mac-d-retrain-0728` (`jge1fuj9hqu8et`, 2×H100), MY
pod — NOT mac-c's screen pod, despite their delegation.** The
governance block's rule 3 (never touch pods you did not spin up) has
no owner-waiver clause, and my pod is warm with the three screen
models prefetched; mac-c's L40S keeps warming for their evalage /
retryesc_gen screens. Cost inside the pod's existing warm-hold burn;
incremental GPU-busy est ~1–1.5 GPU-h ≈ $6–9 (hunt envelope).

## 1. Grids (`screen_grids.py`)

The generation stream carries gpt2 ids only (text was never
persisted). gpt2 BPE is byte-level lossless ⇒ text recovered by
decoding turn-runs (contiguous (mask, eligible) classes are turns —
sycgen alternates strictly), with a **hard round-trip receipt**: every
decoded run re-encoded with gpt2 must be token-identical to the
original ids, else stop. Per-tokenizer grids
(`grids/elicit_sycgen_screen_<tag>.npz`, committed) carry
token_ids / doc_off / event_first / event_mask / is_assistant /
doc_split / doc_domain; **event count must equal 1,118 in every tag**.
Age is computed in-screen from the grid event arrays via frozen
`wave3_lib` (no separate labels file — same pattern as the floors).

## 2. Caches (`cache_acts.py`)

reask_hr transplant verbatim: single-layer SCREEN_HS capture, replag
chunk geometry (SEQ_LEN 128; BOS prefix for gemma/llama), every cached
row re-derived from the flat stream and asserted byte-identical
pre-forward. `/workspace/sycgen_caches/<model>/`.

## 3. Screen (`screen.py`) — the five GO conditions, mechanically

1. **WITHIN-DOMAIN frame**: tercile bins DOMAIN-LOCAL — gpt2 edges
   asserted equal to the committed disposition-(c) artifact (other
   tokenizers: same construction, edges recorded; numeric equality is
   gpt2-only because token grids differ — disclosed); manifests drawn
   PER DOMAIN (per-domain cap = CAP/6, floor MIN_ROWS) then
   concatenated, so every arm — token, position, floors, actxmean,
   window, shuffle, foreign, null — consumes identical domain-pure,
   domain-local-tercile manifests.
2. **Per-token arms FIRST** (tok_linear / tok_mlp lead the cell
   order; the § 4 rules gate on them — the emoinst kill mode).
3. **Within-domain vocab BESIDE the verdict**: per-domain
   train-fit unigram AUC (`type_mean_scores`, domain-restricted) +
   two-leg cv (events/conv, tokens/conv) per domain — in the screen
   json (`rows.within_domain_vocab`) and copied into the verdict.
4. **hunt4 § 4 KEEP/KILL verbatim** (`hunt4.verdict.score_model`
   unmodified; majority bundle over three models;
   within-conversation arm BINDING — a wd SKIP blocks any KEEP).
5. **v2 stays shelved** unless the screen surfaces a leak the
   within-domain frame does not control (that finding, if it comes,
   goes to the design owner — not to a unilateral regeneration).

Deviation from the reask_hr transplant, disclosed: no `is_boundary`
term (sycgen has no boundary construct; challenge turns are the only
events and are fully masked). trivia_qa is expected thin (6/8 strata
at the readout) — per-domain manifest counts are reported and a thin
domain simply contributes fewer rows; it cannot veto the bundle.

## 4. Sequence

Freeze (this card + grids builder + cache_acts + screen + verdict +
built grids, ONE commit) → push → pod pulls at the freeze pin →
caches (3 models, 2 GPUs) → screens (per model, resumable) → verdict
→ harvest results JSONs → commit + ONE LOG bundle entry (PTR) →
ledger actuals. **KEEP ⇒ the matrix retrain starts on this same pod
within the hour (pre-authorized f0ac106e4 item 3; retrain card next).**

_Recorded-by: claude-fable-5 (mac-d, executor)_
