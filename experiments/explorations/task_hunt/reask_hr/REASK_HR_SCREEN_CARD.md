# REASK_HR_SCREEN_CARD — wave-3: hard-refusal-gated re-ask age on the refmark2k substrate

**STATUS: FROZEN AT THIS COMMIT (card + cache_acts + screen + verdict
in ONE commit; pin = this commit's sha from origin history, asserted
by the launcher before any cell). Pre-registration; owner runpod-a
(single-owner discipline); ALL verdicts PENDING TEAM REVIEW.**

Mandate chain: wave-3 directive ae1ce5fb0 → trio pre-measures
ratified c5023d9f3 (reask = the ONE card candidate) → gate census
(my ~19:05 entry, PTR'd) pre-registering `reask_hr` as PRIMARY →
mac-local c6e464881 releasing this build. Label numbers below are
the committed artifacts' — `reask_hr_premeasure.json`,
`reask_gate_census.json`, `wave3_trio_stats.json`,
`refmark2k_stats.json` — quoted, not re-derived.

## § 0 The two binding wave-3 bars, stated first (ae1ce5fb0)

**Clock, stated before anything turn-scale:** the refmark2k
substrate runs 119–137 tokens/message (llama31 119.09 / gemma2
124.23 / gpt2 137.00 — `refmark2k_stats.json` verbatim). The whole
probe ladder T ∈ {4..64} is SUB-MESSAGE-SCALE: T64 ≈ half a message.

**Out-of-window-by-construction:** a `reask_hr` event fires at user
turn u_i only if the re-asked u_{i-2} exists and the intervening
assistant turn a_{i-1} hard-refused — the evidence for "there was a
hard-refused ask, now re-asked" sits ≥ 2 messages (≈ 240–270
tokens) behind the probed assistant token, beyond EVERY ladder
window. Receipts (premeasure, committed): censored-age floors ≤
0.560 at ALL T ∈ {4..64} on all three tokenizers; in-window
event-token-count floors are at-or-below chance (0.41–0.49). The
face's information, if the activations carry it, is trailing state
the window cannot see.

## § 1 The slate (ONE screened face)

| face | disposition |
|---|---|
| **reask_hr** (hard-refusal-gated re-ask age; `wave3_reask_hr_<tok>.npz` VERBATIM — sage_face kernel over event-message first-token flags, support 64) | **SCREEN — the wave-3 primary** (gate census: 233 events, 7.0 % of convs ≥ 1, max 9/conv) |
| pooled reask (apology-persona events included) | **labels-only disclosure, NOT screened** — anti-dup COLLAPSE vs hr: ρ 0.950/0.943/0.940 (gpt2/gemma2/llama31), far over the 0.8 bar ⇒ one face screens for both; hr is the cleaner-gated pre-registered primary |
| refmark rlam (the wave-2 refusal face) | distinct — ρ(hr_age, rlam) −0.60 across tokenizers; no dup conflict |

## § 2 Design

Substrate: the committed refmark2k WildChat grid
(`labels/refmark2k_wildchat_<tok>.npz`: token_ids, doc_off,
turn_idx, is_assistant, is_boundary, doc_split — 2k convs) + the
committed `labels/wave3_reask_hr_<tok>.npz` label arrays, both
REUSED VERBATIM (zero re-tokenization; the dialevel alignment
contract). Models: **all three legs from the freeze** — gpt2/hs7,
gemma2_2b/hs14, llama31_8b/hs14 (replag SCREEN_HS), sequential on
GPU 0.

**COLD caches, priced up front:** `reask_hr/cache_acts.py` chunks
the committed stream (BOS prefix per model, non-overlapping content
chunks, tails dropped, every row byte-verified against the flat
stream before any forward) → ~51k rows × 128 per model.
**Single-layer capture (SCREEN_HS only) — a disclosed deviation
from the wave-2 3-layer convention**: at 51k rows the full capture
is ≈ 280 GB; the screen reads one layer, so one layer is cached
(gpt2 ≈ 10 GB, gemma2 ≈ 30 GB, llama31 ≈ 53 GB; 792 GB free,
receipts in acts_meta.json). Forward cost ≈ 2 / 8 / 20 min on H100.

Probe grid: the hunt4 grid VERBATIM (hunt4w2 transplant): tok
linear+MLP; position floor; visible floor per T ∈ {4,8,16,32,64};
actxmean ± foreign; win + win_shuf linear T ∈ {4,8,16,32};
win_foreign {16,32}; MLP order triple T32; nulls T16; **within-DOC
(= within-conversation) arms BINDING** with the extended wd order
ladder. CAP 4000/1500, MIN_ROWS 300, POS_MIN 64, OFF_MIN 63.

**Eligibility** = assistant tokens only (`is_assistant == 1`),
event tokens masked (`reask_hr_event_mask == 0` — the probe never
reads the event text), turn boundaries masked, position ≥ 64,
in-chunk offset ≥ 63.

**Manifests (position-matched — the tret precedent, BINDING):**
3-class tercile-of-age manifests built in-screen from the committed
arrays via the FROZEN `punctint_lib.pos_strata` (log2 strata) +
`stratified_balanced_manifest` (equal class counts within every
position stratum), seeded by the frozen `_seeded` convention.
Tercile edges recomputed by the premeasure's own `_terciles` on the
premeasure's own eligibility and ASSERTED equal to the committed
`reask_hr_premeasure.json` edges before any probe — a freeze
receipt, not a re-derivation. The manifest-row position AUC is
reported in the row stats (operative number, not assumed 0.5).

**Floor arm (per T):** visible_evidence_floor features =
[`wave3_lib.sage_floor(first, T)` (censored age),
`wave3_lib.dose_window_count(mask, T)` (in-window event tokens)] —
BOTH committed-lib functions of the committed event arrays,
computed in-screen (deterministic; no RNG; a floors-npz artifact
would be 195 MB of pure function output — deviation from the
wave-2 committed-floors convention stated here, function + inputs
both frozen).

## § 3 Label pre-measures (committed JSONs, quoted)

Per tokenizer (gpt2 / gemma2 / llama31): eligible rows 244,636 /
228,060 / 219,317; unigram AUC 0.575 / 0.565 / 0.560 (mild);
**position AUC 0.946 / 0.929 / 0.925 — the named trap** →
position-matched manifests + position-floor arm + BINDING wd arms;
**doc-mean AUC 0.828 / 0.821 / 0.818 — the identity trap** → wd
arms binding. Censored-age floors: ≤ 0.503 at T16, ≤ 0.519 at T32,
≤ 0.560 at T64. In-window count floors 0.41–0.49 (at/below
chance).

**Claim zone = the FULL ladder T ∈ {4..64}** (pre-registered): the
visible floor never exceeds 0.560, so any T may claim — but every
KEEP-qualifying arm must beat its OWN T's combined visible floor,
the width null, and the tok baseline simultaneously (§ 4), and the
wd gate applies regardless of T.

## § 4 KEEP / KILL (frozen; the hunt4 § 4 rules VERBATIM)

Existential KEEP (some window arm: gain ≥ +0.05 over best tok ∧
width null ≥ +0.02 ∧ beats its own-T visible floor, simultaneously)
∧ within-DOC same-direction gain; the four KILL clauses; else WEAK;
SKIP per the approved 6b03b1b06 patch. Order (routing only): wd
win−shuf ≥ +0.03 at any T ∈ {4,8,16,32} with positive wd gain.
Bundle = majority over the three models (3/3 or 2/3; all legs run
from the freeze so no PENDING-THIRD-LEG state is reachable except
by SKIP). Scorer `reask_hr/verdict.py` (hunt4 `score_model`
imported unmodified) committed WITH this freeze — before any cell
runs.

## § 5 Venue, economics, discipline

Pod A GPU 0 (runpod-a), sequential behind the in-flight actmix
tsae_s2 lane; launcher asserts pin (origin-history rev-parse) and
runs the three (cache → screen) legs in slate order gpt2 →
gemma2_2b → llama31_8b, then verdict. Caches ≈ 30 min GPU + probes
≈ 1–2 h wall (mostly CPU fits against mmap'd fp16) → **est ≈ 1.5–3
pod-h ≈ $3–6**, hunt envelope (c1c5c949e, ≈ $160 headroom at
freeze). Ledger line at launch; actuals after. Deliverables:
`reask_hr/results/screen_wildchat_<model>.json` × 3 +
`results/verdict.json` + ONE bundle LOG entry (PTR); the WRITEUP
§ 8 rows for the wave-3 trio kills batch WITH this screen result
(mac-local's line). Caches are pod-scratch (/workspace/
reask_hr_caches), never committed.
