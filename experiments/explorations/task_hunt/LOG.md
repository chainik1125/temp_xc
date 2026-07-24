# Task-hunt log (shared, append-only)

The shared verdict log for the task hunt (`briefings/task-hunt.md`,
`briefings/task-hunt-b.md`). **Append-only; pull-rebase before every
push.** One entry per screen/stage event: date, agent, candidate,
verdict (KEEP / KILL / infeasible), one-paragraph justification, pointer
to the frozen card + results.

Conventions: mini-cards are committed BEFORE their screen runs; every
verdict cites its card; scale-orderings and honest kills are findings,
not failures (prime directive: a sound verdict, never a win).

---

## 2026-07-24 — runpod-d — candidate 1 (backtracking λ̂ intensity) — CARD FROZEN, screen pending

Ward stream + both reader caches rebuilt on the runpod-d volume
(`build_ward_stream.py` verbatim; stream stats reproduce the committed
reference — map_ok 99.97 %, BOS every row; base 219 s / distill 238 s
cache sweeps). λ̂ labels built locally
(`lambda_intensity/build_labels.py`) from the frozen mirror params on
real Sonnet history — runpod-b's `labels/ward_lambda.npz` had not
landed when caches were ready (never-idle rule); it landed mid-session
and the two builds **cross-validate: 99.93 % exact agreement** on
overlapping finite cells, disagreements confined to 277
sentence-boundary token attributions + their padded i<8 sentences
(excluded by my frozen i≥8 rule). Proceeding on my builder (the card's
freeze target); theirs stands as independent confirmation. Label-side
floor check (pre-freeze, disclosed in card): position-only AUC 0.82 on
full-λ̂ terciles vs 0.59 for kernel-only λ̂_hist ⇒ **primary target =
λ̂_hist**. Card: `lambda_intensity/card.md`, frozen at this commit;
screen (`screen.py`) not yet executed on any activation cache.

## 2026-07-24 — runpod-e — replag labels built (pre-screen); one card amendment

Δ labels built inline (runpod-b's parallel builder had not landed;
per-briefing fallback). `replag/build_labels.py` + committed manifests/
stats under `labels/`. All 5 sanity tests pass on all three tokenizers;
every task saturates its caps (4000/1500 per class) under joint
(token-id × position-bucket) matching; realized coverage is the exact
ladder (cov(B,T) = 0 until T crosses the bucket's lower edge).

**Label-stage finding + card amendment (BEFORE any screen ran):** the
card's parenthetical prior "real text far heavier at small Δ than the
shuffled null" is FALSE at the token level — real P(Δ∈[2,4]) ≈ 0.026 <
null ≈ 0.031 on all three tokenizers (grammar avoids near-repetition;
the exchangeable shuffle clumps). Sanity T4 amended from a directional
assert to a two-sided divergence test (TV > 0.02; realized ≈ 0.03) with
direction recorded as a finding. No screen prediction (P1–P5) depends
on this; the KEEP/KILL rules are untouched.

**2026-07-24 · runpod-b · prep (labels + cards) — no verdict.** All four
label artifacts landed under `labels/` (builders committed pre-run, 10
sanity tests in `tests/test_task_hunt_labels.py`): `replag_fineweb_*` ×3
tokenizers, `ward_lambda` (causal mirror λ̂; round-trip validity matches
the committed ward_stream_stats exactly; event rate by λ̂ tercile
0.053/0.081/0.256), `proofops` (+ the clock bridge: median 16
tokens/sentence ⇒ 2-sentence windows need T ≥ 32), `confidence`
(slope4/slope8 + state; slope8 support ≈ 128 tokens — arm-B candidate-2
reachability input). **Duplication note (per briefing):** runpod-e froze
`replag/CARD.md` with its own inline labels before these landed; the
frozen card's scheme differs from `labels/replag_fineweb_*` (within-
sequence Δ, Δ=1 excluded, B32 bucket, identity/position matching vs my
doc-level Δ, buckets ≤4/≤8/≤16/none). The frozen card governs the
screen; the committed artifacts stand as an independent cross-check +
the n=2 robustness reserve. Label-side order receipt worth having: at
doc level, short-lag BIGRAM repeats are ≈13× the within-doc-shuffle
frequency null, unigram short-lags only ≈1.2× in aggregate — and the
Δ∈[1,4] unigram bucket alone sits BELOW the null (20.9k vs 28.5k on
gpt2), the doc-level face of runpod-e's token-level T4 finding above
(`labels/replag_stats.json`). Cards: `proofops/CARD.md` +
`confidence/CARD.md` frozen (science sections; running agents append
screen cells); `forbidden_word/` + `emotional_instability/` CARD.DRAFTs
staged.
