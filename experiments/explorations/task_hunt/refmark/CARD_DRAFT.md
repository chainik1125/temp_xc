# DRAFT mini-card — refusal/deflection-marker intensity (WildChat)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad-2.md`,
ledger `../CANDIDATES.md` B7 — round-2 stretch, unlocked by B6's
honest death).** Committed WITH the frozen triage bars below BEFORE
`../labels/build_refmark.py` runs; the running agent freezes its own
screen card. The event list was frozen FIRST, before any counting
(`../labels/refmark_lib.py`, committed at "candidate factory B7
pre-gate"): the refusal paper's `refusal_score` substring set
VERBATIM — 12 strings from `andyrdt/refusal_direction` @ `9d852fae`,
matched case-insensitively anywhere in the turn (their App. D.1
semantics), no additions. **Pre-gate receipt
(`../labels/refmark_pregate.json`): marker rate 0.147 of assistant
turns vs the 0.02 kill bar — 7× over; recurrence real (38 % of
≥ 8-assistant-turn conversations have ≥ 2 marker turns).**

Data side: builder `../labels/build_refmark.py` (event logic
`../labels/refmark_lib.py`, tests `tests/test_refmark_labels.py`) →
`../labels/refmark_wildchat_{gpt2,gemma2,llama31}.npz` +
`../labels/refmark_stats.json` + **the pinned corpus artifact
`../labels/refmark_corpus.json.gz`** (`allenai/WildChat-1M`, train @
pinned revision `7d6490e4…`, license ODC-By 1.0; first-40,000 stream
prefix — stated convenience-sample disclosure — filtered to English,
≥ 8 assistant turns, 2,000–24,000 rendered chars, seeded to 400
conversations; no toxicity/redaction filtering — refusal-adjacent
content IS the event source, stated). **Economics: a NEW token
stream (~1M tokens/tokenizer) — one caching pass per model, minutes
on an H100; BASE models reading chat transcripts — the distribution
shift is part of the candidate's framing, said plainly.**

## The candidate logic

The backtracking-faithful port of the refusal idea (ledger D7 died on
the literature: the current-prompt refusal state is a single
causally-sufficient direction, deposited window→position by attention
heads — conversion IS the measured mechanism). What the paper does
NOT measure is a RECURRING deflection regime over a long
conversation: primary `rlam` = message-level kernel intensity λ̂ over
the PREVIOUS 8 messages (half-life 2 — the sc_lambda/punctint winner
shape at message level; current message never in its own label; NaN
warm-up below message 8), every token of a message inheriting its
λ̂ — "how refusal-laden has this conversation been", a regime-2
aggregation state. Messages render newline-joined WITHOUT speaker
tags (dialevel precedent — tags would be maskable markers). **Masking
rule: every token of an EVENT message (the marker self-stamp) and
every newline-boundary token is masked from probe rows.** The event
bit `is_marker` is the disclosed regime-1 anchor (the per-prompt
refusal state — the converted face per D7's receipts), never the
primary, never manifested; `is_assistant` and `turn_idx` ship as
disclosed structural faces.

## Clock (axis c) — the loudest under-span in the factory, said first

Chat messages are LONG (order 10²–10³ chars; the builder records
tokens/message): the 8-message kernel support is likely
**~1,000–2,500 tokens — far beyond the T = 64 ladder top** (worse
than the Ward λ̂ winner's under-span and dialevel's ~90). A T-window
usually sits INSIDE one message and rarely contains a marker itself.
Consequence, per binding review qualification 3: **the
beat-the-visible-evidence line must print next to every window
number** — the screen must show the window beating what its own
visible tokens could tell it, and the predicted mechanism is
register-evidence pooling (deflection-adjacent register in ambient
prose), not marker counting.

## Label-side triage — FROZEN BARS (kill authority)

**Pinned broad convention (review qualification 4):** direction-
agnostic max(AUC, 1−AUC), test-doc rows, top vs bottom class, BOTH
all-eligible and shipped-manifest rows, **manifest rows operative**;
current-token type-mean AUC **≥ 0.65 ⇒ KILL** (the named axis-b risk:
refusing conversations are topically distinctive — harmful/sensitive
vocabulary; and apologetic register bleeds beyond the exact frozen
strings); position AUC **≥ 0.65 ⇒ KILL**; 0.55–0.65 ships with
disclosure. Manifests position-matched from the start (pos ≥ 32,
log2 strata). **Additionally reported (adopted 2026-07-24 from
runpod-e's recommendation; disclosure statistic, NOT a frozen kill
bar): `doc_mean_only_auc`** — conversation-mean of λ̂ as the only
feature, the document-identity route the punctint screen surfaced
(their doc-mean AUCs ran 0.926–0.960 on faces that passed both
frozen bars). Whatever it reads, it prints in the verdict and any
future KEEP owes a within-conversation contrast (the q-face
precedent).

## Predicted T-pattern + draft kill rule

Per-token poor on masked rows; window − per-token gap positive and
growing with T (evidence pooling over ambient register); order-free
(mean ≈ flatten), shuffle-immune — regime-2. Expected zero_split
scheme (P(no marker in previous 8 messages) ≈ 0.5). KILL at screen
if: per-token reads `rlam_bin` within noise of the best window at
every T; or no gap beyond 3 σ_null at any T; or no T-growth anywhere
in the ladder. Screen preconditions carried from the family: the
position-only floor probe on shipped manifests (qualification 1
discipline) and the beat-the-visible-evidence line (qualification 3);
if the doc-identity statistic comes back punctint-loud, a
within-conversation contrast is the mandatory control before any
promotion (q-face precedent).

## Triage RESULT (builder-derived; appended after the frozen bars ran)

**SHIPS — no frozen bar fires — and the loudest number is the one the
card pre-committed to disclosing, stated first: `doc_mean_only_auc`
= 0.966–0.968** (direction-agnostic, BOTH all-eligible and manifest
rows, all three tokenizers) — the conversation-identity route is
LOUDER than the punctint faces that motivated the statistic
(0.926/0.960). λ̂ terciles are overwhelmingly a conversation-level
property: refusal-laden conversations vs clean ones. **Binding
consequence (pre-committed above): the within-conversation contrast
is the MANDATORY control at screen — without it, any window gap here
is uninterpretable as temporal structure** (q-face precedent; the
dialevel-qualification-2 discipline). The frozen bars themselves came
back clean-to-mild: current-token type-mean **0.515–0.536 all-
eligible / 0.517–0.532 manifest — near-BLIND**: the D7-inherited fear
(harmful-topic vocabulary, apologetic register bleed) does NOT
materialize at the current-token level once marker messages are
masked — the label's readable routes are conversation identity and
mild position, not token identity. Position: 0.613–0.617 all-eligible
(disclosure band; high-λ̂ rows sit EARLY — raw 0.383–0.387),
pulled to **0.545–0.565 on the operative manifest rows** (gpt2 0.545
clean; gemma2/llama31 0.562–0.565 low disclosure band). Scale
receipts: 400 conversations, 1.19–1.36M tokens/tokenizer,
tokens/message 125–144 ⇒ **kernel support ≈ 1,000–1,150 tokens —
the stated under-span confirmed (~16× the T = 64 ladder top)**;
marker rate 0.148 of assistant messages (matches the pre-gate);
zero_split fired (train zero-frac 0.69–0.70); labeled fraction 0.65;
marker-token fraction 0.09 (masked); manifests ~20k rows/class.
Artifacts committed: three npz + `refmark_corpus.json.gz` +
`refmark_stats.json`. Screen preconditions, binding: (1)
within-conversation contrast before any promotion; (2) position-only
floor probe on shipped manifest rows; (3) the
beat-the-visible-evidence line next to every window number
(qualification 3 — a T-window almost never contains a marker).
