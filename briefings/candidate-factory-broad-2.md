---
status: active
created: 2026-07-24
for: runpod
venue: runpod (32C CPU)
---

# Candidate factory, broad corpus — round 2 (QUANTITY MODE continues)

**You are `runpod`.** Your round-1 factory batch is REVIEWED & APPROVED
(binding qualifications + the screen-queue order: the mac-local
"REVIEW: candidate factories" entry in
`experiments/explorations/task_hunt/LOG.md`). The GPU pods are now
consuming the screen queue. This round: extend the ledger with two
vetted entries from a Han-proposed idea, build B6, and keep the ledger
live as screen verdicts land. **Results by Saturday evening PT**
(Sunday 10:00 PT check-in). All round-1 disciplines govern unchanged:
builders committed before outputs, frozen mini-cards with triage bars,
label-side triage as kill authority, zero-API exact labels, LOG line
per bundle/kill, no reviewer/meeting quotes in tracked files.

## 1. Ledger updates first (append to `CANDIDATES.md`)

- **D7 — refusal-as-posed: DEAD (vetted by mac-local, 2026-07-24;
  record with these receipts).** Proposed as "maybe attention doesn't
  linearize refusal to a single position." The literature answers
  directly (`docs/papers/refusal.md`, Arditi et al.): a difference-in-
  means direction extracted at SINGLE (position, layer) pairs is
  causally sufficient — ablate ⇒ no refusal, add ⇒ refusal — across
  13 chat models to 72B, and its § 5.2 shows attention heads
  performing the window→position deposit (conversion IS the measured
  mechanism); the direction exists even in base models (their App. J).
  Axis (b): harmful-topic vocabulary is a massive unigram leak;
  refusal text is self-stamping. Axis (c): a prompt-level rollout
  boolean — the AVOID class (forbidden-word/emotional precedent).
  Axis economics: needs chat model + instruction corpus (no cache
  applies) and judge labels beyond string match.
- **B7 — refusal/deflection-marker intensity on multi-turn chat
  (WildChat-class): BUILD-if-time, BEHIND B6.** The backtracking-
  faithful port of the refusal idea — requires RECURRENCE, which
  standard refusal data lacks. Sketch to vet properly on the four
  axes before building: events = assistant turns matching a FROZEN
  refusal/deflection substring list (seed it from the refusal paper's
  own substring set; freeze before counting anything); label = λ̂ over
  PREVIOUS turns (kernel per the sc_lambda/dialevel precedent),
  marker-turn tokens masked; corpus = a real multi-turn chat corpus
  with recurring refusals (WildChat-class; CPU-downloadable), shipped
  as a pinned artifact per the new-corpus rule (dialevel precedent:
  transcripts run through the three cached base models — one caching
  pass each, note the cost). Known risks to state: refusing
  conversations are topically distinctive (unigram leak — triage
  decides); event rate may be thin outside filtered subsets (measure
  before building; if <~2% of turns, kill in the ledger for free).
- **Verdict hygiene (standing, cheap):** as d/e post screen verdicts
  in the LOG, append one-line outcomes to the ledger's verdict index;
  re-vet PARKs whose reasons a verdict touches (P2 lifts if
  punctint-list dies specifically on position; P6 lifts if Ward
  verbosity dies on a Ward-specific artifact).

## 2. B6 — OpenWebMath equation-density intensity (the build)

Per the ledger's B6 design, unchanged: math-mode spans by exact LaTeX
delimiter grammar (`$…$`, `$$`, `\[`, `\begin{equation}` — FROZEN in
the card before building); primary = kernel-smoothed trailing
math-token rate from previous sentences/lines, current span excluded,
math tokens masked from probe rows; the in-math bit is the disclosed
regime-1 anchor, not the primary. Axis-b risk to triage: math-notation
vocabulary leaks topic. New-corpus rules: pinned tokenized corpus
artifact (or exact re-pull script) + caching-cost note. Standard
bundle format, triage stats, frozen bars.

## 3. Stretch: build B7 (only if B6 has shipped or honestly died)

## Acceptance gate — stop for review

Ledger updated (D7 + B7 entries + any verdict lines); B6 shipped or
triage-killed with receipt; LOG line per item; STATUS rewritten.
Briefing stays until mac-local review.
