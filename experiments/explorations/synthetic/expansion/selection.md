# Cycle-1 selection (Stage 2, blind)

Rule: tier(under-coverage) desc, then labelability*novelty*predicted_temporalness desc; per-domain floor N/2; max one pick per ledger cell; scorer=claude-opus-4-8 (blind: no data, no arch scores)

| candidate | domain | class | cell | lab | nov | temp | score | picked |
|---|---|---|---|---|---|---|---|---|
| question-answer-adjacency | text-corpus | AC-order-sensitive | empty | 4 | 4 | 4 | 64 | **✓** |
| assumption-then-consequence | reasoning-trace | AC-order-sensitive | empty | 3 | 4 | 4 | 48 | **✓** |
| quotation-burst | text-corpus | bursty/self-exciting | empty | 5 | 3 | 3 | 45 | **✓** |
| uncertainty-hedging-drift | reasoning-trace | DC-slow-drift | empty | 4 | 3 | 3 | 36 | **✓** |
| pronoun-referent-recurrence | text-corpus | long-memory | empty | 3 | 4 | 3 | 36 |  |
| enumeration-cadence | text-corpus | periodic | empty | 5 | 3 | 2 | 30 |  |
| goal-restatement-recurrence | reasoning-trace | long-memory | empty | 3 | 3 | 3 | 27 |  |
| computation-verification-alternation | reasoning-trace | periodic | empty | 4 | 3 | 2 | 24 |  |
| hedge-to-assertion-drift | text-corpus | DC-slow-drift | ABORT-only | 4 | 3 | 2 | 24 |  |
| error-correction-cascade | reasoning-trace | bursty/self-exciting | PROCEED | 4 | 3 | 4 | 48 |  |

## Scorer reasons

- **goal-restatement-recurrence** — Restatement vs fresh-step is a genuine judgment call at boundaries; long-memory/renewal in reasoning is fresh axis coverage vs bursty backtracking; heavy-tailed gaps plausible but restatements may concentrate at start and get killed by N2.
- **uncertainty-hedging-drift** — Ordinal confidence is fairly reliably labelable via clear lexicon cues; DC-drift in reasoning is a new cell but overlaps conceptually with the text hedge-drift card; persistence risks being pure position trend flattened by N2 or per-doc composition.
- **computation-verification-alternation** — Verify-vs-advance is a clean binary with strong lexical anchors; periodic reasoning cell is new but synthetic periodic benches exist; a strict preferred PERIOD in verification is a priori unlikely—more likely clustered than rhythmic, so spectral peak may not survive.
- **assumption-then-consequence** — A/C/neither is somewhat ambiguous since 'so'/'then' bleed across categories; directed order-sensitivity via time-reversal is a genuinely new labeler+mechanism; assume-before-derive asymmetry is a strong, clean AC signal that N1 must equalize.
- **error-correction-cascade** — Concrete-error detection is well-anchored and distinguishable from strategy-switch with the explicit guidance; self-exciting mechanism overlaps SPEC'd backtracking though the event class differs; local recomputation clustering plausibly yields ACF>0/Fano>1 beyond nulls.
- **hedge-to-assertion-drift** — Certainty ordinal has clear lexical cues though neutral-vs-stance is fuzzy; DC-drift in text-corpus is new but nearly duplicates the reasoning hedge card; self-flagged high composition risk means N1 likely explains persistence—triage-risky.
- **question-answer-adjacency** — Question detection is near-trivial and answer-after-question is context-resolvable reliably; directed Q→A adjacency is a fresh order-sensitive labeler/mechanism; forward>>reverse asymmetry is a strong clean AC signal that permutation destroys.
- **enumeration-cadence** — Ordinal/step markers are nearly regex-perfect to label; periodic text cell new but synthetic periodic benches exist; true fixed PERIOD in enumeration is doubtful—items are irregularly spaced by elaboration length, so spectral peak likely weak.
- **quotation-burst** — Quotation/attributed speech is highly reliable to label via punctuation and speech verbs; bursty text-corpus is a new cell mirroring reasoning backtracking mechanism; clustering plausible but flagged composition risk means it may fail to beat N1.
- **pronoun-referent-recurrence** — Main-subject named-entity vs pronoun requires syntactic judgment that Haiku will get inconsistently on complex sentences; long-memory coreference renewal is a distinct axis; heavy-tailed re-anchor gaps are plausible but may be geometric or composition-driven.
