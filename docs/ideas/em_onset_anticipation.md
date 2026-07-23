# EM onset anticipation — the right temporal task for the EM setting

**Idea (2026-07-23, Han, post-depth-ablation).** The paper's § 5.3 detection
target — one aligned/misaligned bit per rollout — is a *global* label, and
the phase-4 decomposition shows its window advantage is mostly order-free
pooling (`g_agg`): the ambience-unfriendly case, designed before the
ambient/non-ambient distinction existed. TXC underperforming there is
unsurprising in hindsight. The temporally-honest EM task is **onset
anticipation**: probe the token sequence *before* the CoT starts behaving
misaligned and predict the onset downstream — the exact analogue of
backtracking anticipation (the paper's TXC success case), and the
safety-relevant formulation (fire before the content exists, don't
recognize it afterwards). The mid-depth position-sensitive slice
(`conversion_depth/RECORD.md` § 4: g_order +0.108 at L13) is circumstantial
evidence that precursor structure exists.

## Design (sketch — needs a card/briefing before execution)

1. **Cohort.** Organism-native rollouts only for the primary claim: in
   α-steered traces the cause is present from t=0 (the steering vector),
   so "onset prediction" risks collapsing into ambient
   steering-strength readout. The existing 1,728-rollout cohort is
   α-balanced (~190 native); generate fresh α=0 rollouts on runpod-c
   (H100) as needed; stratify/report by α if steered traces are used at
   all.
2. **Onset labels.** LLM judge marks the first misaligned sentence/span
   per misaligned rollout. Full PhenomenonBench measurement discipline:
   prereg the labeler, inter-judge κ on a subsample, noise-robustness
   check. This is the main new labeling spend.
3. **Event-anchored recipe** (the backtracking D+ pattern): positives =
   windows at frozen offsets before onset (e.g. D+ = [−13, −8]);
   negatives = matched positions ≥ far from any onset + aligned-rollout
   positions; grouped CV by prompt; per-window shuffle + permutation
   nulls.
4. **Ambience gate FIRST** (§ 8 discipline pointed at the new label):
   raw per-token vs window linear ceilings at the D+ offsets,
   layer-swept — g(ℓ) of the *onset* label, with the g_agg/g_order
   decomposition. Per-token ≈ window everywhere ⇒ the precursor is
   lexical ⇒ STOP with a sound verdict (per-token SAEs suffice); a real
   g_order ⇒ proceed.
5. **Head-to-head at the map-chosen layer** (only after the gate):
   panel incl. TXC variants vs per-token SAE / T-SAE, frozen
   predictions, BOTH readout currencies — raw-probe AUC *and* the
   paper's Wang-selected sparse-probe PR-AUC — so a win (or loss) lands
   in the paper's own metric.
6. **Cache note.** The phase-4 cache truncates at 100 assistant tokens;
   onset work needs a full-span re-cache (runpod-c volume has room).

## Relation to the program

Completes the EM arc: paper's static-detection loss (readout
ambience-unfriendly) → depth ablation (raw headroom exists, internal
ambience gloss falsified) → the *correctly-shaped* task. Natural to pair
with the TXC-tracking session (same pod, same caches for the gate stage;
onset adds labeling + possibly generation). Status: **idea — no card, no
briefing; user proposal recorded.** Prime directive applies: the ambience
gate can kill it honestly at step 4.
