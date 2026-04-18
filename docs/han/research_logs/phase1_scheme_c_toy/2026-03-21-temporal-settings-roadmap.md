---
author: Claude
date: 2026-03-21
tags:
  - design
---

## Temporal data generation settings for TFA evaluation

Our current toy model uses independent Markov chains per feature — each feature's activation follows a two-state chain with parameters (π, ρ). This captures per-feature temporal persistence but misses several aspects of temporal structure in real language. Below are four settings that would provide a more complete evaluation, ordered by implementation feasibility.

### Setting 1: Correlated feature groups (events)

**What it captures.** In real language, features co-activate in groups — a "cooking scene" activates kitchen, ingredients, utensils features simultaneously, and they all persist or vanish together. This is the TFA paper's primary qualitative claim: that predictive codes capture slow-moving event structure.

**Implementation.** Partition features into event groups. Each group shares a single Markov chain that controls the activation of all member features simultaneously. When the group's chain is ON, all features in the group are active; when OFF, all are inactive. Optionally allow overlapping groups (a feature belonging to two events) to create cross-event correlations.

**What it tests.** Whether TFA's attention can exploit the redundancy within events — if it sees one feature from a group active, it should predict the others. This is a stronger signal than per-feature persistence.

**Status.** Partially implemented in `run_event_shuffle_diagnostic.py`. Needs to be run through the unified framework with full TopK + L1 sweeps and AUC analysis.

### Setting 2: Bursty / semi-Markov features

**What it captures.** In real text, features have "bursty" activation patterns — a topic appears, stays active for a variable duration, then vanishes and may never return within the context window. Our Markov chain has geometric holding times (memoryless), but real features have heavier-tailed durations (e.g., a paragraph-length topic).

**Implementation.** Replace the two-state Markov chain with a semi-Markov model: when a feature turns ON, sample a run length from a distribution (e.g., power law, log-normal, or fixed duration) rather than flipping a coin at each step. The OFF duration could similarly be non-geometric.

**What it tests.** Whether TFA benefits from variable-duration features. With geometric holding times, the optimal prediction strategy is the same regardless of how long a feature has been ON. With heavy-tailed durations, a feature that has been ON for 10 steps is more likely to continue than one that just turned ON (positive aging), which TFA could exploit if it can detect run length.

**Status.** Not implemented. Moderate effort — requires a new data generator but no changes to the model or evaluation code.

### Setting 3: Hierarchical temporal structure

**What it captures.** Language has nested timescales — words within sentences within paragraphs within topics. The TFA paper's event boundary analysis (Figure 5) shows predictive codes organizing into hierarchical block structures aligned with sub-event boundaries.

**Implementation.** A two-level generative model: slow "topic" features (low transition rate, long persistence) gate the activation of fast "content" features (high transition rate, conditioned on the current topic). For example, 5 topic features (ρ=0.95) each associated with 4 content features (ρ=0.3, but only active when their parent topic is active).

**What it tests.** Whether TFA's predictive codes capture the slow component (topic) while novel codes capture the fast component (content), as the paper claims. The hierarchy creates a natural pred/novel decomposition that TFA should recover.

**Status.** Not implemented. Requires a new hierarchical data generator and potentially different analysis (measuring pred/novel alignment with topic vs content features separately).

### Setting 4: Cross-feature causal dependencies

**What it captures.** In real language, features interact causally — a "question" feature activating makes an "answer" feature more likely at the next position. Features don't just persist independently; they trigger transitions in other features.

**Implementation.** A coupled Markov chain or small causal model where feature i's transition probabilities depend on the state of other features at the previous timestep. This could be specified via a transition tensor or a small neural network.

**What it tests.** Whether TFA can learn cross-feature temporal predictions — predicting feature j from feature i's history. This is fundamentally different from our current setup where each feature's future depends only on its own past.

**Status.** Not implemented. Hardest to implement and analyze — the space of possible causal structures is large, and the analysis would need to distinguish TFA's cross-feature predictions from within-feature persistence.

### Priority

Settings 1 and 2 are the most feasible and informative next steps. Setting 1 (event groups) is partially implemented and directly tests the TFA paper's primary claim. Setting 2 (bursty features) would test whether the memoryless assumption in our current setup is limiting.
