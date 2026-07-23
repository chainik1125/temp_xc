---
status: active
created: 2026-07-23
for: runpod
venue: runpod
---

# TXC-pro dissection — which loss terms actually help a TXC?

**You are `runpod`** (32C). Parallel agents: `runpod-b` (story pack),
`runpod-c` (EM redo). Shared-branch + commit-citation rules apply.
Prime directive: **a sound verdict, never a win** — the expected outcome
per the team's own experience is "mostly nothing helps"; finding that
cleanly IS the deliverable. Results wanted within ~48 h.

**Context.** The paper's TXC-pro bundles a Matryoshka objective, a
multi-distance contrastive loss, and longer windows — confounded by the
paper's own admission (Limitations ii), and slated to be dropped. The
question: **does ANY component, alone, help the TXC backbone — and on
which regime?** The loss implementations already exist in-repo:
`src/temp_bench/archs/tsae.py` carries the ported matryoshka + AuxK +
contrastive loss machinery (Wasteland's
`TemporalMatryoshkaBatchTopKTrainerLite` lineage) — graft, don't
re-derive.

## Design

1. **Freeze the ablation card first** (commit pre-build,
   `experiments/explorations/synthetic/loss_dissection/CARD.md`):
   - **Variants** (plugin arch files + YAML only — hard rule 3, never
     touch `temp_bench/core/`): `txc_batchtopk_post` backbone ×
     {plain, +matryoshka, +contrastive, +both}. Post is the backbone
     because it is the panel's strongest regime-3 mixing arch; add the
     same 4 variants on `txc_batchtopk_pre` ONLY if the post grid
     finishes early. "Longer windows" is NOT a new variant — it is the
     existing T axis; report T-trends per variant.
   - **Benches** (the discriminating set, one per regime/subtype):
     `backtracking` (regime 2), `frequency` (regime 3 power),
     `phasepair` (regime 3 phase), `recipe_instruction_phase_runs`
     (regime 3 equality, grounded), `multilane` (superposition).
   - **Grid**: canonical slice per bench (d_sae = F, canonical T set,
     k_pos ∈ {1,2,4}, seeds {1,2,42} + untrained), 4 variants — size it
     to the pod; report the exact cell count. Canonical runner,
     leaderboard rows stamped.
   - **Frozen predictions (mac-local priors; sharpen, don't redirect):**
     (i) NO variant improves regime-3 latent recovery beyond seed noise
     (the "TXC-pro is useless" prior, now falsifiable per-component);
     (ii) matryoshka may improve reconstruction/capability metrics
     (NMSE/gAUC) without recovery gains; (iii) contrastive is the most
     plausible recovery helper IF anywhere, and most plausibly on the
     DC/persistent axes (it is T-SAE's signature loss and T-SAE's wins
     are on ambient-shaped tasks — the EM/HH-RLHF pattern);
     (iv) falsifier: a variant that HURTS recovery ≫ seed noise on
     regime 2 (backtracking) indicts the graft, not the loss — check
     the implementation before concluding.
2. **Contract tests** for each variant (loss reduces to plain at
   zero-weight; matryoshka group structure; contrastive pair sampling) —
   committed with the build, before any grid.
3. **Blind verdict vs the card**, per component × bench × metric
   (recovery AND capability), with a one-page component table — the
   deliverable format is "component X: helps/neutral/hurts on regime Y,
   effect size vs seed spread."
4. **Skeptic** (Fable) on any "component HELPS" claim — helps-claims are
   the winner's-curse surface here; persist raw pre-parse.

## Acceptance gate — stop for review

Card + variants + tests + grids + component table + verdict pushed;
STATUS rewritten; spend logged (skeptic only; ≤ $5). No reviewer text in
tracked files. Briefing stays until mac-local review.
