# Card — candidate 5: agreement attraction (number-match XOR)

**Status: RETROSPECTIVE.** ⚠ This card was written *after* its pilot ran, so its
freeze order is **not git-provable**. The predictions below are quoted verbatim
from two artefacts that were timestamped *before* the corrected run: the plan
file (`P(win) 0.35 / P(violent) 0.30`, "expect 0.75 vs 0.92 rather than chance vs
0.9") and the published dashboard revision of 01:32 UTC. Treat the scoring in
§ 5 as honest-but-weakly-provenanced. The sibling card
[`contradiction_xor.md`](contradiction_xor.md) *was* frozen properly
(`74df8f7f`, before its first cell) and is the template to follow.

Agent `runpod`, 2026-07-25. Verdict and numbers: [`../RECORD.md`](../RECORD.md) § 2.

---

## 1. Label

`label = [number(head noun) == number(verb)]` — a genuine equality between two
positions. Balanced 2×2 over (head number) × (verb number), so both marginals are
flat and the mismatch cells are ungrammatical sentences.

> *The inspector noted that **the keys** beside the doors past the checkpoint
> **is** broken.*

Probe row at the verb token. 5,760 distinct items, 24 head-noun groups, split by
group. Head→verb distance 4–12 tokens (median 7), varied by inserting 1–3
modifier phrases whose own number is randomised independently of the label.

## 2. Why this candidate was on the list

- **Theorem-protected.** Equality with balanced marginals puts per-token SAE,
  T-SAE, Stacked, MLC and TXC-pre at a provable chance floor
  (`synthetic/changepoint/bench_record.md` § 3), leaving TXC-post the only
  readable family. That is the reviewers' isolation question answered as a
  measurement.
- **Most legible phenomenon available.** Subject–verb agreement across a
  distractor is the single most-studied syntactic dependency in LM
  interpretability, so a result needs no defending as a construct.
- **Per-token should have been actively misled.** Attraction errors exist because
  the distractor interferes; a per-token readout at the verb was expected to be
  degraded relative to a window containing the head.

## 3. Known risk, stated in advance

Subject number must be maintained to generate agreeing text, so conversion
pressure is high — this was rated the *second* weakest of the five BUILD
candidates for exactly that reason.

## 4. Predictions (as recorded pre-run)

- **P1.** `P(violent) = 0.30`; a *modest* win is likelier than a decisive one.
- **P2.** "Expect 0.75 vs 0.92 rather than chance vs 0.9" — per-token degraded by
  attraction, window materially better.
- **P3.** Layer 0 at chance for every arm (the label cannot be lexical).

## 5. Scoring

| prediction | outcome |
|---|---|
| P1 `P(violent)=0.30` | **FALSIFIED** — no window advantage at any depth ≥ 2 |
| P2 "0.75 vs 0.92" | **FALSIFIED in the more-converted direction**: 1.000 vs 1.000 |
| P3 layer 0 at chance | **CONFIRMED for the linear arms** (0.495 / 0.503) — but *not* for the nonlinear one (window MLP 0.749), which turned out to be the run's most useful finding |

P3's partial failure is the informative one: it produced the gate's **positive
control**, showing the instrument fires when regime-3 headroom exists. Recorded in
RECORD § 2b.

## 6. Kill rules (as applied)

- **K1** per-token within 0.02 of the best window arm at every layer → **fired**.
- **K2** `nonlinear_residual ≤ 3σ_null` in every cell → **fired** from layer 2 on.

**Verdict: KILL.** Agreement equality is converted within four layers.

## 7. Scope

One model (R1-Distill-Llama-8B), one label, English templates. This kills *this*
label on *this* model; it does not establish that no equality latent survives
conversion anywhere.
