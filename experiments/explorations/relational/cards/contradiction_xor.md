# Card — candidate 4: contradiction / fact-consistency XOR

**Status: FROZEN.** Committed before any contradiction cell exists. Git order is
the evidence. Prime directive: a sound verdict, never a win.

Agent `runpod`, 2026-07-25. Exploration `experiments/explorations/relational/`.

---

## 1. The claim being tested

A **balanced-marginal equality label** on real activations is the only latent
class that can separate window architectures from each other, because the
additive-code theorem (`synthetic/changepoint/bench_record.md` § 3) puts
per-token SAE, T-SAE, Stacked, MLC and TXC-pre at a provable chance floor while
leaving TXC-post — the paper's Eq. 1, whose nonlinearity crosses positions — the
only family able to read it.

Candidate 4 asks whether a real LM leaves such a latent *unconverted*.

## 2. Stimuli (frozen; `stimuli.py`, `gen_contradiction`)

Two mentions of the same attribute, separated by 1–3 filler sentences:

> `Note: the meeting was scheduled for Tuesday. <fillers> Confirming: the meeting was scheduled for Thursday`

Balanced 2×2 over (value at mention 1) × (value at mention 2) from a frozen
20-entry fact table, so `label = [value1 == value2]` has flat marginals at both
positions. 4,800 items, **all texts distinct**, 20 lexical groups.

**Probe row.** One row per item, at the token holding the last character of
mention 2's value. Splits are **by lexical group** (16 train / 4 test groups), so
no fact appears in both halves.

**Label-side triage — already PASSED** (`labels/contradiction_stimuli.json`):
AUC from value 1 = 0.500, from value 2 = 0.500, from length = 0.505, from the
inter-mention gap = 0.506, cells exactly equal, zero duplicate texts.

## 3. Arms (identical rows in every cell)

| arm | what it bounds |
|---|---|
| `per_token` linear on `h[p]` | the per-token SAE's ceiling |
| `window_flat` linear on the flattened window | **the additive ceiling** — a linear readout of a per-position decomposition, so it bounds T-SAE, Stacked, TXC-pre and any *pooled* per-token code |
| `window_mean` linear | order-free pooling |
| `window_shuf` linear | within-window order destroyed |
| `per_token_mlp` / `window_mlp` | the **nonlinear** ceilings — presence, so blindness ≠ absence |

Headline statistic: **`nonlinear_residual = window_mlp − max(window_flat, per_token)`**
— the regime-3 headroom that only a position-mixing code can convert into a
linear readout. `g = window_flat − per_token` is reported but is *not* the
regime-3 test, because a linear probe on a flatten is itself additive.

Grid: layers {0, 8, 16, 24} of DeepSeek-R1-Distill-Llama-8B (the paper's § 5.2
subject model) × T ∈ {8, 16, 32, 64} × strata {all, in, out}.

## 4. Controls

1. **Permutation null** (4 label shuffles per cell) → σ_null; every gap is read
   against 3σ_null.
2. **Bootstrap CI** (1,000 resamples) on every AUC.
3. **IN/OUT stratification** — mention 1 is at a known token distance, so the
   same label at the same probe position is scored separately for rows whose
   window reaches mention 1 and rows whose window does not. This is the causal
   control: an advantage present in the OUT stratum cannot be cross-position
   binding. It replaces the shuffle-gap-grows-with-T confound that bit task_hunt
   candidate 2 (`task_hunt/RECORD.md` § 2).
4. **Layer 0 (embeddings)** — separates a lexical signal (readable at hs0) from
   one the model *builds* across depth.

## 5. Predictions (frozen, with reasons)

- **P1.** Per-token at layer 0 sits at chance (≤ 0.55). The label is a relation
  between two positions and cannot be lexical: both values appear in both
  classes by construction.
- **P2.** Per-token at mid-depth rises **materially above chance** — my central
  estimate is 0.70–0.95. Reason: the second mention's value is a high-surprisal
  event when it contradicts, and surprise/prediction-error is linearly
  represented in LMs. This is the same mechanism that just converted candidate 5.
- **P3.** `nonlinear_residual` stays **below 3σ_null at every layer and T.**
  Reason: if the model computes the mismatch at all, it deposits the result at
  the current position, and then the additive ceiling already contains it. This
  is the round-1 conversion prior applied honestly to my own preferred candidate.
- **P4.** The IN and OUT strata differ by less than 0.05 on `g`. Reason: if P2
  holds, the signal lives at the probe position, so reaching mention 1 adds
  nothing.
- **P5.** `window_shuf ≈ window_flat` (|Δ| < 0.02): no within-window order
  effect, because the label is an unordered match.

**I expect this candidate to be KILLED.** P1–P5 together describe a converted
latent. The reason to run it anyway is that the alternative outcome — a real
nonlinear residual — is the single most valuable result available to this paper,
and the cost is minutes on a cache that already exists.

## 6. Kill rules (pre-registered; the card's text governs)

**KILL** if either fires:
- **K1** per-token is within 0.02 AUC of the best window arm at every layer, or
- **K2** `nonlinear_residual ≤ 3σ_null` at every (layer, T) cell.

**KEEP** requires `nonlinear_residual > 3σ_null` in at least two cells at the
same layer, with the effect **larger in the IN stratum than the OUT stratum**
(otherwise it is not binding, and the KEEP is withdrawn).

## 7. Falsifier for the design itself

If `window_flat` reads the label materially above `per_token` while
`nonlinear_residual ≈ 0` **and** the OUT stratum matches the IN stratum, the
stimuli leak through some route the triage did not measure. In that case the
finding is a stimulus defect, not a result: audit and re-freeze, do not
reinterpret.

## 8. Scope

One model, one hookpoint family, English templates. A kill here is scoped to
*this* label on *this* model — it does not establish that no equality latent
survives conversion anywhere, and the record must say so.
