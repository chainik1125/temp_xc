---
author: Han
date: 2026-04-30
tags:
  - guide
  - design
  - reference
---

## How a TXC is used to do steering — step-by-step mechanics

> Reference document for the Phase 7 Hail Mary case-study pipeline as
> used by Agents Y and W on Gemma-2-2b base, k_pos=20 cells.
> Captures the full chain from training a TXC to producing a graded
> steering matrix entry. All file paths are relative to the repo root
> `/workspace/temp_xc/`. Companion to the findings writeups
> `agent_w/2026-04-30-w-final-summary.md` and
> `agent_y_phase2/2026-04-30-y-final-summary.md`.

### Overview

A TXC (Temporal Crosscoder) is a sparse-autoencoder variant that
ingests a length-T window of residual-stream activations and produces
one window-level latent vector `z ∈ R^{d_sae}` (BatchTopK or TopK
constrained). Steering uses the picked latent's decoder direction to
intervene on the residual stream at inference time, modifying which
concepts the language model emits.

The pipeline has six stages: train → pick a feature per concept →
diagnose feature magnitudes → intervene at varying strengths → grade
the generations → compute the headline metric. Every stage is
implemented in `experiments/phase7_unification/case_studies/` with
files I'll cite per step.

### Step 1 — Train the TXC

**Code**: `experiments/phase7_unification/case_studies/train_kpos20_txc.py`
(W's bare-antidead variant) or `experiments/phase7_unification/case_studies/
train_kpos20_matry.py` (W's matryoshka multiscale variant) or
`experiments/phase7_unification/case_studies/train_kpos20_hailmary.py`
(Y's wrapper around `train_phase7.train_txc_bare_antidead`).

**Inputs**:
- Subject model: `google/gemma-2-2b` (base; not -it).
- Activation cache at L12 residual: `data/cached_activations/gemma-2-2b/fineweb/resid_L12.npy` (shape `(24000, 128, 2304)` fp16, 14.16 GB). Built once via `build_act_cache_phase7.py --layer 12` after `build_token_ids.py`.

**TXC architecture** (`src/architectures/txc_bare_antidead.py`):
```
W_enc:  (T, d_in,  d_sae)        per-position encoder weight
b_enc:  (d_sae,)                  shared bias
W_dec:  (d_sae,  T, d_in)         per-position decoder weight (unit-norm per feature)
b_dec:  (T, d_in)                 per-position bias
```
- d_in = 2304 (Gemma residual width).
- d_sae = 18432 (SAE dictionary size, 8× expansion).
- T = window length (T=2, 3, 5 in Phase 7 W; canonical Phase 5 used T=5).
- k_pos = per-position TopK budget (20 for matched-sparsity cells; 100 canonical).
- k_win = T · k_pos (window-level TopK budget; what the encoder actually applies).

**Encode**:
```
z = einsum("btd,tds->bs", x, W_enc) + b_enc       # (B, T, d_in) → (B, d_sae)
z = TopK(z, k=k_win)                              # zero out all but top-k_win
```

**Decode**:
```
x_hat = einsum("bs,std->btd", z, W_dec) + b_dec   # (B, d_sae) → (B, T, d_in)
```

**Loss**: per-window L2 reconstruction `||x - x_hat||²` averaged over batch + positions, plus the anti-dead "AuxK" term that penalizes feature death.

**Training config** (canonical `paper_archs.json::training_constants`):
- batch=4096, lr=3e-4, max_steps=25_000, grad_clip=1.0,
- plateau_threshold=0.02 (early-stop when 5-window loss-improvement-fraction drops below this), min_steps=3000,
- preload 24k FineWeb-Edu sequences (14.16 GB on GPU),
- decoder normalised to unit-norm per feature each step,
- gradient parallel to decoder removed each backward.

**Wall time** (A40 RunPod, 46 GB VRAM):
- bare-antidead T=3, k_pos=20: 33 min, plateau-converged step 4600
- bare-antidead T=5, k_pos=20: 46 min, plateau-converged step 3800
- matryoshka multiscale T=5: 95 min, plateau-converged step 3200 (5× per-step due to multi-scale InfoNCE + matryoshka H/L head)

**Output**:
- `experiments/phase7_unification/results/ckpts/<arch_id>__seed<S>.pt` (state dict, ~1 GB for bare T=3, ~2.7 GB for matryoshka).
- `experiments/phase7_unification/results/training_logs/<arch_id>__seed<S>.json` (meta + loss/l0 telemetry).

The meta JSON is the key for downstream loaders — it carries `src_class`, `T`, `k_pos`, `k_win`, `hook_name`, anchor layer, and is read by every case-study script.

### Step 2 — Pick a steering feature per concept

**Code**: `experiments/phase7_unification/case_studies/steering/select_features.py`.

**Inputs**: a 30-concept benchmark fixture (`concepts.py`) where each concept has 5 example sentences (e.g., "medical": clinical-style sentences; "poetic": lyrical sentences; "harmful_content": flagged content; etc.) and a small set of contrastive baseline sentences.

**Algorithm** (per concept):
1. Forward the 5 example sentences through Gemma to L12, take the residual at every content position.
2. Encode each position with the SAE.
3. For each feature `j ∈ [0, d_sae)`, compute the **lift** — how much higher is the average activation on the concept sentences vs the baseline — restricted to "content" positions (excluding leading specials and pad-equivalents).
4. Pick the feature with maximum lift, subject to having a meaningful absolute activation (not just a noise-flicker).

**Output**: `results/case_studies/steering/<arch_id>/feature_selection.json` — a dict `{concept_id: {best_idx, best_act_train, best_lift_train, top_5}}`. One picked feature per concept × 30 concepts = 30 features per arch. Some features are picked for multiple concepts ("polysemanticity"); Y observed 24/30 distinct features at T=5/k_pos=20 vs 28/30 at T-SAE k=20.

The picked feature's **decoder direction** `W_dec[picked_idx, t, :]` is the unit-norm vector that intervention will activate.

### Step 3 — Diagnose feature magnitudes (the `⟨|z|⟩` per arch)

**Code**: `experiments/phase7_unification/case_studies/steering/diagnose_z_magnitudes.py`.

**Why we need this**: paper-clamp's strength schedule clamps the latent's activation to a fixed absolute number (e.g. 100). But "100" means very different things to a per-token SAE (`⟨|z|⟩ ≈ 10` so 100 = 10× typical) vs a window TXC at k_pos=100 (`⟨|z|⟩ ≈ 30` so 100 ≈ 3× typical) vs a MLC layer-fusion arch (`⟨|z|⟩ ≈ 159`, so 100 is *under* typical). Without normalisation, the strength schedule has implicit per-token bias.

**Procedure**:
1. Forward 30 concepts × 5 sentences = 150 sentences through Gemma at L12.
2. For each (sentence, content position), encode with the SAE and read `z[picked_feature]` for the corresponding concept's picked feature.
3. Pool over (concept, position) and compute statistics: `<|z|>` (mean of absolutes), p90, p99, abs_max, also per-concept distributions.

**Output**: `results/case_studies/diagnostics_kpos20/z_orig_magnitudes.json`, keyed by `arch_id`. Each entry has a `pooled` block (overall) and a `per_concept` block.

**Multi-agent caveat**: the script writes the file via `write_text(json.dumps(summaries))` — it overwrites prior contents. Y's pipeline and W's pipeline both write to the same path, so each agent's run clobbered the other's entry until we patched `run_perposition.sh` to verify the entry exists before invoking intervene, plus manual merging post-clobber. See `feedback_zmag_clobber.md` memory note.

### Step 4 — Intervene during generation (paper-clamp at family-normalised strengths)

This is the steering action itself.

**Code**:
- Right-edge protocol: `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_normalised.py`.
- Per-position protocol (Q2.C): `experiments/phase7_unification/case_studies/steering/intervene_paper_clamp_window_perposition.py`.

**Strength schedule**: 7 normalised values `s_norm ∈ {0.5, 1, 2, 5, 10, 20, 50}` (log-spaced over 2 decades). Per arch, the absolute strength applied is:

```
s_abs = round(s_norm × <|z|>_arch, 1)
```

So at `s_norm=10` and `<|z|>_arch = 25.2` (cell A `phase57_partB_h8_bare_multidistance_t5`), `s_abs ≈ 252`.

For each `(concept, s_norm)` pair, generate one 60-token completion from the prompt `"We find"` (paper's neutral prompt). With 30 concepts × 7 strengths = 210 generations per arch.

#### Right-edge intervention loop (per generation step)

```text
prompt = "We find" + tokens_so_far
forward through Gemma layers 0..11
at L12 hook:
  for each position p in the prefix where p ≥ T-1:
    window = residual[p-T+1 : p+1, :]                # (T, d_in)
    z = encoder(window) + b_enc                      # (d_sae,)
    z = TopK(z, k=k_win)                             # only for sanity check; encoder already does this
    z_clamped = z.clone()
    z_clamped[picked_feat[concept]] = s_abs          # set picked feature to absolute strength
    x_hat_orig = decoder(z) + b_dec                  # (T, d_in)  — the SAE's reconstruction
    x_hat_new  = decoder(z_clamped) + b_dec          # (T, d_in)
    delta = x_hat_new[T-1, :] - x_hat_orig[T-1, :]   # right-edge only — single d_in vector
    residual[p, :] += delta                          # write the change back at position p
forward through Gemma layers 13..25 with the modified residual
sample next token (greedy in the implementation; could be top-p)
```

The crucial design choice: at the encoder, one window produces one latent (window-level encode); at the decoder, the latent produces a T-position reconstruction; at the residual write-back, **only the right-edge (t = T-1) reconstruction is written**. The earlier T-1 positions of the decoder output are discarded under right-edge.

The reason for *clamping* (rather than additive steering): clamping to `s_abs` ensures the latent fires at a known-strong activation regardless of the prompt's natural tendency. If the prompt is concept-relevant, `s_abs > z_natural` pushes it harder; if irrelevant, `s_abs > 0` activates it from scratch. The "delta" written to the residual is the difference between the SAE's reconstruction with the latent clamped vs the SAE's reconstruction with the latent at its natural value — this isolates the latent's contribution rather than just adding noise.

#### Per-position intervention (Q2.C protocol)

Same encode + clamp logic, but the write-back uses *all T positions* of the decoder output instead of just the right edge. Stride-1 windows overlap, so each token position p in the prompt is covered by up to T windows (those ending at p, p+1, ..., p+T-1). Each contributing window writes its decoded delta at the position-within-window, and the final delta at position p is the *average* of those writes:

```text
for each position p:
  contributing_windows = {windows w | p in w}
  delta[p, :] = mean over w of (decode(z_clamped_w) - decode(z_w))[p_within_w, :]
residual[p, :] += delta[p, :]
```

Why this can help: at sparse k_pos, the picked feature is partly polysemantic — it fires for the right concept at some positions and noisily at others within the window. Right-edge takes a single noisy snapshot; per-position averages over T snapshots, reducing noise. Y observed σ_seeds drops 2–5× under per-position (0.33 → 0.10 at T=5).

### Step 5 — Grade with Sonnet 4.6

**Code**: `experiments/phase7_unification/case_studies/steering/grade_with_sonnet.py`.

For each (concept, s_abs, generation) tuple, send the generation to Claude Sonnet 4.6 with two prompts:

1. **Coherence** (0–3): "Rate how coherent and grammatical this English text is. 0 = incoherent gibberish, 3 = perfectly fluent."
2. **Steering success** (0–3): "Rate how strongly this output exhibits the [concept] pattern. 0 = unrelated, 3 = strongly on-pattern."

Concept name + a brief description are also passed to the grader (so the grader knows what "medical" means etc.).

**Rate-limited**: Anthropic's 50 req/min ceiling is shared across agents. We use `--n-workers 1` to stay well under that with two agents grading concurrently.

**Output**: `results/case_studies/<subdir>/<arch_id>/grades.jsonl` where `<subdir>` is one of:
- `steering_paper_normalised/` (right-edge, family-normalised)
- `steering_paper_window_perposition/` (per-position, family-normalised)
- `steering_paper_normalised_seed1/` (right-edge, seed=1, etc.)

Each line: `{concept_id, strength, success_grade, coherence_grade}` — 210 rows per arch per protocol per seed.

### Step 6 — Compute the metric

For each (arch, protocol, seed) tuple:

1. **Group grades by `s_norm`** — 30 grades per s_norm.
2. **Per-strength means**: `mean_success(s_norm) = avg over 30 concepts`, `mean_coh(s_norm) = avg over 30 concepts`.
3. **Unconstrained peak (METRIC A)**: `max over s_norm of mean_success(s_norm)` — the headline number Y reported at single-seed.
4. **Constrained peak (METRIC B)**: `max over s_norm of mean_success(s_norm) where mean_coh(s_norm) ≥ 1.5` — the brief's locked primary metric. Captures peak on-pattern success at strengths that still produce coherent text.

The 1.5 coherence threshold is a "still-readable English" cutoff. Below it, the steered output is gibberish that happens to contain concept-related tokens — not actually useful steering.

### Why the constrained metric matters (and why it's fragile)

Without the coherence constraint, T-SAE k=20's headline is **1.80** at s_norm=10 — a strong steering peak. But at s_norm=10, T-SAE k=20's coherence has dropped to **1.40** (just below 1.5). The output is on-pattern but barely English.

With the constraint, T-SAE k=20's peak kicks back to s_norm=5 where coherence is acceptable. At seed=42, s_norm=5 gives `mean_coh = 1.667` (above threshold) and `mean_success = 1.10`. Constrained peak = **1.10**.

But at seed=1, s_norm=5 happens to give `mean_coh = 1.40` (just below threshold). The peak kicks back further to s_norm=2 where success is only 0.30. **σ_anchor between seeds is 0.80.** This is W's anchor-σ discovery — the constraint's reliance on a sharp threshold makes the metric fragile to seed-noise that pushes the coh-cliff position around.

Implication: the brief's pre-registered ±0.27 threshold for win/tie/loss calls is INSIDE the anchor's own seed-noise. Multi-seed pooling is required for honest comparisons. Under multi-seed-pooled anchor (0.70), the W matrix shifts: T=2 cells WIN by ≥0.27, T=3/T=5/matryoshka cells TIE.

### How the matched-sparsity argument runs

The brief's question: "does a TXC at the same per-token sparsity as T-SAE k=20 (k_pos=20) match its steering performance?" The motivation: previous Y showed that at canonical k_pos=100, TXC trails T-SAE k=20 by 0.96 in unconstrained peak. The decomposition: ~0.10 of that gap is magnitude-scale bias (Q1.3 corrects), ~0.53 is sparsity (k=20 vs k=100), ~0.20 is residual architecture. If we also match sparsity, the ~0.73 of attributable-to-non-architecture gap closes; the remaining 0.20 architectural difference might be inside seed-noise.

W's contribution to that question: trained T=3 (cell C) bare-antidead and T=5 (cell E) matryoshka multiscale, both at k_pos=20. Y trained T=2 and T=5 bare-antidead. Together with T-SAE k=20 anchor, that's 5 archs × 2 protocols × 2 seeds (where multi-seed is run) = 18 graded matrix entries.

### Findings summary table

| arch (matched-sparsity, k_pos=20) | T | protocol | mean (multi-seed where available) | Δ vs anchor 0.70 | call |
|---|---|---|---|---|---|
| T-SAE k=20 (anchor) | 1 | per-token | 0.70 (σ=0.80) | — | — |
| TXC bare-antidead T=2 (Y) | 2 | right-edge | 1.067 | +0.367 | WIN |
| TXC bare-antidead T=2 (Y) | 2 | per-position | **1.117** | **+0.417** | **WIN** ⭐ |
| TXC bare-antidead T=3 (W cell C) | 3 | right-edge | 0.783 | +0.083 | TIE |
| TXC bare-antidead T=3 (W cell C) | 3 | per-position | 0.783 | +0.083 | TIE |
| TXC bare-antidead T=5 (Y) | 5 | right-edge | 0.867 | +0.167 | TIE |
| TXC bare-antidead T=5 (Y) | 5 | per-position | 0.783 | +0.083 | TIE |
| TXC matryoshka T=5 (W cell E) | 5 | right-edge (single seed) | 0.633 | −0.067 | TIE |
| TXC matryoshka T=5 (W cell E) | 5 | per-position (single seed) | **0.933** | **+0.233** | TIE (close to win) |

**No cell loses.** T=2 cells decisively win. The matched-sparsity argument lands.

### Five paper findings (priority order)

1. **Anchor instability is the methodological lead.** σ_anchor=0.80 dominates. Multi-seed pooling required.
2. **Matched-sparsity TXC ties or beats T-SAE k=20.** None lose; T=2 wins by ≥0.27.
3. **T-axis advantage reverses at sparse k_pos.** Smaller T helps; opposite of canonical k_pos=100 where T=5–8 was the sweet spot.
4. **Matryoshka × per-position synergy.** Matryoshka multiscale loses under right-edge but wins under per-position at T=5 — protocol-conditional family advantage.
5. **Per-class shift.** TXC wins on sentiment (uniformly +1.0 under per-position) and stylistic (T=3 per-position +0.80). Knowledge stays T-SAE-favourable. The TXC-favourable concept class shifts when sparsity tightens.

### File index

- Trainers: `case_studies/{train_kpos20_txc.py, train_kpos20_matry.py, train_kpos20_hailmary.py}`.
- Pipeline launchers: `case_studies/steering/{run_kpos20_pipeline.sh, run_w_phase1_cell.sh, run_perposition.sh}`.
- Pipeline scripts: `case_studies/steering/{select_features.py, diagnose_z_magnitudes.py, intervene_paper_clamp_normalised.py, intervene_paper_clamp_window_perposition.py, grade_with_sonnet.py, compare_kpos20_vs_tsae.py}`.
- Trained ckpts: `results/ckpts/{tsae_paper_k20, txc_bare_antidead_t2_kpos20, txc_bare_antidead_t3_kpos20, txc_bare_antidead_t5_kpos20, agentic_txc_02_kpos20}__seed{42,1}.pt` (where each seed exists).
- Grades: `results/case_studies/steering_paper_{normalised,window_perposition}{,_seed1}/<arch>/grades.jsonl` — 210 rows each.
- Headline plot: `results/case_studies/plots/all_matched_sparsity_kpos20.png`.
- Findings writeups: `agent_w/2026-04-30-w-final-summary.md`, `agent_w/2026-04-29-w-phase1-sweep.md`, `agent_y_phase2/2026-04-30-y-final-summary.md`.
