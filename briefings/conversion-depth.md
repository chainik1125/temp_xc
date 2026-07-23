---
status: active
created: 2026-07-23
for: runpod-c
venue: runpod (A40 GPU)
---

# Conversion depth — g(ℓ) curves + the substrate audit's empirical arms

**You are `runpod-c`** — a NEW A40 GPU pod; the user seeded
`/workspace/.agent_id` with `runpod-c`. Read `agents/README.md` (identity
+ shared-branch rules) and `agents/runpod-c/STATUS.md`. Two CPU agents
run in parallel (`runpod`, `runpod-b`) — their briefings are not yours.
Prime directive: **a sound verdict, never a win** — pre-register
predictions before computing curves.

**Governing docs:** `docs/ideas/conversion_depth.md` (the hypothesis +
ablation design — read fully) and `docs/substrate_audit_2026-07.md`
(items 2, 3, 5 are YOUR mandate). Work lives in a NEW exploration:
`experiments/explorations/conversion_depth/` (scripts + `RECORD.md` +
`figs/` + `results/`). **No leaderboard writes** — this session trains no
dictionaries; it probes raw activations (the § 8 ambience machinery
pointed at depth). No `temp_bench/core/` edits; reuse
`src/temp_bench/data/real_lm.py` (`build_activation_cache`) — check its
multi-layer support (`configs/data.yaml` header hints `layers: [...]`);
if single-layer only, loop it per layer rather than modifying core.
New datasource entries in `configs/data.yaml` are append-only.

**Session limits:** ~12 h wall · GPU is for *forward passes only* (pod:
1× H100, 16 vCPU, 251 GB RAM — run probe fits on the GPU, not the CPUs) ·
disk is a **700 GB persistent network volume** — budget: phase-3 caches ≈
72 GB/model at layer-stride 2 (≈ 144 GB for both), phase 4 ≈ 82 GB,
phase 5 ≈ 28 GB, HF weights ≈ 70 GB. **KEEP the phase-3 (Ward) caches on
the volume** — they are the input to the follow-up TXC-tracking session;
delete the phase-4/5 shards after their probe stats are written. Put the
HF cache on the volume (`HF_HOME=/workspace/hf`). Rewrite
`agents/runpod-c/STATUS.md` before any compact.

## Phase 0 — provenance pin (~30 min, blocking for interpretation)

From the read-only branch `origin/aniket-ward-stage-b` (+
`results/c7_backtracking/stage_a/` metadata): **who generated the Ward
math500 traces?** Record the answer in `RECORD.md` § 0 with the evidence.
If provenance cannot be pinned, say so — the base-vs-generator
comparison still runs, but its *interpretation* is conditional and the
record must carry that.

## Phase 1 — the machinery, validated on GPT-2 (CPU-cheap, ~1–2 h)

Build the g(ℓ) probe stack and validate it where the answer is known:
GPT-2 + the day-stride construction (port the pattern from
`origin/dmitry-spectral-sprint2` `gpt2_stride.py`/`bt_freq.py`). For each
layer ℓ: (a) per-token linear ceiling for the stride latent, (b) window
linear ceiling (fixed T per the § 8 convention), (c) MLP presence check.
Acceptance: reproduces the sprint § 4.7 result — non-ambient at hs=0
(per-token ≈ chance, window ≈ 1.0), converted by block 3 (per-token ≈
1.0), position-0 causal control at chance throughout. The stack is
frozen once this passes — no per-target retuning downstream (probe
budget scales with code dim; threshold-optimized ceilings per README).

## Phase 2 — pre-registration (commit before any 8B cache is probed)

Freeze in `RECORD.md` § 2, per target below: predicted g(ℓ) shape
(monotone-shrinking is the hypothesis), the paper-layer bet (is L10/L15
inside the gap?), and the base-vs-generator direction bet (mac-local
prior: the **generator carries the anticipation signal earlier and
stronger** than the base reader; if they coincide everywhere, the § 5.2
claim is reader-predictability, which the paper must then say). Include
a falsifier: g(ℓ) < 0 beyond noise at any ℓ (window probe BELOW
per-token) indicts the probe stack, not the model.

## Phase 3 — backtracking: base vs generator, across depth (the core, ~4–6 h)

Cache **Llama-3.1-8B base** (`NousResearch/Meta-Llama-3.1-8B`) and
**DeepSeek-R1-Distill-Llama-8B** on the IDENTICAL Ward math500 token
stream (the § 5.2 datasource's 4044 × 128 stream; same tokenizer family —
verify token-identity or record the delta), `resid_post`, layer stride 2
(0, 2, …, 30) + embeddings. For each model × layer: the three § 1
ceilings for the Ward anticipation labels. Deliverables:
- **g(ℓ) curves, both models**, with the L10 verdict (is the paper's
  layer near the gap's maximum, or past conversion?);
- **generator − reader gap per layer** (the substrate-audit item-2
  answer);
- EM-style honesty: if the anticipation ceiling is ≈ 0 everywhere for
  both, that is a finding about the labels, not a probe failure — check
  presence (MLP) before concluding.

## Phase 4 — EM depth-confound check (~2–3 h)

The § 5.3 shuffle_gap ≈ 0 verdict exists at ONE layer (L15/28). Locate
the actual medical-organism weights (the § 5.3 datasource; if only stock
`Qwen2.5-7B-Instruct` is resolvable from this repo, STOP this phase and
report — do not substitute the stock model silently). If available:
layer-stride sweep of the per-token vs window ceilings for the EM label.
Frozen prior: flat g(ℓ) ≈ 0 at all depths (ambient everywhere — persona
density leaks into every token). A nonzero early-layer gap would
overturn the EM negative's generality — report either way.

## Phase 5 (stretch) — the § 5.1 probing check

gemma-2-2b **base** vs **-it**, L13, same fineweb-edu stream: do the
sparse-probing per-token ceilings materially differ? One number per
model per probe task, no dictionary training. Substrate-audit item 3.

## Acceptance gate — stop for review

RECORD.md complete (provenance § 0, frozen § 2, curves + verdicts §§ 3–5
with figs), datasource entries committed, STATUS rewritten, pushed.
**Do NOT train dictionaries on the caches** — the TXC-tracking test
(does trained-TXC advantage track g(ℓ)?) is the NEXT session, designed
after review of these curves. Briefing stays until mac-local review.
