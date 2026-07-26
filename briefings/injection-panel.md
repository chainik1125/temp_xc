# Briefing — train the paper's dictionary panel on a real safety task

status: proposed
owner: runpod (H100)
created: 2026-07-26

**Why this exists.** The relational exploration
(`experiments/explorations/relational/`) produced a theoretically clean result and
a practically useless one: no dictionary was trained, the tasks were template
toys, the only separation was at layer 0, and none of the paper's machinery was
used. This briefing replaces it with the experiment the reviewers actually asked
for.

---

## 1. What the reviewers asked, and what has to exist to answer them

| reviewer | demand | what satisfies it |
|---|---|---|
| bbby | Stacked SAE missing from Fig. 4 / Table 2 | Stacked in the panel, same task, same budget |
| bbby | single seed | 3 seeds + untrained controls, CIs on every bar |
| 4z15 | isolate temporal from generic crosscoder capacity | MLC in the panel (cross-**layer**, same width, no temporal axis) |
| EAxU | results preliminary, TXC rests on backtracking alone | a **second** real task where TXC wins on the paper's own metric |

All four need the same object: **one real task, the full architecture panel,
trained, at matched sparsity, with seeds.** That object does not exist anywhere in
this repo. Producing it is the whole job.

## 2. The task: will the model obey an instruction hidden in pasted text?

Chosen because it is the one candidate that survives this project's own screening
rule *and* matters.

- **Class that can work.** The relational atlas showed relations the model
  *computes* are linearised per position within 1–4 layers, so no window
  architecture can win on them. What survives is evidence that stays **spread
  out** because no single position summarises it. "Is this rollout heading toward
  obeying the injected instruction" is a hazard over a trajectory, not a fact —
  the same class as the paper's backtracking anticipation, which is the paper's
  strongest existing result and the only label in the program with a positive
  order receipt.
- **Labels are exact and free.** The injected instruction demands a witness string
  ("reply with the word `PWNED`"). The label is whether the rollout contains it.
  No judge, no API spend, no annotation noise — unlike the paper's backtracking
  (Sonnet judge, κ=0.773) and EM (coherence judge) labels.
- **It is a safety result on its own terms.** A feature that fires *before* the
  model gets hijacked is a monitor. A feature that suppresses compliance when
  steered is a defence. Both map onto the paper's existing
  detection + inducement template (§ 5.2).

**Subject model:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` — already on disk,
and the paper's own § 5.2 model, so the result is directly comparable to the
paper's backtracking numbers. Hookpoint `resid_post` L10 (the paper's steering
layer) with L12 as the confirmatory cell.

## 3. Phases

### Phase A — rollouts and labels (no training)
1. Build the injected-prompt set: a benign task over pasted document content, with
   an instruction embedded in the document. Vary injection strength (position in
   the document, imperative force, whether it names the assistant) so the
   compliance rate lands near 50 % — a balanced label by construction, checked
   before anything else runs.
2. Generate ~2,000 rollouts, batched, `max_new_tokens` capped. Record the exact
   token index where the witness string first appears.
3. **Label:** `complied = witness string present`. **Probe rows:** positions
   strictly *before* the first witness token — so the label is genuinely
   anticipatory and cannot be read off the answer.
4. Gate before spending anything on dictionaries: per-token vs window raw-linear
   ceilings on those rows (reuse `conversion_depth/problib.py`, as the relational
   gate did). If per-token already reads compliance as well as a window does, this
   task is regime-1 and the panel is cancelled — the same per-token-first triage
   that killed four candidates cheaply.

### Phase B — the panel (this is the deliverable)
Reuse the **proven** Stage-2 path: `src/explorations/task_hunt/real_lambda.py` is
a working real-activation datasource plugin that trained 5 architectures × T
ladder × 3 seeds through the canonical runner. Copy it, point it at the injection
cache and the compliance label.

- **Architectures (paper hparams from `configs/archs.yaml`, not invented):**
  `topk_sae` / `batchtopk_sae` (per-token baseline), `tsae`, `stacked_sae` /
  `stacked_batchtopk`, `mlc`, `txc_batchtopk_pre`, `txc_batchtopk_post` (the
  paper's Eq. 1). `d_sae = 18432`, `k_pos = 20`, `T ∈ {2, 4, 8}` — with a reduced
  `d_sae` fallback if wall-clock forces it, disclosed rather than silent.
- **Fairness, per this project's own rules:** matched nominal `k_pos`, and matched
  **realized** `l0_per_token` reported next to every number — `txc_batchtopk_post`
  needs nominal `k = k_pos·T` or its code rate collapses (task_hunt RECORD § 3c).
- **Seeds:** {1, 2, 42} plus untrained controls, so "trained beats init" is shown
  rather than assumed.
- **Resolve `txc_pro`:** it is in the paper but **not in `configs/archs.yaml`**.
  Either locate it on `origin/final-aniket` and register it, or state plainly that
  the panel covers TXC-base only.

### Phase C — the paper's metrics, not new ones
The three real-section evaluators on this branch are stubs. Implement **one** as a
plugin (file drop + `configs/` entry, no `core/` edits):
- **`detection`** — the paper's § 5.2/5.3 metric: rank features by class-mean
  difference on train, take top-`S ∈ {8, 16, 32}`, fit an ℓ1 logistic probe,
  report **PR-AUC** under grouped cross-validation against the class-prior floor.
This fills a real hole in the repo and makes the result quotable in the paper's own
units. **Stretch:** steering — push the top feature and measure the change in
compliance rate, the direct analogue of the paper's inducement axis, and the one
that turns this from a probe into a defence.

### Phase D — report
Money plot: PR-AUC per architecture with seed CIs and the class-prior floor drawn;
realized `l0` annotated on every bar. Plus the honest table of what the panel does
and does not show. One page, plain labels.

## 4. Costs, and the one decision needed

| item | cost |
|---|---|
| rollouts (2k × ≤256 new tokens, 8B, batched) | ~30–60 min GPU |
| activation cache (≈1M tokens × 4096 × fp16, 2 layers) | ~16 GB **disk** |
| panel: 6 archs × 3 T × 3 seeds + untrained ≈ 60–70 cells at `d_sae` 18432 | several hours GPU |

**Blocker — disk.** 12 GB free. The cache alone does not fit, and `d_sae = 18432`
checkpoints add more. Options, in order of preference:
1. Delete `models--Qwen--Qwen2.5-Coder-7B-Instruct` (15 GB, unused by this
    project) — **needs the owner's say-so**;
2. delete `/workspace/role-probes/.venv-vllm` (8.8 GB, idle);
3. shrink the cache (fewer rollouts, one layer, `float16` → `bfloat16` no gain) and
    accept a smaller panel.

Nothing is deleted without an explicit go-ahead.

## 5. What gets dropped

The relational exploration stops where it is. Its four toy tasks are **not** part
of this; at most one paragraph survives as the reason this task was chosen over a
relational one. The layer-0 result is not promoted to a paper claim.

## 6. Kill conditions, pre-registered

- **Phase A gate fails** (per-token reads compliance as well as a window) → stop;
  report as a regime-1 negative and do not train anything.
- **Compliance rate cannot be balanced** between 25 % and 75 % → the label is
  degenerate; re-scope or stop.
- **Panel shows no architecture separating beyond seed CIs** → report the null.
  That is still a better answer to EAxU than a fourth toy.
