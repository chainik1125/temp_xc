---
status: active
created: 2026-07-24
for: runpod
venue: runpod (32C CPU)
---

# Corpus scale-up campaign — panel-grade data for the screen KEEPs (overnight)

**You are `runpod`.** Round-2 factory REVIEWED & APPROVED. This is an
overnight (10+ h) CPU campaign; **results by Saturday morning PT.**
Motivation: the hunt's first screen KEEPs sit on corpora too thin for
panel-grade receipts — runpod-e's punctint-list within-document
control rests on **8 documents**, the pinned fineweb sample is 400
docs, refmark is 400 conversations. If punctint-q (KEEP) or refmark
graduates to Stage-2 on Saturday, the panel should train and control
on 10× the data, and the doc-identity threshold question needs a
distribution, not a point estimate.

## Hard rules for this campaign

- **Never touch the pinned originals** (`fineweb_sample.json`,
  committed corpus artifacts, shipped npz/stats). Every scale-up is a
  NEW versioned artifact (`*_4k`, `*_2k` naming) beside them.
- **Label logic stays FROZEN** — reuse the committed libs
  (`punctint_lib`, `refmark_lib`, `novelty_lib` helpers) unchanged;
  any code change needs its own pre-run commit with stated reason.
- Same pull recipes, same filters, extended stream prefixes, seeded;
  deterministic. Incremental commits: corpus → labels → stats per
  item, one LOG line each.
- **A frozen bar firing at scale is a FINDING, not an embarrassment**:
  disclose it; it binds the Stage-2 design; it does NOT retro-kill
  the shipped small-corpus bundle (different artifact, same logic).

## 1. fineweb 400 → 4,000 docs (punctint faces — the KEEP first)

Extend the fineweb pull (same filters, longer stream scan) to 4,000
docs; rebuild BOTH punctint faces ×3 tokenizers at scale: labels,
position-matched manifests (scale rows/class up to ~100k if the data
supports it — say what it supports), triage on the frozen 0.65/0.65
bars, `doc_mean_only_auc` on both reporting faces. **Bootstrap
receipts are the point:** doc-level bootstrap (≥ 1,000 reps) CIs on
every triage AUC and on doc_mean_only_auc, per tokenizer per face —
the threshold-pinning review consumes this. Also report: how many
documents carry the within-document contrast for each face at scale
(the "8 documents" number, fixed or not).

## 2. refmark 400 → 2,000+ conversations

Same recipe (WildChat pinned revision, same filters, longer prefix).
Same deliverables as item 1, plus: the user-echo exposure at scale
(my review measured 0.22 % of manifest rows on the 400-conv build —
recompute; ship an explicit `is_user_echo` mask array in the scaled
npz so screens can drop those rows trivially), and the recurrence
stats (frac convs ≥ 2 markers) at scale.

## 3. If the night allows: novelty-family bootstrap only

No new corpus — novelty screened NEGATIVE. But its 400-doc triage
numbers feed the same threshold dataset: doc-level bootstrap CIs on
the committed novelty triage AUCs + doc_mean_only_auc equivalent
(cheap, label-side only). Skip if items 1–2 fill the night.

## Deliverables

Versioned corpus artifacts + npz + stats JSONs with bootstrap CI
blocks; a caching-cost table (tokens ×3 models per scaled corpus, for
the GPU pods); one LOG line per item; ledger note under the screen-
outcomes block; STATUS rewritten. Stop for review — briefing stays.
