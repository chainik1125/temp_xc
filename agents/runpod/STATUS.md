# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (night), at the round-2 factory
acceptance gate. **State: `briefings/candidate-factory-broad-2.md`
COMPLETE — stopped for mac-local review (briefing stays until then).**
Nothing mid-flight; repo clean and in sync with origin/arxiv.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, NO GPU. Shared-branch rules: commit STATUS first, `git pull
--rebase origin arxiv` before EVERY push; LOG.md append-only
(conflicts: keep both, upstream first, mine last — 2 more resolved
this session, same python-strip recipe); push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not update the tracking ref — `git fetch origin`
before reading `status -sb`). Tokens in `/workspace/.tokens/`.
`sleep` blocked (`until …; done`); background python needs `-u`.
Beware `pytest | tail` exit-status illusion — capture `$?` from
pytest itself.

## Round 2 — what shipped (all pushed, cite by subject line)

1. **Ledger appends** ("candidate factory r2: ledger appends…"):
   **D7 refusal-as-posed DEAD** with the full `docs/papers/refusal.md`
   receipts (single-(pos,layer) diff-in-means direction causal both
   ways across 13 chat models to 72B; § 5.2 attention heads perform
   the window→position deposit; App. J direction in base models) +
   **B7 BUILD-if-time entry** (four-axis vet, frozen-list sourcing,
   < 2 %-of-turns free-kill pre-gate) + live screen-outcomes block.
2. **B6 eqdens (OpenWebMath equation-density): KILLED at triage —
   free kill, no GPU.** Strict commit-then-run held. The frozen
   unigram bar fired on the OPERATIVE manifest rows: gpt2 0.6530
   ≥ 0.65 (gemma2 0.6430, llama31 0.6298 top-of-band) with ALL math
   tokens masked ⇒ the topic leak lives in the surrounding prose
   register. Position CLEAN (manifest 0.50–0.53 — stratified guard
   worked). Receipts committed: card verdict appendix +
   `labels/eqdens_stats.json` + pinned `labels/eqdens_corpus.json.gz`
   (600 docs, pinned revision `fde8ef8d…`, ODC-By 1.0); the 3 npz NOT
   committed (regenerable; no consumer for a dead bundle). **P3
   inherits the measured technical-register-leak receipt.**
3. **Verdict hygiene** (standing rule, first exercise): folded three
   runpod-e screen outcomes into the ledger — **B2 novelty NEGATIVE**
   (conversion, 23–29 % residue), **B4 punctint-q KEEP (the hunt's
   first)**, **B3 punctint-list WEAK KEEP conditional**; P2 stays
   parked (list did NOT die on position), P6 untouched. **ADOPTED
   runpod-e's recommendation:** every future broad builder reports
   `doc_mean_only_auc` (disclosure statistic; kill authority stays
   with the two frozen bars until a review pins a threshold) + any
   KEEP face owes a within-document contrast.
4. **B7 refmark (WildChat refusal-marker intensity): SHIPPED** —
   three commits in strict order: (a) FROZEN substring list BEFORE
   counting — the paper's `refusal_score` set VERBATIM (12 strings,
   `andyrdt/refusal_direction` @ `9d852fae`, App. D.1 semantics, no
   additions); (b) hard pre-gate: marker rate **0.147 of assistant
   turns vs the 0.02 free-kill bar** on WildChat-1M (pinned rev
   `7d6490e4…`, ODC-By 1.0), recurrence real (38 % of ≥ 8-turn convs
   have ≥ 2 markers) — receipt `labels/refmark_pregate.json`;
   (c) builder + card pre-run, then run. Bundle: 400 English
   ≥ 8-assistant-turn convs, message-λ̂ kernel 2/8, marker-message +
   newline tokens masked, position-matched manifests, 1.19–1.36M
   tok/tokenizer, ~20k rows/class, zero_split. **Triage: unigram
   near-BLIND (manifest 0.517–0.532 — the topic-leak fear does NOT
   materialize at token level), position 0.545–0.565 manifest;
   `doc_mean_only_auc` 0.966–0.968 = the loudest conv-identity route
   in the program ⇒ within-conversation contrast is a BINDING screen
   precondition** (+ position floor probe + beat-the-visible-evidence
   line; kernel support ≈ 1,000–1,150 tok ≈ 16× ladder top, stated).
   All artifacts committed (3 npz + corpus + stats + pregate).

Suite grew 276 → 285 (+7 eqdens grammar/logic, +3 refmark; the
`test_v2_code_version` file is dirty-tree-sensitive — exclude it when
testing with uncommitted files, it passes clean).

## Pinned conventions (carry forward)
- Broad-factory bars, PINNED (review qualification 4): 0.65/0.65,
  direction-agnostic max(AUC, 1−AUC), manifest rows operative,
  0.55–0.65 ship-with-disclosure — stated in every card up front.
- NEW: `doc_mean_only_auc` reported in every builder triage (see 3).
- New-corpus rule held twice more: pinned revision + streamed-prefix
  disclosure + seeded subsample shipped as `<name>_corpus.json.gz`;
  builder doubles as exact re-pull script; license noted (eqdens
  ODC-By + CC ToU; refmark ODC-By).
- Freeze-before-count precedent (B7): event DEFINITIONS commit before
  any corpus measurement, not just bars before builders.

## Waiting on / next
- **mac-local review of round 2** (briefing stays until then). Screen
  queue continues on GPU pods: sc_lambda → oprate → qrate(Ward) →
  vslope → interleave tss → dialevel, now + refmark (with its binding
  preconditions) if the review admits it; novelty NEGATIVE and both
  punctint verdicts already in.
- No other active briefings for me. If idle post-review: verdict
  hygiene stays standing (fold new screen outcomes into the ledger).
- Spend unchanged ($11.52/$25) — zero API calls this round.

## Round-1 summary (APPROVED earlier, do not redo)
Ledger (18 ideas) + B1 interleave + B2 novelty + B3 list-density +
B4 qrate-fineweb + B5 dialevel; 5 binding review qualifications;
bundle-builder pattern + helper inventory in `labels/*_lib.py`
(novelty/punctint/dialevel/interleave/eqdens/refmark libs all
reusable: kernels, zero_split, stratified manifests, type-mean
triage, doc-mean statistic).
