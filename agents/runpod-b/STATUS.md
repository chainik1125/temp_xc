# Working state — agent `runpod-b`

**Last rewrite:** 2026-07-24 (late) — **`candidate-factory-traces.md`
BATCH COMPLETE, all 5 candidates processed, stopped at the acceptance
gate for mac-local review.** Briefing stays in place until that review
(as does the earlier `hunt-support-stats.md`, also awaiting review).
I am idle; next assignment on request.

## Who / where
Second RunPod box, repo `/workspace/temp_xc`, 32 CPU, no GPU, no CFS cap.
`/workspace/.agent_id` = runpod-b. Git identity set (Han); push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
`export ANTHROPIC_API_KEY=$(cat /workspace/.tokens/anthropic_key)`.
HF token at `/workspace/.tokens/hf_token`.

## Factory batch — final tally (all committed + pushed, one LOG line each)

**4 npz bundles / 5 screen-ready labels + 2 honest triage kills**, all
on the canonical Ward grid (4044 × 128), ward_lambda manifest
conventions (20k rows/class primary AND null, pos ≥ 32, by-trace
split), builders committed before outputs, frozen card per candidate:

1. `labels/sc_lambda.npz` — λ̂_sc marker intensity. PASS (tok 0.636 —
   NEAR the 0.65 line, disclose; pos 0.625). Evidence ceiling T32 0.70.
2. `labels/qrate.npz` — λ̂_q question rate. PASS (tok 0.610, pos
   0.586). corr with λ̂_sc only 0.32. Evidence T32 0.74.
3. `labels/oprate.npz` — rate_ver + rate_case, BOTH pass; rate_case
   position-blind (0.51); corr(ver, case) = −0.03, corr(ver, λ̂_sc) =
   0.03 (feared marker overlap did NOT materialize). Evidence T32
   0.83 / 0.78 — high; beat-the-line falsifier has teeth.
4. `labels/verbosity.npz` — vlevel KILLED label-side (tok 0.654 ≥
   0.65: register lexically readable per-token); vslope shipped
   (tok/pos-blind) with the honest caveat that hedging-LEVEL lesson
   makes slopes the hard aggregation face (screen-kill risk accepted).
5. redundancy — KILLED (pos 0.890, tok 0.660; stats JSON is the
   receipt, no npz). The briefing's predicted failure face.

Shared machinery (all tested, `tests/test_factory_labels.py`, 10
tests; full suite 269 green): `labels/factory_lib.py` — frozen
exponential kernel (τ = 3, K = 8, causal, normalized, history guard
i ≥ 4, kernel-only), frozen 17-pattern marker list, event-shuffle
nulls (seeds 101–105 + trace_idx), zero_split bin fallback (min bin ≥
10%), triage kill rule (tok ≥ 0.65 / pos ≥ 0.70 extreme-AUC on
test-split manifest rows), `bundle_core` shared pipeline. Every stats
JSON carries a `visible_evidence_auc` line (in-window event count) —
the screen must beat it at matched T or it is counting visible
evidence tokens.

## For the screening agents (runpod-d/e)
Bundles drop onto the existing Ward base/distill caches unmodified —
same (man_doc = window, man_pos) row convention as ward_lambda;
`man_null_*` manifests probe the shuffled-label null; per-candidate
masks are IN the manifests already (marker/"?" tokens, event-class
sentences, is_rep). Draft T-pattern + falsifier per candidate in
`task_hunt/<name>/CARD_DRAFT.md` — the running agent freezes its own
screen card. Note oprate labels are NaN wherever a kernel-lag sentence
is judge-unlabeled (coverage 0.895 of valid tokens).

## Context that binds future work here
- Sibling factory: runpod (broad corpus) shipped its ledger + B1
  (interleave tss promotion) + B2 (vocabulary-novelty, fineweb); its
  B3/B4 includes a question-rate face on fineweb — my qrate card
  carries the cross-cite. LOG.md conflicts with it twice this session:
  resolution = keep upstream entry, re-append mine (append-only rule).
- hunt-support-stats (previous session) APPROVED upstream with
  consequences: T = 16 dip interpretation RETRACTED to
  cause-not-established; seed top-up first-class for runpod-d.
- Environmental pytest trap: `test_diff_hash_consistent_with_dirty`
  fails when the tree has untracked files — commit/clean before full
  suite runs (traces.json is gitignored, doesn't count).
- Shared branch: 5 agents on arxiv; pull-rebase before EVERY push;
  cite commit SUBJECTS not SHAs. Rewrite this file before any compact.
