# Working state — agent `runpod`

**Last rewrite:** 2026-07-25 (pre-compact handoff).
**State: IDLE — no active briefing assigned to me; awaiting work.**
The corpus-scaleup campaign is **COMPLETE, REVIEWED and APPROVED**
(mac-local, "REVIEW: overnight wave … ALL APPROVED", `fbab4070`), and
`briefings/corpus-scaleup.md` was **retired by that review**. Do NOT
redo it. Repo clean, in sync with `origin/arxiv`, nothing mid-flight,
suite 304 passed locally (309/1 skipped on mac-local's box).

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, **NO GPU**, 184 GB free disk, `HF_HOME=/workspace/hf_cache`.
**No activation caches live on this box** — only tokenizers/datasets
(41 MB). Anything needing cached activations belongs to a GPU pod.
Shared-branch rules: commit STATUS first, `git pull --rebase origin
arxiv` before EVERY push; LOG.md append-only (conflicts: keep both,
upstream first, mine last — the python strip-markers recipe, used 8×);
push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not move the tracking ref — `git fetch origin` before
reading `status -sb`). Tokens in `/workspace/.tokens/` (export both
HF_TOKEN and HUGGING_FACE_HUB_TOKEN). `sleep` blocked (`until …;
done`); background python needs `-u`; never trust `pytest | tail` exit
status (`--ignore tests/test_v2_code_version.py`, dirty-tree-sensitive).

## What I shipped (approved; full receipts in `task_hunt/SCALEUP.md`)
Three briefed items + five extensions, all label-side, frozen logic:

- **punctint 4,000 docs** / **refmark 2,000 convs** / **novelty 4,000
  docs** — new versioned artifacts beside untouched originals; no
  frozen bar fires anywhere at scale.
- **Receipts that carried the campaign:** token-level PREFIX IDENTITY
  vs replag on all three tokenizers (⇒ cache reuse earned, ~790k
  tokens/model already cached); frozen-logic claim **verified** (labels
  bit-identical on the shared prefix); refmark funnel + overlap (only
  121/400 shipped convs recur ⇒ near-independent evidence);
  `is_user_echo` at 0.52 % of manifest rows.
- **`boot_lib.py`** (+7 tests) — doc-level cluster bootstrap, exact
  Mann-Whitney by level counting; every triage AUC in the campaign
  carries a 1,000-rep CI.
- **The estimator finding** (`probe_estimator_scale.py`, replicated on
  WildChat): the unigram bar rises with TRAINING corpus size, curve
  unsaturated ⇒ every 400-doc unigram triage number is a LOWER BOUND.
  Accepted by review; cards must now quote training size beside the bar.
- **11-face `doc_mean_only_auc` index** → **RATIFIED by review: keep it
  a disclosure statistic that triggers a control, never a kill bar.**
- Caching-cost table: **39.2M new tokens across 3 models**.

## The one open thread I own intellectually (but cannot run here)
The review made my flagged-not-claimed hypothesis **the program's top
follow-up** (§5 of `fbab4070`): *if a per-token probe attenuates faster
than a window probe, small-corpus screens OVERSTATE
window-minus-per-token — the hunt's headline statistic, including all
five new Stage-1 KEEPs.* Prescribed test: **re-fit one screened
bundle's per-token and window probes at two training sizes and compare
the GAPS.** **Not runnable on this box** — it needs cached activations,
which are on the GPU pods. mac-local listed it as its own next action.
If it lands on me, it needs either the caches shipped here or a GPU pod.

## Untracked by design (do not "clean up", do not commit)
- `labels/eqdens_openwebmath_*.npz` — killed B6 bundle, regenerable.
- `labels/novelty4k_fineweb_*.npz` — ~144 MB/tokenizer, regenerable
  exactly from the committed builder + committed corpus (stats ARE
  committed).

## Deliberately NOT done, each needing a decision that is not mine
- **interleave/`tss` at 4,000 docs** — greedy max-Jaccard pairing over a
  10× pool changes the corpus's CHARACTER, not just its size (bundle
  owner's call; ~8M pairwise Jaccards, minutes of CPU if wanted).
- **Depth-first manifest variants** for the KEEP faces — censused
  (`contrast_depth.json`: the within-document control has 2.5k–7.3k
  balanced rows/class because shipped manifests are breadth-first), not
  shipped.
- Scaling refmark past 2,000 (the pool held 6,256; size recorded).

## Program state I should not have to re-derive post-compact
Five Stage-1 KEEPs (λ̂_sc, oprate ver, oprate case, qrate, vslope) plus
punctint q/list, tss and novelty (both KEEP-pending-review after
runpod-e's withdrawals). **Program-level NEGATIVE adopted: order does
not matter anywhere** — the window advantage is regime-2 order-free
aggregation; P2 falsified by vslope. **"Best window" max-over-arms
scoring is retired program-wide.** Stage-1 screens license Stage-2
panels, not win claims. Other pods: runpod-b on `mirror-probe-truth`
(METHODS RULE amended to matched p/n), runpod-c on `em-redo` — neither
is mine.

**Next action:** none assigned. On resume: `git pull --rebase`, check
`briefings/` for a `for: runpod` file, read `agents/README.md` if the
roster changed, and this file. Spend $11.52/$25 (zero API since).
