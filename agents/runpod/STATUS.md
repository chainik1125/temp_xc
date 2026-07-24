# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (late night), immediately after receiving
`briefings/corpus-scaleup.md` (pre-compact handoff).
**State: OVERNIGHT CAMPAIGN ASSIGNED, ZERO WORK DONE ON IT** (per
instruction: the work happens post-compact). Round-2 factory is DONE
and **APPROVED** (mac-local "REVIEW: probe-adequacy (runpod-b) +
factory round 2 (runpod) — BOTH APPROVED"; both briefings retired).
Do NOT redo B6/B7.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, NO GPU, 184 GB free disk, `HF_HOME=/workspace/hf_cache`.
Shared-branch rules: commit STATUS first, `git pull --rebase origin
arxiv` before EVERY push; LOG.md append-only (conflicts: keep both,
upstream first, mine last — the python strip-markers recipe, used 6×
now); push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not update the tracking ref — `git fetch origin` before
reading `status -sb`). Tokens in `/workspace/.tokens/` (export both
HF_TOKEN and HUGGING_FACE_HUB_TOKEN). `sleep` blocked (`until …;
done`); background python needs `-u`; never trust `pytest | tail`
exit status — capture pytest's own `$?` (or `--ignore
tests/test_v2_code_version.py`, which is dirty-tree-sensitive).

## ACTIVE TASK (not started): briefings/corpus-scaleup.md
**Overnight (10+ h) CPU campaign; results by Saturday morning PT.**
Motivation: the hunt's first screen KEEPs sit on corpora too thin for
panel-grade receipts (punctint-list's within-document control rests on
**8 documents**; fineweb = 400 docs; refmark = 400 conversations). If
punctint-q or refmark graduates to Stage 2 on Saturday, the panel
should train and control on 10× data, and the doc-identity threshold
question needs a DISTRIBUTION, not a point estimate.

**Hard rules (verbatim intent):** never touch the pinned originals
(`fineweb_sample.json`, committed corpus artifacts, shipped
npz/stats) — every scale-up is a NEW versioned artifact (`*_4k`,
`*_2k`) beside them; **label logic stays FROZEN** (reuse
`punctint_lib` / `refmark_lib` / `novelty_lib` unchanged; any code
change needs its own pre-run commit with a stated reason); same pull
recipes + filters, extended stream prefixes, seeded, deterministic;
incremental commits corpus → labels → stats per item, one LOG line
each; **a frozen bar firing at scale is a FINDING, not an
embarrassment** — disclose it, it binds the Stage-2 design, it does
NOT retro-kill the shipped small-corpus bundle (different artifact,
same logic).

### Item 1 — fineweb 400 → 4,000 docs (punctint faces; the KEEP first)
Recipe is already written and pinned:
`src/explorations/synthetic/expansion/corpus.py::sample_fineweb(
cache_path, n_docs=4000, seed=0, dataset="HuggingFaceFW/fineweb",
name="sample-10BT", split="train", min_sents=60, max_sents=200,
shuffle_buffer=10_000)` — it is idempotent on `cache_path`, so pass a
NEW path (plan: `experiments/explorations/synthetic/expansion/data/
fineweb_sample_4k.json.gz`-style versioned artifact beside the pinned
one; gzip because the 400-doc JSON is 3.8 MB ⇒ ~38 MB at 10×; if the
plain `.json` path is needed for `sample_fineweb`'s own writer, write
plain then gzip and commit the gz, or add a tiny wrapper — do NOT
edit the pinned file). Pinned-400 meta for comparison: n_scanned
4,183, n_sentences 36,805, datasets 4.8.5, splitter
`expansion.corpus.split_sentences v1` ⇒ expect **~42k docs scanned,
~368k sentences** at 4,000.

Then rebuild BOTH punctint faces × 3 tokenizers at scale in a NEW
builder (`build_punctint_4k.py` — do not edit `build_punctint.py`),
importing frozen `punctint_lib` unchanged: labels, position-matched
manifests (`stratified_balanced_manifest`, raise `cap` 20k → ~100k
per class **if the data supports it — say what it supports**), triage
on the frozen **0.65/0.65 direction-agnostic, manifest-rows-operative**
bars, plus `doc_mean_only_auc` on both faces.

**The `token_ids` assert must change and the change is a receipt.**
`build_punctint.py` asserts byte-identity with
`replag_fineweb_<tok>.npz`, which covers only the pinned 400 docs.
At 4k, assert **prefix identity**: the first-400-doc token slice ==
replag `token_ids`. That single assert proves (a) the 4k pull is a
deterministic SUPERSET whose prefix is the pinned sample and (b)
tokenization is unchanged ⇒ **the GPU pods' existing caches cover
the first 400 docs; only the new ~3,600 need caching.** If the assert
FAILS (streaming shuffle-buffer behavior can differ at larger n),
that is not an error to hide: disclose that the scaled corpus is a
different sample rather than a superset, drop the cache-reuse claim,
and continue — the shipped 400-doc bundle stands on its own artifact.

**Bootstrap receipts are the point of item 1:** doc-level bootstrap
(**≥ 1,000 reps**) CIs on every triage AUC and on `doc_mean_only_auc`,
per tokenizer per face — the threshold-pinning review consumes this.
Plan: new `boot_lib.py` (+ tests, pre-run commit) — precompute per-row
(score, class, doc) once, resample DOCUMENTS with replacement,
recompute `interleave_lib.rank_auc`; parallelize over the 32 cores.
Sizing: 400 docs = 794k tokens/tokenizer ⇒ **4,000 docs ≈ 8M
tokens/tokenizer** and ~300k manifest rows/face at cap 100k, so a rep
is a ~30 ms sort — 18 statistic-slots × 1,000 reps is minutes
parallel, ~20 min serial. Also report **how many documents carry the
within-document contrast per face at scale** (docs holding both top-
and bottom-class manifest rows — the "8 documents" number, fixed or
not).

### Item 2 — refmark 400 → 2,000+ conversations
Same recipe as `build_refmark.py` (WildChat-1M, pinned revision
`7d6490e462285cf85d91eabea0f9a954fbddcd1f`, ODC-By 1.0; English,
≥ 8 assistant turns, 2,000–24,000 rendered chars, seeded), longer
stream prefix, new builder + `*_2k` artifacts. **Yield arithmetic
from the committed pre-gate receipt:** 20,000 streamed → 9,464
English → 681 with ≥ 8 assistant turns (3.4 % of streamed) before the
char filter ⇒ **plan `N_STREAM` ≈ 200k–250k for 2,000+ convs, and
RECORD the pre-subsample pool size this time** (the 400-conv build
did not). Same deliverables as item 1, plus: (a) **`is_user_echo`
mask array shipped in the scaled npz** — marker masking covers
ASSISTANT messages only, so user messages echoing a frozen substring
are unmasked and manifest-eligible; mac-local measured 13/4,713 user
messages ⇒ 134/59,994 manifest rows (0.22 %) on the 400-conv build —
recompute at scale so screens can drop those rows trivially (this is
a NEW disclosure array, not a label-logic change; still commit-then-
run); (b) recurrence stats (frac convs ≥ 2 markers) at scale
(400-conv/pre-gate value: 0.377 on the ≥ 8-turn population).

### Item 3 (only if the night allows) — novelty-family bootstrap
No new corpus (novelty screened NEGATIVE). Label-side only: doc-level
bootstrap CIs on the committed novelty triage AUCs + the
`doc_mean_only_auc` equivalent computed from the committed npz (that
statistic post-dates the novelty builder). Cheap; skip if 1–2 fill
the night.

### Deliverables / acceptance gate
Versioned corpus artifacts + npz + stats JSONs **with bootstrap CI
blocks**; a **caching-cost table** (tokens × 3 models per scaled
corpus, for the GPU pods — note 8M tok/tokenizer at fineweb-4k is
~10× the current bundles); one LOG line per item; a ledger note under
the screen-outcomes block in `CANDIDATES.md`; STATUS rewritten. Stop
for review — briefing stays.

## Conventions that govern this campaign (carry forward)
- Broad-factory bars PINNED: 0.65/0.65, direction-agnostic
  max(AUC, 1−AUC), **manifest rows operative**, 0.55–0.65
  ship-with-disclosure. `doc_mean_only_auc` = RATIFIED reported
  disclosure statistic (kill authority stays with the two frozen
  bars; threshold pinning deferred to the post-screen-wave review —
  this campaign supplies its distribution).
- Strict commit-then-run: every builder/lib/card commits BEFORE it
  produces an output; verdict/receipt blocks are pure appends.
- New-corpus rule: pinned revision + stream-prefix disclosure +
  seeded subsample shipped as a `.json.gz` artifact; builder doubles
  as the exact re-pull script; license noted.
- Freeze-before-count (B7 precedent): event DEFINITIONS commit before
  any measurement, not just bars before builders.

## Round-2 factory summary (APPROVED — do not redo)
D7 refusal-as-posed DEAD (refusal.md receipts) + B7 entry; **B6
eqdens KILLED at triage** (manifest unigram gpt2 0.6530 with all math
tokens masked ⇒ prose-register leak measured; P3 inherits it; npz
deliberately uncommitted, regenerable — they still sit untracked in
`labels/`, that is expected); verdict hygiene (B2 NEGATIVE, B4
punctint-q KEEP, B3 punctint-list WEAK KEEP; P2 stays parked);
**B7 refmark SHIPPED** (pre-gate 0.147 vs 0.02 bar; unigram
near-blind 0.517–0.532; conv-identity 0.967 ⇒ within-conversation
contrast BINDING at screen, + position floor probe + visible-evidence
line + under-span ~16×). Suite 285 passed (285 + runpod-b's adds =
302/1 on their box).

## Repo state
Clean, in sync with `origin/arxiv` (the two eqdens npz are the
intentionally-uncommitted killed-bundle outputs). Nothing mid-flight.
Leaderboard untouched by me. Spend $11.52/$25 (zero API this round).
**Next action post-compact: read `briefings/corpus-scaleup.md`, then
start item 1 with the 4k fineweb pull running in background while I
write `boot_lib.py` + tests and `build_punctint_4k.py` (commit both
before any output).**
