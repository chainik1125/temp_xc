# Working state — agent `runpod`

**Last rewrite:** 2026-07-24 (overnight), at the end of the
corpus-scaleup campaign.
**State: CAMPAIGN COMPLETE AND PUSHED — STOPPED FOR REVIEW**
(`briefings/corpus-scaleup.md` stays, per its own instruction). All
three items shipped plus five extensions; nothing is mid-flight; repo
clean and in sync. Suite **304 passed**.

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am
`runpod` (no `/workspace/.agent_id` — do not create it).** 32 CPU /
128 GB, NO GPU, 184 GB free disk, `HF_HOME=/workspace/hf_cache`.
Shared-branch rules: commit STATUS first, `git pull --rebase origin
arxiv` before EVERY push; LOG.md append-only (conflicts: keep both,
upstream first, mine last — the python strip-markers recipe, used 8×
now); push
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`
(URL-push does not update the tracking ref — `git fetch origin` before
reading `status -sb`). Tokens in `/workspace/.tokens/` (export both
HF_TOKEN and HUGGING_FACE_HUB_TOKEN). `sleep` blocked (`until …;
done`); background python needs `-u`; never trust `pytest | tail`
exit status (`--ignore tests/test_v2_code_version.py`, which is
dirty-tree-sensitive).

## What shipped tonight (all pushed; receipts in `task_hunt/SCALEUP.md`)

**Item 1 — punctint 400 → 4,000 fineweb docs.** `pull_fineweb4k.py`,
`build_punctint4k.py`. **Prefix identity PASSES at token level on all
three tokenizers** (`token_ids` AND `doc_off`) ⇒ the scaled corpus is a
deterministic superset and the pods' existing caches cover the first
~780–794k tokens/model. No frozen bar fires. Unigram 0.517–0.534 →
0.574–0.583 (list) / 0.520–0.533 → 0.558–0.563 (q); position FELL;
`doc_mean_only_auc` 0.966 (list) / 0.901 (q). Manifest cap 100k/class
BINDS (support 190k list / 530k q). The "8 documents" within-doc control
becomes 199/173/**56**/3 test docs at ≥1/5/20/50 rows per class (list)
and 504/437/**117**/7 (q).

**Item 2 — refmark 400 → 2,000 WildChat convs.** `pull_refmark2k.py`,
`build_refmark2k.py`. Funnel recorded (250k streamed → 119,458 English →
6,788 ≥8-turn → **6,256 pool** → 2,000); overlap: all 400 shipped convs
are in the pool, only **121** in the sample (near-independent evidence,
NOT a superset). No bar fires; conversation identity **0.974–0.975**
unmoved ⇒ the card's within-conversation control stands, now with **52**
test convs at ≥20 rows/class. `is_user_echo` shipped (**0.52 %** of
manifest rows). Recurrence 0.336; kernel support 1,096 tokens (the ~16×
under-span confirmed).

**Item 3 — novelty bootstrap** (`boot_novelty.py`), shipped point
estimates asserted to reproduce `novelty_stats.json` exactly.

**Extensions (past the gate, all label-side):**
1. `boot_lib.py` + 7 tests — doc-level cluster bootstrap, exact
   Mann-Whitney by level counting, ~3 s per 1.3M-row statistic.
2. `probe_estimator_scale.py` — **the unigram bar rose because the
   type-mean estimator is train-size-limited**, verified by holding
   evaluation rows fixed and varying train documents; **replicated on
   WildChat** (`--bundle refmark2k`). Curve unsaturated ⇒ every 400-doc
   unigram triage number is a LOWER BOUND. The probe-side corollary
   (per-token baselines may attenuate faster than window ones ⇒ 400-doc
   screens may OVERSTATE the window gap) is flagged as an **unverified
   hypothesis** — it needs GPU, so it is not mine to run here.
3. `verify_prefix_labels.py` — the frozen-logic claim CHECKED: per-token
   labels bit-identical on the shared 400-doc prefix, all 3 tokenizers;
   class relabeling quantified (0.57 % list, 0.0000 % q).
4. `boot_docmean_index.py` — `doc_mean_only_auc` over **11 faces** with
   CIs (0.554 Ward vslope … 0.975 refmark). **Recommendation: do NOT
   promote it to a kill bar** — any separating threshold sits below
   punctint q at 0.901, the only unconditional KEEP. Agrees with
   runpod-e's causal dialevel argument.
5. `build_novelty4k.py` — novelty at 4,000 docs, added *because*
   runpod-e withdrew its NEGATIVE verdict mid-campaign. No bar fires;
   autocorrelation structure reproduces at 10× (0.629/0.515 lag 16,
   0.119/0.026 lag 64). **npz NOT committed** (~144 MB/tokenizer;
   exactly regenerable, stats committed) — they sit untracked in
   `labels/` by design, like the killed eqdens bundle.
6. `probe_contrast_depth.py` — the within-document control has
   **2.5k–7.3k balanced rows/class** at scale (manifests are
   breadth-first); a depth-first variant is the screen owner's call.
7. `scaleup_caching_cost.py` — **39.2M new tokens across 3 models**
   (21.1M fineweb4k, 18.1M refmark2k); fineweb's cached prefix is earned
   by the receipt.

**Record correction (mine, appended):** runpod-e withdrew their `tss`
KILL and `novelty` NEGATIVE mid-campaign. My "0.82–0.88 separates
NEGATIVE from surviving" reading is WITHDRAWN (no NEGATIVE anchor);
outcome labels revised in `SCALEUP.md` §7 and the ledger bullet;
**measurements unchanged**; the kill-bar recommendation survives because
it rests on punctint q.

## Deliberately NOT done (and why — do not "fix" without a decision)
- **interleave/`tss` at 4,000 docs**: greedy max-Jaccard pairing over a
  10× pool finds much higher-overlap pairs, so it changes the corpus's
  CHARACTER, not just its size. That is the bundle owner's design call.
  Compute is affordable (~8M pairwise Jaccards, minutes) if wanted.
- **Depth-first manifest variants** for the KEEP faces: censused, not
  shipped — same reason.
- **The probe-attenuation hypothesis**: needs activations; no GPU here.

## Repo state
Clean, in sync with `origin/arxiv`. Untracked by design: the three
`eqdens_openwebmath_*.npz` (killed bundle) and the three
`novelty4k_fineweb_*.npz` (too large; regenerable). Leaderboard
untouched. Spend $11.52/$25 (zero API this campaign).

**Next action:** none pending — await review of the campaign. If a new
briefing lands, read it first; otherwise the highest-value follow-ups
are the three "deliberately NOT done" items above, each of which needs a
decision (or a GPU) that is not mine to make.
