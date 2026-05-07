---
author: agent_paper
date: 2026-05-03
status: locked
---

## Locked decisions, 2026-05-03

These three decisions were made in conversation with Han at session start.
They are now policy. Re-opening any of them requires a new conversation
with Han.

### 1. Two TXC architectures (locked)

- **TXC-base** = `txc_bare_antidead_t5` — vanilla TopK + tsae_paper anti-dead stack.
- **TXC-pro** = `phase5b_subseq_h8` — subseq encoder + matryoshka H8 + multi-distance contrastive.

**Rationale**: these are the only two TXCs with consistent top-3 finishes
across both Phase 5 and Phase 7 probing leaderboards. Steering hill-climb
winners (Galaxy 8/11/18) were considered and rejected because they lose
0.005–0.020 probing AUC vs canonical (`2026-05-02-yw-T8-benchmark.md`).

**Trade-off accepted**: C5 steering becomes a "matches T-SAE at high coh"
result rather than a "beats T-SAE" result. C3+C4 wins are stronger.

**Addendum 2026-05-05** (Han + agent_paper): the canonical paper TXCs
going forward are `txc_base_mw` and `txc_pro_mw` — same encoder/decoder
weights as `txc_base` / `txc_pro`, with `multi_window: true` in hparams
that fixes the per-step training-FLOPs disparity vs per-token archs (see
§ 14). The original `txc_base` / `txc_pro` entries stay registered as
the historical pre-fix baseline; their cells from the in-flight sweep
remain in `leaderboard.jsonl` for diff comparison and don't need
re-running. Paper text continues to refer to them as "TXC-base" and
"TXC-pro"; the `_mw` suffix is implementation detail at the registry
layer. The "two TXCs only" rule still holds at the paper-claim level —
the four registry entries reduce to two architecturally-identical pairs.

### 2. C6 EM is reframed as honest negative

**Finding**: Dmitry's `em_nanda_results_paper.md` shows SAE arditi beats
TXC k=100 at every (steps × organism × α-regime) cell. Arch gap *widens*
to +12.58 align in R32 ext-α regime.

**New paper framing**: report the SAE win honestly. Salvage the
**bundle-null architecture-generality** result — both arches' k=30
bundles peak at align ≈ 41.3, falling 13–23 below their single-feat
champions. This falsifies the "distributed misalignment" hypothesis in
both dictionaries: an interpretive contribution despite the probing loss.

**Coordinator note**: needs to be co-signed with Dmitry. Agent EM should
ping Dmitry's brief in `origin/em-nanda` before launching anything.

### 3. Branch model

- `final` is created from `han-phase7-unification` HEAD (the user's
  current state, including the briefing + paper md).
- All future work commits to `final`.
- Wasteland is the rest of the branch's tree — read-only context.
- Push `final` to origin so worker agents can clone it.

### 4. Cross-branch reads (em-nanda, aniket-ward-stage-b)

- **Do not merge** sibling branches into `final`. They are still being
  updated by Dmitry and Aniket; merging would freeze stale state and
  create conflict surface on every refresh.
- Read directly from origin: `git show origin/em-nanda:<path>`.
- `purified/scripts/wasteland_refresh.sh` does the `git fetch` so
  `origin/<branch>` always resolves to the latest pushed state.
- If we need a frozen snapshot of code (e.g. Aniket's
  `experiments/ward_backtracking_txc/`), copy it once into
  `purified/src/temp_bench/` with the source commit hash in a header
  comment, and stop tracking origin from then on.

### 5. CLAUDE.md scoping

- Subdirectory CLAUDE.md files **auto-load on demand** when an agent
  reads files under that directory (verified). So an agent launched at
  the repo root sees the wasteland CLAUDE.md initially, and
  `purified/CLAUDE.md` loads automatically the moment it touches a
  paper file.
- Added a one-line pointer in the root `CLAUDE.md` directing paper-bound
  agents at `purified/CLAUDE.md`.
- Recommended (not enforced) launch pattern for paper-only agents:
  `cd purified && claude` — keeps `git add -A` scoped to paper files.

### 6. HuggingFace repos

- New, private, paper-dedicated:
  - **`han1823123123/temp-bench-models`** — all checkpoints (locked
    archs + baselines), keyed by `<run_id>` prefix.
  - **`han1823123123/temp-bench-data`** — activation caches, judge
    transcripts, pre-tokenised tasks, synthetic data.
- Provisioned 2026-05-03 with seed READMEs.
- Wasteland repos (`han1823123123/txcdr-base`, `txcdr-it`, `txcdr`,
  `txcdr-base-data`, `txcdr-data`) are **untouched** — they remain as
  historical record. Paper artifacts never go into them.
- Visibility flips to public when the paper draft stabilises.

### 7. Bricken resample is opt-in per component, NOT a locked architecture default

**Context**: Dmitry's data on Qwen-7B medical (`txc_hookpoint_comparison_finding.md`)
shows TXC brickenauxk 30k @ resid_mid (53.87) ties T-SAE 100k @
resid_post (52.39) — the "tied at 30k" Han mentioned. That recipe
co-tunes **six** knobs (resample_every=500, min_fires=1, n_check=2048,
max_resample_fraction=0.5, EMA-AuxK α=1/8, dead_threshold=128k tokens),
all jointly tuned for that organism.

**Decision** (revised after Han pushed back on the original
"trainer-level default for both TXCs" framing):

- The locked architectures TXC-base and TXC-pro **do not** include
  Bricken resample. They include only what's listed in
  `docs/paper/architecture.md` proper.
- Bricken resample is exposed as an opt-in `BrickenConfig` knob in
  `src/temp_bench/training/bricken.py`. Components turn it on
  themselves and disclose the choice in their writeup.
- **C6 only by default** (Dmitry's evidence directly supports it on
  Qwen-7B medical organism).
- C1/C2 keep it off (no dead-feature pressure at $d_{\text{sae}}=40$).
- **C3/C4/C5/C7 keep it off** (revised 2026-05-03 with Han). The
  earlier policy demanded an A/B test (TXC-base ± Bricken at 5k×1seed)
  for each of these components before adopting. Han: "we're locking in
  txc_base and txc_pro, we'll only try Bricken resample if time
  persists at the end." Saves ~8 H100-hours of validation work for a
  maybe-marginal effect; the cost is leaving on the table any
  Δ AUC > σ_seeds that Bricken would have lifted.

**Rationale**: untested interactions — TXC-pro's matryoshka × InfoNCE
might break under hard resets; toy d_sae=40 has no dead pressure;
Gemma activations may not need the recipe. "Default for both" was a
premature commitment to a recipe that's only validated on one
organism.

### 8. Wasteland code deleted; wasteland docs kept

Han: "is it worth deleting the wasteland files in the purified branch
and forcing agents to inspect other branches to see the wasteland?"

Decision: yes — but asymmetrically. **Code wasteland deleted** from
`final` (`src/`, `experiments/`, `references/`, `tests/`,
`scripts/` at root, root `pyproject.toml`, `uv.lock`, `Dockerfile`,
`launch-sandbox.sh`, `torchgpu_packages.txt`, `temporal_crosscoders/`,
root `results/`); **docs wasteland kept** (`docs/`, `papers/`, root
`CLAUDE.md`, `RUNPOD_INSTRUCTIONS.md`, `CONTRIBUTING.md`).

**Why asymmetric**: docs are read often (passively, for context — every
component writeup cites ~5 wasteland research logs). Code is read once
per port (actively, for transcription, ~10 ports total). Delete what's
read once; keep what's read often.

**Benefit**: an accidental `from src.architectures.tfa import …` now
raises `ModuleNotFoundError` immediately rather than silently picking
up wasteland code. The "no wasteland imports" rule (PROTOCOL.md § 2)
becomes git-level enforcement, not policy.

**Cost**: agents porting code use
`git show origin/han-phase7-unification:src/...` to read. Mild — one
extra command per port, ~10 times across the paper. Worked example +
header-comment template in PROTOCOL.md § 2.

3658 → 319 tracked files (~91% reduction). Reversible via `git revert`
or `git checkout origin/han-phase7-unification -- <path>` if a specific
file turns out to be needed.

### 9. Consolidate purified/ docs from 3 → 2 files

Han: "is it necessary to have THREE different files?" (referring to
`purified/` having `README.md`, `CLAUDE.md`, `PROTOCOL.md`).

Decision: drop `purified/README.md`. Merge its unique content (Quick
start, Layout, Components table, TXC summary) into `purified/CLAUDE.md`,
which is the auto-loaded source of truth for agents. `PROTOCOL.md` stays
as the detailed protocol reference.

Final layout:
- **CLAUDE.md** (~250 lines): operating manual + brief overview.
  Auto-loaded by Claude Code when an agent reads any file under
  `purified/`. Self-contained for session start.
- **PROTOCOL.md** (~330 lines): detailed contract (§ 1-11 incl. GPU
  pinning § 11.0, multi-GPU access § 11.1, framework discipline § 11).
  Read on first session, referenced as needed.

### 10. One agent doc — the briefing — owns identity AND rolling state

**Earlier design (rejected)**: per-agent `briefing.md` (Han owns) +
`handovers/<ts>-<slug>.md` (dated archive of state snapshots) +
`log.md` (running chronological narrative). Three docs per agent.

**Failure mode**: in the wasteland, `briefing.md` files went stale
because nobody updated them; `handover_*.md` files accumulated by the
dozen and nobody knew which was current; `agentic_log.md` files grew
to thousands of lines and were never read post-compact. Three docs
per agent ⇒ none stays fresh.

**Decision** (Han, 2026-05-03):

- **One file per agent: `briefing.md`**, with explicit section
  ownership inside the file:
  - `## Identity + mandate (Han owns — agents do not edit)` — Han's
    prose at the top, immutable.
  - `## Current state` / `## What I just did` / `## Next action` /
    `## Don't repeat` / `## Open questions for Han` —
    agent-owned, **overwritten at every compact**.
- **No separate `handover.md` and no `log.md`.** Git history
  (`git log -p purified/agents/<name>/briefing.md`) is the audit
  trail; `decisions.md` captures locked decisions; the briefing's
  "What I just did" captures the last 5–10 actions for the next-life
  instance.
- **Component-vs-agent doc separation codified** in PROTOCOL.md § 7:
  `docs/components/cN.md` owns the technical setup (hypothesis, results,
  caveats); agent briefings own identity + state. Briefings point at
  component docs, do not duplicate.

**Why this beats both prior options**:
- Single file ⇒ no "which doc is current" confusion.
- Section ownership ⇒ Han's mandate doesn't drift into the agent's
  rolling state.
- Git log ⇒ chronological history without manual log.md upkeep.
- Component docs ⇒ technical setup outlives any individual agent.

**Implementation**: PROTOCOL.md § 14 rewritten from "Handover protocol"
to "Briefing maintenance"; `_briefing_template.md` added;
`_handover_template.md` deleted; `agents/agent_paper/handovers/`
deleted; `agents/agent_paper/log.md` deleted. The historical content
of agent_paper's log.md is summarised in `git log` of the deletion
commit + the now-merged briefing's "What I just did".

### 11. C3 task suite is `SAEBench+CT` (n=38)

Han: "I think we should just do whatever SAEBench did ... we should
definitely fix the github-code discrepancy and do the HF permissions ...
the benefit of using the faithful SAEBench task set is reviewers won't
complain about cherrypicking."

**Decision**: C3 evaluates on **SAEBench+CT**, defined as the canonical
upstream SAEBench sparse-probing suite (Karvonen et al., 36 binary
one-vs-rest tasks across 8 datasets) augmented with two cross-token
coreference probing tasks (WinoGrande, SuperGLUE WSC). **Total: 38
tasks.** Phase 5's 36-task and Phase 7's 16-task PAPER subset are
both retired as headline candidates.

The SAEBench composition is fixed by `chosen_classes_per_dataset` in
upstream `sae_bench/sae_bench_utils/dataset_info.py` (verified
2026-05-03):

```
bias_in_bios_class_set1: ["0","1","2","6","9"]      → 5
bias_in_bios_class_set2: ["11","13","14","18","19"] → 5
bias_in_bios_class_set3: ["20","21","22","25","26"] → 5
amazon_reviews_mcauley_1and5: ["1","2","3","5","6"] → 5
amazon_reviews_mcauley_1and5_sentiment: ["1.0","5.0"] → 2
codeparrot/github-code: ["C","Python","HTML","Java","PHP"] → 5
ag_news: ["0","1","2","3"] → 4
europarl: ["en","fr","de","es","nl"] → 5
                                                       ──
                                                       36
+ winogrande_correct_completion + wsc_coreference    → 38
```

`probe_training.py` iterates per-class with no special handling, so
2-class amazon_sentiment yields 2 binaries (verified upstream).

**Three implementation deltas** vs the wasteland's "FULL-36" loader:

1. **github-code provider switch.** Use SAEBench's `codeparrot/github-code`
   with the 5 SAEBench languages `["C","Python","HTML","Java","PHP"]`,
   not our wasteland's `code_search_net` python/java/javascript/go.
   The dataset uses a Python loading script (HF web viewer is disabled
   for that reason but the dataset itself is publicly readable, NOT
   gated). Loader requires `trust_remote_code=True` — already set via
   `os.environ.setdefault("HF_DATASETS_TRUST_REMOTE_CODE", "1")`. Also
   requires `datasets<4` (the `trust_remote_code` mechanism was removed
   in v4); pinned in `purified/pyproject.toml` 2026-05-03.
2. **amazon_sentiment.** Add the 1.0-vs-rest binary (we currently only
   have 5.0-vs-rest as `amazon_reviews_sentiment_5star`).
3. **amazon_categories.** Hardcode the class list to `["1","2","3","5","6"]`
   and use a non-streaming pull large enough to populate all 5; the
   wasteland's streaming-top-5 approach is non-deterministic and
   missed cat6.

**Why SAEBench-faithful + the 2 coref additions**:

- SAEBench is the recognised standard. Saying "we evaluated on
  SAEBench" defends against the "you cherry-picked tasks that favor
  TXC" review on the headline benchmark axis.
- WinoGrande + WSC retained because they are the cleanest single-task
  evidence for TXC's cross-token inductive bias (winogrande T-slope
  +0.0069/T at k=20 — ~100× the next task; from
  `2026-04-29-per-task-tsweep.md`). Reported transparently as a
  "+CT" extension, not folded silently into "SAEBench".
- The 16-task PAPER subset (Phase 7) inherited the same coref-addition
  problem AND added a cluster-balancing decision the paper would have
  to defend separately. Two unforced critique vectors collapsed into
  one ("we extended SAEBench by 2 well-motivated coref tasks").

**Naming convention for the paper**: refer to the suite as
**SAEBench+CT** in tables and figure captions. First mention in prose:
"the standard SAEBench sparse-probing benchmark (Karvonen et al., 36
tasks across 8 datasets) augmented with two cross-token coreference
probing tasks (WinoGrande, SuperGLUE WSC; n=38 binary one-vs-rest
tasks total)."

### 12. TrainingConfig: batch=1024, max_steps=25_000, plateau-stop (Phase-5-faithful)

**Earlier directive (2026-05-04 morning, reverted):** I issued a "batch=256
→ 2048 default" cross-agent re-train, with A40 components at 1024 and H100
components at 2048. Han pushed back: (i) Dmitry/Aniket's wasteland configs
are not validated references either, (ii) re-training only the contrastive
archs (T-SAE, TXC-pro) at higher batch is unfair to the non-contrastive
baselines (TopK-SAE, SAE-arditi, MLC, TXC-base), (iii) the **standard in the
SAE comparison literature is identical training config across every arch**
(T-SAE paper §4.1: "All SAEs are trained with the BatchTopK activation...
chosen to allow for comparability"; TFA paper App. B.1: "all SAEs trained
from scratch... to enable a consistent and fair evaluation"). That earlier
directive was reverted (commit 0beae2bf reverts a9200560).

**Decision (2026-05-04 afternoon, with Han)**: re-train all C3/C4/C5/C6/C7
cells using a **fixed-step protocol**, applied uniformly across all archs
and all pods.

| Knob | Value | Source |
|---|---|---|
| `batch_size` | **1024** uniform (H100 + A40) | Phase 5 summary.md:250 (empirically validated) |
| `n_steps` | **25_000** (binding cap) | Phase 5 summary.md:250 |
| `plateau_early_stop` | **False** (disabled) | SAE-literature standard |

**Why plateau-stop is off**: agent_nlp flagged that the schema's plateau
detection (`max(loss[-5000:]) - min(loss[-5000:]) < 1e-4` per
`training/sae_trainer.py:158-165`) is an absolute threshold over a fixed
window — not Phase 5's "loss drop < 2% per 1K steps" relative criterion.
Han raised the corollary concern: an absolute threshold causes cross-arch
unfairness because the same 1e-4 window-range means very different things
for archs whose losses naturally land at different scales (an arch starting
at loss ≈ 1e-3 would trigger prematurely; an arch starting at loss ≈ 1
might never trigger). The SAE-comparison literature avoids this issue
entirely by using fixed step counts:

- **T-SAE paper (Bhalla/Ye 2025) §4.1**: fixed schedule, no early stopping.
- **TFA paper (Lubana et al.) App. B.1**: "trained from scratch on 1B
  precached activations" — fixed token count.
- **GemmaScope / Anthropic monosemanticity**: fixed token budgets.

Phase 5's "2%/1K" criterion was a Phase-5-internal invention, not a
community standard. Adopting it would require re-implementing the trainer's
plateau logic against an unvalidated relative threshold and isn't worth the
churn given our compute budget already fits a uniform 25K cap. The code
path for plateau-stop stays (gated off by `plateau_early_stop=False`) for
opt-in use; production cells run the full 25K.

**Fairness mechanism**: with plateau-stop off, every arch trains for
exactly **25K steps × batch=1024 = 25.6M activation tokens**. Cross-arch
comparisons are symmetric by construction. No "different archs converged at
different step counts" diagnostic, but no risk of arch-dependent early
stops biasing the comparison either.

**Why uniform across pods**: a non-uniform (batch=2048 H100, batch=1024 A40)
config means C3/C4/C6 archs train under different conditions than C5/C7
archs. Reviewers reading "C3 TXC-base trained at batch=2048 but C7 TXC-base
trained at batch=1024" would (correctly) flag that as a confounder. The
point of locked TXC-base / TXC-pro is that they are *the same architecture
trained the same way* in every component. batch=1024 fits all pods.

**Cross-arch fairness**: every arch in every component re-trains under the
exact same `TrainingConfig`. C6 retains its `bricken_enabled=True` override
per decisions.md § 7 (a published per-component knob, disclosed in the
component writeup), but the (batch_size, n_steps, plateau_*) knobs are
identical.

**Token budget**: 25.6M activation tokens per arch (fixed). Below the
1B-token / GemmaScope-scale field standard but matches Phase 5's empirically
validated convergence point and is what's feasible in the remaining 72-h
window.

**Cache hygiene**: `batch_size`, `n_steps`, and the plateau-* fields are
all in the `TrainingConfig` dump that feeds `compute_train_key` (see
`src/temp_bench/config.py:181-193`). New cells will derive new
`train_key` automatically; old batch=256 cells stay in the leaderboard
under their old keys for diff comparison. **No path collisions.** Analyses
in `experiments/cN_*/analysis.py` should filter the leaderboard to
canonical-config cells via the `temp_bench.report.canonical_train_keys`
helper (added 2026-05-04 PM):

```python
from temp_bench.report import canonical_train_keys, query_leaderboard

valid = canonical_train_keys(
    component="c5",
    archs=["txc_base", "txc_pro", "tsae_paper"],
    seeds=(1, 2, 42),
    datasource_names=["gemma_2_2b_it_l13_fineweb_24k128"],
)
rows = [r for r in query_leaderboard(component="c5") if r.train_key in valid]
```

The helper deterministically computes `train_key` for the canonical
sweep using the *current* `TrainingConfig()` defaults — so when the
defaults change again, every analysis.py automatically picks up the
new keys without a copy-paste churn. Pass an explicit `training_cfg=`
argument only when filtering for an alternate canonical config (e.g.,
a Bricken-on per-component override). Unknown archs / datasources are
silently skipped (lets you list the union without per-cell try/except).
agent_back's `_valid_train_keys` (commit ab02aea2) was the reference
implementation; the shared helper supersedes it.

**Cross-agent re-train directive**: every C3/C4/C5/C6/C7 cell needs new
`train_keys`. Workers update their per-component `_real_training_cfg` (or
just default-construct `TrainingConfig()` since the new values are the
default). Old cells stay; new cells write fresh. agent_em aborts in-flight
batch=256 calibration cells (~12 H100-hours of in-flight work
discarded — Han accepted the cost).

**Compute estimate** (assumes every cell hits the 25K cap, since
plateau-stop is off):
- C3 (24 cells, H100): ~40 H100-hr → 20 wall-hours on 2× H100 pod
- C4 (cache-hit on C3 checkpoints): ~free
- C5 (9 cells, A40): ~15 A40-hr
- C6 (~12 Wang cells, H100): ~20 H100-hr
- C7 (21 cells, A40): ~35 A40-hr
- Total: ~60 H100-hr + ~50 A40-hr → fits in remaining 72-h window with
  margin (parallel pods: ~30 H100 wall-hr + ~13 A40 wall-hr).

### 13. Two extra H100 pods → 100K-iter copy sweeps for C5 and C6

**Context**: Han spun up two additional 1× H100 pods (240 GB system RAM,
1 TB ephemeral /workspace each) on 2026-05-04 PM. The canonical paper
sweep runs at `n_steps=20_000` (Gemma axis: C3/C4/C5) and
`n_steps=25_000` (Qwen axis: C6) — both well below published-SAE-paper
budgets (T-SAE ~4-8B, TFA 1B, Phase 7 ~100M). The new pods give us the
compute headroom to run the same sweeps at `n_steps=100_000`
(~102M tokens, comfortably in the field-standard range).

**Decision** (with Han, 2026-05-04 PM): spin up two new agents that are
**literal copies** of agent_em and agent_steer with the only difference
being `n_steps=100_000`:

- **agent_em_100k** — replicates agent_em's C6 sweep verbatim. Same
  archs (sae_arditi + txc_base with brickenauxk_a8), same Wang full
  protocol, same Sonnet judge, same datasource
  (`qwen_2_5_14b_instruct_finance_l24_resid_post`). Seeds {42, 1}
  (matches agent_em's reduced n=2 sweep). 1× H100 ephemeral pod.
- **agent_steer_100k** — replicates agent_steer's C5 sweep verbatim.
  Same archs (tsae_paper + txc_base + txc_pro), same V7
  tiled-broadcast steering protocol, same Sonnet judge, same
  datasource (`gemma_2_2b_it_l13_fineweb_24k128`). Seeds {42, 1, 2}
  (full n=3 since H100 is fast enough). 1× H100 ephemeral pod.

**Cells coexist cleanly in the leaderboard**: `n_steps` is in the
`train_key` hash via `TrainingConfig.model_dump()` (`config.py:181-193`),
so 20K/25K and 100K cells occupy distinct keys. No path collisions.
Old short-schedule cells stay in `leaderboard.jsonl` for diff comparison.

**Whichever sweep completes first becomes the paper headline**:

- If 100K cells finish before deadline → agent_paper picks them as
  canonical. C5/C6 AUTO-RESULTS render from the 100K cells; the
  short-schedule cells become a "compute-pressure backup" reference.
  Within-component fairness is preserved because every arch in the
  selected sweep is at 100K.
- If 100K sweep is mid-flight at deadline → the short-schedule cells
  stay canonical. Partial 100K cells are kept in the leaderboard for
  a "convergence consistency" caveat in the paper.

**The toggle is mechanical**: agent_paper updates the `training_cfg=`
argument passed to `canonical_train_keys()` in `c5_steering/analysis.py`
and `c6_em/analysis.py` to either `TrainingConfig(n_steps=20_000)` /
`TrainingConfig(n_steps=25_000)` (short-schedule canonical) or
`TrainingConfig(n_steps=100_000)` (long-schedule canonical). One-line
filter swap; the helper does the rest.

**Within-component fairness invariant**: under no circumstances do we
mix 20K and 100K cells in the same C5/C6 AUTO-RESULTS table. The
short-schedule and long-schedule sweeps are separate canonical
universes; one is picked as headline at paper-render time.

**Cross-component independence**: only C5 and C6 get the 100K copy
sweep. C3/C4 (agent_nlp, in-flight) and C7 (agent_back, in-flight)
stay at their respective short schedules — Han's framing is that the
original four agents have already adopted the canonical 20K/25K
schedules and there's no benefit to spinning up additional pods for
those components when their sweeps are mid-flight.

**Compute estimate**:
- agent_em_100k: 2 archs × 2 seeds × ~4 hr/cell (training + Wang full)
  = ~16 H100-hr serial = ~16 wall-hr on the dedicated pod.
- agent_steer_100k: 3 archs × 3 seeds × ~2-4 hr/cell (training + V7
  steering + judge) = ~18-36 H100-hr serial = ~18-36 wall-hr on the
  dedicated pod. txc_pro is the long pole; if compute tight, drop
  seed=2 last.

Both fit in the remaining ~30 hr sprint window with margin.

**Worker briefings**:
- `agents/agent_em_100k/briefing.md`
- `agents/agent_steer_100k/briefing.md`

Both briefings emphasize: (a) re-use agent_em / agent_steer plumbing
verbatim via imports; do not modify their `experiments/cN_*/` code,
(b) write a small driver in `experiments/cN_100k/run.py` that
constructs `TrainingConfig(n_steps=100_000)` and calls the canonical
`runner.run_cell` with the inherited train_fn / eval_fn, (c) cells
land in `leaderboard.jsonl` automatically; agent_paper handles the
canonical-toggle in analysis.py at paper-render time.

### 14. TXC training-FLOPs parity — multi-window sampling toggle [DEPRECATED 2026-05-05 PM]

> **DEPRECATED 2026-05-05 PM (Han + agent_paper).** The MW deployment was
> directionally correct on the FLOPs asymmetry diagnosis, but the wrong
> fix. Closer reading of SAEBench (papers/are_saes_useful.md, App. B)
> showed canonical SAE training is buffer-based, batch=2048 TOKENS/step,
> ~500M tokens total. C3/C5's sequence-based pattern over-batches per-token
> archs by ~5× vs SAEBench (131K vs 2K tokens/step); C6/C7's window-based
> pattern at T=1 is near-canonical (1K). The right fix is to bring the
> per-token baselines DOWN to T=1 window-based (matching SAEBench +
> matching C6/C7), NOT bring TXC up via MW. See § 15 for the replacement
> directive.
>
> **All 4 MW pivots (agent_em C6, agent_em_100k C3, agent_filler C5,
> agent_steer_100k C7) are ABORTED as of 2026-05-05 PM** (commit
> `dd5f773e`). The YAML aliases `txc_base_mw` + `txc_pro_mw`, the
> `multi_window: bool = False` kwarg on `TXCBase` / `TXCPro`, and the
> per-component MW driver scaffolding stay registered as **inert reserves**
> — code is dormant unless explicitly invoked, no behavior change for any
> in-flight cell. Post-paper revisitation is fine; for the sprint we do not
> launch them. The 1 C5 MW cell that landed (eval_key `963df9c69213f998`,
> agent_steer_100k earlier today) and any partial C6/C3 MW rows from
> aborted sweeps stay in `leaderboard.jsonl` — `canonical_train_keys`
> filters them out at paper-render time, harmless.
>
> The remainder of this section is the original MW-deployment writeup,
> kept verbatim for git provenance. Do not act on it.

**Context** (Han caught 2026-05-05): per-step FLOPs differ ~25× between
per-token archs and TXC at the canonical `(batch=1024, n_steps=20K)`
because of how each arch consumes the input batch.

- **Per-token archs** (TopK SAE, T-SAE, MLC, TFA, SAE-arditi, Stacked):
  flatten the `(B, seq_len, d_in)` batch to `(B*seq_len, d_in)` and run
  the autoencoder per token-vector → 131,072 tokens trained on per step
  at B=1024, seq_len=128.
- **TXC archs** (txc_base, txc_pro): `train_step` samples ONE random
  T-window per batch row, giving `(B, T, d_in)` effective rows → 5,120
  token-windows per step at B=1024, T=5.

Net: at the same `(B, n_steps)` the per-token archs see **~25× more
training FLOPs and ~25× more token-vectors** than TXC. This biases
every cross-arch comparison **against** TXC. The headline C6 win
(TXC + Bricken vs SAE-arditi at 14B-finance, +3.24 align) actually
landed *despite* TXC being compute-disadvantaged by 25× — a stronger
result than the leaderboard surface suggests.

**Fix** (landed 2026-05-05): add a `multi_window: bool = False` kwarg
to both `TXCBase.__init__` and `TXCPro.__init__`. When False (default),
`train_step` keeps the original 1-random-window-per-row sampling —
**zero behavior change for in-flight cells**. When True, `train_step`
tiles each input sequence at stride T (TXC-base) or stride
`T_max + max_shift` (TXC-pro) into N non-overlapping window groups,
giving `B*N` effective rows per step. Token throughput then matches
per-token SAEs.

**Cache hygiene**: the kwarg flows through `arch.hparams` (when set
explicitly via `configs/locked_archs.yaml`) into `compute_train_key`'s
hash — verified by `tests/test_txc_multi_window.py` (6 tests, including
direct `compute_train_key` invalidation checks). Toggling False→True at
the YAML level cleanly invalidates all old TXC `train_keys`; new cells
write at fresh keys, old cells stay in the leaderboard for diff. Per-token
archs are unaffected because their hparams don't change. **Only TXC cells
need re-training when we flip the switch.**

**Deployment as separate registry entries** (revised 2026-05-05 PM with
Han): the toggle goes into the YAML by way of two NEW arch entries —
`txc_base_mw` and `txc_pro_mw` — pointing at the same Python classes
as `txc_base` / `txc_pro` but with `multi_window: true` baked into
hparams. **Not** a flip-the-toggle approach on the existing names.

Why separate entries instead of toggling the existing names:

- **Self-documenting leaderboard**: any row with `arch=txc_base_mw`
  is unambiguously a multi-window cell; any row with `arch=txc_base`
  is unambiguously the historical baseline. No need to cross-reference
  hparams to interpret a leaderboard row.
- **Robust against driver-script typos**: a worker writing a one-off
  `TrainingConfig(...)` cell can't accidentally produce a multi-window
  result under `arch_name="txc_base"` — the YAML enforces the
  correspondence between arch name and `multi_window` value.
- **Workers' arch lists become explicit**: their `archs=[…]` argument
  to `canonical_train_keys()` directly says which sweep is canonical
  for an analysis. Switching the canonical from old to new is a list
  edit, not a YAML edit at the policy level.

**YAML implementation** (landed 2026-05-05):

- `txc_base_mw` and `txc_pro_mw` added to `configs/locked_archs.yaml`,
  pointing at `temp_bench.architectures.txc_base:TXCBase` and
  `temp_bench.architectures.txc_pro:TXCPro` respectively. Both have
  `multi_window: true` in hparams.
- All `per_component_hparams` overrides (e.g. `c6: {d_sae: 32768,
  k_pos: 25}` for `txc_base`) are MIRRORED on the `_mw` entries.
  Drift risk: if someone updates `txc_base.per_component_hparams.cN`
  without also updating `txc_base_mw.per_component_hparams.cN`, the
  two archs diverge silently. YAML comment notes this; tests do not
  enforce parity.
- Decision #1 amended: the canonical paper TXCs are `txc_base_mw` and
  `txc_pro_mw`. The original `txc_base` / `txc_pro` stay in the
  registry as the historical pre-fix baseline; their existing cells
  remain in `leaderboard.jsonl`.

**Status (2026-05-05 PM)**: code + tests + YAML aliases landed.
Workers' in-flight sweeps are running on `txc_base` / `txc_pro` (the
historical baseline). The `_mw` archs are registered and ready to
use; the deployment plan (which workers run them, with what compute
budget, against which existing cells) is the next coordination
question with Han.

**Bricken composes orthogonally with `multi_window`** (clarification):
Bricken resample is configured via `TrainingConfig` (a per-cell
training-time choice — `bricken_enabled`, `ema_auxk_alpha`,
`dead_threshold_tokens` etc., per § 7). `multi_window` is an
architecture-level choice baked into the YAML hparams of `txc_base_mw`
/ `txc_pro_mw`. The two are independent.

For C6 (agent_em — TXC + brickenauxk_a8 recipe), the multi-window
canonical cell is therefore:

```python
arch_name = "txc_base_mw"          # NOT a new "txc_base_mw_brickenauxk" entry
training_cfg = TrainingConfig(
    bricken_enabled=True,           # per § 7 (C6 default)
    ema_auxk_alpha=0.125,            # 1/8 per the a8 recipe
    dead_threshold_tokens=128_000,
    bricken_resample_every=500,
    bricken_min_fires=1,
    bricken_n_check=2048,
    bricken_max_resample_fraction=0.5,
    # n_steps, batch_size, plateau_* per the canonical paper schedule
)
```

agent_em's existing `experiments/c6_em/train.py:make_training_cfg`
already builds this dict (the brickenauxk_a8 fields are conditioned
on the arch name). When we deploy, agent_em swaps the arch name they
pass to `make_training_cfg` from `txc_base` to `txc_base_mw`; nothing
in their training config logic needs to change. Same Bricken plumbing,
new sampling under the hood.

**Bricken resample-rate caveat under multi-window** (agent_em
decision-point at C6 MW deploy): `bricken_resample_every` is in step
count, not in token count. Under `multi_window=True` each step
processes N× more token-windows (N=10 for TXC-base at C6's
seq_len=128, T_max=10, max_shift=2). Two equivalent framings:

1. **Keep `bricken_resample_every=500`** (current default). Resample
   triggers every 500 steps regardless of mode. Under MW this is N×
   more aggressive intervention per token — dead features get caught
   ~10× faster (in token units) than under non-MW. Defensible: dead
   features ARE seen 10× more data per step, so 10× more frequent
   intervention is arguably correct.

2. **Scale up to `bricken_resample_every=5000`** (rate-equivalent to
   non-MW). Resample triggers at the same per-token rate as the
   historical baseline, so the Bricken behavior is invariant to the
   sampling-mode change. Cleaner if you want a strictly
   apples-to-apples comparison of "TXC + Bricken at multi-window vs
   single-window" at the algorithm level.

Other Bricken counters (`dead_threshold_tokens`, `bricken_n_check`,
`bricken_max_resample_fraction`) are already in token / fraction
units and don't need adjustment.

agent_em decides at C6 MW deploy time which framing they want; the
choice goes into `make_training_cfg` for the MW cells. The
historical non-MW C6 cells stay at `bricken_resample_every=500` (no
retroactive change). Document the choice in c6.md caveats.

**Estimated re-train cost when we deploy**: ~24 TXC cells across C3,
C5, C6, C7. Per-cell wall-time should be **roughly the same as
before** because the data path stays the same (preloaded `.clone()`
cache); the bigger encode shape (`(B*N, T, d) @ (T, d, s)`) better
utilizes the H100/A40 GPU but the absolute time per step is dominated
by data + matryoshka/contrastive losses, not the encode FLOPs
themselves. The multi-window deployment recovers the FLOPs-parity we
should have had, without proportionally increasing wall-time. ~30-50
GPU-hr total distributed across the 6 worker pods.

### 15. Literature-aligned T=1 baseline re-train (replaces § 14)

**Context** (Han + agent_paper, 2026-05-05 PM): § 14's MW deployment was
solving the right diagnosis ("per-step training-FLOPs asymmetry between
TXC and per-token archs at C3/C5") with the wrong fix ("bring TXC up to
match per-token throughput via stride-T tiling"). Closer reading of the
SAE-comparison literature shows the per-token baselines were *over*-
batched against the field standard, not the other way around:

- **SAEBench** (papers/are_saes_useful.md, App. B): "We use a batch size
  of 2048 tokens... we train each SAE on approximately 500M tokens of
  activations." Buffer-based, batch in TOKENS, ~250K steps total.
- **T-SAE paper** (papers/temporal_sae.md, §4.1): "All SAEs are trained
  with the BatchTopK activation... batch size 4096 tokens... 500M
  activation tokens." Same shape.
- **TFA paper** (priors_in_time.md, App. B.1): "1B precached activations,
  batch size 1024 tokens." Same shape.

Our two patterns and where they sit relative to literature:

| Component | Pattern                                                                | per-step tokens | vs SAEBench 2K |
|---|---|---:|---:|
| C3, C4, C5 | sequence-based: `(B, seq_len, d_in)` → flatten in arch's `train_step` | **131,072** (B=1024 × seq_len=128) | 65× over |
| C6, C7    | window-based: `(B, T, d_in)` with `T=1` for SAE archs                  | **1,024** (B=1024 × T=1)  | within 2× |

The C6 + C7 patterns came from agent_em / agent_back independently
adopting `_build_batch_iter` with a per-row T-window slice. They were
"right by accident" — close to SAEBench's canonical scale. The C3 + C5
patterns came from agent_nlp + agent_steer using the canonical
`preloaded_batch_iter_from_act_cache` which returns whole sequences for
the per-token archs to flatten. They were "wrong by inheritance" — 65×
over canonical.

**Decision** (Han, 2026-05-05 PM): re-train the C3 + C5 per-token
baselines at the per-arch literature-faithful window size:

| Arch          | `train_window_size` | tokens/step | Reference |
|---------------|---:|---:|---|
| `topk_sae`    | **1** | 1024 | vanilla TopK, no temporal — matches C6's sae_arditi at T=1 |
| `tsae_paper`  | **2** | 2048 | Bhalla/Ye 2025 §3.1 "load activations in **pairs** $(\mathbf{x}_t, \mathbf{x}_{t-1})$" — exact paper match |

Both within 2× of SAEBench's 2K canonical scale, each arch trained at
its own paper's intended setup. C6 baselines (`sae_arditi` at T=1) are
unchanged. **C7 T-SAE keeps T=5** — agent_back's `_spec_window_size`
fallback in `experiments/c7_backtracking/run.py:82-97`. Han 2026-05-05
PM: "for C7 can leave at T=5; both need to be supported." So
`tsae_paper` at C3/C4/C5 trains at T=2, while `tsae_paper` at C7 stays
at T=5 — two different `train_keys`, both live in the leaderboard,
component-specific. TXC archs at every component are unchanged (they
sample 1 random T-window per row already, regardless of mode).

**Framework change** (commit `5555e7eb`):

- `temp_bench.data.nlp.cache.preloaded_batch_iter_from_act_cache` gains
  `train_window_size: int | None = None`. ``None`` preserves current
  behavior (full sequences); ``int T`` returns 1 random T-window per row,
  shape `(batch_size, T, d_in)` — vectorised gather, RAM-rate per call.
- `TrainingConfig.train_window_size: int | None = None` field, plumbed
  into `compute_train_key` via `model_dump(exclude_none=True)`. Default
  (None) preserves all existing train_keys; setting an int gets a fresh
  key, so the re-train doesn't cache-hit on the over-batched cells.
- 5 new tests landed (2 in `test_cache_keys.py`, 3 in
  `test_preloaded_batch_iter.py`). 131 → 136 green.

**Re-train scope** (one helper agent each):

- **C3 baseline re-train** — split between **agent_nlp** (TopK) and
  **agent_em_100k** (T-SAE). Han 2026-05-05 PM: "agent_nlp's other
  H100 is free since agent_em is idle; therefore agent_nlp and
  agent_em_100k should work TOGETHER to recover the baselines."
  - **agent_nlp** owns C3 already; takes `topk_sae × 3 seeds × 2
    k_feats` (3 unique trainings + 6 evals) at
    `TrainingConfig(n_steps=20_000, train_window_size=1)`. Their
    existing in-flight `topk_sae` sweep at T=None finishes as the
    diff-reference baseline (~20:21Z); the new T=1 sweep runs on
    GPU 1 (idle, agent_em's spare) immediately, with GPU 0 joining
    after their old sweep wraps. Per-cell wall on H100: ~10-15 min
    train (T=1 → 1024 tokens/step vs 131K full-sequence) + ~30 min
    probing eval × 2 k_feats = ~1.2 hr. 3 cells × 2 GPUs in parallel
    → ~2.4 hr wall.
  - **agent_em_100k** repurposed from the now-aborted C3 MW pivot;
    takes `tsae_paper × 3 seeds × 2 k_feats` (3 unique trainings +
    6 evals) at `TrainingConfig(n_steps=20_000, train_window_size=2)`.
    Per-cell wall on H100: ~15-20 min train (T=2; T-SAE encodes
    anchor + temporal pair, so per-step encoder load = 2048 tokens)
    + ~30 min eval × 2 k_feats = ~1.5 hr. 3 cells serial on 1 H100
    → ~4.5 hr wall.
  - Total C3 baseline re-train: 6 trainings + 12 evals across 2
    agents; wall = max(agent_nlp, agent_em_100k) ≈ ~4.5 hr.
  - MLC unported per § 11 (appendix-only); skipped.
- **C5 T-SAE baseline re-train** (assigned to **agent_filler**,
  repurposed from the now-aborted C5 MW parallel sweep): `tsae_paper`
  × 3 seeds at `TrainingConfig(n_steps=20_000, train_window_size=2)`.
  3 cells. With 8× A40 in parallel, 3 cells simultaneously → ~2-3 hr
  wall. Other C5 archs (txc_base, txc_pro) are unchanged.
- **C4** (qualitative latents): cache-hits on the new C3 checkpoints
  via `train_key`. agent_nlp re-runs C4 evals once C3 baselines land.

**Owner agents notified** (agent_nlp owns C3+C4, agent_steer owns C5):

- agent_nlp's `experiments/c3_probing/analysis.py` already filters
  through `canonical_train_keys()` keyed off `TrainingConfig`. To
  pick up the new per-arch T cells in the headline, the canonical
  filter needs THREE calls (one per training_cfg) unioned:
  - TXC: `TrainingConfig(n_steps=20_000)` for `txc_base, txc_pro`
  - TopK: `TrainingConfig(n_steps=20_000, train_window_size=1)` for `topk_sae`
  - T-SAE: `TrainingConfig(n_steps=20_000, train_window_size=2)` for `tsae_paper`
  The helper accepts a list-of-archs + single training_cfg; agent_nlp
  calls it three times and unions the returned sets. Old over-batched
  cells stay in the leaderboard for diff comparison.
- agent_steer's `experiments/c5_steering/analysis.py` similar — only
  `tsae_paper` rows shift to `train_window_size=2`; the existing TXC
  cells stay unchanged. Two `canonical_train_keys` calls (TXC + T-SAE),
  union.
- agent_back's C7 T-SAE cells **stay at T=5** (their existing
  `_spec_window_size` fallback). C7 T-SAE has no `train_window_size`
  in its TrainingConfig — the windowing is determined by the
  experiment driver, not the schema. So C7 T-SAE cells preserve
  their existing `train_keys` and don't need re-running.
  agent_back's analysis.py is unaffected.

**Expected paper-claim shifts** (uncertain pending re-runs):

- **C3** (per-token archs over-batched 65×, now correctly batched 1×):
  TopK-SAE and T-SAE peak AUCs likely DROP under canonical training.
  Today's C3 headline (TopK > TXC > T-SAE on probing) may flip to
  TXC > TopK / T-SAE — or, more conservatively, to a tighter cluster.
  Either way the result becomes more reviewer-defensible: "we
  trained every arch at SAEBench's canonical compute scale."
- **C5** (only T-SAE re-trained at T=1): T-SAE peak success grade
  may drop slightly. Today's C5 headline (T-SAE > TXC-pro > TXC-base
  on peak grade @ coh ≥ 1.75) may shift toward TXC ties at high coh,
  the original hypothesis. Or may stay refuted, just with a
  literature-aligned baseline.
- **C6 + C7**: no shift expected — these baselines were already at
  the right scale.

**Within-component fairness invariant**: under no circumstances do we
mix sequence-based and T=1-window cells in the same C3 / C5
AUTO-RESULTS table. The re-trained per-token cells become the
canonical baselines; old over-batched cells become the diff-only
"pre-fix" reference (analogous to how decisions § 12's batch=256 cells
are kept). `canonical_train_keys` enforces this by filtering on the
explicit `TrainingConfig(...)` argument — workers update the call
signature in their `analysis.py` to match the new training cfg.

**Cross-component independence**: C6 + C7 are not re-running anything.
agent_em's canonical 8/8 sweep stands as the C6 paper headline;
agent_back's canonical v4 sweep stands as the C7 paper headline.
agent_em_100k and agent_filler's only mission is the per-token baseline
re-train at C3 + C5; everything else is paper-writing or wrap-up.

**Compute estimate**: 9 unique trainings + 15 evals total. Worst-case
~12-13 hr wall on the worker pods (mostly C3's serial 6-training
schedule). Fits comfortably in the remaining sprint window.

**Bricken composes orthogonally (clarification, same as § 14)**: Bricken
resample is a `TrainingConfig` choice (`bricken_enabled`,
`ema_auxk_alpha`, etc.). `train_window_size` is also a `TrainingConfig`
choice. They coexist in the same config; setting both is allowed if a
component decides to. C6 keeps Bricken on for TXC + brickenauxk_a8 (per
§ 7); C3/C5 keep Bricken off for the baseline re-train (per § 7).

### 16. Add MLC + TFA as paper-faithful baselines at C3, C5, C6

**Context** (Han 2026-05-05 PM, post-§ 15 baseline re-train completion):
the C3 + C5 baseline coverage was thin — TopK + T-SAE only. Adding MLC
+ TFA strengthens both the C3 sparse-probing comparison (C3 paper-claim
defensibility) and C5 steering (provides a temporal-aware non-TXC
baseline). C6 EM also benefits from a TFA comparison since TFA's
predictive component aligns naturally with the "stable semantic
features" hypothesis there.

The two baselines' paper-faithful training conventions differ:

**MLC** (Multi-Layer Crosscoder, layer-axis analog of TXC):
- Native data format: ``(B, L, d_in)`` where L = number of adjacent
  layers (paper default L=5).
- Wasteland reference (`94119bc0:src/architectures/mlc.py` +
  `experiments/phase7_unification/train_phase7.py:246-260`):
  shared latent across L layers; per-step encoder load = B × L tokens.
- For paper-faithful training at C3, we need a 5-layer Gemma cache.
  Build via the new multi-layer activation cache framework
  (commit `d859daef`):
  - New datasource `gemma_2_2b_it_l11to15_fineweb_24k128`
    (`layers: [11,12,13,14,15]`).
  - `build_activation_cache` detects `layers: list[int]` and registers
    N forward hooks in one model pass → captures (N, L=5, seq_len,
    d_in) into a single .npy. Cost: ~70 GB on disk, ~3 H100-hours to
    build.
  - `preloaded_batch_iter_from_multilayer_cache(act_cache_key, seed)`
    samples 1 random (seq, token) per row → returns (B, L=5, d_in).
  - `build_probe_cache` extended analogously for MLC eval: per-task
    X arrays at (N, L=5, S=32, d_in). Same multi-layer hooks.
- TrainingConfig: `B=1024, n_steps=20_000, train_window_size=None`
  (the L axis lives outside the train_window_size system; MLC's data
  shape is determined by the multi-layer datasource).
- Per-step encoder load: B × L = 5120 tokens (similar to TXC at T=5).

**TFA** (Temporal Feature Analysis):
- Native data format: ``(B, T_seq, d_in)`` where T_seq is the full
  sequence length. TFA's attention attends over preceding context
  tokens; T_seq=128 is the design intent.
- **Wasteland-faithful training** (`94119bc0:experiments/phase7_unification/train_phase7.py:312-353`):
  ```
  TFA_BATCH = 32
  Phase 5 default tfa_batch_size = 64
  gen(B): returns full sequences (B, L=128, d) — sample whole context
  Per-step tokens: 32 × 128 = 4096 (close to SAEBench's 2K canonical)
  ```
  TFA's per-step memory is heavy (B × T × d_sae attention tensor at
  fp32 ≈ 36 GB at B=4096, T=128, d_sae=18432). Phase 7 used B=32 to
  fit within an H200 holding multilayer buffers; we reuse that
  convention. The wasteland TFA paper (priors_in_time.md, App. B.1)
  doesn't pin a specific batch but shows context > 100 tokens helps
  reconstruction (Fig. 2(d) — variance explained increases to 80% at
  ~500 tokens of context).
- **TrainingConfig per-arch override**: `B=32, n_steps=20_000,
  train_window_size=None` (full sequence). The B=32 is the only
  cross-arch B exception (decisions § 12 notes "uniform across pods"
  was for the non-TFA baselines; TFA was identified earlier as a
  per-arch outlier in Phase 7 too).

**Re-train assignments** (all per-arch literature-faithful):

| Agent | Component | Arch | TrainingConfig | Notes |
|---|---|---|---|---|
| agent_em_100k | C3 | MLC × 3 seeds | `B=1024, train_window_size=None`, datasource=l11to15 | Build 5-layer cache first (~3 H100-hr); also extend probe_cache to multi-layer for eval |
| agent_nlp | C3 | TFA × 3 seeds × 2 k_feats | `B=32, train_window_size=None`, datasource=l13 (existing) | Wasteland-faithful B=32 + full seq |
| agent_filler | C5 | TopK × 3 seeds + TFA × 3 seeds | TopK: `B=1024, train_window_size=1`; TFA: `B=32, train_window_size=None` | 6 cells parallel on 8 A40s; ~1.25 hr wall |
| agent_steer_100k | C6 | TFA × 2 seeds × 2 organisms | `B=32, train_window_size=None`, datasources: 14B-finance + 7B-medical | 4 cells serial on 1 H100; ~12-16 hr |

**Cross-arch B uniformity note** (decisions § 12 amendment): TFA is the
single arch with a B=32 override (paper-faithful per-arch convention);
all other archs continue at B=1024. This is the same exception we'd
make for any arch whose per-step memory pressure precludes B=1024 at
the d_sae we use. Documented in c3.md / c5.md / c6.md caveats so
reviewers see the per-arch B is a deliberate paper-faithful choice,
not a config drift.

**Within-component fairness invariant**: every arch within a component
trains for ``n_steps=20_000``. Per-step token throughput VARIES per
arch (TopK 1024, T-SAE 2048, TFA 4096, TXC 5120, MLC 5120) — each at
its source paper's intended setup. Total-token-budget per arch ranges
from 20.5M (TopK) to 100M (MLC/TXC). This isn't strict token-count
fairness; it's "each arch trained at its paper's intended per-step
scale, for the same number of gradient steps." Reviewer-defensible
because every arch's training matches its primary reference.

**No re-train for already-completed cells** (§ 15 stays):
- C3 TopK T=1 (agent_nlp, `9b9d6cc5`)
- C3 T-SAE T=2 (agent_em_100k, `82674a75`)
- C5 T-SAE T=2 (agent_filler, `3a654fab`)
- C6 baselines (sae_arditi T=1 + TXC T=5; agent_em's canonical 8/8)
- C7 baselines (all at T=5; agent_back's v4)
- All TXC cells (every component, T=5 + max_shift)

### Non-decisions (to revisit later)

- **MLC scope** — competitive with TXC-base at C3 k=5. Include as related
  work / appendix? Decide before the paper goes to draft.
- **A third agent on the A40 pod** — slot is open. Could be a "synthetic
  helper" agent that runs C1/C2 multi-seed at larger scale (50+ features).
  Defer until C3/C4/C7 land.
- **Bumping the 25K cap if loss is clearly still descending** — agents
  should observe the loss curves on the first round of cells. If multiple
  archs end at 25K with loss still falling steeply (e.g., final-1K-step
  drop > 5% of the loss value), revisit the cap. Any bump must be uniform
  across all archs (fairness).
