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

### 14. TXC training-FLOPs parity — multi-window sampling toggle

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

**YAML convention** (important — avoid a footgun):

- Do NOT add `multi_window: false` to YAML hparams. The default kwarg
  is False; the absence of the key produces the same behavior, and the
  hash includes only present hparams (so no key = same hash as before).
- When ready to flip, ADD `multi_window: true` to the `hparams` block
  for `txc_base` and `txc_pro` in `configs/locked_archs.yaml`. This
  changes the hash → invalidates old cells.

**Status (2026-05-05)**: code + tests landed. YAML NOT YET FLIPPED —
default off, in-flight runs untouched. The flip is a coordinated
worker-team event:
1. agent_paper edits `configs/locked_archs.yaml` to set
   `multi_window: true` for both TXC archs.
2. Workers pull → their `canonical_train_keys()` filter now points at
   new train_keys → AUTO-RESULTS auto-filters out the old TXC cells.
3. Workers re-run TXC cells only (per-token archs are unaffected; their
   train_keys are stable).
4. agent_paper updates this § with the flip date + cell-count summary.

**Estimated re-train cost when we flip**: ~24 TXC cells across C3, C5,
C6, C7. Per-cell wall-time should be **roughly the same as before**
because the data path stays the same (preloaded `.clone()` cache);
the bigger encode shape (`(B*N, T, d) @ (T, d, s)`) better utilizes
the H100/A40 GPU but the absolute time per step is dominated by data
+ matryoshka/contrastive losses, not the encode FLOPs themselves. So
the multi-window flip recovers the FLOPs-parity we should have had,
without proportionally increasing wall-time. ~30-50 GPU-hr total
distributed across the 6 worker pods.

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
