# Camera-ready evidence and figure workspace

Prepared 2026-09-27. Start here before editing manuscript numbers or figures.
The paper checkout has been fast-forwarded to Overleaf `b27b3fc`; this audit
does not revise its scientific content. The outer branch was synchronized
with `origin/neurips-aniket` at `00b45a8a7` before the archive was added.

The recoverable experiment code and compact numbers are now collected in
[`../../../artifacts/camera_ready_2026/`](../../../artifacts/camera_ready_2026/).
The manifest verifies 805 copied source files (25,140,985 bytes), alongside
127 already tracked native artifacts indexed without duplication.
The archive is **not complete experimental reproduction**: the final Shamir
run, private Stacked payloads, and some original figure inputs remain missing.
Do not turn a readable summary table into an assertion that its raw run was
verified. No activation caches, datasets, model checkpoints, or new training
runs were needed. Only 2,675,556 bytes of JSON/Markdown were downloaded from
Hugging Face; the remaining recovered files came from existing local Git
objects and the small Overleaf update.

## Where to look

| Need | Entry point |
|---|---|
| What was accessible on OpenReview | [Access receipt](openreview-access.md) |
| Prioritized manuscript changes, with exact lines | [Manuscript audit](manuscript-audit.md) |
| Experiment identity, corrected results, and missing payloads | [Rebuttal artifact audit](rebuttal-artifact-audit.md) |
| All 24 active figure inputs and cheapest redraw path | [Figure source audit](figure-source-audit.md) |
| Actually posted numerical tables, preserved without correction | [Nine OpenReview tables](../../../artifacts/camera_ready_2026/posted_rebuttal_tables.json) |
| Recovered branch/HF files, immutable revisions, byte counts, SHA-256 | [Source manifest](../../../artifacts/camera_ready_2026/manifest.json) |
| Already tracked native code/results, indexed without duplication | [Native artifact index](../../../artifacts/camera_ready_2026/native_artifact_index.json) |
| Small real-task rows from Han's rebuttal branch | [472 verbatim probing/RLHF/EM rows](../../../artifacts/camera_ready_2026/derived/arxiv_real_task_leaderboard.jsonl) |
| Small files to request from Dmitry | [Recovery handoff](dmitry-artifact-request.md) |

`sources/<branch>/...` retains each source's original layout and content.
These are historical snapshots, not replacements for today's package APIs.
The manifest gives the original commit for every copied file; code presence
does not imply its training inputs were downloaded or that every historical
script runs unchanged in the current environment. `sources/paper/` contains
the current plotting scripts, numeric sidecars, and editable diagram sources.

## Scientific decisions to settle first

1. **Reconcile the Backtracking experiment identities.** The posted
   0.27/0.27/0.26 row matches rounded values from the older 20K window sweep
   at T=6, S=32; that source attribution remains an inference from the
   numerical match. Corrected 300K TXC-base at S=8 is
   0.1874 ± 0.0080, below the two T-SAE references. At S=32 it is
   0.2568 ± 0.0048. Retain the full budget curve, distinguish TXC-base from
   TXC-pro, and keep detection and steering replications separate.
2. **Replace the abstract's broad performance percentages.** The 40% figure
   is a selected-cell relative PR-AUC increase, not a detection-rate
   improvement, and combines a batch-256 TXC-pro cell with a batch-1024 SAE.
   The submitted steering winner is TXC-base. Report named metrics,
   variants, budgets, seeds, and absolute values from one canonical table.
3. **Show what temporal evidence actually establishes.** Add the completed
   Stacked control, context-length curves, and shuffled controls prominently.
   A fixed-probe shuffle gap is representation sensitivity under distribution
   shift; it is not proof that learned temporal order caused the gains.
   The Shamir information bound says when the task becomes recoverable,
   not that TXC is the only architecture that could recover it.
4. **Make robustness and negative results visible.** Use actual training
   seeds rather than treating folds or bootstrap samples as seeds.
   Increasing backtracking counts does not establish improved reasoning:
   the existing rescue and coherence tables are relevant main-text context.
   Preserve the negative EM/HH-RLHF results and disclose unmatched realized
   sparsity in Stacked EM.
5. **Repair implementation/reporting contradictions.** Resolve the 36/38
   probing-task count, TXC-base/pro labels, dictionary widths, evaluation
   windows, seed checklist, bias and matrix conventions, and conference
   citations. A compact architecture/protocol table should replace the
   blanket claim that all models were matched.

The probing task-count issue is numerical: the saved curves match the
38-task aggregation, not the declared 36-task subset. The recovered
per-task, per-seed rows are sufficient to recalculate the intended subset
and seed-aware integrated uncertainty on CPU; no model rerun is needed.

The medical TXC steering row 17/20/23 is supported by recovered exact
extrema 16.71875/20.359375/22.875, including the completed seed-2 aggregates.
The separate Stacked sprint also finished EM steering after the posted
response: its summary reports 13.3 versus TXC 17.1 on a common grid. These
are distinct summaries; do not compare 13.3 directly with the wider-grid
17/20/23 row. The raw Stacked payload remains in an inaccessible private
archive.

## Cheap figure workflow

From the outer repository root:

```bash
python3 purified/scripts/camera_ready/collect_sources.py --verify
python3 purified/scripts/camera_ready/replot.py
```

The second command uses saved numbers only and writes PNG/SVG previews plus
source tables to `purified/artifacts/camera_ready_2026/previews/`. It requires
NumPy and Matplotlib, not PyTorch, a GPU, a model download, or API judging.
Its shared palette is in `purified/scripts/camera_ready/palette.json`.
The previews demonstrate reproducibility; they are not replacements for the
paper's final figures or a claim that unresolved experiments are validated.

The available inputs support immediate redraws of synthetic recovery,
sparse-probing curves/heatmaps, both Backtracking window sweeps, corrected
300K detection, C8 feature statistics, cached UMAP coordinates, medical
three-seed summary points, and optional writing-revision results. Historical
C7 continuous curves can be redrawn only to the precision of their saved
rounded tables; exact count tables retain integer precision. Some original
bootstrap intervals, final C6 extension curves, and the exact sentence
activation heatmap need the small exports listed in the handoff.

For the overhaul, replace the rose chart with aligned metric panels: its
current inputs include an old C7 value and a different RLHF endpoint.
Use one architecture palette, stable method ordering, markers as a second
visual channel, explicit seed points, and vector exports. Rework the editable
TikZ architecture figure to show the same input window under token SAE,
Stacked SAE, and TXC, with position-specific weights and the shared latent
clearly distinguished. Add a benchmark/label diagram for the exact
pre-onset window. Neither diagram requires experimental data.

Do the provenance/numerical corrections and figure redraws before deciding
whether new experiments are necessary. The present task did not run any.
