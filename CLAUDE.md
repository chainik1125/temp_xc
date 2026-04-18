# Temporal Crosscoders

Research repository combining documentation (Obsidian vault) and Python code for temporal crosscoder experiments.

Welcome. You are an AI agent working on a research project to develop temporal SAE architectures that can discover features spanning multiple token positions in language models. This document will get you up to speed on the motivation, background, codebase, and working conventions.

## Repository structure

- `src/` -- reusable backend (architectures, data, training, eval, plotting, pipeline, utils)
- `experiments/phaseN_<shortname>/` -- phase-specific one-off experiment scripts + their `results/` output
- `tests/` -- pytest test suite (covers `src/` only)
- `references/` -- vendored paper code (e.g. TemporalFeatureAnalysis); reference-only, not imported
- `papers/` -- markdown paper summaries for agents to ingest
- `docs/` -- Obsidian documentation vault (research logs, project brief)
- `docs/han/research_logs/phaseN_<shortname>/` -- per-phase briefings, plans, experiment writeups, and summaries; `<shortname>` must match the `experiments/` sibling
- `docs/templates/` -- document templates (Base.md, Guide.md)
- `docs/Tags.md` -- tag taxonomy index
- `run-checks.sh` -- runs all quality checks locally
- `check-tags.sh` -- validates tags in documentation
- `get-tags.sh` -- extracts tags from a markdown file
- `.markdownlint.jsonc` -- markdown linting rules
- `pyproject.toml` -- Python project configuration

**Layer rule:** `experiments/` may import from `src/*`; `src/*` never imports from `experiments/`. The filesystem split enforces this — keep it that way.

**Adding a new architecture:** drop a new file in `src/architectures/`, subclass `ArchSpec`, register it in `src/architectures/__init__.py`. It then plugs into both toy and NLP pipelines automatically.

## Phase convention

Each research phase has a **paired** directory at two locations, and they must match:

- `experiments/phaseN_<shortname>/` — scripts that run the phase's experiments and the `results/` subtree those scripts write to.
- `docs/han/research_logs/phaseN_<shortname>/` — writeups for the phase: briefings, pre-registered plans, experiment logs, and summaries.

The `<shortname>` slug must be identical across the two paths (e.g. `phase4_nlp_comparison` appears under both `experiments/` and `docs/han/research_logs/`). This lets any agent locate a phase's code and its documentation from knowing either one.

**Canonical filenames within a phase's research-log dir:**

- `brief.md` — pre-phase briefing: context, motivation, candidate directions, architecture menu, open questions. Written before the phase starts. The target audience is a fresh agent picking up the phase cold.
- `plan.md` — pre-registered experimental plan (hypotheses, success criteria, figures to produce). Written before any checkpoint is trained.
- `YYYY-MM-DD-<topic>.md` — individual experiment writeups. One file per experiment.
- `summary.md` — end-of-phase synthesis across all experiments.

When starting a new phase, create the paired `experiments/phaseN_*/` and `docs/han/research_logs/phaseN_*/` directories simultaneously and draft `brief.md` first.

## Environment setup

The repo uses a single **`uv`-managed virtualenv** at `.venv/`, reproducible on any machine (local, RunPod, CI). Python ≥ 3.12 + PyTorch CU128.

Install `uv` if missing: https://docs.astral.sh/uv/getting-started/installation/ — standalone binary, no system deps.

**On RunPod:** see `RUNPOD_INSTRUCTIONS.md` at the repo root for persistent-volume paths, HF/GitHub/Anthropic auth, and reconnection workflow.

First-time setup (clones a fresh `.venv` from `pyproject.toml` + `uv.lock`):

```bash
uv sync
```

From then on, run commands via `uv run` (auto-activates the venv) or directly through `.venv/bin/python`:

```bash
uv run python src/experiments/analyze_activation_spans.py
# or
.venv/bin/python src/experiments/analyze_activation_spans.py
```

Dependency source-of-truth is `pyproject.toml`. `uv.lock` pins exact versions. To add a package:

```bash
uv add <package>         # adds to [project.dependencies] and relocks
uv add --dev <package>   # adds to [dependency-groups.dev]
```

Do not `pip install` into the venv directly — it'll drift from the lockfile.

## Running code

All scripts run from the repository root. Example:

```bash
uv run python src/experiments/toy_model.py
```

### Disabling progress bars

**You must disable tqdm progress bars** when running code. Set the following environment variable before execution:

```bash
export TQDM_DISABLE=1
```

Or inline:

```bash
TQDM_DISABLE=1 uv run python src/experiments/toy_model.py
```

This keeps logs clean and avoids flooding output with progress bar updates.

### Saving figures

Save all figures as both `.png` and `.html` (for interactive plotly figures) to a `results/` directory within the relevant experiment folder. Use descriptive filenames that include the experiment name and key parameters, e.g. `toy_model_l0_sweep_corr0.4.png`.

**Use `save_figure()` from `src/utils/plot.py`** instead of calling `fig.savefig()` directly. This saves both a high-res PNG (150 DPI) and a low-res thumbnail (`.thumb.png`, ≤288px wide, 48 DPI):

```python
from src.utils.plot import save_figure
save_figure(fig, os.path.join(results_dir, "my_plot.png"))
# → my_plot.png       (high-res, for docs)
# → my_plot.thumb.png (thumbnail, for agent inspection)
```

**AI agent rule:** When you need to inspect a plot, **always read the `.thumb.png`** file, never the full-res `.png`. Full-res images can be many megabytes and will blow up the context window. Markdown docs should link to the full-res `.png` for human readers.

## Rules and conventions

1. **Disable tqdm.** Always set `TQDM_DISABLE=1` before running any Python code.
2. **Read recent research logs first.** Before starting any task, check `docs/han/research_logs/` for the latest state of the project.
3. **Document your work.** If you run an experiment or reach a finding, write it up. Record the command you ran, the config/hyperparameters, and the result.
4. **Reproduce before extending.** If you are building on a previous result, reproduce it first to confirm the baseline.
5. **Sanity check everything.** When training SAEs, always check basic health metrics: fraction of alive features, reconstruction loss, L0, and decoder cosine similarity ($c_{\text{dec}}$). If something looks off, investigate before proceeding.
6. **Use seeds.** All experiments should set random seeds for reproducibility. Use `torch.manual_seed()`, `np.random.seed()`, and any other relevant RNG seeding.
7. **Keep code modular.** Separate data generation, model definitions, training loops, and evaluation into distinct files/functions. Avoid monolithic scripts.
8. **No unnecessary dependencies.** Work with what is pinned in `pyproject.toml` + `uv.lock`. If you need a new package, use `uv add <pkg>` and commit the updated `pyproject.toml` + `uv.lock` together.
9. **GPU memory awareness.** We work on single-GPU setups. Be mindful of memory — use `torch.no_grad()` during evaluation, clear caches when needed, and report GPU memory usage for large experiments.
10. **Ask if uncertain.** If the task is ambiguous or you are unsure about a design choice, ask rather than guessing.
 

## Markdown documentation guide

### Headings

- **No H1 headings** -- Obsidian shows the filename as the page title. Start with `##`.
- ATX style only (`##`, not underlines).

### Lists and Emphasis

- Dash (`-`) for unordered lists.
- Asterisks for emphasis: `*italic*`, `**bold**`.

### Links

- **Internal**: Obsidian wikilinks -- `[[Page Name]]` or `[[Page Name|display text]]`.
- **External**: Standard markdown -- `[text](url)`.
- No bare URLs.

### Code Blocks

- Fenced with backticks, always specify language.
- Use `text` for plain output.

## Frontmatter

Every doc in `docs/` requires YAML frontmatter:

```yaml
---
author: Name
date: YYYY-MM-DD
tags:
  - tag-name
---
```

- `author` -- required
- `date` -- required, ISO format
- `tags` -- required, at least one tag from `docs/Tags.md`

Tags must be `kebab-case` and listed in `docs/Tags.md`. Nested tags are valid if the parent exists.

## Running Checks

```bash
./run-checks.sh       # all checks (markdown lint + tag validation + link check)
./run-checks.sh -m    # markdown lint only
./run-checks.sh -t    # tag check only
./run-checks.sh -l    # link check only
```

## Available Commands

Contributors can run these Claude Code slash commands at any time to get AI-assisted reviews of their work:

- `/review` -- auto-detect content type and run appropriate reviewers
- `/review-doc` -- documentation quality review (structure, cross-refs, frontmatter, clarity, formatting)
- `/review-math` -- mathematical correctness review (notation, proofs, equations, definitions)
- `/review-code` -- code quality review (bugs, tests, design, performance)
- `/review-experiment` -- experimental methodology review (reproducibility, controls, metrics)
- `/review-data` -- data quality review (schemas, formats, configs, pipelines)

Each command accepts an optional file path argument (e.g., `/review-math docs/Theory.md`). Without one, it diffs against `main` to find relevant changed files.

## CI Agent Review (currently disabled)

There is an automated agent review workflow in `.github/workflows/agent-review.yml` that can run these reviews on pull requests. It is currently disabled. To enable it, set `ANTHROPIC_API_KEY` in your GitHub repo secrets and change `if: false` to `if: github.event.pull_request.draft == false` in the workflow file.

## Tag Taxonomy

See `docs/Tags.md`. Current categories:

- **Type**: guide, design, proposal, results, reference
- **Status**: todo, in-progress, complete
