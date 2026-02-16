# Temporal Crosscoders

Research repository combining documentation (Obsidian vault) and Python code for temporal crosscoder experiments.

## Repository Structure

- `src/` -- Python source code
- `tests/` -- pytest test suite
- `docs/` -- Obsidian documentation vault
- `docs/templates/` -- document templates (Base.md, Guide.md)
- `docs/Tags.md` -- tag taxonomy index
- `run-checks.sh` -- runs all quality checks locally
- `check-tags.sh` -- validates tags in documentation
- `get-tags.sh` -- extracts tags from a markdown file
- `.markdownlint.jsonc` -- markdown linting rules
- `pyproject.toml` -- Python project configuration

## Key Rules

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

## Python

- Python 3.12+, managed with `uv`
- Setup: `uv sync`
- Run tests with `uv run pytest`
- Source code in `src/`, tests in `tests/`

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
