# Contributing

## Code

1. Branch off `main`
2. Run `uv sync` to set up the environment
3. Make your changes in `src/`
4. Run `uv run pytest` to verify
5. Open a PR against `main`

## Writing docs

1. Copy a template from `docs/templates/`
2. Fill in frontmatter (author, date, tags from `docs/Tags.md`)
3. Start with `##` headings (no H1 — Obsidian shows the filename as the title)
4. Use `[[wikilinks]]` to reference other docs
5. Run `./run-checks.sh` before pushing

## AI review commands

Use these Claude Code slash commands to get AI-assisted reviews:

- `/review` -- auto-detects content type and runs the appropriate reviewers
- `/review-doc <path>` -- documentation quality (structure, frontmatter, cross-refs, clarity)
- `/review-math <path>` -- mathematical correctness (notation, proofs, equations)
- `/review-code <path>` -- code quality (bugs, tests, design, performance)
- `/review-experiment <path>` -- experimental methodology (reproducibility, controls, metrics)
- `/review-data <path>` -- data quality (schemas, formats, configs, pipelines)

The `<path>` argument is optional. Without it, each command diffs against `main` to find relevant changed files.

## Obsidian setup (one-time, optional)

1. Open the `docs/` folder as a vault in Obsidian
2. Go to Settings > Community plugins > Turn on community plugins
3. Install "Obsidian Git" (the config is already checked in)
4. You can now pull/commit/push from within Obsidian

This step is optional — you can write docs in any editor. Obsidian just gives you the graph view and wikilink navigation.
