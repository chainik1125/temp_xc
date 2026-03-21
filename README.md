# temp_xc

## Setup

```bash
uv sync
git config core.hooksPath .githooks
```

## Documentation

Docs live in `docs/`. Open that folder as an Obsidian vault for connected browsing, or read the markdown directly.

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to write and submit docs.

## Running checks

```bash
./run-checks.sh       # all checks (markdown lint + tag validation + link check)
./run-checks.sh -m    # markdown lint only
./run-checks.sh -t    # tag check only
./run-checks.sh -l    # link check only
```
