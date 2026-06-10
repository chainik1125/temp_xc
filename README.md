# temp_xc — Temporal Crosscoders

The `arxiv` branch: the post-submission, paper-ready codebase for the Temporal
Crosscoders study. Everything lives at the repo root (the historical `purified/`
subtree was lifted up).

Start with [`CLAUDE.md`](CLAUDE.md) — the operating manual (framework hard rules,
read order, layout). Key entry points:

- [`docs/framework.md`](docs/framework.md) — the framework spec.
- [`src/explorations/synthetic/README.md`](src/explorations/synthetic/README.md) — the synthetic-benchmark program
  (prime directive, the measure→mirror→bench loop + validity gates, conventions,
  and the benchmark index) + [`src/explorations/synthetic/STATUS.md`](src/explorations/synthetic/STATUS.md) (the
  living scratchpad).
- [`RUNPOD_INSTRUCTIONS.md`](RUNPOD_INSTRUCTIONS.md) — bringing a pod online.

Run the framework from the repo root:

```bash
uv sync
.venv/bin/python run.py validate
.venv/bin/python -m pytest tests/ -q
```

Historical research code lives on the wasteland branches
(`han-phase7-unification`, `aniket-ward-stage-b`, …) — read-only context, never
imported into this work; read via `git show origin/<branch>:<path>`.
