---
author: Aniket Deshpande
date: 2026-04-18
tags:
  - reference
  - venhoff-eval
---

## Venhoff code provenance

Attribution + change tracking for every file ported from
`cvenhoff/thinking-llms-interp` into this repo.

**Upstream repo**: https://github.com/cvenhoff/thinking-llms-interp
**Upstream paper**: Venhoff et al., "Base Models Know How to Reason,
Thinking Models Learn When", arXiv:2510.07364
**Upstream license**: (check their repo — this doc gets the SPDX tag
when porting starts)

## Ported files

Populated as each integration step completes. Entries are:
- upstream path
- our local path
- commit hash of upstream source at time of port
- what we changed + why

| upstream | ours | upstream-commit | changes |
|---|---|---|---|
| _(none yet — integration hasn't started)_ | | | |

## Files we deliberately did not port

- `train-vectors/` — Phase 2, stub for now. Would duplicate their
  steering vector training verbatim.
- `hybrid/` — Phase 3. Only needed if Phase 2 shows steering works.
- `human_eval/` — not in the default pipeline per their README.

## Files we wrapped, not ported

Listed here are files we call as-is via subprocess or import, without
local copies. Subject to their upstream license.

| upstream | how we use it | why not ported |
|---|---|---|
| _(tbd)_ | | |

## Upstream configs we pre-register as invariants

When we swap architectures, these MUST stay fixed to keep the metric
comparable:

- The 5 hand-written reasoning-category examples in
  `utils/autograder_prompts.py::categories_examples` (Generating
  Hypotheses, Expressing Uncertainty, Planning Future Steps,
  Stating Assumptions, Recalling Mathematical Definitions).
  Changing these changes the benchmark. Held fixed across SAE /
  TempXC / MLC comparisons.
- GPT-4o judge model (not gpt-4o-mini, not Anthropic) — their paper's
  default.
- Judge prompts in `utils/autograder_prompts.py` — all three
  (accuracy, completeness, semantic-orthogonality) used as-shipped.
- n_autograder_examples = 100, accuracy_target_cluster_percentage =
  0.2 — their defaults. If we tune these, we're not running their
  benchmark anymore.

Any deviation from these requires a note in this file with a reason.
