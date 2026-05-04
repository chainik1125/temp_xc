---
author: aniket
date: 2026-05-04
tags:
  - reference
  - in-progress
---

## Cross-component findings

This directory holds findings that span multiple `cN` components. Per
PROTOCOL.md § 7 (results live in state) the AUTO-RESULTS blocks of
`docs/components/cN.md` are owned by each component's `analysis.py`,
so cross-component analysis lives here instead.

Conventions:

- Frontmatter `author` / `date` / `tags` per
  [[docs/Tags|Tags]] taxonomy.
- These docs **point at** code under `src/temp_bench/` and scripts
  under `experiments/det_steer/`. They do not own training compute.
- Component agents adopt cross-component findings by importing the
  shared infra (`temp_bench.eval.detection`,
  `temp_bench.eval.steering_hooks`,
  `temp_bench.eval.steering_protocols`,
  `temp_bench.utils.shuffles`) and rendering the resulting numbers
  into their own `cN.md` AUTO-RESULTS via
  `temp_bench.report.render(component="cN")`.

## Index

- [[det_steer_detection]] — sparse linear probe + within-window shuffle
  ablation as the detection protocol for C5 / C6 / C7.
- [[det_steer_steering]] — TXC steering audit: V0 mean-decoder is
  TopK-SAE-equivalent; V1 / V2 / V3 / V4 alternatives + when each
  matters.
- [[det_steer_summary]] — methodology validation results +
  per-component integration TODO list for the case-study agents.
