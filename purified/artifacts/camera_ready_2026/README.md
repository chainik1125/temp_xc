# Camera-ready artifact archive

Start with [the evidence and figure workspace](../../docs/aniket/camera-ready-2026/README.md).

- `manifest.json` pins copied source files to Git/HF revisions and SHA-256.
- `native_artifact_index.json` indexes existing tracked code and results in place.
- `posted_rebuttal_tables.json` preserves nine posted tables as claims, not verified results.
- `sources/` contains small branch-specific code, configs, figure inputs, and provenance notes.
- `hf_reviewer_results/` contains only small JSON/Markdown from the reviewer archive.
- `derived/arxiv_real_task_leaderboard.jsonl` keeps verbatim real-task rows from the larger leaderboard, with the filter recorded in the manifest.
- `previews/` contains CPU-only reconstruction examples and their source tables.

This is deliberately not a dataset/checkpoint release. Missing raw artifacts
and conflicting experiment identities are documented in the linked workspace.
Historical code is preserved unchanged for provenance; running training,
cache-building, or judge scripts is not needed to recolor the figures.
