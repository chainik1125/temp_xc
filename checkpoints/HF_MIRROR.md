# Checkpoint mirror on HuggingFace (interim A40 pod, 2026-07-25)

The 2026-07-25 force majeure destroyed every pre-existing checkpoint
(old pods' volumes; the manifests in this directory reference weights
that no longer exist anywhere for runs BEFORE that date). To prevent a
repeat, the two Stage-2 panels trained on the interim A40 pod were
uploaded at the funding cutoff:

- **Repo:** `han1823123123/temp_xc_a40_checkpoints` (PRIVATE, Han's
  HF account; write token required for changes)
- `stage2_fineweb_checkpoints/` — runpod-e's panel + replications
  (gemma-2-2b primary 84 cells, gpt2 + llama31 replication cells);
  134 files, ~17 GB — the checkpoints dir of the runpod-e clone,
  verbatim.
- `stage2_oprate_checkpoints/` — runpod-d's oprate rate_case panel
  (84 cells); 86 files, ~35 GB — the checkpoints dir of the runpod-d
  clone, verbatim.
- Upload verified file-count-exact against the pod (134/134 + 86/86)
  before the pod was released. Both panels' leaderboard rows carry
  paired v1+v2 probe columns, so no re-eval is *required* — this
  mirror exists for future diagnostics on these exact dictionaries.

**Standing rule (from the force-majeure lesson):** any plan described
as "eval-only" must verify weight existence FIRST — check this file
and the HF repo before assuming a checkpoint exists.
