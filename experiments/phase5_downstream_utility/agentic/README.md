## Agentic autoresearch loop

Scope marker for the Claude-driven TXC + MLC exploration phase. The
actual research log and operating rules live at:

`docs/han/research_logs/phase5_downstream_utility/2026-04-21-agentic-log.md`

This directory only holds per-cycle metadata if we end up needing it
(snapshots of training logs, ad-hoc probe outputs, etc.). The primary
execution path reuses `run_autoresearch.sh` — each cycle is a new
named variant in `train_primary_archs.py`, so the infrastructure is
unchanged.

### What goes here

- `cycles/NN_<name>/` — if a cycle needs a scratch directory for
  intermediate artifacts (training logs beyond the default, probe
  dumps, etc.). Not mandatory; most cycles won't need this.

### What doesn't go here

- Model classes — those belong in `src/architectures/`.
- Dispatcher branches — those stay in `train_primary_archs.py`.
- Probe routing — stays in `run_probing.py`.
- Commit bookkeeping — stays in `results/autoresearch_index.jsonl`.
