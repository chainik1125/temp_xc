# C7 — Backtracking (Ward Stage B)

Per-component scripts for the Ward et al. 2025 Stage B backtracking case
study on Gemma-2-2b BASE. See `docs/components/c7.md`.

## Files (TODO — Agent BACK fills in, coordinating with Aniket)

- `mine_features.py` — per-arch feature mining for "begin backtracking" direction
- `steer.py` — per-arch steering at densified magnitude grid
- `judge_sonnet.py` — Sonnet judge for coh + backtracking + looping (κ-validated)
- `detection_probe.py` — linear probe with PR-AUC (NOT F1)
- `blind_judge.py` — 20-transcript blind validation CSV
- `run.sh` — full pipeline

## Locked metrics

- **Inducement**: peak Δgc (gain-in-keyword-rate). TXC's expected ~3× win.
- **Detection**: PR-AUC (Aniket open todo: switch from F1).
