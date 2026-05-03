# C5 — RLHF / sentiment steering

Per-component scripts for the steering case study from T-SAE § 4.4.
See `docs/components/c5.md`.

## Files (TODO — Agent STEER fills in)

- `gen.py` — generate prompts + steered completions at α grid
- `judge_gemini.py` — Gemini coh + success grading
- `aggregate.py` — coh-vs-success curves at thresholds {1.5, 1.75, 2.0, 2.25, 2.5}
- `run.sh` — full pipeline; one row per (arch, seed, threshold)

## Reminder

Do **not** chase the Y/W steering hill-climb winners. Only TXC-base + TXC-pro
+ T-SAE here.
