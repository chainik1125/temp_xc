# tscale — the T-scaling hill-climb (runpod-c)

Mission (Dmitry 07-27, Han allocation): make TXC T-scaling actually
improve with window size on § 5.1 sparse probing — custom losses,
training tricks, whatever works — under a pre-registered dev/holdout
split. ARCH R&D, not claim production: nothing leaves this directory
without an L3 holdout run + mac-local ratification.

- **`CARD_SPLIT.md`** — the FROZEN dev/holdout split + pyramid gates +
  candidate-1 pre-registrations. Read it before touching anything.
- **`make_split.py`** — asserts the frozen split reproduces from the
  committed baseline rows (read-only).
- **`RESULTS.md`** — append-only candidate ledger (every candidate, its
  config hash, L0–L2 numbers, verdict; negative results are data).
- **`results/`** — scratch JSONL + checkpoints (gitignored ckpts);
  L3 rows go through the canonical runner instead.

Baseline truth being attacked (P1 btk-only, 36-task CT-excl, 3 seeds):
TXC-pre k=20 declines 0.9264 → 0.9033 over T1→T16; k=5 only recovers
to the SAE band. Dev-8 reproduces both phenomena (CARD § 2).
