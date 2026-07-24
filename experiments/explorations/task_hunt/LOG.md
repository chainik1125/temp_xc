# Task-hunt log (shared, append-only)

The shared verdict log for the task hunt (`briefings/task-hunt.md`,
`briefings/task-hunt-b.md`). **Append-only; pull-rebase before every
push.** One entry per screen/stage event: date, agent, candidate,
verdict (KEEP / KILL / infeasible), one-paragraph justification, pointer
to the frozen card + results.

Conventions: mini-cards are committed BEFORE their screen runs; every
verdict cites its card; scale-orderings and honest kills are findings,
not failures (prime directive: a sound verdict, never a win).

---

## 2026-07-24 — runpod-d — candidate 1 (backtracking λ̂ intensity) — CARD FROZEN, screen pending

Ward stream + both reader caches rebuilt on the runpod-d volume
(`build_ward_stream.py` verbatim; stream stats reproduce the committed
reference — map_ok 99.97 %, BOS every row; base 219 s / distill 238 s
cache sweeps). λ̂ labels built locally
(`lambda_intensity/build_labels.py`) from the frozen mirror params on
real Sonnet history — runpod-b's `labels/ward_lambda.npz` had not
landed when caches were ready (never-idle rule); it landed mid-session
and the two builds **cross-validate: 99.93 % exact agreement** on
overlapping finite cells, disagreements confined to 277
sentence-boundary token attributions + their padded i<8 sentences
(excluded by my frozen i≥8 rule). Proceeding on my builder (the card's
freeze target); theirs stands as independent confirmation. Label-side
floor check (pre-freeze, disclosed in card): position-only AUC 0.82 on
full-λ̂ terciles vs 0.59 for kernel-only λ̂_hist ⇒ **primary target =
λ̂_hist**. Card: `lambda_intensity/card.md`, frozen at this commit;
screen (`screen.py`) not yet executed on any activation cache.
