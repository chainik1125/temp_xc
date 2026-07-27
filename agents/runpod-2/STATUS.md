# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2`). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops added
— not in pyproject).

## Where things stand (2026-07-27 ~08:10 London)

**RLHF ABLATION: COMPLETE + RATIFIED (mac-local 01c5244fc ~08:15)
with quote licence** — LEAD with R-E1 (shipped-ckpt shuffle control
confirms the paper; gap +0.012; length-spurious 3→1 under shuffle
while AUC holds); T-sweep = order-free inverted-U; R-E5 quotable as
"sparse RANDOM projections carry the preference signal above every
trained dictionary at this budget class" WITH the l0-mismatch
disclosure beside it; joins the untrained-boundary story as the 5th
substrate. My tsae serving pointer (36df9ffb6) was accepted and
restores runpod-1's trained tsae column (their amendment 2b).
Verdict originally posted PTR (LOG ~08:05). Card `actmix_rlhf/CARD.md` freeze 72b0ca729 (approved
ba8af7bf9 + c4595d533) executed in full:
- paper-match arm (eval-only, 4 shipped seed-42 ckpts, case-study
  artifact `results/papermatch.json`): headline = the missing
  shuffle control CONFIRMS the paper's reading (agentic gap +0.012;
  3-length-spurious reproduced exactly). Provisionally ratified
  c4595d533; FINAL ratification rides the full verdict.
- btk-only arm (canonical runner, 25 leaderboard cells): T-sweep =
  order-free inverted-U (0.578→0.626→0.611, gaps ≈ 0 every T, both
  seeds, 100·T/tok parity held). R-K1..3 ✓, R-E1..E4 ✓ (E4 on seed
  mean, per-seed split disclosed), R-E5 informative MISS: untrained
  k500 twins 0.659 > every trained cell (random projections carry
  the currency).
- Artifacts: `actmix_rlhf/{results/{papermatch,rlhf_table}.{json,md},
  figs/rlhf_tsweep.*}`; analysis = `analyze.py` (mechanical),
  `render_figs.py`. Ledger actuals ≈ $30.

**EM: closed** under Han's full stop (dca32ce6b) at ~00:30 — 3
landed cells NON-QUOTABLE-pending-Dmitry, close-out LOG entry +
handover preserved (caches/ckpts/wall-logs on volume; threshold-
transfer flag acked by mac-a in their HUNT3 note). Nothing EM
relaunches from this fleet.

**GPU 2: IDLE** as of ~08:05 (all lanes drained ok: r 14/14,
rs 4/4, s1 7/7; zero FAILs post-relaunch). No further runs planned
by me — standing priority test (task hunt / RLHF / sparse probing)
satisfied on my side; awaiting team-review outcomes + the 9am PT
meeting.

## Standing watchers (session-local; re-arm on wake)

- origin listener (archs/, task_hunt LOG, COMPOSITION_AUDIT,
  briefings/) — poll 150 s.
- No lane monitors active (nothing running).

## Ledger

RUNPOD section in briefings/MODAL_SPEND.md — runpod-2 weekend
actuals ≈ 14.5 GPU-h ≈ $44 (EM ~$13.5 + RLHF ~$30) of $150/day cap.

## If resuming after compact

Read this file + the LOG tail (my ~08:05 verdict + any team
ratifications after it). Possible follow-ups the team may ask for:
seed-1 T8/T16 (pre-declared stretch, ~4 GPU-h), autointerp stage
(needs API budget), s1 panels in the figure, REBUTTAL_PACK-format
rows (mac-b's d9df9d0c9 format) for the one-pager. Tokens rotate
post-weekend — nothing token-valued is in git.
