# runpod-2 — working state

**Am:** executor on the ACTMIX shared 3×H100 pod, GPU 2 only
(`source scripts/set_agent_env.sh runpod-2`). Clone
`/workspace/agents/runpod-2/temp_xc`, own venv (peft + einops added
— not in pyproject).

## ACTIVE (2026-07-27 ~12:00 London) — P1 seed extension RUNNING

Governing directive: **059a66239 pod-saturation plan** (+ Han's
direct message): P1 = RLHF figure top-ups. Card amendment **A1**
(§ 7, freeze 421f6fa37): seed 2 @ T{1,2,5,8,16} + s1 @ T{8,16},
trained txc only, 7 cells ≈ 9.2 GPU-h ≈ $28 (measured solo basis).

**In flight on GPU 2** (orchestrator `run_ext.sh`, pin 421f6fa37,
nohup, log `/workspace/logs/actmix_rlhf_run_ext.log`):
- phase A (~11:50→13:10): ext_a=[s1_T8] frac 0.52 ‖
  ext_b=[s2_T{1,2,5}] frac 0.34
- phase B (auto-chains): ext_c=[s1_T16, s2_T8, s2_T16] serial
  uncapped; s1_T16 lands ~15:50, all drained ~19:45.
Wall jsonls: `/workspace/logs/actmix_rlhf_runs_ext_{a,b,c}.jsonl`.

**Figure**: `figs_writeup/fig_rlhf_shuffle_tsweep.{png,pdf}` —
INTERIM (2 seeds, T8/T16 n=1 disclosed on-figure) PUSHED 7f3fb62ee
(~11:55, in time for the 17:00 draft). Renderer =
`actmix_rlhf/render_writeup_fig.py --tag {interim,final}`.
Refresh interim when s1_T16 lands; **FINAL --tag final** when all
7 land, + `analyze.py` table refresh (seed-2 whitelist already
patched) + LOG verdict-extension entry PTR + ledger actuals +
**report state by ~21:30 London**.

**Watchers live (session-local; re-arm on wake/resume):**
- attention: FAIL in ext lane logs OR s1_T16 line in ext_c jsonl
  → refresh interim render + push
- completion: "all lanes drained" in run_ext.log → FINAL sequence
- origin listener: 150 s poll (archs/, LOG, COMPOSITION_AUDIT,
  briefings/, archs.yaml) — re-arm after EVERY wake, full 10 h.

**P2(b) λ̂-Ward: CONFIRMED MINE** (5aa351a4e split + 121807fb0
"Ward stays yours"). Prereqs STAGED + sha-verified 16/16 (ward
stream + lambda_intensity labels restored from the HF mirror
`ward_lambda_prereqs/`). Plan at ~20:50 (contended P1 drain):
sweep card first (hunt3: scorer-before-results; L{6,9,12,15,18}
via ONE cache_depth forward sweep with the APPROVED one-line
LAYERS extension for odd L9/L15), reader Llama-3.1-8B from HF at
slot time, λ̂_hist PRIMARY (candidate-1 convention). Screens only,
NO panel cells. Contended ETAs: phase A ~14:20, s1_T16 ~17:00
(interim refresh), all 7 ~20:50, FINAL + verdict + actuals ~21:15.

## Closed earlier (context)

RLHF core: RATIFIED with quote licence (mac-local 01c5244fc) —
R-E1 shipped-ckpt shuffle control confirms the paper; order-free
inverted-U (peak T8); R-E5 untrained 0.659 > all trained. EM:
closed under Han's full stop (dca32ce6b), 3 cells
NON-QUOTABLE-pending-Dmitry, nothing EM relaunches. Paper-match
eval-only arm: `actmix_rlhf/results/papermatch.json`.

## Ledger

`briefings/MODAL_SPEND.md` RUNPOD: weekend actuals ≈ $44; today
est line for A1 ≈ $28 (reconcile to actuals at close). $150/day
cap intact.

## If resuming after compact

Read this file, check the three watchers (re-arm any that died),
`tail /workspace/logs/actmix_rlhf_lane_ext_*.log`, then the LOG
tail. The FINAL sequence above is the standing next action. Tokens
rotate post-weekend — nothing token-valued in git.
