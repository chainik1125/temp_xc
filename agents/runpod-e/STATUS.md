# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-25 (**PRE-COMPACT handoff.** Briefing
`task-hunt-r2-e.md` fully discharged and **RETIRED by mac-local's
review**; all my work APPROVED. Nothing running, tree clean, all
pushed, **no briefing assigned**. Next action is mine to claim — see
the bottom.)

## Who / where / env (verified working this session)
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Git `runpod-e-agent`, creds `store --file=/workspace/.git-credentials`
(token from `/workspace/.tokens/gh_token`); `export HF_TOKEN=$(cat
/workspace/.tokens/hf_token)` per command; anthropic key at
`/workspace/.tokens/anthropic_key`. Pull-rebase before EVERY push
(shared `arxiv` branch). venv `.venv/bin/python`. Hygiene at last
check: **309 tests pass / 1 skipped**, `run.py validate` OK,
leaderboard 8,822 rows / 0 dup eval_keys / 0 null metrics.
**Activation caches on this box (this is why the open escalation is
mine):** `/workspace/{replag,interleave,dialevel,conv_depth,emo}_caches`
— 365 GB free.

## TWO OPERATIONAL GOTCHAS
1. **Never write a waiter whose own command line matches its own
   `pgrep -f` pattern** — it matches the waiter shell and loops
   forever. Use `kill -0 $PID` or grep the log for a DONE marker.
2. **Every LOG.md push conflicts on rebase.** Strip the three marker
   lines, keep BOTH sides, `git add`, `git -c core.editor=true rebase
   --continue`. Done 6× total; never lost a sibling entry.

## REVIEW OUTCOME (mac-local, `fbab4070`) — all approved
- **My two withdrawals ACCEPTED** and called "the best work of the
  night". `tss` KILL → **KEEP-PENDING-REVIEW**; `novelty` NEGATIVE →
  **KEEP-PENDING-REVIEW**; `dialevel` **WEAK stands**; `punctint`
  q/list margins are **LOWER BOUNDS pending re-quote**.
- **ADOPTED PROGRAM-WIDE: no card may score against a max-over-arms
  "best window" again — fix the probe class and control width.** My
  matched-class comparison + foreign-context nulls are now "the
  convention of record".
- "Conversion is broader than next-token prediction" **WITHDRAWN**;
  the corrected reading is the theory-consistent one (remove the
  generative payoff and the window *does* carry the state).
- **`doc_mean_only_auc` RATIFIED as a disclosure statistic that
  TRIGGERS A CONTROL — explicitly NOT a kill bar**, my causal
  `dialevel` datapoint cited as beating the correlational anchor.
- Standing caveat on everything I produced: these are **Stage-1
  screens** (window-vs-per-token on raw activations), **not**
  TXC-vs-T-SAE. They license Stage-2 panels, not win claims.

## TWO OPEN THREADS AIMED AT ME (do not lose these)
1. **The escalation — the top program follow-up, and this is the only
   box that can run it.** runpod measured that **76–91 % of the unigram
   rise at 10× corpus is estimator sample size** (curve unsaturated at
   3,200 docs). The flagged hypothesis: *if a screen's per-token probe
   attenuates faster than its window probe, a small-corpus screen
   **overstates window-minus-per-token — the hunt's headline
   statistic***. It touches every screen gap including the five new
   KEEPs. **The prescribed test: re-fit ONE screened bundle's
   per-token and window probes at two training sizes and compare the
   GAPS.** runpod's STATUS says it is not runnable there (no caches, no
   GPU). Mine are listed above. Cheap, no forward passes.
2. **Cross-pod tension mac-local named, binding on any Stage-2 I
   propose:** on Ward a LINEAR mean-pool carries the whole gain
   (`g_agg ≈ g`); on fineweb `tss`/`novelty` a linear mean-pool sees
   ≈ +0.04 while an **MLP on the window sees +0.06…+0.13**. A sparse
   linear dictionary can capture the Ward-type gain and **may not**
   capture the fineweb-type one. **Any Stage-2 promotion of
   tss/novelty must state which of the two it is betting on.**

## MY OWN UNRECONCILED FLAG (posted `2026-07-25`, LOG)
The review adopted runpod-d's "**order does not matter, anywhere**"
(Ward `g_order` at T32 spans −0.004…+0.008). **`dialevel` measured the
opposite on its substrate**: flatten − shuffled at identical width =
**+0.056/+0.062/+0.035 at T32** and +0.031/+0.025/+0.028 at T16, three
models, consistent sign, both arms far above the foreign null (so not
the flatten-overfit mode the review flagged on qrate). **Reconciliation
hypothesis (mine, untested):** my shuffle scatters the slots *adjacent
to the anchor*, so this may be **recency / distance-to-anchor, not
sequence order** — a rate is uniform in slot index (Ward: buys
nothing), a trailing LEVEL is dominated by the nearest turn boundaries
(dialevel: buys +0.06). If so the finding should read "sequence order
does not matter; **distance-to-anchor sometimes does**". Cheap test,
no forward passes: shuffle only the far half vs only the near half,
and compare an exponentially-recency-weighted context mean to the flat
one.

## §3 SCREEN QUEUE — 5 of 5, briefing RETIRED
| bundle | verdict of record |
|---|---|
| `novelty` | **KEEP-PENDING-REVIEW** (withdrawn NEGATIVE) |
| `punctint` **q** | **KEEP** — margins are lower bounds |
| `punctint` list | **WEAK KEEP**, disclosed |
| `tss` | **KEEP-PENDING-REVIEW** (withdrawn KILL) |
| `dialevel` | **WEAK — no rule fires as written** |

## What I contributed that outlived the candidates
1. **The "best window" card defect** → adopted convention change.
2. **The window-instrument bias**: `win_mean` dilutes the anchor to
   1/T (lost **9 of 9** to `anchor ⊕ context-mean`, +0.051 AUC mean);
   flatten width costs up to 0.15 AUC against its foreign-context null.
   **Adopt `anchor ⊕ context-mean` as the standard order-free arm and
   the foreign-context null as the standard width control.**
3. **`doc_mean_only_auc`** + within-document contrasts — ratified.
   `dialevel/screen.py` balances **per document** (identity carries
   exactly zero label information); reuse it over
   `qrate_fineweb/within_doc.py`'s global balancing.
4. **Conversion fraction — AMENDED**: needs `tok > floor` to be
   defined; on `dialevel` per-token sits BELOW the label-side floor on
   2 of 3 models. Where it fails, say "the activation probe does not
   beat the label-side floor".
5. **Probe capacity** (Stage 2) — never compare a wide arm to a narrow
   one without a width null.
6. **`source` anchor spec defect** — a within-pair *role* label a
   global probe cannot learn; recorded UNEVALUABLE.

## NEXT ACTION — nothing assigned; claim one, in this order
1. **The estimator-attenuation escalation** (thread 1 above) — highest
   program value, explicitly blocked on every other box, cheap here.
   Post a claim-line in `task_hunt/LOG.md` first.
2. **Re-screen `tss` + `novelty` on the corrected grid** — the two
   withdrawn verdicts deserve a clean pre-registered answer
   (`anchor ⊕ context-mean` + foreign null + scoring arm named per
   probe class + growth tested across the full ladder). Zero forward
   passes.
3. **The recency-vs-order test** (my flag above) — cheapest of the
   three, settles a program-level wording.
4. Re-quote `punctint` q/list on the corrected grid.

Recipes to reuse verbatim: `dialevel/{screen,capacity_check,
actxmean_null}.py`, `interleave/anchor_arm_recheck.py`,
`novelty/anchor_arm_recheck.py`.

## Protocol reminders
Freeze the card AND the script in one commit BEFORE any cell; commit
post-hoc diagnostics before running them, each with a **pre-registered
outcome rule** (did this 5× — it is what made the withdrawals
defensible); **name the scoring arm per probe class in the card** — the
omission cost two verdicts; report "no rule fires as written" rather
than inventing one; always run the label-side floor next to window
numbers; Stage-1 screens write NO leaderboard rows.

## Sibling context
**mac-local**: reviewing; next up is the estimator escalation, then the
mirror receipt, then Sunday distillation. **runpod**: corpus scale-up
APPROVED and retired; **IDLE, and cannot run the escalation** (no
caches, no GPU). **runpod-d**: factory screens APPROVED — five Stage-1
KEEPs + the order-NEGATIVE; briefing discharged. **runpod-b**:
`briefings/mirror-probe-truth.md` ACTIVE — the λ-readout METHODS RULE
was **amended at its catch** (branches evaluate at **matched p/n swept
through 1.0**, not canonical mirror budget; p/n is the x-axis).
Active briefings not mine: `mirror-probe-truth.md`, `em-redo.md`.
