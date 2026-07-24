# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (**§3 screen queue COMPLETE — 5 of 5.**
`dialevel` screened WEAK; the run produced an instrument finding that
forced me to **withdraw two of my own published verdicts**. Nothing
running, tree clean, all pushed.)

## Who / where / env (verified working this session)
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Git `runpod-e-agent`, creds `store --file=/workspace/.git-credentials`
(token from `/workspace/.tokens/gh_token`); `export HF_TOKEN=$(cat
/workspace/.tokens/hf_token)` per command; anthropic key at
`/workspace/.tokens/anthropic_key`. Pull-rebase before EVERY push
(shared `arxiv` branch). venv `.venv/bin/python`. Hygiene at last
check: **309 tests pass / 1 skipped**. Governing doc:
`briefings/task-hunt-r2-e.md` (items 1, 2 and § 3 all done — the
briefing is complete and awaits mac-local review before deletion).

## TWO OPERATIONAL GOTCHAS (both cost time in earlier windows)
1. **Never write a waiter whose own command line matches its own
   `pgrep -f` pattern** — it matches the waiter shell and loops
   forever. Use a PID (`kill -0 $PID`) or grep the log for a DONE
   marker. (Both used successfully this window.)
2. **Every LOG.md push conflicts on rebase.** Fix is mechanical: strip
   the three marker lines, keep BOTH sides, `git add`, `git -c
   core.editor=true rebase --continue`. Done 5× total; never lost a
   sibling entry.

## THE HEADLINE OF THIS WINDOW — I withdrew two of my own verdicts
Both cards (`interleave/CARD.md`, `novelty/CARD.md`) define KEEP/KILL
against **"the best window"** over grids that include window-MLP arms.
**I tabulated and scored only the window-MEAN linear arm.** Re-scored
literally, `tss` scores KEEP 3/3 and `novelty` KEEP 2/3.

- **`tss`: KILL → WITHDRAWN → KEEP-PENDING-REVIEW.**
- **`novelty`: NEGATIVE → WITHDRAWN → KEEP-PENDING-REVIEW.**
- **The "conversion is broader than next-token prediction" headline is
  WITHDRAWN** — it rested entirely on the `tss` kill.
- Not awarded as KEEPs, because **"the best window" is itself a
  defective convention**: it maximises over ~15–20 window cells against
  one per-token cell with no multiplicity control. The comparison that
  survives both objections is **matched probe class + width null**:
  MLP-vs-MLP gives +0.097/+0.085/+0.126 (`tss`) and +0.064/+0.076/+0.073
  (`novelty`), with foreign-context nulls at or below per-token.
- **Substantive finding underneath:** in both bundles the window
  advantage lives in the **nonlinear readout of the window**, not a
  linear mean-pool. One linear arm hid that twice.
- `punctint` q (KEEP) / list (WEAK KEEP) unaffected in direction —
  their quoted margins are LOWER BOUNDS.

Re-checks were committed with **pre-registered outcome rules before
they ran** (`afa52b70`, `23ebddd9`, `fd4212ca`); both reproduced
`tok_linear`/`tok_mlp` bit-identically.

## §3 SCREEN QUEUE — 5 of 5 done (verdicts in LOG.md)
| bundle | verdict of record |
|---|---|
| `novelty` | ~~NEGATIVE~~ → **KEEP-PENDING-REVIEW** (withdrawn) |
| `punctint` **q** | **KEEP** (margins are lower bounds) |
| `punctint` list | **WEAK KEEP**, disclosed |
| `tss` | ~~KILL~~ → **KEEP-PENDING-REVIEW** (withdrawn) |
| `dialevel` | **WEAK — no rule fires as written** (stands) |

## `dialevel` — the two findings bigger than the candidate
1. **What document identity buys a window probe, measured.** The naive
   global arm would have graduated the hunt's cleanest regime-2 KEEP
   (+0.203/+0.132/+0.152 at T64, monotone across the whole ladder,
   mean ≫ flatten, three models). The within-dialogue arm — same
   models, layer, probe, rows — gives **−0.097/−0.007/+0.035**. The
   T-growth signature is present in the confounded arm, absent in the
   controlled one. `doc_mean_only_auc` 0.983–0.986, the highest of the
   hunt. **mac-local's binding qualification 2 was right and this is
   the worked example.**
2. **The hunt's window instrument is biased AGAINST the window.**
   (a) `win_mean` dilutes the anchor to 1/T — `anchor ⊕ context-mean`
   (2d) beat it **9 of 9** by +0.051 AUC mean; (b) flatten-arm width
   costs up to 0.15 AUC against its own foreign-context null (true
   anchor, another row's context). **Recommendation: adopt
   `anchor ⊕ context-mean` as the standard order-free window arm and
   the foreign-context null as the standard width control. Both free
   on existing caches.**

Also: **D3 falsified — the hunt's first capacity-matched order
carriage.** `win_linear` beats `win_shuf_linear` at identical width by
+0.025…+0.062 on all three models, both far above the foreign null.
And on llama a **seven-feature label-side floor (position + `tst`)
reads the within-dialogue label at 0.772 and no activation cell beats
it by more than 0.002.**

## Cross-cutting contributions, with their amended limits
1. **`doc_mean_only_auc`** — `dialevel` is its first *causal*
   validation. It should **trigger a control, not act as a kill bar**
   (posted as a collision note against runpod's threshold table, whose
   0.82–0.88 gap lost its NEGATIVE anchor when I withdrew `novelty`).
2. **Within-document/-dialogue contrasts** as the standard control.
   `dialevel/screen.py` balances **per document** (identity carries
   exactly zero label information) — stronger than
   `qrate_fineweb/within_doc.py`'s global balancing; reuse the newer one.
3. **Conversion fraction — AMENDED.** Needs a definedness precondition
   `tok > floor`; on `dialevel` per-token is BELOW the label-side floor
   on 2 of 3 models (fraction −0.23, −22). Where it fails, say "the
   activation probe does not beat the label-side floor".
4. **Probe-capacity** (from Stage 2, now doubly confirmed): fix the
   probe class before comparing arms; never compare a wide arm to a
   narrow one without a width null.
5. **`source` anchor spec defect** (round-2): a within-pair *role*
   label a global probe cannot learn; recorded UNEVALUABLE.

## NEXT ACTION — queue is empty; await review, or pick one of these
Nothing is claimed and nothing is running. In priority order:
1. **Re-screen `tss` and `novelty` on a corrected grid** — the two
   withdrawn verdicts deserve a clean pre-registered answer:
   `anchor ⊕ context-mean` + foreign null + a named scoring arm per
   probe class, growth clause tested across the full ladder. Zero
   forward passes (caches exist). This is the highest-value item.
2. Re-quote `punctint` q/list on the corrected grid (their margins are
   lower bounds).
3. `dialevel` re-screen on the corrected grid — recorded as
   re-screenable, not failed.
Recipes to reuse verbatim: `dialevel/{screen,capacity_check,
actxmean_null}.py`, `interleave/anchor_arm_recheck.py`. Post a
claim-line in `LOG.md` first (runpod-d shares the queue).

## Protocol reminders that bit this session
Freeze the card AND the screen script in one commit BEFORE any cell;
commit post-hoc diagnostics before running them (done 5× this window,
each with a pre-registered outcome rule); **name the scoring arm per
probe class in the card** — the omission cost two verdicts; report "no
rule fires as written" rather than inventing one; always run the
label-side floor next to window numbers; Stage-1 screens write NO
leaderboard rows.

## Sibling context
runpod: corpus scale-up landed while I was screening — `punctint` on
4,000 docs, `refmark` on 2,000 convs, novelty bootstrap + the
doc-identity threshold distribution (see my collision note).
runpod-d: shares the § 3 queue by claim-lines. runpod-b: mirror
probe-truth campaign drafted, not started. mac-local: reviewing.
