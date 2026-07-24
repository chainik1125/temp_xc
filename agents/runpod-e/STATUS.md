# Working state — agent `runpod-e`

**Last rewrite:** 2026-07-24 (**PRE-COMPACT handoff.** Round-2 items
1+2 done; §3 screen queue 4 of 5 bundles done. Nothing running, tree
clean, all pushed. Next action: `dialevel`, the last bundle.)

## Who / where / env (verified working this session)
H100 pod, `/workspace/temp_xc`, `/workspace/.agent_id` = `runpod-e`.
Git `runpod-e-agent`, creds `store --file=/workspace/.git-credentials`
(token from `/workspace/.tokens/gh_token`); `export HF_TOKEN=$(cat
/workspace/.tokens/hf_token)` per command; anthropic key at
`/workspace/.tokens/anthropic_key`. Pull-rebase before EVERY push
(shared `arxiv` branch). venv `.venv/bin/python`. Hygiene at last
check: **280 tests pass / 1 skipped**, `run.py validate` OK,
leaderboard **8796 rows, 0 dup eval_keys, 0 null metrics**.
Governing doc: `briefings/task-hunt-r2-e.md` (items 1, 2 done; §3 is
the live queue). HEAD at rewrite: `d6f5ace3`.

## TWO OPERATIONAL GOTCHAS (both cost time this session)
1. **Never write a waiter whose own command line matches its own
   `pgrep -f` pattern** — `until ! pgrep -f "foo.screen"; do sleep;
   done` matches the waiter shell itself and loops forever after the
   job exits. Killed two of these. Use a PID (`kill -0 $PID`) or grep
   the log for a DONE marker.
2. **Every LOG.md push conflicts on rebase** (all agents append to the
   same file). The fix is mechanical and safe: strip the three marker
   lines, keep BOTH sides, `git add`, `git -c core.editor=true rebase
   --continue`. Done 4× this session; never lose a sibling's entry.

## ROUND-2 ITEMS 1+2 — DONE, awaiting mac-local review
- **Item 1, hedging-LEVEL Stage 2 = NEGATIVE.** 84/84 cells, all archs
  budget-matched (l0 6.3–8.1). Recovery peaks at T=4 and declines
  (within-seed trend p=0.73/0.96/0.50). One bounded positive: TXC-pre/T4
  beats both token archs (+0.055 CI[+0.007,+0.103]; +0.037
  CI[+0.012,+0.062]). RAW per-token reference 0.221 exceeded by only
  1 of 14 cells. **Lesson recorded: Stage-2's unmatched sampling
  re-admits the ambient route the Stage-1 screen matched away — the
  screen's premise does not transfer to the panel convention.**
- **Probe-capacity finding (matters program-wide).** The evaluator's
  unregularized OLS on p=d_sae with n∝1/T is **biased against large T**:
  under ridge on identical codes the T-ordering REVERSES (pre T16
  0.134→0.324). Stacked takes the biggest lift (+0.239; it is read at
  T·d_sae=32768 features), which reframes the "Stacked pathology" as
  probe loading, in MY panel and runpod-d's λ̂ panel. runpod-d found the
  same defect independently the same day. **Any future Stage-2 must use
  a capacity-adequate probe.**
- **Item 2, early-layer addendum.** lag4: A2/A3/A4 confirmed, **A1
  falsified** → new atlas shape *present-then-discarded*; **A5 refined**
  — short-T `g_order` is mostly anchor-vs-context separation, not order.
  slope8: B1/B2/B3 confirmed (g_agg>0 in all 34 cells; per-token never
  exceeds 0.483 at ANY depth), **B4 falsified** (mid-depth valley).

## §3 SCREEN QUEUE — 4 of 5 bundles done (all verdicts in LOG.md)
| bundle | verdict | one-line |
|---|---|---|
| `novelty` | **NEGATIVE** | gap ≤+0.045 vs +0.05 bar, peaks mid-ladder; 71–77 % converted |
| `punctint` **q** | **KEEP** (first of the hunt) | gap → +0.114/+0.127/+0.143 at T64, tracks kernel mass, anchor LOSES from windows |
| `punctint` list | **WEAK KEEP**, disclosed | anchor GAINS from windows; doc-AUC 0.960; control rests on 8 docs |
| `tss` | **KILL (converted)** | engineered anti-conversion corpus converts anyway (74/104/73 %); null receipt proves the state is REAL |

**Three cross-cutting contributions to carry forward (all in LOG):**
1. **Conversion fraction** `(tok − floor)/(best_window − floor)` should
   replace the absolute window−token gap as the screen diagnostic — an
   absolute threshold cannot separate "converted with a residue" from
   "genuinely window-only".
2. **Doc-identity triage bar**: proposed `doc_mean_only_auc` for the
   factory (novelty 0.792 / q 0.926 / list 0.960 / tss 0.664 — a
   batch-wide route the frozen unigram+position bars cannot see), plus
   within-document contrasts as the standard control for any KEEP. My
   `qrate_fineweb/within_doc.py` is the reference implementation; the q
   face SURVIVED it (+0.101/+0.132/+0.183 over 24–26 test docs).
3. **`source` anchor spec defect**: it is a *within-pair role* label
   (per-pair mean 0.14–0.71), so a global probe cannot learn it by
   construction; its at-chance reading says nothing about the model.
   S3 recorded UNEVALUABLE; factory advised to respecify or drop.

**`tss` is the scientific headline**: conversion is broader than "the
model linearizes what predicts the next token" — removing the
generative payoff did not stop per-position carriage, while the
shuffled-block null corpus (real−null +0.086…+0.116 at T16, +0.092…
+0.143 per-token) proves the state is genuine and coherence-dependent.
Untested hypothesis on record: `tss` has an incidental per-position
correlate (coherence recovers as a block continues after a switch).

## NEXT ACTION — `dialevel`, the last bundle in my queue
Queue position 9. Needs (a) a caching pass — corpus
`labels/dialevel_dailydialog_{gpt2,gemma2,llama31}.npz` +
`dialevel_corpus.json.gz`, NOT the replag caches — and (b) mac-local's
**binding qualification 2**: the screen precondition is within-dialogue
contrasts or dialogue-length matching PLUS position/doc-length floor
probes (all-row position AUC 0.93 means the naive screen is foreclosed).
License note (CC BY-NC-SA) travels with any figure that graduates.
Recipe to reuse verbatim: `interleave/cache_acts.py` (chunk the flat
stream to 128-token rows mirroring replag geometry, verify the
flat↔windowed mapping BEFORE caching) + `interleave/screen.py` +
`qrate_fineweb/within_doc.py`. Post a `screening <bundle>` claim-line
in LOG.md first (claim-lines rule; runpod-d shares the queue).

## Protocol reminders that bit this session
Freeze the card AND the screen script in one commit BEFORE any cell;
commit post-hoc diagnostics before running them; report "no rule fires
as written" rather than inventing a verdict (novelty's card lacked that
clause — later cards add it); always run the position floor next to
window numbers; Stage-1 screens write NO leaderboard rows.

## Sibling context
runpod-d: budget-matched TXC-post re-run done (its k=8·T convention is
validated by my panel's falsifier); shares the §3 queue by claim-lines.
runpod-b: hunt-support-stats complete (its variance-aware renderer is
what `confidence/render_stage2.py` wraps). runpod: candidate factories
+ `probe-adequacy` briefing (`briefings/probe-adequacy.md`) — that work
is downstream of my probe-capacity finding.
