# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ELICITATION HARNESS OWNER + hunt candidate screening
**Last update:** 2026-07-28 03:15 BST (read from `date` — see stamp
corrigendum `a49324ce0`; my earlier stamps ran up to 99 min fast)

---

# RESUME HERE

## State in one paragraph

**The `evalage` lane is CLOSED.** Both blockers cleared, the screen ran,
and the bundle verdict is **WEAK (3/3 legs, 0 KEEP, 0 KILL)** — full
writeup in `evalage/RESULT.md`. **No KEEP ⇒ mac-d's pre-authorized
matrix retrain does NOT trigger.** My pod is **TERMINATED and
API-verified (0 mac-c pods remaining)**. Both harness fixes I owed are
**discharged as code**. `sycgen` is mac-d's to screen on their pod.
**`retryesc_gen` is the only thing left in my queue and is not started.**

## The verdict (`305fada43`)

| leg | tok | window best | gain | wd gain | verdict |
|---|---|---|---|---|---|
| gpt2 | 0.3907 | 0.4307 T64/actx_lin | **+0.0400** | +0.0370 | WEAK |
| gemma2_2b | 0.3970 | 0.4430 T64/actx_mlp | **+0.0461** | +0.0410 | WEAK |
| llama31_8b | 0.4025 | 0.4332 T32/actx_mlp | **+0.0307** | +0.0588 | WEAK |

Bar is gain ≥ +0.05 (with null ≥ +0.02 and own-T floor). Null and floor
cleared on every leg; **the +0.05 was not**, missing by 0.004–0.019.

**Two things to carry forward, one good one bad:**
* **The harness changed the FAILURE MODE.** Same face family on an
  organic corpus (`reask_hr`, **KILLED 3/3**) had within-conversation
  gains +0.017 / **−0.060** / **−0.006** — erased, then reversed, by the
  wd frame, i.e. it was reading position. `evalage`'s wd gains are
  positive on **all three** legs. It is not dying of that confound; it is
  just a small effect.
* **NO order information anywhere.** `actxmean` beats ordered-window on
  all legs; win−shuf ≈0 or negative; `order_pass_wd` False 3/3. For a
  temporal-structure program, a face that cannot beat its own shuffle is
  not evidence of ordered structure even if it had cleared the bar.

`position_floor` at chance 3/3 (0.330/0.322/0.336) despite label-side
Spearman 0.4226 — the balanced manifest controls position by construction.

**Disposition (mine, as design owner): no more GPU on `evalage` as
specified.** The obvious knob (larger T) is barred by the apparatus
(`gather_win` needs anchor ≥ T−1, `OFF_MIN`=63 in 128-tok chunks), not by
a choice. Any follow-up is a NEW card with its own freeze.

⚠ **Do not repeat this excuse:** I nearly published "windows too short
to reach terciles at ages 429/1021" and killed it myself — every age
face here has separation ≫ 64 (`reask_hr` 1021, `sycgen_age` 408,
`retryesc` 2120), so separation ≫ T is the *intended* regime for a
trailing functional.

## Owed fixes: BOTH DISCHARGED (`aa95c4eb1`)

1. **Raw transcripts persisted** — `elicit_lib.save_transcripts` wired
   into `run_elicit`, **and recovered retroactively** for the existing
   corpus via `evalage/dump_transcripts.py` (400 docs / 22,412 turns /
   1,542 event turns / sha256 `8a87fc3154fb3913…`, per-run round trip
   re-asserted).
2. **`vocabulary_control_check` reports BOTH legs** (events/conv,
   tokens/conv) with a `stop` flag. ⚠ **The cv bar 0.35 is PROPOSED, NOT
   RATIFIED** — calibrated only on sycgen 0.749 (fail) vs evalage 0.1346
   (pass). Someone should confirm or move it.

Still standing: **checkpointing is BLOCKING for every generation card**;
mechanism exists in `elicit_lib`, `run_evalage`/`run_sycgen` still need
the 3-line wiring.

## Queue

1. **`retryesc_gen`** — mine, design NOT started. Enters **UNTESTED, not
   rescued** (all its passing bands were label-side, no probe ran) and is
   checkpointing-blocked. No corpus, so no pod is needed yet.
2. `sycgen` — mac-d executing on THEIR pod; I sequence, they run.
3. `evalage` — CLOSED (WEAK). Nothing further unless a new card is cut.

## Corrections I posted this stretch (both mine, both material)

* **Stamp corrigendum** (`a49324ce0`): three LOG entries stamped up to
  **99 min fast**; commit time is authoritative. Root cause — I wrote
  stamps from an elapsed-time estimate instead of reading the clock.
  **Fix adopted: stamp from `date` at write time.**
* **Ledger correction** (`7e614f0b9`): my pre-screen idle burn was
  **$2.24, not the $0.74 I claimed** — API uptime is authoritative. The
  warm-hold cost **7× the screen it was warming for** ($0.30). A
  warm-hold only pays if staging happens DURING the wait; mine ran
  serially after the corpus landed, so it bought nothing.

---

# Reference

## Artifacts (committed + pushed)

Harness: `labels/elicit_lib.py`, `labels/run_elicit.py`.
evalage: `evalage/CARD.md` (§9 backend/provenance), `evalage/SCREEN_CARD.md`,
**`evalage/RESULT.md` (the verdict writeup)**, `evalage/results/`
(3 screen jsons + `verdict_evalage.json`),
`evalage/{screen_grids,cache_acts,screen,verdict,dump_transcripts}.py`,
`evalage/grids/`, `evalage/transcripts_receipt.json`,
`labels/evalage_lib.py`, `labels/build_evalage_premeasure.py` (3-leg),
`labels/evalage_premeasure{,_3leg}.json`.
Transcripts (`labels/elicit_evalage_v1_transcripts.json`) are gitignored
and **deliberately NOT on HF** — unlike the npz they are DERIVABLE, and
`dump_transcripts.py` regenerates them exactly against the committed
sha256.
Corpus **durable on HF**:
`han1823123123/temp-bench-data/hunt_corpora/evalage_20260728/`
(npz sha256 `b5cd16b98e92299ea6e4…`; the `.npz` is gitignored — do not
re-add it).

## Provenance caveat that must travel with any evalage quote

Generated by `claude-haiku-4-5-20251001` (seed 0, temp 0.8) via the
MATS key. The pin is **model-id + API version, not a weight sha** ⇒
**reproducible-in-expectation, not bit-exact**. Labels are unaffected
(the scaffold knows every cue position). Regenerable on open weights
from the same frozen scaffold if bit-exactness is ever required.

## Closed this session (all ratified)

`msdose_r1` KILLED; `sycgen` geometry-passed after my own clock-bar
self-demotion; `dharm` KILLED on document length (155.6 tok/chain);
`warddebt` no-screen (Ward sentence-kernels unreachable at our T);
`retryesc` KILLED (task-vocabulary leak); menu-exhaustion report;
harness scope estimate; kill triage (3 corrections accepted, incl. the
new **structurally-unscreenable** class); corpus-split arbitration;
checkpointing mechanism; `sycgen` disposition as design owner;
**`evalage` SCREENED → WEAK 3/3 (lane closed)**.

⚠ **Calibration reference, stated precisely because I once wrote it
loosely:** `reask_hr` is the in-repo *band* calibration (position AUC
0.925–0.946, doc-mean 0.818–0.828, unigram 0.560–0.575). **The `reask_hr`
CANDIDATE was KILLED 3/3 at screen** — "surviving" in my older notes
meant its label-side runs, never the candidate. Its wd gains
(+0.017/−0.060/−0.006) are the contrast that makes `evalage`'s positive
wd gains meaningful.

## Spend (all pods TERMINATED + API-verified; 0 mac-c pods remaining)

~$0.85 sunk on two generation pods that produced nothing (vLLM would
not build against the image torch). evalage generation inside its $40
cap. **Screen pod `4dztelehvj8l5n`: 00:39 → 03:13 BST = 2h34m ≈ $2.54**
(est $3–6). Of that the actual work — caches + 3 screens + verdict — was
≈18 min ≈ **$0.30**, and **≈$2.24 was idle warm-hold: the hold cost 7×
the work it was holding for.** Ledgered as waste, not absorbed. The rule
I'd give anyone: **a warm-hold only pays if staging runs DURING the
wait.** Mine ran serially after the corpus landed, so it bought nothing.

## Pod / key governance

Key: keychain `dmitrys-runpod-api-key`, **env-inject only**, never
printed/filed/argv. $10/h cap per agent. **Never touch pods I did not
spin up** (mac-d's `jge1fuj9hqu8et` is theirs). Prefer TERMINATE;
verify by API query after; ledger at spin-up AND termination.

Pod staging gotchas, both bit me: `ssh -n` nulls stdin, so **piping a
file into `ssh -n 'cat > …'` silently writes 0 bytes** — verify with
`wc -c`/`md5sum`, never trust the `echo staged`. And zsh does not
word-split `$S 'cmd'` — use a shell function `s() { ssh … "$@"; }`.

## Git / hygiene

Branch `arxiv`, identity `mac-c-agent`. LOG collisions are routine —
union resolve (`sed -i '' -e '/^<<<<<<< HEAD$/d' -e '/^=======$/d' -e
'/^>>>>>>> /d'`, `git add`, `GIT_EDITOR=true git rebase --continue`),
verify with anchored `grep -n '^<<<<<<<'`.
**Listener: re-arm after EVERY wake** —
`zsh <scratchpad>/listener.sh` as a background task. On fire: read the
output file, fetch+rebase, act only if addressed to mac-c, re-arm.
