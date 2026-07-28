# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ⚑ **THE WHOLE TASK HUNT, END TO END** (Han 13:56, briefing
`hunt-mac-c-takeover.md`) — was: elicitation harness + screening
**Last update:** 2026-07-28 14:16 BST (read from `date` — see stamp
corrigendum `a49324ce0`; my earlier stamps ran up to 99 min fast)

---

# RESUME HERE

## State in one paragraph

**I own the entire hunt now**, and **item 7 of Han's deliverable list
has NO candidate** — `sycgen` filled item 6, `struqpos` was killed
honestly and took item 7's only other contender with it. **I am the
only lane that can close it.** `retryesc_gen` is the candidate; its
**GENERATION CARD IS FROZEN** (`3f6ba0d3d`) with a *derived* density
target. Spend so far on this lane: **$0** — no pod, no generation, no
API. **Next action: the ~20-doc pilot**, which measures `floor_excess`
+ Tier T/R and can kill the card at pilot cost.

## ⚑ The finding that reorganized the lane (`density_gain_survey.py`)

**Screen gain tracks IN-WINDOW EVENT MASS.** Recomputed $0 from 150
face×model cells / 48 committed screen artifacts (49 cells, 15 faces
carry a floor): Pearson **+0.699** cell-level, **+0.820 face-level**
(Spearman +0.696 / **+0.882**).

And it is **derivable, not just empirical**: `visible_evidence_floor`
is fit on exactly `(censored_age, in_window_event_count)`, so for a
balanced 3-class age face **`floor_excess` ≡ `f` ≡ P(event inside the
T-window)** — verified exact (worst err **2e-6**) in
`retryesc_gen/verify_floor_identity.py`. The x-axis is a **design
parameter I can aim at before generating.**

| face | floor-excess | gain | verdict |
|---|---|---|---|
| `sycgen_age` | **+0.210** | **+0.117** | **KEEP** (gold) |
| `evalage_age` | +0.045 | +0.039 | WEAK |
| `reask_hr` | +0.034 | +0.018 | KILL |

Band is **two-sided**: every face in +0.15…+0.25 cleared every cell
(4/4); 1 of 11 outside did. Above +0.25, 3/5 cells **lose to their own
floor** (`qd` margin −0.034). ⚠ Band edges are **POST HOC** — the
correlation is the evidence, the bands are a design target.

## Two retractions of my own, both material

1. **The age-face objection (withdrawn 14:07).** I said no age face
   passes the order ladder (0/9, true) and inferred `retryesc_gen`
   shouldn't be an age face (**false**). `sycgen_age` **is** an age
   face, `order_pass_wd` **False**, and it is the **gold** at +0.115.
   Order is **Q3 table-routing, NOT in the hunt4 § 4 KEEP rule.** The
   hub had already folded my argument into the overnight map — the
   retraction is what to act on. **Right lever: sparse vs dense, not
   age vs rate.**
2. **"Capped near 1/3" (wrong, caught by simulation pre-freeze).** I
   wrote that the identity holds only while `T ≤ e1`. It survives to
   `f = 2/3`. The false version said the floor "cannot run away" and
   would have licensed a **much denser corpus as safe**. Truth is the
   opposite: **the floor is computed from GROUND TRUTH and climbs
   toward 1.0, while the arm only reads activations — density hands the
   floor a bar the arm cannot reach.**

## `evalage` — CLOSED (WEAK), and now explained

Verdict `305fada43`, writeup `evalage/RESULT.md`: gains +0.040 /
+0.046 / +0.031 against a +0.05 bar; null + floor clean on every leg.
**The mechanism I could not give at the time is the corpus clock:**
terciles at ages **429 / 1021** = 6.7× the T=64 ceiling ⇒ floor-excess
+0.045 ⇒ dead band predicts +0.032, it scored **+0.039 — on the
curve.** I had the gap median 862 in my own receipts and never
connected it. Disposition unchanged: **no more GPU on `evalage`.**

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
  all legs; win−shuf ≈0 or negative; `order_pass_wd` False 3/3.
  ⚠ **Read with retraction 1 above** — this is a real negative about
  *ordered* structure and about **table routing**, and I wrongly let it
  imply the candidate family couldn't be gold.

`position_floor` at chance 3/3 (0.330/0.322/0.336) despite label-side
Spearman 0.4226 — the balanced manifest controls position by construction.

**Disposition (mine, as design owner): no more GPU on `evalage` as
specified.** The obvious knob (larger T) is barred by the apparatus
(`gather_win` needs anchor ≥ T−1, `OFF_MIN`=63 in 128-tok chunks), not by
a choice. Any follow-up is a NEW card with its own freeze.

## ⚠ Two claims that sound alike — do NOT conflate them

I killed a "windows too short to REACH terciles at 429/1021" excuse
before publishing, and I was right to. The density finding is **not**
that excuse coming back. Keep these apart:

| | claim | status |
|---|---|---|
| **REACH** | separation ≫ T means the face is unreachable / uncomputable | ❌ **still wrong.** A T2 age face is well-defined at any distance; the window reads accumulated state, not the event. `sycgen_age`'s median low edge is **180 > 64** and it is the **gold** |
| **DENSITY** | the fraction `f` of probe rows with an event *inside* the window predicts gain **magnitude** | ✅ the finding — ρ_face +0.88, and `floor_excess ≡ f` exactly |

The lever is the **mean inter-event gap `g`**, not tercile separation
(the edges are a *consequence* of `g`). Back-solved: `sycgen` **g ≈
271**, `evalage` **g ≈ 862**, organic `retryesc` **g = 886**. So my
old instinct that "the clock matters here" was right; I reached for
*reach* when the right variable was *density*, and I should have
pursued it then.

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

Checkpointing: wired into `run_evalage` (`720234442`).
**`run_sycgen` still lacks the 3-line wiring** — the rule is standing
and blocking for every generation card, so wire it before any
`retryesc_gen` generation reuses that path.

## Queue

1. **`retryesc_gen` — CARD FROZEN `3f6ba0d3d`, next action = the
   ~20-doc PILOT.** Enters **UNTESTED, not rescued** (every passing
   band was label-side; no probe ever ran). Design in
   `retryesc_gen/GENERATION_CARD.md`:
   * face = **repeat-failure escalation**, § 1.2-shaped two-timescale
     — indicator needs out-of-window memory (*is this a repeat?*),
     kernel support inside T.
   * **construction rule that makes density safe (binding):** failure
     text is drawn from a fixed pool **independent of repeat-status**,
     so a repeat and a first-time failure are textually
     indistinguishable. If the generator ever makes repeats
     distinctive, the candidate is dead.
   * **vocabulary fix is structural:** failure schedule drawn FIRST,
     independent of task ⇒ difficulty *assigned*, not intrinsic. This
     is the bar `retryesc` actually died on (0.69–0.72 vs 0.60).
   * target **`floor_excess` ∈ [+0.15,+0.25]**; `g` is the only knob
     permitted to move post-freeze, planning centre **170–290 tok**.
   * odds on record **before** the result: magnitude ~70–75 %, leak
     gate ~65–75 % (**the dominant risk**), **joint ~50–55 %**.
   * **A pilot outside the band is a NO-GO I report, not a band I
     widen.**
2. `sycgen` — **DELIVERED** (item 6, FINAL 15/18 at handover).
   Maintenance only; do not reopen.
3. `struqpos` — closed, KILLED SOUND 3/3. Salvage = amendment-window
   only. Triage row applied to `KILL_TRIAGE.md` (runpod-a's wording).
4. `evalage` — CLOSED (WEAK). Nothing further unless a new card is cut.

## Standing rules that bind this lane (from the takeover briefing)

* **Prime directive: a sound verdict, never a win.** A second honest
  KILL is a fine outcome; a soft-pedalled KEEP is not.
* **⚑ GOLD-VISIBILITY (Han, standing, asked twice):** if a gold task is
  found it goes into `REBUTTAL_HANDOFF.md` **the same beat**, not at
  the next tidy-up.
* Generation on `dmitry-mats-claude-api-key`, **$300 cap**, ledger both
  ends, **mac-only — never seeded to a pod**.
* **Hardware:** mac-c and mac-d are both sessions on **one MacBook**
  (M5 Pro, 18 cores, 48 GB unified). The RLHF grid moved to a pod so
  the laptop is effectively mine — but it is still **one 48 GB
  machine**, and GPU-hour budgets go through the hub.

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
