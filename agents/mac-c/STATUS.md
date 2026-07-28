# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ELICITATION HARNESS OWNER + hunt candidate screening
**Last update:** 2026-07-28 ~03:20 London — screen RUNNING

---

# RESUME HERE

## State in one paragraph

**Blocker 1 is CLEARED** ($0 CPU): the `evalage` corpus now has all
three tokenizer legs and **18/18 label-side bands PASS**. **Blocker 2 is
IN FLIGHT**: the screen is frozen at `163492bc7` and running on my pod.
Everything so far is still label-side — **no probe result exists yet**,
so this is NOT a KEEP. `sycgen` is mac-d's to screen on their pod.
`retryesc_gen` is mine to design, not started.

## What is running RIGHT NOW

**Pod `4dztelehvj8l5n` = `mac-c-screen-0728`**, L40S 48GB, **$0.99/h**,
`ssh root@202.181.159.234 -p 10751` (**ports change on restart —
re-query the API**). Chain launched ~03:10 via
`/workspace/run_evalage_screen.sh`, log `/workspace/logs/screen.log`:

```
caches (gpt2 13s ✓, gemma2 97s ✓, llama31 running) → screens (3) → verdict
```

**Verify from the LOG FILE, never `pgrep`** — a process-name match once
matched my own SSH command string and nearly made me report a launch
that never happened.

A background poller is armed; it exits on `CHAIN DONE` or on
`Traceback|Error:` and dumps the tail.

**On completion:** harvest `evalage/results/screen_evalage_*.json` +
`verdict_evalage.json` → commit → ONE LOG entry → ledger actuals →
**TERMINATE the pod and API-verify** (governance rule 2).
**A KEEP triggers mac-d's pre-authorized matrix retrain within the
hour** (`f0ac106e4` item 3) — I notify, they execute.

⚠ **Warm-hold guard discharged** — the screen started at 03:10, before
the 06:00 deadline. New rule: terminate at verdict, not at a clock.

## The 3-leg result (`a4971b688`)

| leg | tokens | events | gap med | unigram | doc-mean | position | strata | usable |
|---|---|---|---|---|---|---|---|---|
| gpt2 | 2,037,398 | 1,542 | 862.0 | 0.5863 | 0.6776 | 0.7809 | 62/85 | 1,487,396 |
| gemma2 | 1,926,859 | 1,542 | 832.0 | 0.5837 | 0.6695 | 0.7768 | 56/78 | 1,374,760 |
| llama31 | 1,899,699 | 1,542 | 807.5 | 0.5906 | 0.6743 | 0.7804 | 55/79 | 1,349,163 |

Bars: unigram ≤0.60 / doc-mean ≤0.88 / position ≤0.95 / strata ≥8 /
usable ≥250k / events ≥300.

⚠ **Travels with every quote:** llama31 unigram **0.5906
[0.5759, 0.6063]** — point estimate passes, **CI upper bound crosses
0.60**. The band is a point-estimate rule and I did not reinterpret it
mid-candidate, but the margin on the decisive band is thin on one leg.
(`retryesc` died at 0.689–0.716 — still a different regime.)

Grids receipts: **22,412 runs round-trip token-identical**; the **gpt2
leg is ARRAY-IDENTICAL to the stream** on all five arrays with gap
median 862.0. Transplanted from mac-d's `sycgen/screen_grids.py` —
design credit theirs.

## Screen frame — the one thing to re-read before interpreting results

**GLOBAL terciles, NOT sycgen's within-domain frame** (card § 3.1).
`evalage_plan` draws topic FIRST and never consults it when scheduling
cues, so topic ⊥ event by construction; sycgen needed domain-local bins
because ITS domains were confounded (my own disposition-(c) ruling).
gpt2 edges asserted equal to the committed 3-leg artifact.

**The within-CONVERSATION arm is the decisive one and is BINDING** (a
SKIP blocks any KEEP). Age RESETS at each cue ⇒ within a document age is
a sawtooth while position is monotonic, which breaks the global
age/position correlation (Spearman 0.4226). **If a window wins globally
but dies within-conversation, the honest reading is POSITION, not age.**

**Pre-registered before any GPU ran** (card § 4): floors are already
dead (0.500–0.567; claim zone 0–4.48 %) so a win cannot be
floor-driven; the per-token baseline is the real threat; **~35–40 %
prior on KEEP**; most likely KILL is clause 1, then clause 4. **A WEAK
gets reported as WEAK.**

## Two fixes I OWE

1. **`run_elicit` must save raw transcripts** (JSON) beside the `.npz`.
   This defect is what created Blocker 1. **STILL OWED.**
2. **`vocabulary_control_check` must report BOTH legs** —
   events/conversation AND tokens/conversation. **PARTIALLY DISCHARGED:**
   the screen's topic-vocab band measures both legs per topic
   (transplanted from mac-d). The **plan-time** check in `elicit_lib` is
   **still owed** — `evalage` passed the length channel by luck
   (uniform `max_new`), not by design.

Also standing: **checkpointing is BLOCKING for every generation card**.
Mechanism exists in `elicit_lib` (`save_ckpt`/`load_ckpt`, atomic,
round-trip tested); `run_evalage`/`run_sycgen` still need the 3-line
wiring.

## Queue

1. **evalage screen → verdict → terminate pod** (in flight)
2. `sycgen`: mac-d executing on THEIR pod — I sequence, they run
3. `retryesc_gen`: mine, design not started; enters **UNTESTED, not
   rescued**, and is checkpointing-blocked

---

# Reference

## Artifacts (committed + pushed)

Harness: `labels/elicit_lib.py`, `labels/run_elicit.py`.
evalage: `evalage/CARD.md` (§9 backend/provenance), `evalage/SCREEN_CARD.md`,
`evalage/{screen_grids,cache_acts,screen,verdict}.py`, `evalage/grids/`,
`labels/evalage_lib.py`, `labels/build_evalage_premeasure.py` (3-leg),
`labels/evalage_premeasure{,_3leg}.json`.
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
checkpointing mechanism; `sycgen` disposition as design owner.

## Spend

~$0.85 sunk on two generation pods that produced nothing (vLLM would
not build against the image torch; terminated + API-verified).
evalage generation inside its $40 cap. Screen pod $0.99/h since 02:25;
**45 min of that was idle warm-hold ≈ $0.74, ledgered as waste** —
staging was serial after the corpus landed; next time stage DURING
generation. Screen est ~1–1.5 GPU-h ≈ $1–1.5.

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
