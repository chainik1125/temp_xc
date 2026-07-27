# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane (CHANGED 07-27 19:15, `47040da59`):** HUNT EXECUTION — no longer
read-only. dharm end-to-end + my two re-entry cards, CPU-first, one
self-provisioned GPU pod only when a GPU stage is actually reached.
**Last update:** 2026-07-27 ~19:20 London (lane ACK pushed; starting
msdose amendment card)

## The assignment (LOG 19:15, mac-local relaying Han)

1. **`dharm` end-to-end** — the ONE approved corpus pull
   (`YuehHanChen/DecomposedHarm`), pulled under the `pull_pg19.py`
   rules I own; then the four $0 gates from menu entry #3 (unigram
   triage AUC FIRST — it can kill for $0; identity-in-kind =
   within-decomposition readout only; subtask-index↔position;
   boundary floor per T); screen behind a frozen card iff gates pass.
2. **Both re-entry cards I authored** (`WAVE3_SECOND_SOURCE.md`
   addendum): msdose per-doc-scale amendment + realised pre-measure
   (CPU, $0); sycpress generator-mode card freeze ($0 part —
   elicitation spend is a separate pre-registered decision).
3. **GPU screens for survivors**: hunt4-clone harness, scorer-first,
   cold-cache priced in-card.

**Ownership split**: runpod-a keeps reask + original trio + wave-3
GPU behind RM. Anti-dup binds BOTH ways. Cards + verdicts PTR;
mac-local reviews on push.

## Pod governance (BINDING, actmix-shared § RunPod API)

Dmitry's key, keychain service `dmitrys-runpod-api-key` — env-inject
only (`export RUNPOD_API_KEY="$(security find-generic-password -s
dmitrys-runpod-api-key -w)"`), NEVER echo/print/file/argv. $10/h max
across my pods; name `mac-c-hunt-<mmdd>`; ledger line in
`briefings/MODAL_SPEND.md` § RUNPOD at spin-up (pod id, config, $/h,
purpose) AND termination (actuals); prefer TERMINATE over stop;
verify state change by API query after; NEVER write to pods I did
not spin up (incl. Han's three hand-provisioned pods). Key NEVER
seeded to any pod. L40S/A100-class ≈ $1–2/h target.

## Progress ledger (update as stages close)

- [x] Lane ACK pushed (LOG 19:19)
- [ ] msdose: amendment card committed (commit-then-run) → realised
      pre-measure vs the pre-registered simulated bound (ρ 0.844,
      10/66 strata, 397,481 usable tokens; miss = kill)
- [ ] dharm: `pull_dharm.py` + receipt committed → pull → 4 gates
- [ ] sycpress: generator-mode card frozen ($0)
- [ ] GPU stage decision (only if survivors exist)

**Substrate notes:** runpod-a's frozen-plan msdose streams exist
(`labels/wave3_msdose_{gpt2,gemma2,llama31}.npz`) — those are the
KILLED construction; my amendment builds fresh plan + streams, does
NOT overwrite theirs. Their instrument: `labels/build_wave3_trio.py`
+ `wave3_trio_stats.json`; match its measurement conventions so
numbers are comparable.

## Round 1+2 (closed, all ratified)

Menu (`d44843ae7`) + §10 addendum, txc_pro dig (`a2d0745b1`),
second-source + re-entry packets (`8a51347d5`, `248049349`), Tier-C
designs (`c36fe9c62`). Round 2 ratified `4d9a900ed`; emoinst erratum
CONFIRMED and WRITEUP §8 row corrected by mac-local (my flagged open
item — now closed). Standing rule proposal on record: where §8 and
the LOG disagree, the LOG wins.

## Security / hygiene (in force)

HF token path `~/.tokens/hf_token_datasets`, RunPod key in keychain —
paths/service-names only, values NEVER printed/logged/committed/argv.
All tokens rotate post-weekend. clew read-only (no sync/register/
--refresh). S2 key untouched. No Modal spend from mac-c (pods are
RunPod, ledgered separately).

## Git position

Branch `arxiv`, identity `mac-c-agent`. LOG collisions: union resolve
(`sed -i '' -e '/^<<<<<<< HEAD$/d' -e '/^=======$/d' -e '/^>>>>>>> /d'`,
`git add`, `GIT_EDITOR=true git rebase --continue`), verify with
anchored `grep -n '^<<<<<<<'`.

## If resuming from compact

Read this file, LOG tail, then the progress ledger above — continue
the first unchecked stage. Listener: re-arm after EVERY wake
(`zsh <scratchpad>/listener.sh` background, 10h). On fire: read
output file, fetch+rebase, act only if addressed to mac-c,
re-arm.
