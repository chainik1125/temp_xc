# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane (CHANGED 07-27 19:15, `47040da59`):** HUNT EXECUTION — no longer
read-only. dharm end-to-end + my two re-entry cards, CPU-first, one
self-provisioned GPU pod only when a GPU stage is actually reached.
**Last update:** 2026-07-27 ~20:32 London (LANE CLOSED — all three
candidates resolved for $0; no pod ever spun up)

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

## Progress ledger — LANE CLOSED

- [x] Lane ACK (LOG 19:19)
- [x] **`msdose_r1` — KILLED $0** (`86f5ce0f8`). Freeze approved
      `04b179c31`. Absolute legs all passed (ρ 0.838, 15/74 strata,
      489k usable — beat my own sim bound); **ratio legs missed 3/3**
      because my § B baseline sim understated the frozen plan 2.3×
      (86.6k sim vs 201.5k realised ⇒ "4.6× gain" is really 2.43×).
      Erratum recorded; frozen rule not overturned. My recommendation:
      **no third msdose entry.**
- [x] **`sycgen` — RATIFIED, single face** (`51bf6fabc` freeze,
      `10362af34` result; ratified `8a7c722b2`). Jittered scaffold
      gives the best trap numbers in wave-3 (age pos-AUC 0.689 /
      doc-mean 0.747; rate ρ(face,pos) −0.020) — **and I demoted my
      own rate face** on the binding clock bar (8-msg kernel = 1,014
      tok vs T ≤ 64 = refmark's death mode; its 0.624 floor is
      doc-identity in a costume). `sycgen_age` carries. mac-local
      adopted my "geometry can kill but not clear — **per-token
      baseline binding first on any generated corpus**" as a STANDING
      RULE for all generated-corpus faces.
- [x] **`dharm` — card + pre-measure FROZEN BEFORE ACCESS EXISTS**
      (`d731f3411`). Primary face changed from my own menu entry to
      **`dharm_thage`** (age since `harmful_index` crossing) — escapes
      both the doc-identity trap and the msdose position trap, and
      satisfies § 1.2 by construction. Two predictions pre-registered
      from a synthetic-stand-in smoke (artifacts deleted, none
      committed): `dharm_dose` dies on position (stand-in AUC 1.000);
      `dharm_bage` risks a floor-solve (1.000 at T ≥ 16) ⇒ clock
      reported before any AUC.
- [x] **`dharm` — KILLED $0** (`a85724000`). Han cleared the gate
      ~20:15; pull clean (4,641 chains, zero funnel losses). **Schema
      amendment pushed BEFORE the run**: the HF README is wrong
      (`decomposition` encoded differently per modality, `harm_index`
      1-BASED, `id` not unique) and **764 ids span the shipped splits**
      ⇒ split grouped BY ID, shipped splits discarded. Faces/bands
      untouched. **The clock killed it: 155.6 tokens per DOCUMENT**,
      18.2 tok/subtask, **3 position strata in the whole corpus**. All
      3 faces dead on all 3 tokenizers — `thage` doc-mean 0.993
      (sycpress severity), `dose` position 0.993, `bage` floor-solved
      at 1.000 for T ≥ 8. **Both my pre-registered predictions
      confirmed**; unigram alone (0.712–0.883 vs 0.60) would have
      killed all three.
- [x] GPU stage — **CLOSED WITHOUT ACTION, no pod ever spun up.** No
      GPU-needing stage was reached: two candidates died on $0 CPU
      gates, the third needs a generation decision that is not mine.
      Correct outcome under the governance rule.

## Lane closed — total spend $0

Three candidates resolved, no compute bought. **The one thing still
live from my work is `sycgen_age`**, which passed geometry and awaits
someone's decision on the shared elicitation harness
(`TIERC_PIPELINE_DESIGNS.md` § 3 — four candidates want it). Not mine
to fund.

**Standing bar proposed to the menu (from dharm's death):** *before
recommending any corpus pull, measure tokens-per-document against the
T values we screen.* My § 8 inventory ranked substrates by availability
and label quality, never by length — that gap cost a gate request
(and $0 of compute). `dharm` § 7.1 carries the full statement.

**Gate-terms flag is now MOOT for dharm** (corpus gitignored, not
committed; receipt + pre-measure carry no prompt text) but stays live
for any future gated pull.

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
