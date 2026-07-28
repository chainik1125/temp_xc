# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`)
**Lane:** ELICITATION HARNESS OWNER (authorized `63864ae66`, 2026-07-28)
**Last update:** 2026-07-28 ~01:35 London

## Current task: harness build — cards FROZEN + RATIFIED, generation NOT yet running

**Frozen & pushed `1a955344c`; RATIFIED by mac-local** ("what
full-throttle looks like done right"; vocabulary control as a STOP
condition singled out; CPU-stub claim-zone validation before GPU spend):

- `labels/elicit_lib.py` — harness core. Tonight's three kill-lessons
  are **scaffold parameters, not comments**: event spacing chosen for
  the clock (`realised_gaps` receipt), topic drawn independently of the
  event schedule, no sentence-scale kernels.
  **`vocabulary_control_check` is a STOP condition** — spread ⇒ the
  `retryesc` leak is being rebuilt ⇒ do not trust the corpus.
- `labels/evalage_lib.py` + `evalage/CARD.md` — corpus 2 pick.
  **Deliberately NOT menu #12** (a marker-RATE face I predicted dies);
  bars-first **T2 age** redesign. CPU stub: claim zone
  **0/0/0.8/5.5/14.7 %** at T=4/8/16/32/64; plan vocab cv **0.048**.
- `labels/run_elicit.py` — turn-major batched runner (all docs advance
  one turn per step; causally correct within a document).

## Pod + spend (governance-compliant)

**`vyp2zlq13cf7df` = `mac-c-hunt-0728`**, L40S 48GB, **$0.99/h**.
SSH `root@103.196.86.47 -p 42839` (key `~/.ssh/id_ed25519`).
Ledgered at spin-up AND the correction in `MODAL_SPEND.md § RUNPOD`.
**Spend so far ≈ $0.5 of the ≤$100 slice.** I own this pod alone and
have touched no other. Key: keychain `dmitrys-runpod-api-key`,
env-inject only, never printed/filed/argv.

**Backend = pod-hosted OPEN WEIGHTS** (`Qwen/Qwen2.5-7B-Instruct`).
The Anthropic personal key was withdrawn (`a073c3913`); my choice was
already open-weights and is unchanged — pinned weights give exactly
reproducible provenance, and the probe target is our own models reading
the TEXT, so generator choice affects realism, not what we measure.

## Bring-up: 4 attempts, all POD-ENVIRONMENT mechanics (design untouched)

1. First pod had no `PUBLIC_KEY` env ⇒ sshd refused ⇒ **terminated at
   ~3 min ≈ $0.05**, recreated with the key. Ledgered as waste, not
   absorbed.
2. `pgrep -f run_elicit` **false positive** (matched my own SSH command
   string) — nearly reported a launch that never happened. **Verify
   from the LOG FILE, never a process-name match.**
3. zsh does not word-split `$S 'cmd'` ⇒ staging was a silent no-op.
   Fixed with a shell function.
4. `pip install vllm==0.6.3.post1` **downgraded torch to cu121 and
   broke the system transformers** (`DTensor` ImportError). Fix in
   flight: isolated venv at `/workspace/hunt/venv` so vLLM pulls its
   own matched torch; system image left alone.

**Staged on pod (committed files verbatim, no retyping):**
`/workspace/hunt/pkg/{__init__,elicit_lib,evalage_lib,run_elicit,lib}.py`
(`lib.py` = a 6-line `doc_split` shim so package-relative imports
resolve).

## LIVE STATE (2026-07-28 ~02:30) — two lanes running in parallel

**1. GENERATION — `evalage` v1 RUNNING on the Claude API** (confirmed
from the log, not pgrep). Local process, log at
`<scratchpad>/evalage_v1.log`; writes
`labels/elicit_evalage_v1.npz` + `_receipt.json`.
Model `claude-haiku-4-5-20251001`, 400 docs, seed 0.
Cost ~$25–34 of a $40 pre-registered cap (revised UP from $10–25:
each turn re-sends the transcript). Smoke (4 docs) passed: gaps
244–1731 tok, **vocab spread 0.0004** on generated text.

**2. SCREEN POD — `4dztelehvj8l5n` = `mac-c-screen-0728`**, L40S
48GB, **$0.99/h**, 150GB vol, `PUBLIC_KEY` injected at create.
**Stated purpose (required by the amended warm-hold policy):**
pre-stage gpt2/gemma2/llama31 + cache builders so the `evalage`
screen starts within minutes of the corpus landing; then screen
evalage → sycgen_age → retryesc_gen. **Warm-hold until the LANE is
done, not between stages.** Get its SSH port via the API `pod`
query (ports change per pod). NOT yet staged — that is the next
pod-side task.

**Generation pods are GONE** (`tbxn8b3rsk1hnt`, `vyp2zlq13cf7df`):
terminated, API-verified, ~$0.85, zero output — vLLM would not
build against the image torch.

## ⚑ TOP PRIORITY ON RESUME: SCREEN `sycgen` (GO given; I own execution)

Order `dc3cb8fd9`. My (c) disposition worked — mac-d's within-domain
analysis RESCUED sycgen without regeneration (within-domain doc-mean
**0.636–0.795** vs 0.858 pooled/confounded; position 0.608–0.731;
**511,907 usable tokens** ≥ 2× bar; 158 strata; trivia_qa thinness
disclosed). **v2 stays shelved.**

**SCREEN POD IS WARM AND WAITING:** `4dztelehvj8l5n` =
`mac-c-screen-0728`, L40S, $0.99/h,
**ssh `root@202.181.159.234 -p 10751`** (ports change on restart —
re-query the API if refused). Uptime ~80 min at handoff. NOT yet
staged with tokenizers/cache builders.

**FIVE BINDING IN-CARD CONDITIONS (verbatim from the order):**
1. **WITHIN-DOMAIN frame is the pre-registered readout** — all arms,
   all floors, all baselines within-domain.
2. **PER-TOKEN BASELINE FIRST** (generated corpus, standing rule —
   this is what killed `emoinst`).
3. **Vocab band re-measured WITHIN-DOMAIN as part of the screen** —
   the STOP fired on the pooled frame, so the screen must carry the
   within-domain vocab numbers BESIDE the verdict.
4. **hunt4 § 4 KEEP/KILL verbatim.**
5. v2 shelved unless the screen surfaces a leak the frame does not
   control.

**KEEP ⇒ mac-d's warm-pod matrix retrain within the hour
(pre-authorized).** mac-d supports; execution is mine.

**Handoff note:** I ran out of context before staging the pod. Nothing
is half-done — no screen started, no partial artifacts. Start from
staging.

## NEXT ACTION on resume

1. `ssh -p 42839 root@103.196.86.47 'tail -20 /workspace/hunt/evalage_v1.log'`
   — confirm generation from the LOG, not `pgrep`.
2. If not running: `cd /workspace/hunt && HF_HOME=/workspace/hf nohup
   ./venv/bin/python -m pkg.run_elicit --scaffold evalage --model
   Qwen/Qwen2.5-7B-Instruct --n-docs 400 --out-tag v1 >
   evalage_v1.log 2>&1 &`
3. On drain: scp back `elicit_evalage_v1.npz` +
   `elicit_evalage_v1_receipt.json`; **check `vocabulary_control` and
   `realised_gaps` against the card BEFORE any screen**; then the
   label-side bands (unigram ≤0.60, doc-mean ≤0.88, position ≤0.95,
   ≥8 strata, ≥250k usable, ≥300 events) on all 3 tokenizers.
4. **TERMINATE the pod at drain + API-verify**, actuals line to
   `MODAL_SPEND.md`.
5. Then corpus 1: `sycgen_age` scaffold (constants already frozen in
   `sycgen_lib`; needs `are_you_sure.jsonl` seeds from the pinned
   `meg-tong/sycophancy-eval` repo).
6. **Delete `briefings/safety-hunt-continuation.md`** once generation
   is confirmed running (its closing line).

## Standing rules on this build

Per-token baseline **binding** on every generated corpus; "geometry can
kill but not clear"; bands **absolute only** (`msdose_r1` lesson);
cards frozen before generation; full generation provenance in-receipt;
corpus is **model-generated and disclosed as such** (exhibit-vs-appendix
is the paper owner's call).

## Closed earlier (all $0, all ratified)

Round 1–2 (menu, txc_pro dig, second-source, re-entry packets, Tier-C
designs). Execution lane: `msdose_r1` KILLED, `sycgen` ratified
single-face after my own clock-bar self-demotion, `dharm` KILLED on
document length (155.6 tok/chain). Hunt continuation: `warddebt`
no-screen (Ward sentence-kernels unreachable at our T), `retryesc`
KILLED (task-vocabulary leak), menu-exhaustion report → this build.

## Git / hygiene

Branch `arxiv`, identity `mac-c-agent`. LOG collisions: union resolve
(`sed -i '' -e '/^<<<<<<< HEAD$/d' -e '/^=======$/d' -e '/^>>>>>>> /d'`,
`git add`, `GIT_EDITOR=true git rebase --continue`), verify anchored
`grep -n '^<<<<<<<'`. Listener: re-arm after EVERY wake.
