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
