---
agent: agent_paper
ts: 2026-05-03T20:30:00Z
type: handover
prev_handover: first
component: orchestration
---

## Identity

`agent_paper` — orchestrator + paper drafting + C1 + C2 (toy
synthetics). Pod: local 5090. GPU: 0 (only). Mode: `persistent`
(local SSD). Briefing: [`briefing.md`](../briefing.md). I am the
paper-coordinator agent; I do not own H100/A40 training compute.

## State of the world (verified 2026-05-03T20:30Z)

- `git HEAD` to be set by next commit (this handover is being written
  before the framework-cleanup commit lands; `git log -1` after push).
- All 38 framework tests passing in `purified/.venv` (just rebuilt
  from `uv sync`; root `.venv` deleted as orphan).
- `results/leaderboard.jsonl`: empty (no cells run yet — framework
  scaffold only).
- `checkpoints/manifest.jsonl`: empty.
- Locked decisions: `agents/agent_paper/decisions.md` #1–9. Don't re-open.
- HF repos provisioned: `han1823123123/temp-bench-{models,data}` (private).
- Token store unified: `~/.tokens/{hf_token,anthropic_key}` on local;
  `/workspace/.tokens/...` on RunPod. `get_token()` resolves both.
- No agents besides me are provisioned. Five worker briefings (NLP, EM,
  STEER, BACK, EM_H200) are TODO — see `agents/README.md` Active roster.

## What I just did (newest first)

- Wrote handover protocol (PROTOCOL.md § 14), template at
  `agents/_handover_template.md`, and this very handover.
- Reordered PROTOCOL.md sections: 1, 2, ..., 9, 11.0, 11.1, 11, 10
  → 1, 2, ..., 14 in numeric order. Updated cross-refs in CLAUDE.md,
  `agents/README.md`, `runner.py`, `utils/__init__.py`, `gpu_locks.py`
  (§ 11.1 → § 13).
- Trimmed `agents/README.md`: removed duplicated GPU-isolation section
  (now lives only in PROTOCOL.md § 12); fixed concurrency-budget
  table for 4× A40 / 38 vCPU; dropped Agent QA / SYNTH refs.
- Unified Quick start in CLAUDE.md (one block, not two), removed dead
  reference to root CLAUDE.md (now `@purified/CLAUDE.md`).
- Consolidated `.gitignore`: merged `purified/.gitignore` into root
  with `purified/`-prefixed patterns; verified ignore rules with
  `git check-ignore -v`.
- Token storage unification: added `temp_bench/utils/tokens.py`
  (`tokens_dir()` / `get_token()` / `require_token()` /
  `token_status()`); `bootstrap_local.sh` for the local equivalent
  of `bootstrap_runpod.sh`; `cache.py:_push_checkpoint_to_hf` uses
  `require_token("hf")`; sync_from_hf.sh resolves via the unified
  chain. 9 new tests, 38/38 passing.
- Migrated `.env_autointerp` (the legacy local Anthropic key) into
  `~/.tokens/anthropic_key`, validated via Anthropic API, deleted
  the loose file.
- Wasteland code deletion: `src/`, `experiments/`, `references/`,
  `tests/`, `scripts/`, `pyproject.toml`, `uv.lock`, etc. — gone from
  `final`. Wasteland docs (`docs/`, `papers/`, `RUNPOD_INSTRUCTIONS.md`)
  retained because component writeups cite them. 3658 → ~320 tracked
  files. Reading wasteland code = `git show origin/han-phase7-unification:<path>`.
- Replaced root `CLAUDE.md` with single-line `@purified/CLAUDE.md`
  import. Replaced root `README.md` with brief stub. Deleted root
  wasteland tooling (`run-checks.sh`, `.markdownlint.jsonc`, etc.).
- RUNPOD_INSTRUCTIONS.md rewritten for the `final`-branch workflow.

## What I'm in the middle of

Nothing in flight; this is a clean break. The most-recent edits are
all staged and about to be committed in one bundle:
- `.gitignore` consolidation (root .gitignore + delete purified/.gitignore)
- Start-location enforcement (`set_agent_env.sh` rejects non-purified cwd)
- PROTOCOL.md § 13 ↔ § 14 reorder
- Cross-ref fixes for § 13 (multi-GPU)
- Handover protocol (PROTOCOL.md § 14, template, my own first handover)

After commit, `git push origin final` and then I'm done with the day's
framework work. Han may have more directives next session.

## Next action for my successor

1. `cd $(git rev-parse --show-toplevel)/purified`
2. `source scripts/set_agent_env.sh agent_paper`
3. `bash scripts/agent_smoke_test.sh` — expect 38/38 tests + 8 expected
   arch-class import gaps (those are the architectures we haven't
   ported yet — see `configs/locked_archs.yaml`).
4. `git pull --rebase origin final` — pick up anything Han or worker
   agents pushed.
5. Read this handover and `decisions.md` #1–9.
6. **Begin C1+C2 implementation work.** This is agent_paper's actual
   research mandate (not just orchestration). Steps:
   - Port `temp_bench.architectures.{topk_sae,tsae,tfa,txc_base,txc_pro}`
     from origin/han-phase7-unification's `src/architectures/` (one
     file each, with header attribution + `git show` of the source
     commit hash).
   - Implement `temp_bench.data.toy.markov_chain_support` and
     `coupled_hmm` (Phase 2/3 generators).
   - Write `experiments/c1_synthetic_topk/run.py` from the
     `_runner_template.py`.
   - First run: `txc_base` × seed 42 × k=2, ~10 min on 5090. Verify
     `run_cell` writes a leaderboard row + saves a checkpoint.
   - Then full sweep.
7. After C1+C2 are running, draft worker-agent briefings for NLP / EM /
   STEER / BACK so Han can spin them up.

## Don't repeat

- **Don't re-create the root wasteland.** `src/`, `experiments/`,
  etc. are gone on `final`. Read via `git show origin/han-phase7-unification:<path>`.
- **Don't add a third TXC.** decisions.md #1: TXC-base + TXC-pro only.
- **Don't auto-add Bricken.** decisions.md #7: opt-in per component;
  only C6 enables by default.
- **Don't operate from repo root.** `set_agent_env.sh` will refuse;
  always `cd purified` first.
- **Don't allocate run_ids manually.** `runner.run_cell` computes
  `train_key`/`eval_key` deterministically. Bumping `arch_version`
  invalidates train cache; bumping `EVAL_PROTOCOL_VERSION` invalidates
  eval cache.
- **Don't push without `git pull --rebase` first.** PROTOCOL.md § 1.

## Open questions for Han

(none — Han has been driving the framework shape; if there's nothing
new in the chat, proceed to C1+C2 implementation.)
