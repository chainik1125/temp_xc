# mac-local — STATUS · SNAPSHOT #6 (2026-07-29 15:3x BST)

**Supersedes SNAPSHOT #5 (07-28 22:0x), whose live facts had gone false**
— it read *"3 unattributed pods at $6.42/h"*. **Fleet is empty.**

I am the **hub**: review, ratify, rulings, the binding LOG
(`experiments/explorations/task_hunt/LOG.md`), ledger oversight, and the
handover surfaces (`REBUTTAL_HANDOFF.md`, `REBUTTAL_CODE_GUIDE.md`,
`REBUTTAL_CELL_CENSUS.md`). **No compute of my own.** I coordinate
`mac-c` and `mac-d` **only via git pushes + the LOG**.

---

# RESUME HERE

## Fleet & spend
**0 pods, $0.00/h — API-verified 15:28 07-29** (`scripts/pod_inventory.py`;
**stdin is the curl RESPONSE, not the key**). Re-query before quoting:
the audit gives volatile `API-verified` claims a **1-hour** budget and
fails the build when they age out. That is working as intended —
**re-query, never re-stamp without checking.**

## Nothing is queued for either agent
Item (1) — the grid-vs-cache blast radius — is **CLOSED on both
questions**: 0 results data-corrupted, and no value from the radius
reached any reviewer-bound surface. The real defect was **provenance,
not corruption**. See `facecmp/results/PROVENANCE.md`.

## ⚑ WAITING ON HAN — do not act on these unilaterally

1. **Send the proposal to Dmitry.** Finished and verified:
   `docs/dmitry/reviewer_responses/PROPOSED_sycgen_excerpt_reviewer1.md`
   (trained TXC only, no untrained control, 6/6 rows exact against
   `frontier.json`).
2. **`dmitry-txcwins-10h` carries an unsubmitted sycgen section** — my
   `cc9274b6c`, added on Han's instruction; Dmitry submitted none of it.
   Three options in LOG `340a255b9`: leave / revert my block / sync to
   the excerpt. **The section is numerically CLEAN** — mac-d's 0.537
   report was refuted and they retracted it (`4919deeb3`); `4a1f7c735`
   had already landed the fix on that branch. Scope question, not a
   correctness one. **Branch untouched.**
3. **Token rotation** — the three pod-staged tokens (`gh`, `hf_token`,
   `hf_token_datasets`). Still Han's call.

## Deliberately unspent until after Aug 3
**No cache builder in the repo records a grid — zero, not one.** One
cache has the field only because `facecmp/cache_evalage_512.py` (now
committed; it was untracked and would have died with the scratchpad)
wrote it. When spent: add `"grid"` + a **derived** `substrate` to
`evalage/cache_acts.py`, `facecmp/cache_local_mps.py`,
`facecmp/cache_local_mps_512.py`. Nothing shipped, no number moved.

## Frozen / hands-off
- **EM section** — I synced OUR arxiv copy's steering cells to the
  confirmed `17 / 20 / 23` (`:187`, `:398`) after establishing seed 2
  completed (Dmitry rewrote the prose to *"full Wang protocol for all
  three seeds"*). My 01:52 freeze is **lifted for our copy only**.
  **Dmitry's branch stays untouched.**
- **Pushing to a collaborator's branch is a deliberate act** even where
  authorised.

---

# THE RULE THAT COST THE MOST TODAY

**An instrument reporting truthfully about a state that is not the one
under test.** Seven instances across all three agents in ~15 hours —
every instrument worked correctly:

| check | measured | claimed |
|---|---|---|
| `rev-parse` mid-rebase | detached position | the shipped state |
| conflict counter | `<<<<<<<` | any residue (`>>>>>>>` shipped) |
| `assert` after filter | surviving rows | all rows |
| mtime / first-commit | branch-switch / path birth | content landing |
| my greps (×3) | expected literal shape | the property |
| subject grep | prose quoting a commit | the commit on origin |
| commit read | one commit's content | the branch's state |

**Before trusting a check, say out loud which state it observes, and
confirm that is the state in question.** Corollary, earned twice today:
**a negative grep is the direction that lies — run the positive control
on the same files with the same command first.**

**Hub-specific:** I quote agent commit subjects verbatim in the LOG, so
**name-based push verification is unreliable by construction here.**
Verify by content on origin: `git show origin/arxiv:<path>`.

**Also standing (00:23 ruling):** no guard work unless a guard actually
fails on real input. Preserving an existing artifact is not construction
and is exempt.

---

# STANDING CONSTRAINTS (unchanged)

- Keychain: `dmitrys-runpod-api-key` (**mac agents only, NEVER seeded to
  pods**), `dmitry-mats-claude-api-key` (mac-only), `s2-api-key`.
  Never echo, never argv, never into files/logs/scripts.
- **S2:** 1 req/sec **cumulative across all users**; space ≥1.1 s, use
  `fields=`, honour `Retry-After`, cap ~3 retries, fail loudly. Don't
  run a direct S2 workload while a `clew sync` is running.
- `private/**` never enters tracked files.
- **Agents must not modify pods they did not spin up** (look-don't-touch)
  — and per mac-d's `2aabbb3af`, **that extends to files, not just pods.**
- Pod naming `<agent>-<purpose>-<mmdd>`; terminate at lane end +
  API-verify; ledger both ends. Never `set -x` near a secret.
- **Rebuttal voice: clear, plain language, NO AGENTIC JARGON.** 10K char
  limit per response, **no links, no images.** Amendable to **Aug 3**.

# BEFORE ANY COMMIT
`.venv/bin/python scripts/handoff_audit.py` — and **`&&` it, don't `;`
it**. I `;`-separated it at 15:4x and pushed through a failing gate.
