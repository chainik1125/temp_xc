# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`; read-only archaeology + literature lane, $0 compute)
**Briefings:** none active — `briefings/safety-task-research.md` DELIVERED and DELETED (`a2d0745b1`)
**Last update:** 2026-07-27 ~20:00 London (safety menu + txc_pro dig both closed)

## Current posture: NO OPEN ASSIGNMENT — watch + review-support

Every dispatched item is closed. Standing obligations only:

1. **Re-arm the listener after every wake** (Han's explicit rule, 10h
   window): `zsh <scratchpad>/listener.sh` as a background task.
2. Support the one-pager / meeting on request. PTR discipline on pushes.

**SAFETY_TASK_MENU review landed (`ae1ce5fb0`, mac-local): ACCEPTED as
the wave-3 source, nothing routed back to me.** Both menu principles
were adopted as **binding review bars** for every wave-3 safety card —
(1) out-of-window-by-construction, (2) clock-stated-first — with
runpod-a directed to bounce cards on them. All four § 7 $0-kill
recommendations endorsed (formal kill lines stay runpod-a's), and the
**ethics note is BINDING** (no crisis-escalation face without Han's
explicit sign-off AND a synthetic substrate). Execution is runpod-a's:
CPU pre-measures for the zero-pull trio (`sycpress`, `reask`, `msdose`)
starting during RM shard-2, anti-dup vs `refmark` first, event-mass
check for `reask`; **`DecomposedHarm` approved as the one new pull**.
One item went to Han, not me: add a fetchable URL for *Many-shot
Jailbreaking* in Zotero (my § 9 registry-gap note).

## Delivered this session (both briefing items)

- **`SAFETY_TASK_MENU.md`** (`d44843ae7`) — 16 ranked safety-relevant
  trailing-state candidates in 3 feasibility tiers, from the 2026-07-27
  meeting briefing. Research inventory, NOT a freeze. Spine: (a) the
  design principle that separates `tret`/`sage` (survived) from
  `refmark` (died) — **the event indicator must depend on out-of-window
  information while the kernel support stays inside the window**;
  (b) the clock argument — turn-scale safety events at 125–144
  tok/message are reach-limited under a rate face, so **the T2 age face
  (`sage`-class) is the workhorse**; (c) four reusable label templates,
  each with an audited in-repo precedent. Tier A (runnable on committed
  data): `sycpress`, `reask`, `dharm`, `msdose`. Recommends
  `DecomposedHarm` as the one pull worth taking, and 4 candidates for
  $0 kills at design review. clew-only sourcing; **the on-loan S2 key
  was never touched**.
- **`TXC_PRO_RECOVERY.md`** + **`docs/recovered/txc_pro_phase5b_subseq_h8.py`**
  (`a2d0745b1`) — the dig. **The implementation SURVIVED in git**
  (`git show 5dd7337b2^:purified/src/temp_bench/archs/txc_pro.py`, blob
  `480f3755d`, 496 lines, already v2-ported), correcting the "impl lost
  in purification" reading. Two hparam corrections (`n_matryoshka: 8` is
  a **phase id, not a level count** — real control is `h_size =
  d_sae//5 = 3686`; `k_pos: 20` was missing), the train-vs-inference
  budget asymmetry flagged for any T-sweep, and the A12-aware verdict:
  **zero real probing T-scaling evidence for txc_pro anywhere** (the
  phantom T-replicas were `txc_base`, not this arch).

## Earlier deliverables still live (receipts in LOG + the files)

- **HF mirrors COMPLETE + RATIFIED** — `hunt_payload_bundles/` = 455
  files + manifest.jsonl (sha256 + source volume:path + mtime) + README,
  remote-verified 457/457. **Token rotations unblocked from my side.**
  Two process notes adopted as house practice: plain `modal volume ls`
  TRUNCATES (use `--json`); `modal volume get` can wedge indefinitely
  (120s watchdog, 3 attempts, `--force` after `rm`).
- **`GEN4_CORPUS_SCOUT.md`** — consumed by wave-2, which closed: `sage`
  landed in WRITEUP § 8 as a new intensity-family breadth row, `drev`
  took a $0 kill citing my § 4, `tretd`/`tret_wt` died on the merits.
- **`COMPOSITION_AUDIT.md`** — the reference record for paper
  compositions; A12 (the c3 T5-replica phantoms) is the load-bearing
  finding, cited again by this session's dig. `tbm_census.jsonl` (1,283
  ckpt configs), ONEPAGER_SKELETON.md, WRITEUP § 9 R30 staging.

## Security / hygiene constraints (still in force)

Token at `~/.tokens/hf_token_datasets`, S2 key in the macOS keychain —
**paths only; values NEVER printed, logged, committed, or passed as
argv**. All tokens rotate after the weekend (team action). `clew` is
read-only to agents (no `sync`, `register`, `--refresh`). No Modal spend
from mac-c.

## Git position

Branch `arxiv`, pushed through `a2d0745b1`. Identity `mac-c-agent`. My
pushes touch: `experiments/explorations/task_hunt/*` (my own docs +
append-only LOG), `docs/recovered/`, `agents/mac-c/`. LOG collisions are
routine — resolve by **union** (`sed -i '' -e '/^<<<<<<< HEAD$/d' -e
'/^=======$/d' -e '/^>>>>>>> /d'`, `git add`, `GIT_EDITOR=true git
rebase --continue`), then verify with `grep -n '^<<<<<<<'` (note: bare
`grep -c '<<<<<<<'` false-positives on older entries that *quote* marker
syntax in prose).

## If resuming from compact

Read this file, then the LOG tail. **No open task** — do not start new
work without a listener hit or a direct request. On a listener fire:
read the output file, `git pull --rebase --autostash origin arxiv`, read
the triggering LOG/briefings content, act only if addressed to mac-c
(otherwise note-and-hold), and **immediately re-arm the listener**.
