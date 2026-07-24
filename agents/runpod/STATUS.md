# Working state — agent `runpod`

**Last rewrite:** 2026-07-24, immediately after receiving
`briefings/hunt-support-synthetic.md` (pre-compact; NOT started yet).
**State: NEW TASK ASSIGNED, ZERO WORK DONE ON IT.** The previous task
(TXC-pro dissection) is DONE + REVIEWED & APPROVED (briefing retired).

## Who / where
Remote CC on RunPod (Linux), repo root `/workspace/temp_xc`. **I am `runpod`
(original box — `/workspace/.agent_id` does NOT exist; do not create it).**
Parallel agents: runpod-b/-c/-d/-e (task-hunt + em-redo arms) — their
briefings are NOT mine. Shared-branch rules (agents/README.md): `git pull
--rebase origin arxiv` before EVERY push (commit this STATUS first); shared
files append-only; leaderboard/manifest union-merge; **cite commits by
SUBJECT LINE or re-verify SHAs post-push**. Tokens in `/workspace/.tokens/`
(`gh_token`, `anthropic_key` → export ANTHROPIC_API_KEY). Push:
`git push https://x-access-token:$(cat ../.tokens/gh_token)@github.com/chainik1125/temp_xc.git arxiv`.
32 CPU / 128 GB, NO GPU; harness-tracked background Bash; python -u.

## ACTIVE TASK (not started): briefings/hunt-support-synthetic.md
Two synthetic mechanism receipts for the task hunt's Stage-2 λ̂ result
(real Ward activations, TXC-pre 0.13→0.19→0.21 over T=2/4/8, dip at
T=16). **Deadline: results by Saturday morning PT.** Read order when
starting: (1) the briefing itself; (2) `task_hunt/LOG.md` mac-local
review entry (the briefing discharges its two review notes: T=16-dip
interpretation + T-SAE-fairness rejoinder); (3) `task_hunt/RECORD.md`
§ 3b (the dip sentence at stake).

- **Item 1 — budget-dilution receipt.** On `toy_backtracking_selfexcite`
  (λ̂ mirror, provable DPI floor, shared `lambda_recovery` evaluator):
  Arm A fixed budget, TXC-pre × T∈{2,4,8,16,32}, canonical d_sae/k_pos
  slice, 3 seeds + untrained — frozen prediction: rise then DECLINE
  (reproduces real dip). Arm B budget-scaled (state the knob; match on
  MEASURED realized l0) — prediction: decline flattens/vanishes.
  Falsifiers in the card: no dip in A ⇒ real dip needs another
  explanation (finding); dip persists in B ⇒ RECORD's dilution clause
  must be RETRACTED (finding). Deliverable: two-arm figure (recovery vs
  T) + LOG paragraph backing or retracting the dip sentence.
- **Item 2 — T-SAE fairness receipt.** Sweep the registered T-SAE's own
  temporal knob (whatever `src/temp_bench/archs/tsae.py` exposes — its
  contrastive pair is consecutive-token; if the knob isn't YAML-exposed,
  add a plugin-arch variant file, hard rule 3, never edit core) across
  ≥3 settings at canonical budget, 3 seeds + untrained, on the mirror.
  Frozen prediction: λ̂ recovery FLAT in the knob (per-token decode is
  binding). Falsifier: recovery rises ⇒ flag IMMEDIATELY in LOG + to
  runpod-d (its round-2 T-SAE cell would need a re-run before any
  rebuttal figure ships).
- **Process rails:** ONE short card per item at
  `experiments/explorations/task_hunt/support_synthetic/CARD.md`,
  frozen/committed PRE-RUN; results + figs under `support_synthetic/`;
  every trained cell through the canonical runner; leaderboard 0 dup
  keys; verdict paragraphs APPENDED to shared `task_hunt/LOG.md`; no
  reviewer/meeting quotes in tracked files; STATUS rewritten; stop for
  review (briefing stays until mac-local deletes it).

## Reusable know-how from the dissection (same box, same substrate)
- Grid engine: `design.uniform_cells(ds, F, n_steps, archs=…,
  k_pos_sweep=…, d_saes=…)` → `grid.run_pool` (16 workers good).
  Backtracking canon: DS `toy_backtracking_selfexcite_d64`, F=20,
  N_STEPS=30000, seeds {1,2,42}; window-mode cells ~7 s each.
  NOTE T∈{16,32} at eval_window_L=32: L must tile by T — T=32 needs
  eval_window_L≥32 (=32 ok, exactly one tile); d_sae ≥ k_pos·T for the
  pooled pre family (F=20 ⇒ k_pos=1 only at T=16/32 unless d_saes
  raised — the card must pick the slice consciously; arm B raises the
  budget knob anyway).
- TXC-pre = `txc_batchtopk_pre` (pooled BatchTopK, budget k_pos·T per
  window). Budget-scaling knob candidates: k_pos (raises atoms/window
  ∝ T at fixed per-token rate — arm A IS fixed k_pos so atoms/window
  already grows ∝ T… careful: pre's k_win = k_pos·T ALREADY scales with
  T; the fixed-BUDGET arm therefore likely means fixed k_win, i.e.
  k_pos = C/T. Resolve this in the card BEFORE running — read how the
  real Stage-2 panel matched budgets (RECORD § 3b / runpod-d's scripts)
  and mirror THAT convention; match on measured realized l0.)
- T-SAE knob: `tsae.py` samples consecutive-token pairs (t, t+1)
  internally; no explicit distance hparam today ⇒ likely needs a
  variant arch file exposing pair distance (e.g. (t, t+Δ), Δ∈{1,2,4,8})
  — precedent: `txc_post_dissect.py` (one class, many YAML entries).
  `contrastive_alpha` is YAML-exposable too (α∈{0,1} as a second knob
  axis if cheap). Dissection lesson: sequence-mode cells cost ~115 s.
- Skeptic pattern if needed: `loss_dissection/skeptic_dissect.py`
  (cache-guarded, raw pre-parse, Meter → expansion spend.json;
  cumulative $11.52/$25).

## Just completed (REVIEWED & APPROVED 2026-07-24; do not redo)
**TXC-pro dissection** — bundle mostly selection on noise; ONE surviving
component (multi-distance contrastive → frequency velocity T=8, post
decode only; prediction (v) confirmed = decode-structure-contingent).
Record: `loss_dissection/RECORD.md`; § 7 has the rebuttal-safe sentence.
Earlier: C7 close, C6, stage-6 #3b, C5 (all reviewed).

## Repo state
Clean, in sync with origin/arxiv at last check after pulling the
task-hunt round-1 + round-2 commits. Nothing mid-flight. Next action
post-compact: read `task_hunt/LOG.md` review entry + `RECORD.md` § 3b,
then freeze the support_synthetic CARD.

## Gotchas (this box)
- Harness blocks `sleep` (use `until …; done` or background tasks);
  background python needs `-u`.
- Sequence-mode cells ~115 s vs ~7 s window-mode (SequenceBuffer
  regenerates per step).
- 5-family models reject `temperature`; calibrations SEQUENTIAL.
- Skeptic verdicts: persisted raw pre-parse, cached, NEVER re-rolled.
- Rebase rewrites SHAs — cite subjects or re-verify post-push.
- Mid-grid the leaderboard/manifest are live-appended — never
  rebase/stash while a grid runs.
