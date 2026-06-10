# STATUS — synthetic-benchmark program (living briefing / pre-compact handoff)

**This is the one-stop briefing.** Update *this file only* before a compact; it
is the canonical current-state of the synthetic-benchmark program. Read it
top-to-bottom, then the linked per-benchmark docs as needed.

Last updated: 2026-06-10.

---

## 0. TL;DR — what's active right now

- **ACTIVE → changepoint bench, GRID RUNNING.** All build stages DONE +
  committed + pushed (2026-06-10): § 8 gating **PASSED** (`553ed9d1`), generator
  + evaluator + datasource + tests (`76ae09fc`), grid driver + renderer + record
  skeleton (`e5586b58`). The 198-cell BatchTopK grid is running under nohup
  (24 workers, ~2–4 h; log:
  `changepoint/results/grid_run.log`). **Next actions when it finishes:**
  (1) `.venv/bin/python -m experiments.explorations.synthetic.changepoint.render_figs`,
  (2) finalize the narrative stubs in `changepoint/bench_record.md` (sections
  2–9) honestly from the numbers, (3) update README.md benchmark tables +
  this file, (4) commit + push. Details + key gating numbers in § 4.
- **Backtracking BatchTopK redo: DONE.** Verdict POSITIVE
  and it survives a uniform BatchTopK backbone. Per-token pinned at the DPI floor
  λ≈0.41; all three window families (TXC-pre, TXC-post, Stacked) recover λ
  0.87→0.95. New findings: **TXC-pre > TXC-post** (post slips at large T) and the
  **shared-code crosscoder ≫ Stacked on eAUC** at matched λ. Full result:
  [`backtracking/bench_record.md`](backtracking/bench_record.md). (Design rationale
  archived in § 5.)

(Running on RunPod now. Repo root = `/workspace/temp_xc`; work from there.
Git creds: token at `/workspace/.tokens/gh_token`, wired into
`~/.git-credentials` (helper=store); repo-local user.name/email set to Han.)

---

## 1. Where everything lives (post-restructure)

**`src/` is importable library code only**; experiments live under `experiments/`.

- **The framework:** `src/temp_bench/` — core (never edit `core/`), interfaces,
  and the **registered plugins**: archs in `src/temp_bench/archs/`, the data
  generators in `src/temp_bench/data/synthetic.py`, evaluators in
  `src/temp_bench/evals/`. New archs/evals/generators for an exploration go here
  (referenced by `class_path` / generator-name in `configs/`).
- **This program:** `experiments/explorations/synthetic/` — the synthetic-benchmark
  program (one exploration under `experiments/explorations/`). Self-contained:
  [`README.md`](README.md) (the single governing doc — prime directive, the
  measure→mirror→bench loop, § 3 validity gates, conventions, benchmark index),
  this `STATUS.md`, then one subdir per benchmark with docs + scripts + `figs/` +
  `results/`. The DC/AC lens is at
  [`../../../docs/ideas/frequency_lens.md`](../../../docs/ideas/frequency_lens.md).
- **Run scripts** from the repo root as
  `.venv/bin/python -m experiments.explorations.synthetic.<bench>.<script>`.
- **Canonical results store (single source of truth):** `results/leaderboard.jsonl`
  at the repo root — every cell, code-version-stamped, via the runner. Real-label
  inputs (Ward backtracking) stay at `results/c7_backtracking/stage_a/`.
- `src/explorations/<name>/` is *reserved* for exploration **library** code that
  isn't ready for `temp_bench` — empty today (synthetic has none; its archs/evals/
  generators graduated into `temp_bench`).

### Single-source record pipeline (built for backtracking — the template)
`results/leaderboard.jsonl` →
`-m experiments.explorations.synthetic.backtracking.render_figs` → paper-quality
`figs/*.{pdf,png}` + `results/backtracking_bench_stats.json` + **auto-filled**
`<!-- AUTO:* -->` blocks in `bench_record.md`. Idempotent; no hand-typed numbers;
figures embedded (`![...]`) so they render in VS Code preview. Reuse this pattern
for changepoint.

---

## 2. Benchmark status

| benchmark | dynamics class | verdict | state |
|---|---|---|---|
| **backtracking** | self-exciting (AC) | **POSITIVE** | DONE — 198-cell BatchTopK grid; record+figs regenerated, committed, pushed |
| **signed_motion** | order-sensitive (AC) | **NEGATIVE** | done; leave as published (memorization confound at `#windows=2F`) |
| **topic_switching** | change-point/sticky | **ABORT** | measured; composition-dominated + labeler inadequate; no bench. BUT it *did* measure a valid dwell (≈geometric, mean run 1.73) — the anchor for changepoint |
| **changepoint** | change-point / dual-latent | — | **GRID RUNNING** (§ 4); gating PASSED; generator/evaluator/grid built + committed; spec frozen + amendments A1–A4 ([`changepoint/bench_spec.md`](changepoint/bench_spec.md)) |

---

## 3. Key facts that carry over

- **The BatchTopK arch family (reuse for changepoint):** `batchtopk_sae` (per-token),
  `tsae` (per-token + contrastive), `stacked_batchtopk` (per-position independent
  dicts), `txc_batchtopk_pre` / `txc_batchtopk_post` (shared-code crosscoder,
  pre/post squash). All on the strong backbone (BatchTopK-train → JumpReLU-eval +
  AuxK + decoder unit-norm + grad-orth). Registered in `configs/archs.yaml`. The
  grid driver normalizes throughput (`batch_size = 1024 if T==1 else 1024//T`) so
  every arch sees equal tokens/step + an equal `B·T=1024` BatchTopK pool.
- **Latent-recovery metric pattern:** held-out **linear** probe on the arch's code
  → the hidden latent, per-tile at the tile's leading edge, normalized to
  [chance, oracle]. Linearity is mandatory (measures what the code makes *linearly*
  available). Per-token gets a provable floor where possible (e.g. backtracking's
  DPI floor). `eAUC` = local feature-direction cosine recovery.
- **Conventions** (full detail in [`README.md`](README.md) Part II): `d_sae` + `k_pos`
  equal across archs, anchored on `F`, swept into the scarce regime (`d_sae ≤ F` is
  the object of study); powers-of-two windows tiled into a common `L=32`;
  memorization-free per-tile probes (features = one tile's `d_sae` code, never
  concatenated — the signed-motion lesson); report the frontier, not a cell.

---

## 4. ACTIVE TASK — the change-point bench

> **2026-06-10 build-state: everything below is BUILT; the grid is RUNNING.**
> - **Gating PASSED** (`changepoint/gating.py` →
>   `results/changepoint_gating_stats.json`): per-token mode oracle **1.000**;
>   per-token AC exactly chance (c balacc 0.500, τ corr ≈ 0); window τ info
>   ceilings **0.76/0.96/1.00** at T=2/4/8. Bonus (load-bearing): the
>   **raw-linear** window ceiling is ≈ chance for both AC latents
>   (mode-symmetry → equality patterns are XOR-like) — an AC win on a trained
>   code is *learning*, not linear access. Untrained per-token control lands
>   mode≈0.49, tss≈0 (early grid rows).
> - **Spec amendments A1–A4** dated in `bench_spec.md` (geometric-dwell
>   ungating + uniform Π + mode-independent content; τ primary AC latent;
>   BatchTopK family; gating result).
> - **Built:** `semi_markov_modes()` (synthetic.py), `toy_changepoint_modes_d64`
>   (data.yaml), `evals/changepoint_recovery.py` (mode/tss/cp probes, dispatched
>   from `synthetic_recovery.py` on `extra['mode_labels']`; protocol stays
>   1.2.0), `tests/test_changepoint_bench.py` (8 tests; suite 62 passed),
>   `changepoint/run_grid.py` (198 cells), `changepoint/render_figs.py`
>   (single-source: figs + stats + AUTO blocks), `bench_record.md` skeleton.
> - **When the grid finishes** (log `changepoint/results/grid_run.log`): run
>   `render_figs`, finalize record narrative (sections 2–9, incl. prereg
>   verdicts P1–P4 + controls + caveats), update README/STATUS, commit + push.

*(original task description below, kept for reference)*

**What it is** (frozen spec: [`changepoint/bench_spec.md`](changepoint/bench_spec.md)):
a **dual-latent** substrate scored on **two axes that should split**:

| latent | type | axis | predicted winner |
|---|---|---|---|
| **mode `m_t`** (the global hidden state, categorical `K_m=8`) | persistent | **DC** | per-token (it's stamped into every token of the dwell) |
| **change-point / time-since-switch** (the boundary structure) | order-sensitive | **AC** | window |

The headline is the **split**: on identical data, per-token should win
`mode_recovery` (DC) and the window archs should win the AC latent. That two-way
prediction (not "window always wins") is what makes it strong. F=20 directions
(`K_m=8` mode-signature + `C=12` content), `spread=3`, `seq_len=64`, `n_seqs=4096`.

**Why it's ungated now (pre-run amendment to the spec's gating):** the spec was
gated on "a validated real dwell to set the persistence knob." topic-switching
ABORTED as an *order-sensitive* phenomenon, but it **measured a valid dwell** —
≈geometric, mean run ≈1.73 (matches Markov-1). Anchor the persistence knob on that
**measured geometric dwell** → grounded, not arbitrary. The DC/AC split doesn't
need stickiness, so the bench proceeds at the geometric setting; optionally sweep
the knob (geometric → heavy-tailed → absorbing) as a robustness axis. The
heavy-tailed/EM variants remain gated (need a better labeler / paid judge — § 6).

**AC-latent choice (design decision):** `c_t = [m_t ≠ m_{t-1}]` (adjacency) is the
*simple-floor companion*, but it risks being "too easy" (pure architectural access:
an untrained window may already solve it). Make **time-since-switch** (a scalar —
how many tokens since the last boundary) the **primary AC latent**: it needs more
than adjacency (counting since the boundary), so a window win reflects learning,
not just access. Report `c_t` alongside as the minimal floor. *(This was the
agreed steer; confirm if revisiting.)*

**Order of work (do NOT skip the gate):**
1. **§ 8 gating due-diligence FIRST** — the analogue of `backtracking.gating`.
   From the generator at `K_m=8` + the geometric dwell + `Π`: (i) confirm the best
   *linear* predictor of the AC latent from `m_t` alone sits ≈ chance (else the
   split is uninformative — rebalance `Π`/`K_m`); (ii) confirm `mode_recovery`
   oracle is reachable by a per-token probe on the noiseless emission. Write a
   `changepoint/gating.py`; commit the stats JSON. Only proceed if the ceilings
   are well separated on both latents.
2. **Generator:** implement `semi_markov_modes()` in `src/temp_bench/data/synthetic.py`
   (specified in the spec, not yet built) + a `toy_changepoint_modes` datasource in
   `configs/data.yaml`. Expose `mode_labels`, `changepoint_labels`,
   `time_since_switch` in `extra` (like backtracking exposes `lambda_labels`).
3. **Evaluator add-on:** `mode_recovery` (multinomial-logistic probe → `m_t`, DC) +
   the AC probe (linear → time-since-switch, logistic → `c_t`), dispatched from
   `SyntheticRecovery` when `extra` carries the changepoint labels (mirror
   `lambda_recovery.py` / the dispatch in `synthetic_recovery.py`; keep protocol at
   1.2.0 — no-op for other benches).
4. **Grid:** reuse the BatchTopK arch family + the `run_grid.py` / `render_figs.py`
   pattern from backtracking (copy into `changepoint/`). Same capacity sweep
   (`d_sae` anchored on F=20, scarce regime), `L=32`, seeds {1,2,42}, untrained
   control, k_pos robustness.
5. **Record + figs** via the single-source pipeline; **prereg/bench_spec stay
   frozen** except dated pre-run amendments (the geometric-dwell ungating + the
   time-since-switch primary-latent choice are exactly such amendments — note them
   transparently, like the backtracking K=8→2 and TopK→BatchTopK amendments).

**Honest-outcome reminder (prime directive):** the AC latent could be (a) pure
access (untrained window already solves it → report it as access, not learning) or
(b) a hard bilinear interaction the scarce-`d_sae` code can't linearly expose
(→ a real negative, like signed_motion). Both are complete, citable verdicts. The
DC half (per-token *wins* mode) is the novel, robust claim regardless.

---

## 5. DONE — backtracking BatchTopK redo (archived rationale)

Completed 2026-06-09. The fairness problem: T-SAE already used BatchTopK (Bussmann
et al., the strong backbone) while the other archs used plain TopK — an
uncontrolled confound. Fix: put every arch on the same BatchTopK→JumpReLU backbone
(+ AuxK + decoder unit-norm + grad-orth), normalize throughput (equal tokens/step),
and correct the post-squash budget to `k_pos` per window (`= k_win // T`, since
each squashed atom is reused at all T positions). Built 4 new archs (§ 3), ran a
198-cell grid (132 trained + 33 untrained control + 33 k_pos=2 anchor, 0 gaps).
Full numbers + narrative + figures: [`backtracking/bench_record.md`](backtracking/bench_record.md);
frozen spec + amendments: [`backtracking/bench_spec.md`](backtracking/bench_spec.md).
The pre/post-squash and crosscoder-vs-Stacked design notes live there + in git.

---

## 6. Roadmap beyond changepoint

- **Heavy-tailed / sticky changepoint:** needs a stronger topic labeler (LLM
  segment tagging / validated topic model) that passes the temporal-ness gate, to
  justify a heavy-tailed dwell. Gated until that measurement exists.
- **EM (emergent misalignment) instantiation** of the changepoint generator
  (`K_m=2`, state 2 absorbing, ramping entry-hazard precursor): needs a **paid
  per-span judge labeler** (`evals/em.py` is a stub; `experiments/em/` is the §5.3
  real-LM scaffold). Out of scope until the spend + labeler are authorized.
- The fair-backbone grid + single-source-record pipeline is the template for any
  future bench.

---

## 7. Hard rules + run reference + git

- `TEMP_BENCH_ALLOW_DIRTY=1`, `.venv/bin/python`, never edit `temp_bench/core/`,
  plugin-only (new arch/eval/generator = file drop + `configs/` entry), everything
  through the canonical runner (code-version stamped), paper-section names. Prime
  directive: a sound verdict, never a "win".
- **Run (from repo root `/workspace/temp_xc`):**
  `TEMP_BENCH_ALLOW_DIRTY=1 .venv/bin/python -m
  experiments.explorations.synthetic.backtracking.<gating|kernel_order|measure|mirror|run_grid|render_figs>`.
  Canonical leaderboard: `results/leaderboard.jsonl`. Verify env:
  `bash scripts/agent_smoke_test.sh`.
- **Git:** branch `arxiv`, **pushed to `origin/arxiv`** (HEAD = `e5586b58`). Recent
  chain: backtracking redo (`6d406e19` archs → `d64e7c4e` results) → RunPod infra
  restore (`4c54908f`) → restructure (… → `c9e457e2`
  `→experiments/explorations/synthetic`) → STATUS rewrite (`0fae2afe`) →
  **changepoint** (`553ed9d1` gating PASS → `76ae09fc` generator+evaluator →
  `e5586b58` grid driver+renderer+record skeleton). An empty untracked
  `src/explorations/` shell may linger locally (cosmetic; absent from the repo).
