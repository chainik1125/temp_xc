# STATUS — synthetic-benchmark program (living briefing / pre-compact handoff)

**This is the one-stop briefing.** Update *this file only* before a compact; it
is the canonical current-state of the synthetic-benchmark program. Read it
top-to-bottom, then the linked per-benchmark docs as needed.

Last updated: 2026-07-10.

---

## 0. TL;DR — what's active right now

- **Grounded-benchmark expansion loop: SCOPED — Cycle 1 queued for runpod
  (2026-07-13).** An autonomous, *gated* pipeline to expand the list of GROUNDED
  benchmarks (measure→mirror on real LM data), generalizing the proven
  backtracking loop (Claude-judge labeler + N1/N2/N3 null gate + mirror). Covers
  **both domains** (reasoning-trace + text-corpus) with an anti-drift **coverage
  ledger** ([`expansion/LEDGER.md`](expansion/LEDGER.md), 12 cells, 10 empty — only
  backtracking PROCEED + topic_switching ABORT filled): per-domain floor (≥⌊N/2⌋
  each) + under-covered-cell-biased selection. Guardrails (prime directive):
  abort-is-success, blind-to-architecture, null gate load-bearing, labeler
  noise-floor mandatory, adversarial skeptic pass on every PROCEED, cost cap.
  Deliverable = frozen bench_specs + calibration records (or aborts), NOT arch
  evals. **Calibration is text-only** (label-stream signature; no activations / no
  local model / no arch — those are the later blind eval). **Cycle 1** = Claude-API
  client + factory harness (generalize `backtracking/measure.py`) + prove the loop
  on N=4 (2/domain): reasoning-trace half runs on the **300 stored traces**
  (re-label, no generation); text-corpus half stands up fineweb + a labeler (the
  heavier half). Cost cap **$25/cycle**; then STOP for review. Briefing:
  [`briefings/grounded-benchmark-expansion.md`](../../../briefings/grounded-benchmark-expansion.md);
  prereg card: [`expansion/prereg_template.md`](expansion/prereg_template.md).
  (This is the *grounded* arm of the program; the abstract benches — signed_motion,
  frequency — are out of its scope.)

- **Program-level B×A report + full clean-room rerun: DONE (2026-07-11).** The
  entire synthetic result set was rebuilt from scratch at **protocol 1.3.0** under
  the locked uniform design and the stale rows purged; [`REPORT.md`](REPORT.md)'s
  per-token matrix is now **fully filled (36/36 cells)** with both companion panels
  (NMSE + eAUC) and **three interp figures** (recovery heatmap, capacity frontiers,
  capability gate — `report_figs.py`, embedded in REPORT.md, `a0cb5424`).
  **Design (locked):** 6 fair-backbone archs (batchtopk_sae, tsae;
  stacked_batchtopk, txc_batchtopk_pre/post, **spectral_txc**) × `d_sae∈{F//2,F,2F}`
  × `T∈{1,2,4,8}` × `k_pos∈{1,2,4,8,16}` × seeds{1,2,42} + untrained. Matrix cell =
  per-token-matched (`T_can=4`, realized `l0_per_token≈B*=2`) at `{F, F//2}`. F
  per-bench (bt/cp 20, sign 19, freq 101=M). ~2239 cells, **0 failures**.
  **Provenance:** run on **CPU** (`runpod` A40 is pathologically slow for these
  tiny d_in≤128 models — kernel-launch-latency-bound, ~14% util; CPU ~7h). Cached
  checkpoints made overlapping cells reproduce the old numbers **exactly**
  (drift-sanity: max|Δ|=0.000 on the primary recovery metrics at overlapping
  cells). **Purge:** leaderboard now 2239 fresh synth (1.3.0 only, seeds{1,2,42}
  only) + 356 non-synth preserved byte-for-byte (backup `.prepurge`). **New from
  the spectral column:** it recovers λ on backtracking (0.94–0.96), exposes the
  changepoint AC boundary and — unlike TXC-post, whose AC code is scarcity-forced
  and vanishes at k_pos=2 — its recovery is **k_pos-robust** (τ 0.59→0.57 at
  k=1→2, T=2), and it is near-oracle on frequency (**0.96 at T=8**; the uniform
  design drops T=16, where it was 1.00). Per-bench verdicts all **preserved**.
  Infra: single-source `design.py` (uniform grid) + `test` (9/9); the rerun
  briefing is retired.

- **Changepoint bench: DONE — verdict SPLIT (two-way), committed + pushed
  (2026-06-10).** Full chain: § 8 gating PASS → generator/evaluator/tests →
  198/198-cell BatchTopK grid (zero failures, 108 min) → figures + record.
  **Headline:** per-token pins the DC mode at oracle (1.00 at d_sae=8) and
  sits *exactly* on the provable AC chance floor; **only the post-squash
  crosscoder** linearly exposes the boundary (τ 0.66/0.60/0.52 at T=2/4/8,
  d=20, vs ceilings 0.76/0.96/1.00; c_t 0.90 at T=2), paying for it in
  mode (0.67) + content (eAUC 0.11), and the AC code vanishes at k_pos=2
  (scarcity-forced specialization). **TXC-pre + Stacked are at chance on AC
  *provably*: their eval-time codes are additive over per-position features,
  and the gating symmetry argument extends to any additive code.** Untrained
  TXC-post has a real access residual (τ 0.20 at T=2) — trained gain is
  learning on top of it. Full result:
  [`changepoint/bench_record.md`](changepoint/bench_record.md).
- **Frequency / cyclic-tone bench: DONE — verdict POSITIVE (periodic axis),
    committed + pushed (2026-07-08).** Port of Dmitry's FrequencyBench
  (`origin/dmitry-spectral-sprint2`) — the one uncovered axis. Full chain: § 8
  gating PASS (`9d074219`) → `cyclic_tones()` generator + `spectral_txc` DCT-band
  arch + `frequency_recovery` (velocity + S(f)) evaluator + 14 tests → **298-cell
  main grid + 120-cell matched-budget band-partition addendum** (0 failures) →
  figures + single-source record. Frozen design (amendments A1–A6): `M=101` prime,
  `d_in=128`, `Ω={0,1,2,4,8,16,24,32,40,50}` (chance 0.1), `σ=0.10`, `seq_len=64`,
  `L=32`, `T∈{2,4,8,16}`, `d_sae {32,64,101,256}`, memorization `|Ω|·M=1010`.
  **Headline (3-seed means, d=101):** the discriminator is the **S(f) *shape***,
  and it splits the crosscoders by **where the nonlinearity sits**. Codes that
  **mix positions before the nonlinearity** (TXC-post `relu(Σ_t W_t x_t)`,
  Spectral) recover the tone with a **high-pass / Rayleigh-resolved** S(f) —
  **Spectral near-oracle 1.00 at T=16**, TXC-post 0.53. The **additive-over-
  position** code (TXC-pre `Σ_t g(x_t)`) caps at **0.27 with a flat S(f)** (bag-
  level, not spectral estimation — each token's marginal is Y-independent → no
  frequency ordering). **Per-token = 0.00** (provable DPI + raw-linear). The
  **DCT-band inductive bias is decisive and largely an *access* prior**: untrained
  Spectral already reads 0.64 (its bandpass kernels are tone-detectors at init;
  TXC-post 0.18, TXC-pre 0.02), training lifts to 1.00. **P3 (matched budget):**
  multiband ≈ 1-band DCT (`full`) ≈ 2-band at T=16 (all saturate — a **tie**, as
  preregistered); multiband's edge shows at the scarcest window T=2 and in the
  untrained access (band-limiting is the prior). **Null:** random-embedding S(f)
  is flat (no Δf ordering); circle 1.00 vs random 0.57 at T=16 (geometry makes Y
  resolvable); above `|Ω|·M` the null jumps to ~1.0 by template memorization —
  caught + flagged. **Amendment A5:** Stacked dropped (its concatenated `T·d_sae`
  code memorizes above `|Ω|·M`, the signed-motion confound). Full result:
  [`frequency/bench_record.md`](frequency/bench_record.md).
- Other options (not started): the § 6 roadmap (heavy-tailed/sticky changepoint
  — gated on a better labeler; EM instantiation — gated on a paid judge), an
  atom-level case study of the TXC-post boundary pair-atoms (record § 9), or a
  dwell-knob robustness sweep.
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
| **changepoint** | change-point / dual-latent | **SPLIT (two-way)** | DONE — 198-cell BatchTopK grid; per-token: DC oracle + provable AC chance; AC exposed only by the post-squash crosscoder (additive codes provably blind); record+figs committed ([`changepoint/bench_record.md`](changepoint/bench_record.md)) |
| **frequency** (cyclic tones) | periodic / frequency (AC / 2nd-moment) | **POSITIVE** | DONE — 298-cell BatchTopK grid + 120-cell band addendum; position-mixing crosscoders recover the tone with high-pass S(f) (Spectral near-oracle 1.00 at T=16), additive/per-token blind (flat S(f)/chance); DCT-band inductive bias decisive (untrained access 0.64); null flat + memorization flagged ([`frequency/bench_record.md`](frequency/bench_record.md)) |

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

## 4. DONE — the change-point bench (archived rationale)

Completed 2026-06-10. Full chain: § 8 gating PASS (per-token mode oracle
1.000; per-token AC exactly chance; window τ info ceilings 0.76/0.96/1.00 at
T=2/4/8; **raw-linear window access ≈ chance by mode-symmetry**) → spec
amendments A1–A4 → `semi_markov_modes()` + `toy_changepoint_modes_d64` +
`evals/changepoint_recovery.py` (dispatch on `extra['mode_labels']`, protocol
1.2.0 unchanged) + 8 tests → 198/198-cell grid (no failures) → single-source
record. **Verdict SPLIT (two-way), with the AC half architecturally specific**
(post-squash only; additive codes provably blind — the new theory result) —
see § 0 TL;DR and [`changepoint/bench_record.md`](changepoint/bench_record.md).
Key extras for future benches: (i) the *additive-code corollary* of the gating
symmetry argument (any code additive over per-position features is blind to
equality-pattern latents — applies to TXC-pre and Stacked at eval time);
(ii) the k_pos=2 anchor showed the AC specialization is *scarcity-forced*
(vanishes at T=2 when budget doubles).

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
- **Git:** branch `arxiv`, **pushed to `origin/arxiv`**. Recent
  chain: backtracking redo (`6d406e19` archs → `d64e7c4e` results) → RunPod infra
  restore (`4c54908f`) → restructure (… → `c9e457e2`
  `→experiments/explorations/synthetic`) → STATUS rewrite (`0fae2afe`) →
  **changepoint** (`553ed9d1` gating PASS → `76ae09fc` generator+evaluator →
  `e5586b58` grid driver+renderer → `e288999b` **grid results, verdict SPLIT**
  → README/STATUS wrap-up commit after it). An empty untracked
  `src/explorations/` shell may linger locally (cosmetic; absent from the repo).
