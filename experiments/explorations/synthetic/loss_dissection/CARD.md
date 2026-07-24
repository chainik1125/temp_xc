# Loss-dissection card — which TXC-pro loss component helps the TXC backbone?

**FROZEN pre-build** (this commit precedes every implementation, test, and
grid commit; commit-order is the freeze evidence). Prime directive: **a sound
verdict, never a win** — the team's prior is "mostly nothing helps"; finding
that cleanly IS the deliverable. Mandate: `briefings/txcpro-dissection.md`.

## 1. Question and provenance

The paper's TXC-pro bundles three additions to the TXC backbone —
a Matryoshka objective, a multi-distance contrastive loss, and longer
windows — confounded by the paper's own admission (Limitations ii) and
slated to drop. Locked definition (`paper/appendix.tex` ¶ TXC-pro): (i)
subseq slabs `T_max=10`, `t_sample=5`; (ii) Matryoshka with **H=8 nested
feature groups**, group G reconstructing from the first `G·d_sae/8`
features; (iii) **inverse-distance-weighted InfoNCE at shifts Δ∈{1,2}**.
At toy scale the paper *disabled* matryoshka (`h_size=d_sae`), so
toy-scale TXC-pro's contrastive ran on the **full code**.

**Provenance fact that shapes the design (Han, 2026-07-23):** the
component combination was selected by hill-climbing on sparse probing —
a metric where all architectures differ by ~0.001 AUC amid noise. So (a)
the original selection is weak evidence for anything; this is the FIRST
clean per-component measurement, not a re-test; (b) probing is NOT a
dissection venue — the synthetic set discriminates by construction with
effect sizes 10–100× the probing noise floor. **No probing arm.**

"Longer windows" is NOT a variant — it is the existing T axis; we report
T-trends per variant. The paper's `T_max/t_sample` slab sub-sampling is
not re-implemented (entangled with T; the T-trend is the clean question).

## 2. Variants — one class, four registry entries (plugin-only)

Class `TXCPostDissect` in `src/temp_bench/archs/txc_post_dissect.py`,
subclassing `TXCBatchTopKPost` (imported, never edited; `temp_bench/core/`
untouched — hard rule 3). `consumes="sequence"`: receives `(B, seq_len,
d_in)` full sequences and slices its own windows (see § 3 for why).

Per train step, for every batch row: one anchor offset `p ~ U{0, …,
seq_len − T − S_max}` with **S_max = 2 for ALL variants** (identical
anchor-window distribution family-wide); anchor window `x[p:p+T]`;
positive windows `x[p+Δ : p+Δ+T]` for `Δ ∈ {1,2}` (materialized only when
the contrastive term is on; positives are deterministic shifts of the
anchor offsets — no extra RNG, so all four variants consume identical RNG
streams and identical training windows at equal seeds until gradients
diverge).

Backbone loss on the **anchor only**, byte-identical in op order to
`txc_batchtopk_post.train_step`: post-squash BatchTopK (pool B, budget
k_pos/window), recon MSE, AuxK on the full-reconstruction residual,
threshold EMA, dead-feature tracking, decoder unit-norm + grad-parallel
removal (inherited). Positives contribute NO recon/AuxK/threshold/dead
side effects — only the InfoNCE terms — so zero-weight reduction to
plain is exact.

Total loss:

```
loss = l_recon_full(anchor)
     + mat_alpha · Σ_{G=1}^{H−1} l2( x_anchor, decode_prefix(z_anchor, n_G) )
     + auxk_alpha · l_auxk(anchor)                      # unchanged, full residual
     + ctr_alpha · Σ_{Δ∈{1,2}} [1/(1+Δ)] · InfoNCE( z_anchor, z_pos_Δ )
```

- **Matryoshka** (paper-faithful): H = 8 nested prefixes `n_G =
  ⌊G·d_sae/8⌋` (G=1..8, `n_8 = d_sae`, deduped if degenerate); the G=8
  term IS the plain recon term, so the added sum runs G=1..7; equal group
  weights at `mat_alpha=1` = the paper's equal-sum objective.
  `decode_prefix(z, n) = einsum(z[:, :n], W_dec[:n]) + b_dec` (the
  `_decode_prefix` graft from the Phase-6.2 lineage). The paper's own
  toy-scale caveat (small prefixes vs the k budget) is noted; the
  dissection ENABLES the head — measuring the component is the point.
- **Contrastive** (TXC-pro lineage graft, `_info_nce` from
  `txc_bare_matryoshka_contrastive_antidead` @ `2fa9bdab`):
  cosine-normalized symmetric InfoNCE — `F.normalize` both codes, `sim =
  ẑ_a @ ẑ_b.T`, `0.5·(CE(sim, I) + CE(simᵀ, I))` — between the anchor's
  gated shared code and each shifted positive's gated shared code (each
  gated by training-path BatchTopK over its own B-pool). Weights `w_Δ =
  1/(1+Δ)`, shifts `Δ∈{1,2}` (paper), overall `ctr_alpha = 1.0`.
  **On the full code** — paper-faithful at toy scale (h_size=d_sae) and
  makes `+both` exactly compositional (`= plain + mat-term + ctr-term`).
  The large-scale bundle's high-prefix coupling is named, not run.

Registry entries (same class, four YAML hparam sets — the `spectral_txc`
precedent): `txc_post_plain` (mat_alpha=0, ctr_alpha=0), `txc_post_mat`
(1, 0), `txc_post_ctr` (0, 1), `txc_post_both` (1, 1). Family `"post"`
(dict constraint `d_sae ≥ k_pos`). Variants share constructor, parameter
shapes, and init RNG draws exactly (loss-only differences) ⇒ **untrained
rows must be numerically identical across variants** at equal (T, seed) —
checked, not assumed.

## 3. Pipeline decision + the bridge anchor

`WindowBuffer` feeds i.i.d. random windows — shifted-window pairs are
unobtainable in window mode, and fixing that in core is forbidden. To
avoid a train-data-pipeline confound BETWEEN variants, **all four**
consume sequences and slice windows internally, matching WindowBuffer's
uniform-(seq, pos) window marginal (up to the S_max=2 top-offset
exclusion) at the same windows-per-step (`batch_size(T) = 1024//T`).

The existing window-mode **`txc_batchtopk_post` leaderboard rows at the
identical cells** are the bridge anchor (Gate B, § 5) — all 135 slice
rows verified present in `results/leaderboard.jsonl` pre-freeze; they are
read, not re-run.

## 4. Benches and grid (exact)

Five benches — the discriminating set, all synthetic, canonical toy
substrate, primary datasource only (no null/memo/T16 addenda), canonical
per-bench `n_steps`:

| bench | datasource | F | n_steps | primary metric | secondary | regime |
|---|---|---|---|---|---|---|
| backtracking | toy_backtracking_selfexcite_d64 | 20 | 30000 | lambda_recovery | — | 2 (additive-in-window) |
| frequency | toy_cyclic_circle_M101_d128 | 101 | 6000 | velocity_recovery | — | 3 power |
| phasepair | toy_phasepair_M101_d24 | 101 | 30000 | sign_recovery | pair_recovery | 3 phase |
| recipe_instruction_phase_runs | toy_recipe_instruction_d64 | 20 | 30000 | equality_residual_recovery | phase_recovery (DC) | 3 equality (grounded) |
| multilane | toy_multilane_circle_M101_d24 | 101 | 30000 | multilane_recovery | — | superposition |

Capability metrics everywhere: `nmse`, `gauc` (l0 reported).

Slice (per briefing): `d_sae = F` only; `T ∈ {2,4,8}`; `k_pos ∈ {1,2,4}`;
seeds `{1,2,42}`; automatic untrained control per (variant, T, seed)
(`n_steps=0`, `k_pos=1`, `d_sae=F`). Enumeration through the locked
engine: `uniform_cells(ds, F, n_steps, archs=DISSECT_ARCHS,
k_pos_sweep=(1,2,4), d_saes=[F])` → `grid.run_pool` → canonical
`run_experiment`; every row leaderboard-stamped.

**Exact cell count: per bench 4·3·3·3 = 108 trained + 4·3·3 = 36
untrained = 144; total 720.** Driver:
`experiments/explorations/synthetic/loss_dissection/run_grid.py`
(one positional `max_workers`). A `txc_batchtopk_pre`-backbone extension
runs ONLY if the post grid finishes early, as a separate amendment to
this card — not part of these verdicts.

## 5. Frozen decision rules

Per component C ∈ {mat, ctr, both} × bench × metric m, over the 9 (T,
k_pos) trained cells:

- Paired per-seed differences `Δ_s(cell) = m_C(s) − m_plain(s)`, s ∈
  {1,2,42} (pairing is real: same data stream, same init per seed).
  `D(cell) = mean_s Δ_s`, `SE(cell) = std(Δ_s, ddof=1)/√3`.
- **Cell passes the bar** iff `|D| > max(2·SE(cell), δ_floor)` with
  `δ_floor = 0.05` for normalized [chance=0, oracle=1] recovery metrics
  and `gauc`; `δ_floor = 0.02` absolute for `nmse` (lower is better —
  sign flipped so "helps" = decrease).
- **Bench verdict for (C, m):** HELPS iff ≥ 2/9 cells pass positively AND
  0 cells pass negatively; HURTS mirror; MIXED iff cells pass in both
  directions (reported as such, never a salvage claim); NEUTRAL
  otherwise. No seed-escalation: a boundary case is reported
  NEUTRAL/MIXED, never upgraded (C6/C7 margin discipline, adapted).
- **Headline table** = primary recovery metric per bench; capability
  metrics reported in the same format, labeled separately — a
  capability-only HELPS is prediction (ii), NOT a recovery salvage.
- **Interaction** (descriptive only): `I(cell) = D_both − D_mat − D_ctr`.
- **Untrained guard:** untrained rows identical across variants per
  (T, seed) — equality asserted in analysis; any inequality ⇒ config
  drift ⇒ STOP and investigate.
- **Gate B (bridge / graft-validity), evaluated BEFORE any verdict is
  interpreted:** per bench, on the primary metric, `|mean_s plain −
  mean_s anchor| ≤ max(2·SD_pool, 0.10)` must hold on ≥ 7/9 cells, where
  anchor = the canonical `txc_batchtopk_post` leaderboard rows and
  SD_pool = pooled cross-seed SD of the two arms at that cell. Gate B
  FAIL on any bench ⇒ the graft/pipeline is indicted ⇒ NO component
  claims for that bench until resolved.
- **Falsifier (iv) rule:** a HURTS on backtracking primary is interpreted
  only after Gate B passes and the contract tests are re-examined; if
  Gate B failed, the conclusion is "graft defect", never "component
  hurts".

## 6. Frozen predictions (mac-local priors, sharpened — scored at verdict)

1. **(i)** NO variant beats plain on any regime-3 primary
   (velocity / sign / equality-residual / multilane) beyond the § 5 bar —
   the "TXC-pro is useless" prior, now per-component falsifiable.
2. **(ii)** matryoshka may improve capability (`nmse` and/or `gauc`)
   somewhere with NO primary-recovery HELPS anywhere.
3. **(iii)** contrastive, if it helps anywhere, helps a DC/persistent
   metric — sharpened to: recipe `phase_recovery` (the slice's only
   DC-tagged metric). Sharpened converse: InfoNCE at shifts is
   shift-invariance pressure, which is anti-AC — if contrastive moves
   phasepair `sign_recovery` (the purest odd/AC latent) at all, it HURTS
   it.
4. **(iii-b) interpretive fork, in advance:** any real synthetic
   recovery HELPS ⇒ a genuine salvage — name the component and the regime
   it helps; nothing ⇒ the original hill-climb was selection on noise and
   TXC-pro drops with nothing to salvage — equally publishable, cleaner.
5. **(iv)** a HURTS ≫ noise on backtracking indicts the graft first
   (Gate B + contract-test re-exam before any conclusion).

## 7. Skeptic + spend

Fable skeptic (`claude-fable-5`) on **every (component, bench) HELPS
claim on a recovery metric** — the winner's-curse surface. Runner
`loss_dissection/skeptic_dissect.py` mirrors `skeptic_c7_close.py`:
cache-guarded (refuses to re-roll if raw exists), verdict persisted raw
pre-parse to `loss_dissection/records/`. Spend metered to
`expansion/results/spend.json`; cap **$5** this session. Capability-only
HELPS (prediction ii) does not trigger the skeptic (explicitly predicted;
noted in the record). No HELPS claims ⇒ $0.

## 8. Order of operations (strict commit-then-run)

1. This card — committed pre-build (this commit).
2. Build commit(s): arch file + 4 YAML entries + contract tests
   (`tests/test_loss_dissection.py`) + `run_grid.py` + `analyze.py` +
   `skeptic_dissect.py` — ALL committed before any execution.
3. Contract tests pass (frozen list below) before any grid.
4. Grids (5 benches; report exact wall-clock and any failed cells).
5. `analyze.py` → `results/dissection_table.{json,md}` — the mechanical
   application of § 5; verdicts are read off, not chosen.
6. Skeptic on HELPS claims (if any).
7. `RECORD.md` narrative + research STATUS § 0 bullet + agent STATUS
   rewrite + push. STOP for mac-local review.

**Contract tests (frozen):** (1) plain-reduction — `_loss_on` at
mat_alpha=ctr_alpha=0 equals `TXCBatchTopKPost.train_step` outputs on an
identical state/batch to ≤1e-6 (drift guard for the reimplemented
assembly); (2) zero-weight exactness — supplying positives with
ctr_alpha=0 changes nothing; (3) matryoshka structure — prefix ladder
`⌊G·d_sae/8⌋` deduped, ends at d_sae, prefix-G recon insensitive to
zeroing decoder rows ≥ n_G, correct at d_sae ∈ {20, 101}; (4) contrastive
pairs — offsets satisfy `p+Δ+T ≤ seq_len`, positive windows equal
`x[p+Δ:p+Δ+T]` exactly, weights (1/2, 1/3), InfoNCE matches the ported
`_info_nce` on a fixed example; (5) parameter identity — all four
variants' state_dicts identical at init under one torch seed; (6) window
distribution — slicer offsets uniform on `{0,…,seq_len−T−2}` (seeded
statistical smoke).

Budget: 720 cells (wall-clock measured at a smoke cell first and
reported); build+tests+analysis inside the 48 h window; API spend ≤ $5
(skeptic only).

_Frozen-by: claude-fable-5 (runpod agent, txcpro-dissection briefing),
2026-07-23, pre-build._

---

## 9. AMENDMENT (frozen pre-build of the pre extension, 2026-07-24) —
## the briefing-authorized `txc_batchtopk_pre` extension

The post grid finished early (720/720 in ~2.1 h; briefing: "add the same
4 variants on `txc_batchtopk_pre` ONLY if the post grid finishes early").
Post-grid state at this freeze: mechanical table + skeptic done — ONE
surviving HELPS (ctr on frequency `velocity_recovery`, T=8), the recipe
mat-HELPS killed on e_metric_leak. This amendment is frozen BEFORE any
pre-variant code exists.

**Variants.** `TXCPreDissect` — subclass of `TXCPostDissect` overriding
ONLY the two squash hooks with the PRE implementations
(`_compute_post` / `_to_shared` assigned from `TXCBatchTopKPre` — the
hooks ARE the entire pre/post difference per the backbone's contract, so
`_slice`/`_loss_on`/matryoshka/InfoNCE are inherited unchanged, not
copy-pasted). Registry entries `txc_pre_plain` / `txc_pre_mat` /
`txc_pre_ctr` / `txc_pre_both`; family `"pre"` (pooled dict constraint
`d_sae ≥ k_pos·T` — `uniform_cells` drops infeasible cells and logs; at
F=20 the (T=8, k_pos=4) column is infeasible, so backtracking and recipe
have 8/9 cells; verdict thresholds unchanged, counted over PRESENT
cells with the same ≥2-positive / 0-negative rule).

**Everything else identical to §§ 4–5**: same benches, slice, seeds,
n_steps, metrics, decision rules, untrained guard; Gate B bridges
`txc_pre_plain` to the canonical `txc_batchtopk_pre` leaderboard rows
(129/129 feasible anchor cells verified present pre-freeze). Contract
tests parametrized over both families, incl. plain-reduction vs
`TXCBatchTopKPre`. Analyzer/skeptic gain a family switch; outputs to
`dissection_table_pre.{json,md}`; skeptic policy unchanged (recovery
HELPS only; same $5 session cap shared).

**Frozen predictions (pre extension):**

1. **(v)** ctr does NOT lift pre's frequency `velocity_recovery` beyond
   the bar — the additive-over-positions decode cannot exploit the
   phase-washing/tone-preserving code pressure; the post-side lift is
   decode-structure-contingent. This is the extension's headline
   falsifiable claim: if ctr lifts pre too, the mechanism story
   ("contrastive helps only where the decode can mix positions") is
   wrong and must be reported as such.
2. **(vi)** no variant beats pre-plain beyond the bar on any other
   primary (the § 6 (i) prior transferred).
3. **(vii)** matryoshka's frequency HURTS pattern may replicate on pre
   (capacity split fragments an already-additive code); scored but not
   load-bearing.

_Amendment frozen-by: claude-fable-5 (runpod agent), 2026-07-24,
pre-build of the pre extension._
