# TXC-pro dissection — which loss component actually helps the TXC backbone?

**Headline: ONE component survives — the multi-distance contrastive loss
helps the post-squash backbone on exactly one regime (regime-3 power,
long windows), and nothing else in the TXC-pro bundle helps anything.**
Matryoshka helps recovery nowhere and hurts in four places; the
surviving contrastive win does NOT transfer to the additive pre backbone
(prediction (v) CONFIRMED, § 6), pinning it to the decode structure, not
the loss alone. The paper's TXC-pro bundle — selected by
hill-climbing on a ~0.001-AUC probing noise floor — is therefore mostly
selection on noise, with a single salvageable ingredient, precisely
localized.

Mandate: `briefings/txcpro-dissection.md`. Frozen protocol: `CARD.md`
(commit "loss dissection: ablation card FROZEN (pre-build) …" strictly
before the build commit; § 9 pre-extension amendment frozen before the
pre build). All 45 post-family verdicts and gates are the mechanical
output of `analyze.py` (committed pre-run): `results/dissection_table.md`
(+ `.json`); pre family: `results/dissection_table_pre.md`. Raw grids:
`results/<bench>_dissect[_pre]_grid_results.json`; every cell also runs
through the canonical runner into `results/leaderboard.jsonl`
(code-version stamped).

## 1. Design in one paragraph

Four loss-only variants of `txc_batchtopk_post` — plain / +matryoshka
(H=8 nested prefixes, the paper's spec, enabled at toy scale for the
first time) / +multi-distance contrastive (cosine InfoNCE at window
shifts {1,2}, weights 1/(1+Δ), full code — the paper's own toy-scale
convention) / +both — one sequence-mode class, identical params/init and
identical per-seed training streams, so component effects are
paired-by-seed. Five discriminating benches (backtracking = regime 2,
frequency = r3 power, phasepair = r3 phase, recipe = r3 equality,
multilane = superposition) at the canonical slice d_sae=F, T∈{2,4,8},
k_pos∈{1,2,4}, seeds {1,2,42} + untrained. "Longer windows" is the T
axis, not a variant. No probing arm — probing is a noise floor, not a
dissection venue (briefing provenance fact).

## 2. Gates — every guard passed, everywhere

- **Gate B (graft validity):** the sequence-fed plain twin matches the
  canonical window-mode `txc_batchtopk_post` leaderboard anchors on
  **9/9 cells on all five benches** (and the pre twin likewise on its
  feasible cells, § 6). The reimplementation is exonerated; briefing
  falsifier (iv) never fired (backtracking primary: all three components
  NEUTRAL).
- **Untrained guard:** max |metric diff| across variants = **0.00e+00**
  on every bench — the four variants are bit-identical at init, so any
  effect is the training loss and nothing else.
- **Grids:** post 720/720 ok (~2.1 h, 16 workers); pre 696/696 feasible
  cells ok (~2.4 h; the pooled family loses (T=8, k_pos=4) at F=20 —
  dropped and logged, never silent). Zero failures across 1416 cells.
- Sequence-mode cell cost ≈ 115 s vs ~7 s window-mode (SequenceBuffer
  regenerates per step) — noted for future dissections.

## 3. The component table (post backbone — primary metrics)

| bench (regime) | +matryoshka | +contrastive | +both |
|---|---|---|---|
| backtracking λ (r2) | NEUTRAL | NEUTRAL | NEUTRAL |
| frequency velocity (r3 power) | **HURTS** (to −0.094) | **HELPS — survives skeptic** (T=8: +0.084, +0.093; 0.69→0.78, 0.31→0.40 absolute) | NEUTRAL (mat cancels ctr) |
| phasepair sign (r3 phase) | **HURTS** (−0.16…−0.19 at calm cells) | NEUTRAL (−0.081±0.091, direction consistent with the anti-AC prediction but below bar) | NEUTRAL |
| recipe equality residual (r3 equality) | ~~HELPS~~ **KILLED by skeptic** (e_metric_leak: both arms far below chance — "fails less badly" ≠ extraction) | NEUTRAL | MIXED |
| multilane (superposition) | NEUTRAL (−0.049 max) | NEUTRAL (+0.031 max) | NEUTRAL |

Capability metrics: ctr HELPS backtracking `eauc` (+0.099) and HURTS its
`nmse` (recon degrades while feature geometry improves); mat HURTS
recipe `nmse` and backtracking `eauc`; everything else NEUTRAL. Full 45
verdicts + per-cell D/SE in `results/dissection_table.md`.

Interaction is near-additive everywhere (|I| ≤ 0.057 on frequency
velocity): `+both` on frequency = ctr's gain minus mat's harm — the
bundle's own components partially cancel each other on the one bench
where one of them works.

## 4. The surviving finding, and its mechanism

**+contrastive lifts frequency velocity_recovery at T=8 only** (k=1:
D=+0.084, all seeds +0.052/+0.107/+0.094; k=2: D=+0.093, all seeds
+0.060/+0.126/+0.092; bar = max(2·SE, 0.05) ≈ 0.05; T=2 nothing, T=4
borderline-positive below bar). Skeptic (`claude-fable-5`) recomputed
D/SE from the per-seed deltas, checked all five kill items, no kills:
"absolute levels show real latent extraction (0.69 → 0.78)".

Mechanism (stated in the card's prediction commentary, sharpened by the
pre extension): InfoNCE between Δ-shifted windows is **phase-washing,
tone-preserving pressure** — shifted windows of a cyclic tone share the
velocity latent but not phase, so shift-invariant codes align with tone
identity. That is a *code* pressure; reading it still needs a decode
that can mix positions. Hence prediction (v): the additive pre backbone
should NOT benefit — confirmed in § 6, which upgrades the finding from
"a lift on one bench" to a mechanism with a tested scope: **contrastive
helps only where the decode can exploit position mixing, and only at
window lengths long enough for shift structure to bind (T=8).**

## 5. Predictions scored (frozen § 6, honest ledger)

- **(i) "no variant improves regime-3 recovery beyond seed noise" —
  FALSIFIED in one place**: ctr on frequency velocity (the surviving
  claim). Held everywhere else (12 of 13 other primary-metric component
  verdicts NEUTRAL/HURTS; the 13th was the killed recipe claim).
- **(ii) matryoshka improves capability without recovery — WRONG**: mat
  improved nothing anywhere, capability included (recipe nmse HURTS,
  backtracking eauc HURTS). The observed capability gain belongs to ctr
  (backtracking eauc), not mat.
- **(iii) ctr helps DC/persistent if anywhere — WRONG venue**: recipe
  phase_recovery (the DC target) is NEUTRAL; the actual help is AC
  regime-3 power. The sharpened converse (ctr hurts phasepair sign if it
  moves it) was directionally right (−0.081) but below the bar — scored
  as not triggered.
- **(iii-b) fork resolution: genuine salvage.** Component: multi-distance
  contrastive. Regime: regime-3 power (spectrally-concentrated), post
  backbone, T=8. Everything else in the bundle: drop with nothing to
  salvage — the hill-climb was selection on noise.
- **(iv)** never fired (backtracking all-NEUTRAL, Gate B all-PASS).
- **(v) CONFIRMED, (vi) CONFIRMED, (vii) partially wrong — § 6.**

## 6. Pre-backbone extension (CARD § 9 amendment)

Grid 696/696 feasible cells ok (132 at F=20 benches — the (T=8, k_pos=4)
column is dict-infeasible for the pooled family and was dropped with a
log line; 144 elsewhere). Gate B PASS on all five benches (8/8 or 9/9
cells); untrained guard exact-zero again. Full table:
`results/dissection_table_pre.md`.

**Every one of the 15 primary-metric verdicts is NEUTRAL.** No
recovery-metric HELPS anywhere on the pre backbone — the skeptic was not
triggered ($0), exactly per the frozen policy.

- **(v) CONFIRMED — the headline of the extension.** ctr on pre's
  frequency velocity: NEUTRAL, max |D| −0.047±0.039 (directionally
  *negative*). The post backbone's surviving contrastive lift does NOT
  transfer to the additive-over-positions decode: the salvage is
  **decode-structure-contingent** — contrastive helps only where the
  architecture can exploit position mixing (post's coincidence code),
  and only at T=8. Had ctr lifted pre too, the mechanism story would
  have died; it did not.
- **(vi) CONFIRMED**: no variant beats pre-plain beyond the bar on any
  primary.
- **(vii) partially wrong, honestly scored**: mat's post-side frequency
  *velocity* HURTS did not replicate on pre (NEUTRAL −0.019); mat's harm
  on pre shows up on capability instead — nmse HURTS on frequency,
  recipe, and multilane, plus eauc HURTS on recipe.
- Side observations (capability only, no skeptic per policy): ctr HELPS
  backtracking eauc on pre too (+0.129 — same pattern as post: feature
  geometry improves while nmse is flat/worse), and mat HELPS multilane
  eauc (+0.246±0.005) while HURTING its nmse — a geometry/recon
  trade-off, not a recovery salvage (multilane recovery itself NEUTRAL).

## 7. What this means for the paper

TXC-pro as bundled does not survive dissection: of its three components,
longer windows were already the T axis (and the T-trend is where the one
real effect lives), matryoshka is neutral-to-harmful at every tested
cell, and the multi-distance contrastive loss is a real but narrow
ingredient — it buys regime-3 power recovery on a position-mixing decode
at long windows, and nothing else. The rebuttal-safe sentence: *"we
dissected TXC-pro per-component on a synthetic suite whose effect sizes
are 10–100× the probing noise floor; the bundle's original selection was
noise, and its single active ingredient (multi-distance contrastive) is
salvageable in isolation with a stated mechanism and scope."*

Spend: $0.51 (two skeptic claims on the post family; pre family $0 —
skeptic not triggered; cumulative $11.52/$25; session cap $5 respected).
Wall-clock: card→verdict inside one day, ~11 h including both grids
(720 + 696 cells, zero failures).

_Recorded-by: claude-fable-5 (runpod agent, txcpro-dissection briefing),
2026-07-24. Card and amendment frozen pre-build; scripts committed
pre-run; verdicts mechanical; skeptic raw persisted pre-parse, never
re-rolled._

_Reviewed (2026-07-24, mac-local): **APPROVED** — freeze chain verified
by git forensics (card → build → grid → skeptic; amendment → pre build
→ pre grid); the surviving claim checked against
`results/dissection_table.md` (+0.093 ± 0.038) and the skeptic raw
files; 1,416 leaderboard rows accounted for in the canonical artifact.
The § 7 rebuttal-safe sentence is endorsed as written._
