# tscale RESULTS — append-only candidate ledger

Convention: one section per candidate attempt, appended in run order,
never edited after its verdict line. Config hash = sha256 of the
candidate's full hparam+training dict (printed by the runner). Dev
numbers are dev-8 s42 k20 unless stated. Comparators per CARD_SPLIT § 3
(L1 → matched-steps baseline twin; L2 → P1 s42 rows). Negative results
stay on the record.

| id | candidate | level reached | dev Δ16 (best) | verdict |
|---|---|---|---|---|
| C1 | txc_pro_r1 (+_btkonly twin) | L1 + A1 diag COMPLETE | **+0.1168** (base −0.0134) | slope PASS ×, T1-level FAIL → NO PROMOTE as-is; diag: mechanism CONFIRMED-partial (T1 0.7985→0.8974 w/ AuxK live), 20k curve MONOTONE RISING (0.8974→0.9103→0.9171), but T1 floor + k5-preservation both FAIL ⇒ **A2 NOT triggered** |
| C2 | txc_btk_pre_subseq_btkonly | L1 | −0.0303 | KILL — curriculum alone does not transfer; threshold under-admit datum |
| C3 | r1-min (r1_btkonly, aux losses off) | L1 (+ts attribution) | **+0.1180** | slope PASS, T16 PASS (program-best both k), T1-level FAIL → NO PROMOTE as-is; ts sweep: INTERIOR MAX at ratio rule (t8), tokens-confound killed |
| C4 | r1-min + k_train anneal (8×/2000) | L1 interim (T16 in flight) | — | T1 0.8171: +0.010 via doubled diversity (dose–response) but FAILS floor — transient exposure decays ⇒ H-fail-T1; needs sustained pressure |
| C5 | r1-min + train_select=batch | PRE-REGISTERED | — | sustained pool pressure (twin's healthy T1 dynamic in the r1 frame); § 3 gates as written; launching |

*(index updated as sections land below)*

---

## C1 — txc_pro_r1 / txc_pro_r1_btkonly (recovered recipe, faithful revival) — L1 verdict 2026-07-27 ~20:20 London

Tags `r1-paper-4k` / `r1-btkonly-4k` (+ comparator `baseline-4k`), rows
in `results/l1_rows.jsonl`; config hashes there. 4k steps, b1024
sequences, ratio-rule t_sample, dev-8 s42.

**Curves (k20 | k5):**

| arch | T1 | T4 | T16 | Δ16 (k20) |
|---|---|---|---|---|
| baseline twin (pre_btkonly) | 0.8944 \| 0.7978 | 0.9099 \| 0.8419 | 0.8810 \| 0.8267 | −0.0134 |
| txc_pro_r1 (paper comp.) | 0.7985 \| 0.7291 | 0.8633 \| — | **0.9153** \| **0.8711** | **+0.1168** |
| txc_pro_r1_btkonly | 0.7985 \| 0.7291 | 0.8634 \| 0.8035 | **0.9148** \| **0.8610** | **+0.1162** |

**Findings (all PTR; dev-only, 1 seed, 4k steps — NOT claim-grade):**

1. **First monotone-rising TXC T-curve in the program** — and the T16
   level beats the matched-steps baseline twin at BOTH k (k20 +0.034
   over its T16, +0.005 over its best-anywhere; k5 +0.034/+0.029),
   at the program sparsity convention (l0 = 20·T exactly for btkonly:
   20.0/80.0/320.0).
2. **Mechanism read:** the rise is COLLAPSE RECOVERY + genuine win.
   L0 census: frac latents active in a full batch = 0.021 (T1) →
   0.133 (T4) → 0.363 (T16) — the recipe collapses at low T (2 % of
   dictionary carrying k=20 probes ⇒ depressed T1 anchor 0.7985) and
   window growth restores feature diversity (larger t_sample ⇒ more
   tokens/step + bigger k_train). AuxK was INERT at T1/T4 (tokens
   seen < 10 M dead threshold — `frac_dead_threshold = 0.0`); at T16
   it engaged late (0.239).
3. **The T16 gain is ORDER-FREE**: shuffled twin ≥/≈ ordered
   (k20 gap −0.0007 paper / −0.0001 btkonly) — pooled composition
   information, not sequence structure (same regime as the k=5
   inversion mechanism, fcf62963b).
4. **Composition twins are training-identical** (bit-identical loss
   traces at T1/T4/T16 through step logs; final AUC Δ ≤ 0.0005; the
   only fingerprint difference is the paper arm's zero-picks,
   l0 319.6 vs 320.0 at T16) — the per-sample-TopK analog of
   runpod-1's P1-RM arm-identity finding (d4645c242). **DECISION:
   carry `txc_pro_r1_btkonly` ONLY from here** (the baseline's arm);
   the paper twin remains registered as the faithfulness receipt.

**Formal gates (CARD § 3, k20 vs matched-steps twin):** slope PASS
(+0.1168 ≫ −0.0054 required); level T16 PASS (0.9153 ≥ 0.8810); level
T1 **FAIL** (0.7985 < floor 0.8844) ⇒ **L1→L2 PROMOTE: NO** as-is.

**A1 mechanism-exception INVOKED (this statement is the § A1 (i)
pre-declaration):** L0 receipt = `frac_dead_threshold 0.0` at T1/T4
with `frac_latents_active_batch 0.021/0.133` — the 4k screen
structurally disabled the recipe's anti-collapse machinery. ONE
L2-shaped diagnostic launches next: `txc_pro_r1_btkonly`, 20k steps,
dev T-grid {16, 1, 4} (T16 first), s42, both k, tag `r1b-L2diag-20k`.
Questions: (a) does the T1 collapse resolve once AuxK engages
(20.5 M tokens at T1 > threshold)? (b) does the T16 win hold at
canonical step count? Its result CANNOT reach L3 without passing the
§ 3 L2→L3 gates as written.

**In flight (GPU 0, plain L1 cells):** ingredient attribution at 4k —
`r1b-nocontr-4k` (contrastive_alpha=0) and `r1b-nomatr-4k`
(h_size=d_sae, contr_prefix=3686) at T {1, 16}: which ingredient
drives the collapse, which carries the slope.

---

## C2 — txc_btk_pre_subseq_btkonly (subseq curriculum on the btk backbone) — L1 verdict 2026-07-27 ~22:57 London

Tag `subseq-btk-4k`; 4k steps, WindowBuffer serving (Amendment-1 batch
rule), ratio-rule t_sample, dev-8 s42.

| | T1 | T4 | T16 | Δ16 |
|---|---|---|---|---|
| k20 | 0.8944 (≡ twin, receipt) | 0.8928 | 0.8641 | **−0.0303** |
| k5 | 0.7978 | 0.8163 | 0.7815 | −0.0163 |

**Verdict: NO TRANSFER — L1 KILL at 4k.** The curriculum grafted onto
the BatchTopK backbone declines MORE steeply than the plain baseline
(twin Δ16 −0.0134). The T=1 bit-identity receipt worked exactly as
designed (0.8944, l0 13.4 ≡ twin). Additional mechanism datum: T16
realized l0 283.8 vs 320 nominal — the JumpReLU threshold calibrated
on the SAMPLED pool (t_sample=8) under-admits at full-T serve; the
graft pays a real threshold-path cost for the train/serve asymmetry
that the r1 family (exact per-sample TopK at serve, no threshold)
never pays.

**Attribution state after C1+C2:** the rising T-curve requires the r1
TRAINING COMBINATION (subseq × per-sample-window-TopK × sequence
serving with random anchor offsets × exact-k serve) — not the
curriculum alone on a BatchTopK/threshold backbone. Next completing
cell: `r1b-min-4k` (r1 btkonly with contrastive AND matryoshka both
removed — the minimal subseq+TopK+auxk recipe), T {1,16}, launched
~22:55 on GPU 0. If r1-min holds ≈0.918 at T16, the minimal recipe is
the L2 candidate; its T1 collapse remains the open problem (L2 diag
answers the AuxK-live half overnight).

---

## C3 — r1-min (`txc_pro_r1_btkonly` + contrastive_alpha=0, h_size=d_sae, contr_prefix=3686) — L1 verdict 2026-07-28 ~00:06 London

Tag `r1b-min-4k`; 4k steps, b1024 sequences, ratio-rule t_sample,
dev-8 s42. Config hashes 7629e123d0679ac0 (T1) / ec681f27912395a0
(T16). Family: C1 (txc_pro_r1) for A1 accounting.

| | T1 | T16 | Δ16 |
|---|---|---|---|
| k20 | 0.8071 (shuf ≡, T=1 identity) | **0.9251** (shuf 0.9258) | **+0.1180** |
| k5 | 0.7326 | **0.8763** (shuf 0.8776) | +0.1437 |

**Findings (PTR; dev-only, 1 seed, 4k steps — NOT claim-grade):**

1. **Program-best T16 at BOTH k.** k20 0.9251 (twin 0.8810, full r1
   0.9148/0.9153, nocontr 0.9177, nomatr 0.9185); k5 0.8763 (twin
   0.8267, full r1 0.8610/0.8711). Context only (mismatched steps,
   dev-8, 1 seed): these 4k cells sit ABOVE the 20k-step P1
   references for the first time anywhere in the program (k20 P1 pre
   T16 0.8985, SAE band 0.9111; k5 P1 0.8651, SAE band 0.8450).
2. **Aux-loss harm at T16 is super-additive.** Removal deltas vs
   full-recipe btkonly 0.9148: contrastive-off +0.0029, matryoshka-off
   +0.0037, both-off **+0.0103** (> +0.0066 sum). Both aux losses
   actively tax the T16 representation; the minimal
   subseq+TopK+auxk recipe is the strongest family member.
3. **T1 collapse is NOT the aux losses' fault** (they're exonerated
   for the collapse too): T1 ladder nocontr 0.7955 < full 0.7985 <
   nomatr 0.8024 < min 0.8071, all ≈ 0.80 vs twin 0.8944; census
   frac_latents_active_batch 0.0237 (≈ full's 0.021),
   `frac_dead_threshold = 0.0` at T1 (4k·1024·1 = 4.1 M tokens < 10 M
   — the SAME structural AuxK-inert artifact as C1).
4. **Still order-free at T16** (shuf − ord: k20 +0.0007, k5 +0.0013)
   — pooled composition, not sequence structure; the binding claim
   caveat carries.
5. **Budget receipts exact:** train-path l0 20.0 / 160.0
   (= k_pos·t_sample), probe-path realized_l0 20.0 / 320.0
   (= k_serve; § 3 L0 sanity gate passes).

**Formal gates (CARD § 3, k20 vs matched-steps twin):** slope PASS
(+0.1180 ≫ −0.0054); level T16 PASS (0.9251 ≥ 0.8810); level T1
**FAIL** (0.8071 < floor 0.8844) ⇒ **L1→L2 PROMOTE: NO** as-is.

**A1 accounting — NO second diagnostic.** The C1-family A1 slot is
consumed by the in-flight `r1b-L2diag-20k` (full recipe, GPU 1); per
§ A1 (iii) ("one diagnostic per candidate family, not per config")
r1-min does NOT get its own 20k diagnostic, despite being the
stronger config. The family diag still answers the config-shared
mechanism question (is the T1 collapse a 4k-screen artifact of AuxK
inertness?).

**A2 decision tree (pre-stated NOW, before any diag cell has
landed):** when `r1b-L2diag-20k` completes —

- IF the diag's T{16,1} cells pass the § 3 L2 slope+level criteria
  against the § 2 P1 s42 row (T16 ≥ 0.8985 with slope, T1 ≥ 0.9035)
  → the mechanism exception is VALIDATED for the family; propose
  amendment **A2**: the family's one L2 slot re-runs as a FULL L2
  (20k, dev {1,2,4,8,16}, both k) on **r1-min**, the
  attribution-selected config — appended to the card before launch,
  loudly PTR to mac-local (A1 append-then-run precedent). The
  full-recipe diag numbers themselves graduate nowhere; r1-min's L2
  would face the § 3 L2→L3 gates fresh.
- IF the diag T1 does NOT recover to level → the collapse is
  structural at 20k too; NO A2. Low-T fixes (per-position k floors,
  k-anneal, t_sample/k_train floor at small T) enter as NEW
  candidates through L1 as written.
- IF the diag T16 fails to hold → the family's win is a 4k artifact;
  kill the lane, move down the menu.

**Launching meanwhile (GPU 0, plain L1 attribution cells at T16,
pre-declared here):** `r1min-ts16-4k` (t_sample=16 = NO subsampling:
is the curriculum necessary AT ALL within r1-min, given C2 killed
curriculum-alone on the btk backbone? — if ts16 ≈ ts8's 0.9251 the
win is per-sample-window-TopK + sequence serving, not subseq) and
`r1min-ts5-4k` (absolute t_sample=5, asymmetry 3.2 — the
phase5b-locked instance, the CARD § 4 pre-registered ablation).
Confound note, stated up front: t_sample scales tokens/step and
k_train together at matched steps (ts16 sees 2× the tokens of ts8);
matched-steps is the program's L1 convention — read with that lens.

---

## C1-D — A1 family diagnostic `r1b-L2diag-20k` (full recipe, 20k steps): T{16,1} verdict + A2 walk — 2026-07-28 ~00:50 London

Dev-8 s42, 20k steps, b1024 sequences; hashes 9d9567ddd6a4ef6e (T16) /
72a9f0a979cf575c (T1); T4 cell still running (interior point — cannot
change this verdict; appends below when it lands).

| | T1 | T16 | Δ16 | P1 s42 20k (T1 / T16 / Δ) |
|---|---|---|---|---|
| k20 | 0.8974 (shuf ≡) | 0.9171 (shuf 0.9139) | **+0.0197** | 0.9135 / 0.8985 / −0.0150 |
| k5 | 0.7555 | 0.8487 (shuf 0.8489) | +0.0932 | 0.8417 / 0.8651 / +0.0234 |

Census: T1 `frac_latents_active_batch` 0.120 (4k: 0.021 — 6×
recovery), `frac_dead_threshold` 0.352 (AuxK LIVE and working); T16
active 0.423 / dead 0.312. l0 receipts exact (serve 20/320, train
20/160).

**Findings (PTR; dev-only, 1 seed):**

1. **Mechanism CONFIRMED, recovery PARTIAL.** With AuxK live past the
   10 M-token threshold, T1 recovers 0.7985 → 0.8974 (+0.099) — most
   of the low-T collapse WAS the 4k-screen artifact, vindicating A1's
   premise. The residual −0.0161 vs P1's T1 (0.9135) co-occurs with
   residual concentration (12 % active vs 42 % at T16): per-sample
   TopK at T1 (k_train=20 drawn from ONE position per row) still
   concentrates; AuxK fights it but does not fully win by 20k.
2. **First RISING k20 curve at canonical steps.** Δ16 +0.0197 (P1:
   −0.0150); T16 0.9171 sits +0.0186 over the P1 row and above the
   SAE band (0.9111 ± 0.0042). The 4k→20k T16 change is +0.002 —
   the win was not a step-count artifact.
3. **k5 REGRESSION at 20k.** T16 k5 0.8487 vs 0.8711 at its own 4k
   (−0.022, opposite sign to k20) — BELOW the § 3 preservation bar
   (0.8651 − 0.010 = 0.8551). Longer training buys k20 slope partly
   at k5 fidelity's expense — precisely the failure mode § 3 guards
   ("do not destroy the k=5 recovery to buy the k=20 slope").
4. **Order-sensitivity switches ON at 20k** (k20 T16 shuf gap +0.0032
   ordered-over-shuffled; ≈ 0/negative at every 4k cell; P1 baseline
   +0.0305). Small, but the first order-positive signal in the r1
   family — longer training begins to use sequence structure.

**A2 walk (triggers verbatim from C3, pre-stated before any cell
landed):** T16 ≥ 0.8985 ✓ (slope ✓, +0.0197 vs required ≥ −0.0070);
T1 ≥ 0.9035 ✗ (0.8974, short −0.0061) ⇒ **A2 NOT TRIGGERED.** The
k5-preservation miss (finding 3) lands on the same side
independently. The family's one diagnostic slot is spent; there is no
exception-lane L2 for r1-min (A1 (iii)). r1-min may still EARN L2
only by passing the L1 gates as written.

**Lane forward (per the pre-stated tree): C4+ low-T fixes through L1
as written.** Design constraints from tonight's mechanism data:
(i) the fix must not depend on AuxK — structurally inert at the 4k
screen (A1 artifact), so an AuxK-dependent fix can never pass L1
honestly; candidates that act from step 0: k_train-anneal (wide→20,
attacks across-row concentration directly), batch-diverse selection
at small T, k_train floors. (ii) The T1 problem is ACROSS-ROW latent
concentration (same latents win every row when k_train=20 from one
position), not within-row budgeting — per-position floors do NOT
address it. (iii) C4 must track k5@T16 alongside T1 (finding 3
failure mode). Decision on the specific C4 design waits for the
ts16/ts5 attribution cells (in flight) — whether the curriculum is
even necessary shapes the search space.

---

## C4 — k_train anneal on r1-min (`c4-kanneal-4k`) — PRE-REGISTRATION 2026-07-28 ~01:15 London (before launch)

**Design (menu item "sparsity scheduling"; mechanism-targeted per
C1-D):** anneal the TRAINING admission width linearly from
`k_anneal_mult · k_train` down to `k_train` over `k_anneal_steps`
steps, then constant; serve path (exact k_serve = k_pos·T) untouched.
Implemented as `# r1-c4:` tagged deviations in `txc_pro_r1.py`
(defaults OFF = bit-identical pre-C4 behavior, tested; anneal progress
is a plain attr — scratch screens never resume, state_dict/ckpt compat
untouched; knobs flow via arch_hparams_override and hash into
config_hash).

**Mechanism target:** C1-D showed the T1 problem is ACROSS-ROW latent
concentration (k_train=20 drawn from ONE position per row → the same
high-bias latents win every row → 2 % active at 4k), and that healthy
diversity appears exactly where per-row admission is wide (T16 trains
at k_train=160 → 36–42 % active). The anneal gives EVERY T that wide
early exposure (spread gradient → diverse dictionary) without touching
the serve budget and WITHOUT AuxK (structurally inert at the 4k
screen — C4's fix must act from step 0, per C1-D constraint (i)).

**Pre-registered point (ONE, no sweep before signal):**
`k_anneal_mult = 8` — mirrors the 8× admission ratio that separates
healthy T16 (k_train 160) from collapsed T1 (k_train 20);
`k_anneal_steps = 2000` — half the L1 screen, leaving 2000 steps at
the nominal budget before eval. Cells: T {1, 16}, 4k steps, dev-8
s42, both k, r1-min backbone (contrastive_alpha=0, h_size=18432,
contr_prefix=3686), tag `c4-kanneal-4k`. Launch: on next GPU drain.

**Gates: § 3 L1→L2 as written, no exceptions** — T1 ≥ 0.8844,
T16 ≥ 0.8810, slope vs twin. Also REPORT k5@T16 (the 20k regression
mode from C1-D finding 3 is an L2 concern, but a 4k k5 collapse would
be an early kill signal). Hypotheses on record: (H-pass) anneal
restores T1 dictionary diversity → T1 clears the floor while T16
keeps ≈ 0.925 → first full-gate L1 PASS of the program → L2 proper.
(H-fail-T1) concentration re-forms after the anneal ends → T1 stays
≈ 0.81 → the fix needs sustained pressure, not initialization-time
exposure (points to batch-diverse selection at small T as C5).
(H-fail-T16) wide early admission degrades the T16 win → the anneal
trades the win for the floor — record and reassess.

---

## C1-D completion — diag T4 cell — 2026-07-28 ~02:10 London

T4 @ 20k (cfg c29ea51f2aaaa3c4): k20 **0.9103** (shuf 0.9078, order
gap +0.0025 — consistent with the 20k order-positive signal), k5
0.8472; census active 0.221 / dead 0.365; budgets exact (train 40,
serve 80). **The full-recipe 20k curve is MONOTONE RISING: 0.8974 →
0.9103 → 0.9171** (P1: 0.9135 → 0.9181 → 0.8985). Interior-dip check
vs P1: T4 −0.0078 (inside the 0.010 band). k5 curve 0.7555 → 0.8472 →
0.8487 (T4 above P1's 0.8434; T16 below 0.8651 — the C1-D finding-3
regression). No verdict change: A2 remains NOT triggered (T1 level +
k5@T16). C1 lane closed at diagnostic; family continues via C4/C5.

---

## C3 addendum — t_sample attribution at T16 (`r1min-ts16-4k` / `r1min-ts5-4k`) — 2026-07-28 ~02:10 London

r1-min backbone, T16, 4k, dev-8 s42 (hashes ae40dc52ad72d3e8 /
040b9f18e5d919ae):

| t_sample (asym k_serve/k_train) | k20 | k5 | active_frac |
|---|---|---|---|
| 5 (3.2×) | 0.9167 | 0.8652 | 0.223 |
| **8 = ratio rule (2×)** | **0.9251** | **0.8763** | **0.407** |
| 16 = no subsample (1×) | 0.9149 | 0.8664 | 0.233 |

1. **INTERIOR MAXIMUM at the ratio rule** (both k) — the CARD § 4
   pre-registration is validated; both extremes lose ≈ 0.008–0.010.
   Mirrors phase5b's own t-sweep (t = T_max/2 optimal at T_max 10).
2. **The tokens/step confound is DEAD as an explanation:** scored
   tokens/step rise monotonically with t_sample (5.1k / 8.2k / 16.4k)
   — a data-volume story predicts monotone gains in t; the observed
   interior max contradicts it. The subseq curriculum's contribution
   (+0.010 over full-window at MATCHED steps, with FEWER tokens) is
   real.
3. **Dictionary diversity tracks probe quality a third time**
   (active-frac 0.22 / 0.41 / 0.23 aligns with AUC ordering) — same
   direction as the C4 T1 dose-response and the T-grid census. The
   family's operative variable is looking like TRAINED DICTIONARY
   DIVERSITY, however induced.
4. Curriculum-necessity read (ts16 vs ts8): subsampling carries
   +0.0102 of the win; the remaining +0.034 over the twin comes from
   per-sample-window TopK + sequence serving (ts16 0.9149 ≈ full
   recipe 0.9148, coincidentally).

---

## C4 interim — k_train anneal T1 datum — 2026-07-28 ~02:10 London (T16 cell in flight)

T1 (cfg 37b53d95ea3d0289): k20 **0.8171** (r1-min 0.8071; twin floor
0.8844), l0 serve 20.0 exact. Census: active_frac **0.0542** vs
r1-min's 0.0237 — the anneal DOUBLED trained diversity and bought
+0.0100 AUC (clean dose–response), but the effect decays after the
anneal ends (2000 nominal steps re-concentrate). **H-fail-T1
CONFIRMED pending T16: transient wide admission is insufficient — the
mechanism needs SUSTAINED pressure.** T1 gate: FAIL (0.8171 <
0.8844). T16 cell lands ~02:50; C4 verdict line then.

---

## C5 — train-time batch-pool admission on r1-min (`c5-batchsel-4k`) — PRE-REGISTRATION 2026-07-28 ~02:12 London (before launch)

**Design (the C4-H-fail arrow, pre-stated in C4's hypotheses):**
`train_select="batch"` — training admission becomes a pooled B·k_train
budget over the whole (B, d_sae) batch (BatchTopK-style: rows COMPETE
each step; per-row counts vary, total is exact) — the baseline twin's
proven-healthy T1 dynamic, imported into the r1 frame as SUSTAINED
diversity pressure. Serve path per-row exact-k UNCHANGED in both
modes. Arm conventions held: btk-only arm selects by RAW SIGNED value
and passes survivors through signed; paper arm selects then ReLUs.
Implemented as `# r1-c5:` tagged deviations in `txc_pro_r1.py`;
default `"row"` is bit-identical pre-C5 (tested; 29/29 suite).
Anneal OFF in C5 — one variable at a time.

**Pre-registered cells:** T {1, 16}, 4k steps, dev-8 s42, both k,
r1-min backbone (contrastive_alpha=0, h_size=18432, contr_prefix=3686,
train_select=batch), tag `c5-batchsel-4k`, T1 first. Launch: GPU 0 on
this commit.

**Gates: § 3 L1→L2 as written, no exceptions** (T1 ≥ 0.8844,
T16 ≥ 0.8810, slope; k5@T16 reported). Hypotheses: (H-pass) sustained
pool pressure holds T1 diversity → T1 clears the floor; T16 keeps
≥ ≈ 0.92 → first full-gate L1 PASS. (H-fail-T16) the pooled budget
destroys the per-sample-window-TopK win at T16 (echoing C2's
batch-pool component) → selection rule must differ by T — record as a
structural finding. (H-fail-T1) even sustained pressure leaves T1
≈ 0.82 → concentration is not admission-driven at T1; re-examine
(single-position encoder slab? b_enc bias?) before C6.

---

## C5 interim — batch-pool T1 datum + TWIN CENSUS — 2026-07-28 ~02:24 London (C5 T16 in flight)

C5 T1 (cfg cfdac03dcaa38b97): k20 **0.8221** — best collapsed-family
T1 yet still FAILS the floor (0.8844). **Census surprise:**
`frac_latents_active_batch` = **0.0073** — the MOST concentrated
dictionary in the program (r1-min 0.024, C4 0.054) at the best
r1-family T1 score. Diversity is NON-MONOTONE with AUC within the r1
family at T1 — the earlier three-context correlation gets this caveat
on the record.

**Twin census (measurement, this session):** the baseline twin's T1
ckpt (83a57e4412200a37) under its TRAIN-path selection on a matched
1024-row batch: `active_frac = 0.1276`, top-20 latents carry 11.4 %
of fired slots, 676 latents cover 50 %, l0/window 20.0 exact. So the
twin IS ~5–17× more diverse than every r1-family T1 — but C5 shows
the SAME pooled selection rule yields 0.128 active on the twin
backbone and 0.0073 on r1's. **The selection rule is exonerated; the
BACKBONE (what feeds selection) drives concentration.** Remaining
diff list at T1: (a) geometric-median b_dec centering (r1 yes / twin
no), (b) shift-window reconstruction (r1 sums recon over anchor + 2
shifted windows = 3× per-step recon gradient scale; twin has one
window), (c) sequence serving with random anchor offsets. C5's
H-fail-T1 branch is CONFIRMED at T1 (sustained pool pressure does not
heal it); its T16 cell still decides H-fail-T16 (~03:30).

---

## C6 — backbone diff-ablations at T1 (`c6-nobdec-4k` / `c6-noshiftrecon-4k`) — PRE-REGISTRATION 2026-07-28 ~02:24 London (before launch)

**Design:** isolate the two cheap suspects from the C5 diff list on
the r1-min backbone, T1-ONLY screens (6 min each; T16 follow-up ONLY
for a cell that moves T1 materially): (a) `bdec_geom_median_init=0`
(existing hparam; passed as int 0 — the string "false" would be
truthy) — kills the geometric-median centering diff; (b)
`recon_shifts=0` (new `# r1-c6:` tagged knob in txc_pro_r1.py,
default-on bit-identical, tested 32/32 incl. a shift-exclusive
blindness check) — anchor-only reconstruction, killing the 3×
recon-gradient-scale diff. Cells: T1, 4k, dev-8 s42, both k, r1-min
backbone, tags above, GPU 1 on C4-T16 drain. Serving diff (c) is not
cheaply isolable (consumes='sequence' is structural) — deferred.

**Read rules:** T1 ≥ 0.8844 (gate floor) on either cell → that diff
IS the collapse driver → T16 confirmation cell immediately (the win
must survive the removal). Material move (≥ +0.03 over r1-min's
0.8071) short of the floor → partial driver; combine both knobs in a
follow-up cell. Both flat → the driver is the serving/init residue —
re-think before spending more (C7 would need the serving isolation).

---

## FREEZE — hill-climb halted cleanly (Han order 11227ce0d) — 2026-07-28 02:40 London (date-verified) — RESUME PLAYBOOK

**Why:** fleet paper-faithful sprint; this pod's 2×H100 take probing
shards at runpod-1's card-pin (~05:00 target). Frozen, not abandoned
(order text: "the program resumes post-rebuttal; C4 stays
pre-registered").

**State at freeze:**

- C1–C5 index table above is current. Two cells in flight at order
  time, kept per the "minutes-from-done" clause and both draining
  ≥1.7 h before card-pin: **C4-T16** (GPU 1, lands ~02:50 London) and
  **C5-T16** (GPU 0, lands ~03:15). Their verdict lines append below
  on drain (append-only discipline unchanged). ts-attribution chain
  fully drained (~02:15 entries above).
- **C6 pre-registered above and NOT LAUNCHED — that is the freeze
  line and the resume point.** Nothing else launches.

**Resume playbook (one card):**

1. Read `CARD_SPLIT.md` (§ 3 gates — unchanged, still frozen) +
   the RESULTS index + this block.
2. Launch the C6 pair exactly as pre-registered (§ C6): r1-min
   backbone, T1-only 4k dev-8 s42, tags `c6-nobdec-4k`
   (`bdec_geom_median_init=0`) and `c6-noshiftrecon-4k`
   (`recon_shifts=0`); pass CLI booleans as `0`/`1` (parser casts
   int→float→string; `"false"` is truthy). Apply the § C6 read rules
   verbatim.
3. Winner path unchanged: floor-clearing knob → T16 confirmation →
   full § 3 L1 gates → L2 (20k, dev {1,2,4,8,16}, both k, L2→L3
   gates incl. k5 preservation bar 0.8551) → L3 ONLY via canonical
   `run_experiment`, seeds {1,2,42}, eval_cfg namespacing.
4. Open menu (unchanged): multi_window exposure fix, position-loss
   reweighting. Ruled out: per-position k floors (C1-D),
   curriculum-alone (C2, KILLED + ratified), k-anneal transient
   (C4 T1), batch-pool alone (C5 T1).
5. Standing science: backbone-not-selection drives T1 concentration
   (C5 twin census 0.1276 vs r1 ≤0.021); AuxK-live recovers most of
   T1 at 20k but k20 slope was partly bought at k5's expense (C1-D).

**Durability:** all lane ckpts mirroring to HF `temp-bench-data`
under `ckpts/tscale/<cfg_hash>/model.safetensors` (decision-grade
first; receipts → `results/hf_durability_receipts.jsonl`; C4/C5-T16
stragglers mirrored on drain). `l1_rows.jsonl` committed in-tree.

**Ledger:** overnight hill-climb actuals ≈ $16 at freeze (00:00 →
~03:15 drain, 2×H100 pod-wall) vs $35–40 est — under, frozen early.
Hours after ~03:15 bill to the probing sprint.
