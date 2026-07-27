# tscale RESULTS — append-only candidate ledger

Convention: one section per candidate attempt, appended in run order,
never edited after its verdict line. Config hash = sha256 of the
candidate's full hparam+training dict (printed by the runner). Dev
numbers are dev-8 s42 k20 unless stated. Comparators per CARD_SPLIT § 3
(L1 → matched-steps baseline twin; L2 → P1 s42 rows). Negative results
stay on the record.

| id | candidate | level reached | dev Δ16 (best) | verdict |
|---|---|---|---|---|
| C1 | txc_pro_r1 (+_btkonly twin) | L1 (+A1 family diag in flight) | **+0.1168** (base −0.0134) | slope PASS ×, T1-level FAIL → NO PROMOTE as-is; A1 exception invoked; mechanism = low-T activation collapse |
| C2 | txc_btk_pre_subseq_btkonly | L1 | −0.0303 | KILL — curriculum alone does not transfer; threshold under-admit datum |
| C3 | r1-min (r1_btkonly, aux losses off) | L1 | **+0.1180** | slope PASS, T16 PASS (program-best both k), T1-level FAIL → NO PROMOTE as-is; A1 (iii) bars a 2nd family diag; A2 tree pre-stated |

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
