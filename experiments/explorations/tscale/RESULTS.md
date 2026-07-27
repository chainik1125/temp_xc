# tscale RESULTS — append-only candidate ledger

Convention: one section per candidate attempt, appended in run order,
never edited after its verdict line. Config hash = sha256 of the
candidate's full hparam+training dict (printed by the runner). Dev
numbers are dev-8 s42 k20 unless stated. Comparators per CARD_SPLIT § 3
(L1 → matched-steps baseline twin; L2 → P1 s42 rows). Negative results
stay on the record.

| id | candidate | level reached | dev Δ16 (best) | verdict |
|---|---|---|---|---|
| C1 | txc_pro_r1 (+_btkonly twin) | L1 | **+0.1168** (base −0.0134) | slope PASS ×, T1-level FAIL → NO PROMOTE as-is; A1 exception invoked; mechanism = low-T activation collapse |

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
