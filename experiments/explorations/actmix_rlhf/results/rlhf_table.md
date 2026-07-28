# ACTMIX RLHF — the both-arm exhibit table

Protocol: preference_auc_k20 primary (5-fold CV, top-20 signed |mean_rejected − mean_chosen| projection); within-window shuffle seed 42 pre-encode; T = 1 archs' shuffle ≡ identity by construction. l0 = realized nonzero per encode unit over response positions (A2 caveat: this family is the k500-dev regime — cross-section sparsity not comparable to c3's k20).

## paper-match (EVAL-ONLY, shipped seed-42 ckpts; case-study artifact, not leaderboard)

| cell | auc | shuffled | gap | mass@20 | l0/unit | len-spurious |
|---|---|---|---|---|---|---|
| topk_sae | 0.6129 | ≡ | — | 0.084 | 500.0 | 0 |
| tsae_paper_k500 | 0.6306 | ≡ | — | 0.087 | 547.1 | 1 |
| tsae_paper_k20 | 0.6097 | ≡ | — | 0.142 | 17.5 | 0 |
| agentic_txc_02 | 0.6096 | 0.5975 | 0.0121 | 0.064 | 500.0 | 3 |

## btk-only (canonical runner, datasource = the shipped ckpts' own training stream)

### seed 42

| cell | auc | shuffled | gap | mass@20 | l0/unit |
|---|---|---|---|---|---|
| batchtopk_sae@T1/k500 | 0.6250 | ≡ | — | 0.087 | 535.3 |
| batchtopk_sae_btkonly@T1/k100 | 0.6130 | ≡ | — | 0.108 | 108.3 |
| batchtopk_sae_btkonly@T1/k500 | 0.6250 | ≡ | — | 0.087 | 535.3 |
| tsae_btkonly@T1/k20 | 0.5997 | ≡ | — | 0.286 | 19.4 |
| tsae_btkonly@T1/k500 | 0.6163 | ≡ | — | 0.095 | 549.6 |
| txc_batchtopk_post@T5/k500 | 0.6229 | 0.6196 | 0.0033 | 0.071 | 516.6 |
| txc_batchtopk_post@T16/k1600 | 0.6108 | 0.6130 | -0.0022 | 0.052 | 1645.9 |
| txc_batchtopk_post_btkonly@T1/k100 | 0.5777 | ≡ | — | 0.098 | 108.5 |
| txc_batchtopk_post_btkonly@T2/k200 | 0.6196 | 0.6107 | 0.0088 | 0.098 | 210.6 |
| txc_batchtopk_post_btkonly@T5/k500 | 0.6229 | 0.6196 | 0.0033 | 0.071 | 516.6 |
| txc_batchtopk_post_btkonly@T6/k600 | 0.6350 | 0.6328 | 0.0022 | 0.070 | 617.3 |
| txc_batchtopk_post_btkonly@T8/k800 | 0.6262 | 0.6262 | 0.0000 | 0.064 | 831.5 |
| txc_batchtopk_post_btkonly@T10/k1000 | 0.6218 | 0.6207 | 0.0011 | 0.057 | 1041.4 |
| txc_batchtopk_post_btkonly@T16/k1600 | 0.6108 | 0.6130 | -0.0022 | 0.052 | 1645.9 |
| batchtopk_sae_btkonly@T1/k100 (untrained) | 0.5899 | ≡ | — | 0.055 | 2.8 |
| batchtopk_sae_btkonly@T1/k500 (untrained) | 0.6588 | ≡ | — | 0.028 | 91.5 |
| tsae_btkonly@T1/k20 (untrained) | 0.5000 | ≡ | — | 0.000 | 0.0 |
| tsae_btkonly@T1/k500 (untrained) | 0.6588 | ≡ | — | 0.028 | 91.5 |
| txc_batchtopk_post_btkonly@T1/k100 (untrained) | 0.5998 | ≡ | — | 0.058 | 2.9 |
| txc_batchtopk_post_btkonly@T2/k200 (untrained) | 0.6125 | 0.6047 | 0.0077 | 0.065 | 13.1 |
| txc_batchtopk_post_btkonly@T5/k500 (untrained) | 0.6483 | 0.6477 | 0.0006 | 0.045 | 95.4 |
| txc_batchtopk_post_btkonly@T8/k800 (untrained) | 0.6119 | 0.6251 | -0.0132 | 0.027 | 226.8 |
| txc_batchtopk_post_btkonly@T16/k1600 (untrained) | 0.6207 | 0.6273 | -0.0066 | 0.015 | 816.6 |

### seed 1

| cell | auc | shuffled | gap | mass@20 | l0/unit |
|---|---|---|---|---|---|
| batchtopk_sae_btkonly@T1/k100 | 0.5986 | ≡ | — | 0.095 | 107.6 |
| batchtopk_sae_btkonly@T1/k500 | 0.6174 | ≡ | — | 0.087 | 530.9 |
| tsae_btkonly@T1/k20 | 0.6019 | ≡ | — | 0.271 | 19.4 |
| tsae_btkonly@T1/k500 | 0.6251 | ≡ | — | 0.081 | 552.3 |
| txc_batchtopk_post_btkonly@T1/k100 | 0.5975 | ≡ | — | 0.111 | 108.2 |
| txc_batchtopk_post_btkonly@T2/k200 | 0.6162 | 0.6185 | -0.0022 | 0.097 | 209.2 |
| txc_batchtopk_post_btkonly@T5/k500 | 0.6218 | 0.6119 | 0.0099 | 0.073 | 510.0 |
| txc_batchtopk_post_btkonly@T6/k600 | 0.6328 | 0.6317 | 0.0011 | 0.071 | 615.8 |
| txc_batchtopk_post_btkonly@T8/k800 | 0.6251 | 0.6328 | -0.0077 | 0.063 | 822.7 |
| txc_batchtopk_post_btkonly@T16/k1600 | 0.6196 | 0.5998 | 0.0198 | 0.053 | 1643.5 |

### seed 2

| cell | auc | shuffled | gap | mass@20 | l0/unit |
|---|---|---|---|---|---|
| tsae_btkonly@T1/k20 | 0.5986 | ≡ | — | 0.260 | 19.4 |
| tsae_btkonly@T1/k500 | 0.6217 | ≡ | — | 0.093 | 536.6 |
| txc_batchtopk_post_btkonly@T1/k100 | 0.6008 | ≡ | — | 0.104 | 108.5 |
| txc_batchtopk_post_btkonly@T2/k200 | 0.6096 | 0.6085 | 0.0011 | 0.094 | 211.2 |
| txc_batchtopk_post_btkonly@T5/k500 | 0.6185 | 0.6041 | 0.0143 | 0.073 | 517.0 |
| txc_batchtopk_post_btkonly@T8/k800 | 0.6262 | 0.6163 | 0.0099 | 0.064 | 825.1 |
| txc_batchtopk_post_btkonly@T16/k1600 | 0.6317 | 0.6086 | 0.0231 | 0.056 | 1647.5 |

## Mechanical R-scoring (CARD § 4, as frozen)

```json
{
 "R_K1": {
  "values": {
   "papermatch_topk_sae": 0.6128953919009168,
   "btk_sae_k500": 0.6250440167567239
  },
  "pass": true
 },
 "R_K2": {
  "pass": true,
  "note": "cache meta integrity_gate.pass=True (36.232/28.573/9.76e-10, phase-7 verbatim)"
 },
 "R_K3": {
  "n_spurious": 3,
  "pass": true,
  "note": "paper's own '3 length-spurious' = 3 observed"
 },
 "R_E1": {
  "gap": 0.012118268471859595,
  "holds": true
 },
 "R_E2": {
  "holds": true,
  "note": "T=1 shuffle == identity by construction; not simulated"
 },
 "R_E3": {
  "btk_T5": 0.6228765709428693,
  "shipped": 0.6096472588185295,
  "holds": true
 },
 "R_E4": {
  "per_seed": {
   "42": {
    "txc_T1": 0.577688057798555,
    "sae_k100": 0.6129621759456014,
    "delta": -0.035274118147046396
   },
   "1": {
    "txc_T1": 0.5975168477930909,
    "sae_k100": 0.5985914637848339,
    "delta": -0.0010746159917429665
   }
  },
  "delta_seed_mean": -0.01817436706939468,
  "holds": true
 },
 "R_E5": {
  "untrained_aucs": {
   "batchtopk_sae_btkonly/T1/k500": 0.6587578167688666,
   "batchtopk_sae_btkonly/T1/k100": 0.5898822172302836,
   "txc_batchtopk_post_btkonly/T5/k500": 0.6482666504765953,
   "txc_batchtopk_post_btkonly/T1/k100": 0.5997692914819985,
   "txc_batchtopk_post_btkonly/T2/k200": 0.6124612956104668,
   "tsae_btkonly/T1/k500": 0.6587578167688666,
   "tsae_btkonly/T1/k20": 0.5,
   "txc_batchtopk_post_btkonly/T8/k800": 0.6118936312306478,
   "txc_batchtopk_post_btkonly/T16/k1600": 0.6206787687450671
  },
  "holds": false,
  "note": "MISS is informative: k500-class untrained twins reach 0.659 > every trained cell \u2014 the currency is carried by sparse random projections; sae/tsae k500 untrained twins coincide exactly (shared init \u2014 coincidence-by-design receipts check)"
 }
}
```
