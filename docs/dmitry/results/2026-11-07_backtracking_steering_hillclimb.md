---
author: Dmitry
date: 2026-11-07
tags:
  - results
  - complete
---

## Backtracking steering hill-climb: TXC vs T-SAE vs TFA

Goal: find a steering arrangement where the TXC beats T-SAE and TFA at inducing genuine backtracking in DeepSeek-R1-Distill-Llama-8B. Each architecture is reported at its own optimum, with an equal search budget per architecture.

**Bottom line.** Across three rounds and about 2,500 judged continuations, no arrangement gives the TXC a robust advantage. Each architecture's best setting adds about 5–8 coherent backtracking events over a 2–5-event baseline on 20 questions. The architectures differ by 0–2 events, adjacent magnitudes swing by about ±4 events, and every paired confidence interval includes zero. Detection (sparse-probe PR-AUC) favours T-SAE at both S=8 and S=32.

## Setup (shared by all rounds)

- Steering vectors come from dictionaries trained on Llama-3.1-8B layer-10 activations (20K-step checkpoints, d_sae=32768) and are applied to DeepSeek-R1-Distill-Llama-8B.
- Evaluation questions: 20 MATH-500 development questions. The unsteered trace is cut at 25% and the model continues greedily under steering, with a single BOS and the direction calibrated to L2 = 0.4139768.
- The 40 historical test questions and the 80 reserved confirmation questions were never touched.
- Metric: genuine backtracking events summed over globally coherent continuations (incoherent ones score 0), minus a paired no-hook baseline. Labels come from a blinded judge using a frozen rubric.
- Optima are selected on all 20 questions, then applied unchanged to a 19-question subset where the TXC has the lowest last-token reconstruction error.
- **Caveat:** all optima are selected on the same development questions they are scored on. Nothing here is confirmatory.

## Round 1: protocol and direction search (Luna-high judge)

Arms: decode-only steering, alternative directions, and a fine magnitude grid. Global best per architecture:

| Arch | Best setting | Full 20: events vs baseline | Incoherent | Subset 19 delta |
|---|---|---|---|---|
| TXC | averaged direction, prefill+decode, −4 | 9 vs 4 (+5) | 0 | +5 |
| T-SAE | decode-only, −16 | 10 vs 5 (+5) | 0 | +5 |
| TFA | decode-only, −12 | 9 vs 5 (+4) | 2 | +5 |

## Round 2: magnitudes, timing, detection

**Steering.** Round 2's generations were judged here with claude-haiku-5-5, using the same rubric and blinded IDs. These labels came from a different judge than rounds 1 and 3, so compare deltas only within this round. 1–3 items per setting failed to parse and are excluded.

| Track | Arch | Settings (delta vs baseline) | Best |
|---|---|---|---|
| Magnitudes (baseline 2) | TXC prefill+decode | −3.5: −1, −4: +6, −4.5: −1, −5.5: +1 | +6 |
| | T-SAE decode-only | −14: +4, −16: +5, −18: −1, −20: +2 | +5 |
| | TFA decode-only | −10: +6, −11: +2, −12: +6, −14: +3 | +6 |
| Timing (baseline 3) | TXC | first32: 0, first128: +3, full: +3 | +3 |
| | T-SAE | first32: +3, first128: −1, full: +3 | +3 |
| | TFA | first32: +4, first128: +2, full: +4 | +4 |

TXC's +6 at −4 is an isolated spike: −3.5 and −4.5 each score −1.

**Detection** (fresh 5-fold, question-grouped sparse-probe PR-AUC; 17,497 rows from 213 problems; prevalence 0.139):

| Readout | S=8 | S=32 |
|---|---|---|
| T-SAE native (last token) | **0.227** | **0.264** |
| TXC native (last 5) | 0.206 | 0.251 |
| TFA native (last 5) | 0.193 | 0.261 |
| T-SAE last-5 max (secondary) | 0.234 | 0.275 |

At S=32, TXC minus T-SAE per fold is −.049, +.003, +.007, −.004, −.017. Shuffling token order within the window does not hurt TXC or TFA (gaps −0.005 and −0.006 at S=32), so this readout shows no sign that TXC uses temporal order.

## Round 3: temporal steering arrangements (Luna-high judge)

1,080 continuations, decode-only. The TXC rows, the TXC mean direction, T-SAE and TFA all got the same magnitude grids. Paired baseline: 3 events.

- **tile5:** the TXC cycles through its five per-offset decoder rows, one per generated token. T-SAE and TFA, which have one decoder row, use their own vector at every token.
- **pulse5:** a 5-token pulse starts at each sentence boundary, roughly one per 19 tokens. The TXC uses rows 0–4 across the pulse; T-SAE and TFA use their own vector at the same positions.
- **Dropped analytically:** applying each row at every window position reduces exactly to the mean direction (row cosines with the mean are 0.21–0.71). Sum vs mean of rows gives the same direction after norm calibration. Latent clamping is addition along the same decoder row, because decoders are linear.

| Arch | Optimum | Full 20 events / delta / incoherent | Subset 19 delta |
|---|---|---|---|
| TXC | pulse5 rows, −20 | 11 / +8 / 0 | +8 |
| T-SAE | pulse5, −32 | 10 / +7 / 3 | +7 |
| TFA | constant vector, −12 | 9 / +6 / 2 | +6 |

**TXC optimum minus each competitor's optimum** (per question, 10k-resample paired bootstrap):

| Comparison | Full 20 | Subset 19 |
|---|---|---|
| vs T-SAE | +0.05 [−0.30, +0.45] | +0.05 [−0.32, +0.47] |
| vs TFA | +0.10 [−0.20, +0.40] | +0.11 [−0.21, +0.42] |

Why this is not an advantage:

- **Spiky optima:** TXC pulse −16/−20/−24 scores +4/+8/+1, and T-SAE pulse −28/−32/−40 scores +3/+7/+1.
- **Concentrated gain:** one question supplies 3 of TXC's events; TXC improves on 5 questions, T-SAE on 8.
- **Split-half re-selection** (2,000 splits; choose on 10 questions, score on the other 10): mean held-out delta is TXC 1.63, T-SAE 1.58, TFA 1.35. TXC wins outright in 30% of splits, T-SAE in 23%, TFA in 18%, with 29% ties.

The one TXC-specific pattern: at strong magnitudes, cycling the per-offset rows stays coherent longer than the TXC's own mean direction. This is a coherence/dose effect within the TXC, not an advantage over the other architectures.

## Interpretation and next step

With 20 questions, the noise between neighbouring settings (±4 events) is larger than any gap between architectures (≤2 events). Resolving a real difference would need more power: more fresh questions excluding the test and reserved sets, several sampled continuations per question, and a smoothed or held-out optimum. Without that, the defensible claim is that the three architectures are tied on backtracking inducement at their respective optima. T-SAE leads detection.

## Artifacts

- Local notes, small files only (temp_xc_tex repo): `notes/backtracking_astra_search/20261107/`
  - Round 1: `judging/FINAL_COMPARISON.txt`, `judging/selected_comparison.json`
  - Round 2: `hillclimb_round2/ROUND2_CONTRACT.txt`, `hillclimb_round2/detection_features/DETECTION_REPORT.txt`, `detection_results.json`
  - Round 3: `hillclimb_round3_opus/HANDOFF.md`, `analysis_A2_B.json`, `hooks.py`, `run.py`, `PROTOCOL*.txt`
- Prior audited evaluation and protocol: `notes/backtracking_luna_eval/20261107_eval20k/` (`PROTOCOL.txt`, `JUDGE_RUBRIC.txt`, `asset_manifest.json`)
- Full generations, token captures and judge transcripts on simplex3: `/data/users/dmitry/backtracking_astra_search_20261107/` (`runs/`, `hillclimb_round2/`, `hillclimb_round3_opus/`)
- Haiku re-judge of round 2 (script, labels, summary): `temp_xc/local_data/backtracking_haiku_rejudge_round2/` (gitignored)
