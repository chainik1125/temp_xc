# Cohen's κ + raw agreement: Aniket (blind) vs Sonnet 4.6 judge

n = 20 transcripts

| field | raw agreement | Cohen's κ | n_disagree | per-row breakdown |
|---|---|---|---|---|
| coherence | 0.85 | 0.749 | 3/20 | id0(3/2), id3(1/0), id7(1/2) |
| backtracking_present | 0.95 | 0.773 | 1/20 | id11(0/1) |
| looping_present | 1.00 | 1.000 | 0/20 | (perfect) |

## Targets
- raw agreement ≥ 0.80 ✅ acceptable
- Cohen's κ ≥ 0.6 ✅ substantial
- Cohen's κ < 0.4 ❌ refine judge prompt or document limitation

## Per-transcript scores

| id | arch | mag | coh (A/J) | bt (A/J) | loop (A/J) | aniket notes |
|---|---|---|---|---|---|---|
| 0 | MLC | 5.0 | 3/2  ⚠️ | 0/0 | 0/0 | clean |
| 1 | SAE | -6.0 | 3/3 | 0/0 | 0/0 | let me double-check but verification only, no error caught -> bt=0 |
| 2 | MLC | -12.0 | 3/3 | 0/0 | 0/0 | clean |
| 3 | TXC | -16.0 | 1/0  ⚠️ | 0/0 | 1/1 | severe pseudo-bt, 'Wait, perhaps I should use the fact that 1 is a root' repeate |
| 4 | MLC | 12.0 | 1/1 | 1/1 | 1/1 | edge case - genuine constraint detection (duplicates not allowed -> switch to 4, |
| 5 | TXC | 12.0 | 0/0 | 0/0 | 1/1 | total LaTeX collapse |
| 6 | MLC | -8.0 | 3/3 | 0/0 | 0/0 | clean |
| 7 | MLC | 16.0 | 1/2  ⚠️ | 0/0 | 1/1 | wrong answer (should be 7, says 6), then loops the wrong conclusion |
| 8 | SAE | -4.0 | 3/3 | 0/0 | 0/0 | clean |
| 9 | TFA | -5.0 | 3/3 | 0/0 | 0/0 | clean |
| 10 | TFA | 8.0 | 2/2 | 1/1 | 0/0 | clearest bt=1: volume can't be negative, must have made a mistake -> switches to |
| 11 | TSAE-paper | 6.0 | 2/2 | 0/1  ⚠️ | 0/0 | thinking trace gets -3pi/3pi; formal solution silently switches to 49pi-16pi=33p |
| 12 | TFA | -8.0 | 3/3 | 0/0 | 0/0 | clean |
| 13 | TFA | 16.0 | 2/2 | 0/0 | 0/0 | x^2-9x=48 => x=3 is wrong, but presentation fluent |
| 14 | TFA | 16.0 | 3/3 | 0/0 | 0/0 | clean |
| 15 | TXC-H8 | -8.0 | 3/3 | 0/0 | 0/0 | clean |
| 16 | TXC-H8 | 12.0 | 3/3 | 0/0 | 0/0 | clean |
| 17 | MLC | -4.0 | 3/3 | 0/0 | 0/0 | clean |
| 18 | TFA | -16.0 | 2/2 | 0/0 | 0/0 | wrong (subtracts boundary as 11+9, should compute 9x7=63), but coherent prose |
| 19 | TXC-H8 | -12.0 | 3/3 | 0/0 | 0/0 | clean |