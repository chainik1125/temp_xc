# Task-hunt label artifacts (built by runpod-b, `briefings/task-hunt-prep.md`)

CPU-side label engineering for the hunt pods (`runpod-d`: λ̂ + proof-op;
`runpod-e`: repetition-lag + confidence). Every artifact is built by the
committed script next to it (scripts committed BEFORE their outputs); all
numbers in the stats JSONs are script-derived. Pure label logic lives in
`lib.py` and is covered by `tests/test_task_hunt_labels.py`.

## The alignment contract

- **Fineweb artifacts** (`replag_fineweb_<tok>.npz`) carry the exact
  `token_ids` (+ `doc_off` prefix offsets). **Feed these ids — do not
  re-tokenize.** Positions index the no-special-tokens sequence; if your
  cache prepends BOS or windows the doc, map positions accordingly on
  your side.
- **Ward-grid artifacts** (`ward_lambda.npz`, `proofops.npz`,
  `confidence.npz`) are in canonical Ward cache coordinates
  (4044 × 128, window w × position p) — aligned by the same round-trip
  identity check as `conversion_depth/build_ward_stream.py` (`valid`
  masks the verified positions). They drop onto any cache built from the
  canonical stream.
- `-1` (int) / `NaN` (float) = undefined. Never train on them.
- **Split by document/trace** (`doc_split` / `trace_split`, 20% test,
  seed 0) — never split one doc's rows across train/test.
- Manifests (`man*_doc/pos/cls`) are class-balanced probe rows
  (subsampled to the smallest class, seed 0), restricted to `pos ≥ 32`
  so a trailing window of any screened T ≤ 32 fits. For the Ward grids,
  `man*_doc` is the WINDOW index; group by `trace_idx[man_doc]`.

## Artifacts

| artifact | builder | target (screen) | control baked in |
|---|---|---|---|
| `replag_fineweb_{gpt2,gemma2,llama31}.npz` | `build_replag.py` | Δ-bucket (1–4/5–8/9–16/none>64; guard 17–64 excluded) for 1- and 2-grams | `delta*_shuf` = within-doc-shuffle Δ (frequency-only null); `logfreq` |
| `ward_lambda.npz` | `build_ward_lambda.py` | λ̂ regression (`lam`, causal mirror intensity) + terciles (`lam_bin`); `lam_hist` = position-term-free | `is_bt` (current-event detection ≠ intensity recovery) |
| `proofops.npz` | `build_proofops.py` | `time_in_run` bins + `is_run_start` (boundary) | `op` 5-class (per-sentence readable by construction) |
| `confidence.npz` | `build_confidence.py` | `slope8` terciles (hedging→commitment trend) | `hedge` state 3-class (lexically stamped) |

Clock bridge (substrate-audit item 6): tokens-per-sentence stats +
sentences-per-window at each screen T are in `proofops_stats.json`
(`clock_bridge`) — choose T ranges from these numbers.

Mini-cards live one dir per candidate (`../replag/CARD.md` — frozen by
runpod-e with its own inline labels; `../proofops/`, `../confidence/` —
frozen by runpod-b; `../forbidden_word/`, `../emotional_instability/` —
DRAFTs, the running agent freezes its own).

## Methodology

`PREMEASURE_METHODOLOGY_NOTE.md` (runpod-a, hub-licensed `7a7ee52c8`) —
**read before designing a premeasure for an arrangement/position face.**
What label-side bands certify and what they cannot: `struqpos` passed all
5 bands on all 3 legs and was still killed 3/3 by a confound computable
from the tokenizer + embedding table alone. Proposes two new
premeasure-grade tiers (token-side bag probe; readout-side adjacency
floor) and the design rules for paired-arrangement corpora. PTR.
*(Index line added by runpod-a for discoverability — the surrounding
document is runpod-b's and is otherwise untouched.)*
