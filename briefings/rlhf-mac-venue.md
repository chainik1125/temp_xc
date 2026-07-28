---
status: retired
owner: mac-d
issued-by: mac-local (hub)
issued: 2026-07-28 13:5x London
supersedes: all pod-based wave-1 lane maps (4e04ae0e3 item 3, 31930ad8c)
---

# RLHF pf grid — the mac venue

Han, 2026-07-28: *"I've killed all the pods. No more runpod chaos.
local mac agents own RLHF from now on."* and *"mac-c and mac-d are both
still alive; mac-c should continue gold task hunting; mac-d handles the
RLHF cells."*

**You own the RLHF cells.** Everything below is already done for you or
is a decision you own. The hub does review + LOG, not execution.

## 1. What is already unblocked (hub did these — do not redo)

| thing | state |
|---|---|
| activation cache | **on this Mac**, `~/caches/rlhf/txcdr-base-data/` (14.16 GB fp16, from HF `han1823123123/txcdr-base-data` — the pods only ever held a hardlink) |
| keyed install | **done**, `results/data_cache/44b72320bc3a56e2` (13 G, hardlink) |
| `convert_train_cache.py` | `SRC` now reads `ACTMIX_RLHF_CACHE_SRC`; run it on any machine after fetching |
| MPS device | **`_select_device()` now returns `mps`** (`503f726f1`) — see § 5, it is a declared rule-3 exception |
| `run.py validate` | green on this tree |
| warmup fidelity | `PF_WARMUP_STEPS = 0`, already in `cells.py` (runpod-2) |
| anchor freeze | `PF_ANCHOR_FROZEN`, already in `cells.py` (mac-d) |
| T5 anchors | 3 corrected base-l12 rows in the leaderboard |

## 2. Measured on this MacBook — real buffer, not synthetic

M5 Pro, 18 cores, **48 GB unified**, torch 2.8.0. Vendored port, paper
batch, Adam + grad-clip 1.0, fresh window pairs drawn from the real
14 GB cache every step:

| T | s/step | feed | feed % of wall | peak | h per 8 000 steps |
|---|---|---|---|---|---|
| 2 | **0.611** | 0.114 | 18.7% | 5.59 GiB | 1.4 h |
| 4 | **1.314** | 0.143 | 10.9% | 12.55 GiB | 2.9 h |

**Against the H100's measured 1.49–1.55 s/step at T2, this Mac is
~2.4x faster.** The reason: on RunPod the feed was **95% of wall**
(818 MiB/s over the host→device bus); here it is **11–19%**, because
unified memory has no such bus. The workload's dominant cost simply
does not exist on this hardware.

*(A compute-only probe gave 0.495 s/step at T2 and a "3x" figure. That
was optimistic by ~20%; the table above supersedes it. Quote only these.)*

## 3. The 48 GB ceiling and what it means for scope

| T | est. peak | on a 48 GB Mac |
|---|---|---|
| 1 / 2 / 4 | 3.6 / 5.6 / 12.6 GiB | measured, fine |
| 6 | ~22 GiB | fits |
| 8 | ~34 GiB | fits **alone** (raise `PYTORCH_MPS_HIGH_WATERMARK_RATIO`) |
| 10 | ~48 GiB | **no** |
| 16 | ~69 GiB (OOMs an 80 GB H100) | **no** |

Extrapolating s/step ∝ params^0.95: T1 ≈ 0.30, T6 ≈ 2.4, T8 ≈ 3.6 s/step.
**Per seed, T{1,2,4,6,8} ≈ 18 h. Three seeds ≈ 55 h sequential.**
Against an Aug-3 horizon that fits, but only if it starts now.

**Concurrency will not save you.** Feed is 11–19%, so this is
compute-bound on one GPU; parallel cells time-slice rather than
overlap. Run **sequentially**, seed-column-first (s42 across all T
first, so a full-shape curve is renderable early — the sycgen pattern).

## 4. ⚑ Traps — read before editing `cells.py`

1. **DO NOT change the module-level `N_STEPS`.** It is shared with the
   btk arm (`_cell` defaults to it), so moving 25 000 → 8 000 re-mints
   **every btk cell's train_key** and orphans the completed btk grid.
   Add a **separate** `PF_N_STEPS = 8_000` and use it only in the
   `pf()` sweep branch. (Budget rationale: upstream's own per-seed
   `upstream_final_step` was 4 200 / 4 600 / 5 200; 8 000 is ~1.5x the
   slowest, generous in the safe direction.)
2. **Anchors must stay frozen.** `PF_ANCHOR_FROZEN` exists because a
   sweep-recipe change silently re-mints anchor keys and the runner
   would **retrain** the paper weights instead of loading them. Keep
   the hard-assert that the staged file resolves.
3. **`precision: bf16` is declarative only** — no autocast, no
   `.half()`, and `SequenceBuffer` casts to fp32. Every row we have
   says bf16 and trained fp32. Disclosure item, not yours to fix.
4. **grad_clip 1.0** is in upstream and absent from our core trainer.
   Disclosed, not patched (rule 3).

## 5. The rule-3 exception you are inheriting

`src/temp_bench/core/trainer.py::_select_device()` had no `mps` branch,
so on a Mac it fell through to **CPU** — roughly 50x slower, i.e. the
whole venue was a non-starter. Hub added two lines (`503f726f1`).
**CUDA still wins wherever CUDA exists**, so no existing pathway can
change device and no past result is affected. It is declared, isolated
to one hunk, and Han can veto it with a revert. If he does, the venue
needs a different answer before any cell runs.

## 6. What the hub needs back from you, first beat

1. **Machine topology — the one thing nobody knows.** Are you on *this*
   MacBook (M5 Pro / 48 GB) or your own? Report chip + RAM. If mac-c
   and mac-d are sessions on one machine, the fleet is **one** 48 GB
   GPU and § 3's 55 h is contended by the hunt; if you are on separate
   hardware — especially anything with more unified memory — T10
   re-enters scope and the wall time changes materially.
2. **A real end-to-end smoke** through `run_experiment` (not my
   standalone probes): one short pf cell, MPS, cache-expect check
   passing, one row written. That is the last unproven link; my probes
   exercised the arch and the buffer but not the canonical pathway.
3. Then launch **s42 × T{1,2,4,6,8}** and report the first landing.

## 7. Scope question that is Han's, not ours

T10 and T16 cannot run at 48 GB. Before proposing pods for them, note
what the upstream source says: the T-sweep archs are
`t2, t3, t6, t7, t8, t10, t15, t20` — **there is no `t16`**, so our T16
is our own interpolation, not a paper cell. And upstream's batch
schedule is commented as a **48 GB A40** accommodation while our port
needs **69.3 GiB** at T16 for params + Adam alone — so our large-T
cells may not be the paper's cells at all. **Surface this to Han as a
fidelity finding before anyone spends money to chase T16.**
