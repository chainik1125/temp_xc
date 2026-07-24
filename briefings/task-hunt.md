---
status: active
created: 2026-07-24
for: runpod-c
venue: runpod (H100)
---

# The task hunt — a real-world task where TXC beats T-SAE and scales with T

**You are `runpod-c`** (H100, 700 GB volume: Ward caches, EM caches,
probe stack). **This briefing SUPERSEDES `em-redo.md`** (team decision,
2026-07-24): EM stays a negative case in the paper; PAUSE em-redo
wherever it is — append a dated PAUSED note to
`conversion_depth/TRACKING.md` (the frozen prereg stays valid for a
future session; do NOT delete artifacts; any Phase-A results already
produced get recorded as-is, unscored). Your cache builders, panel
driver, and detection-eval port are exactly the machinery this hunt
reuses.

**The goal (the one sentence):** find a real-activation task where
**TXC recovery/detection improves systematically with window size T
while T-SAE and the per-token SAE stay flat** — plus a within-window
shuffle ablation showing order/structure matters. Prime directive
still rules: a candidate that dies at screening is a success of the
process. **Fail fast — selection over depth.** Check-in deadline:
results wanted by 2026-07-26 morning PT.

## The two-stage protocol (per candidate)

**Stage 1 — the screen (raw activations, NO dictionary training;
~2–4 h/candidate).** Freeze the candidate's mini-card first (one
committed page: label definition, why the latent should be
regime-2/3-shaped i.e. non-ambient, predicted T-pattern per STORY.md
§ 7, falsifier). Then: cache a small activation slice (one mid-depth
layer per the g(ℓ) precedent — L13-equivalent), and run the frozen
probe stack: per-token linear vs window linear at
T ∈ {2, 4, 8, 16, 32}, plus the window-MEAN decomposition
(g_agg/g_order) and the within-window shuffle control.
**KEEP iff** the window−token gap grows with T over some range AND
the pattern matches the card. **KILL otherwise — one paragraph in the
hunt log (`experiments/explorations/task_hunt/LOG.md`), move on.**

**Stage 2 — the head-to-head (survivors only).** Panel through the
canonical runner on real_lm datasources (append-only data.yaml):
per-token BatchTopK SAE, T-SAE, Stacked, TXC-pre, TXC-post ×
T ∈ {2, 4, 8, 16} × seeds {1, 2, 42} + untrained, matched realized
l0_per_token. Frozen per-arch predictions before training. Deliverable
per survivor: **the T-scaling figure** (recovery vs T, one line per
arch — the money plot is TXC rising while T-SAE stays flat), the
shuffle-ablation bar, and the record.

## The candidate queue (mac-local priors; runpod-b is prepping labels — pull its specs as they land)

1. **Backtracking intensity λ̂ (grounded regime 2 — top prior).**
   Latent: the local event intensity (the real parent of the synthetic
   backtracking bench's λ) over the Ward stream — labels ON YOUR VOLUME
   (`probe_rows.npz` events + `results/c7_backtracking/stage_a/`).
   Frozen prior: every window arch ≫ T-SAE ≈ per-token; T-scaling
   pattern A/B saturating near the Hawkes kernel support (the fitted
   kernel: `backtracking/results/` mirror params; note the Ward
   ~18-position intervention distance as the natural scale — window
   spanning it is the story). Distinct from Aniket's detection/
   inducement readout — this is intensity RECOVERY; flag any overlap
   in the record.
2. **Repetition-lag Δ (exact labels, threshold T-scaling — top prior).**
   Latent: distance Δ to the previous occurrence of the current
   n-gram in natural text (fineweb slice; labels computed exactly from
   tokens — zero labeling cost). Provably non-ambient (no single token
   knows Δ); recovery of lag-Δ structure needs T > Δ ⇒ **built-in
   threshold scaling: sweep Δ ∈ {4, 8, 16} and show each Δ turns on as
   T crosses it.** Subject model: a BASE model (gemma-2-2b base or the
   Llama-3.1-8B base you have cached machinery for).
3. **Proof-operation run structure (grounded backup).** Time-in-
   current-phase / run-rate over the R1-Distill traces (labels from
   the expansion corpus; three-timescale structure is C5-confirmed
   model-independently). Screen only if 1–2 leave GPU headroom.
4. **Confidence-trend (grounded backup, clock-mismatch risk).**
   Windowed hedging→commitment slope; sentence-clock vs token-clock
   bridging per the substrate-audit item 6 — screen last.

Do NOT pursue bracket/indentation state-tracking (tried before this
program; dies under strict per-position baselines) or forbidden-word
(owned elsewhere).

## Also wanted (cheap, after the first survivor)

The **shuffle ablation for the existing backtracking case study**: on
your Ward caches, per-token vs window vs SHUFFLED-window raw ceilings
at L10, T = 16 — the order-sensitivity receipt for the existing paper
task (the g_order machinery run once more, ~30 min).

## Acceptance gate — stop for review

Hunt log with every screen verdict; ≥ 1 survivor taken through Stage 2
with the T-scaling figure (or the honest "all candidates died" log);
em-redo PAUSED note; STATUS rewritten; leaderboard hygiene (canonical
runner, 0 dup keys). No reviewer/meeting quotes in tracked files.
Briefing stays until mac-local review.
