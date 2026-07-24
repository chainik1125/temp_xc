---
status: active
created: 2026-07-24
for: runpod-d
venue: runpod (GPU, new pod)
---

# The task hunt — a real-world task where TXC beats T-SAE and scales with T

**You are `runpod-d`** — a NEW GPU pod (`/workspace/.agent_id` =
`runpod-d`, seeded by the user; see `agents/README.md` + your
`agents/runpod-d/STATUS.md`) with your **own independent 700 GB
volume** (no access to runpod-c's caches — different datacenter).
**First task: rebuild the Ward stream + the reader caches you need**
from the committed builders (`conversion_depth/build_ward_stream.py`,
`cache_depth.py` — ~1 h on the H100; single mid-depth layer suffices
for screens, per model). The frozen probe stack (`conversion_depth/`
scripts) is in the repo — reuse, do not re-derive. Keep everything you
build on the persistent volume so a repurposed pod inherits it.

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
**Model axis (screen-only):** non-ambience is a (task, MODEL) property
— a model whose attention converts the structure leaves no window
advantage (the depth-ablation lesson). Screen each candidate on every
model that is cheap to obtain (for your trace candidates, cache BOTH
readers — Llama-3.1-8B base + R1-Distill — one mid-depth layer each
during your Ward rebuild; the marginal cost of the second model is
minutes); Stage 2 runs ONLY the single best (task, model) cell (at
most two).
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

## Your candidate queue (the hunt is SPLIT: you own the Ward-cache-dependent candidates; `runpod-e` owns repetition-lag + confidence-trend via `task-hunt-b.md`. runpod-b preps labels — pull its specs as they land)

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
2. **Proof-operation run structure (grounded).** Time-in-current-phase
   / run-rate over the R1-Distill traces (labels from the expansion
   corpus via runpod-b; three-timescale structure is C5-confirmed
   model-independently). Screen after candidate 1's screen is in.
3. **Forbidden-word violation onset (NEW, Han 2026-07-24 — CoT-Control).**
   Latent: imminent-violation pressure — the model is instructed to
   suppress a keyword in its CoT and usually fails; label positions by
   offset to the FIRST occurrence of the keyword (**exact string match —
   programmatic ground truth, zero judge cost**). Data: generate
   R1-Distill-Llama-8B rollouts on the public CoT-Control
   keyword-suppression split (`docs/papers/reasoning_models_struggle.md`;
   github.com/YuehHanChen/CoTControl — keywords pre-validated per
   question; R1-family controllability ≈ 0.1 % ⇒ violations are the
   common case, labels are dense). Screen = the onset-anticipation gate
   at frozen horizons D ∈ {4, 8, 16} before first occurrence: per-token
   vs window vs shuffled + g_agg/g_order per the main protocol. Ambience
   is genuinely CONTESTED here and that is the card's crux: the model
   "circles" the concept pre-violation (semantic-neighborhood tokens are
   individually informative), so per-token ≈ window at ALL horizons is
   the pre-registered kill; surviving it with g_order > 0 is the
   non-ambient anticipation story. Frozen T-prediction: recovery at
   horizon D turns on as T crosses D (a second threshold ladder, like
   arm B's per-Δ plot), saturating at the typical circling span.
   **SILOED (Han 2026-07-24):** run this candidate entirely
   independently — do NOT condition the design, priors, screens, or
   verdict on Aniket's parallel forbidden-word work, and do not wait
   on or consume his results; treat the two efforts as separate silos.
   Queue discipline: generation may run in the background while
   candidate 1 trains; screen only after candidates 1–2 have committed
   verdicts.

Do NOT pursue bracket/indentation state-tracking (tried before this
program; dies under strict per-position baselines). Forbidden-word is
in scope per candidate 3 — run it siloed from Aniket's parallel work
(his results are not an input to ours).

## Parked — reviewed and REJECTED for this window (recorded so nobody re-derives)

- **Hint-use faithfulness** ("Reasoning Models Don't Always Say What
  They Think", `docs/papers/reasoning_models_dont_always_say.md`):
  "hint used" is a rollout-level boolean (the EM trap); the moment of
  use has no observable ground truth (the paper argues it is a single
  forward pass); and hint-*presence* is trivially ambient — the hint
  tokens sit in the prompt, so a window spanning them reads the label
  off the input. Post-rebuttal at most.
- **Introspection / injected thoughts** (Anthropic 2025): detecting an
  injection we ourselves added to the residual stream is a
  perturbation-detection confound, not a temporal-structure result;
  the genuine noticing phenomenon is capability-gated beyond
  panel-feasible open models. Post-rebuttal at most.

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
