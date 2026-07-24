---
status: active
created: 2026-07-23
for: runpod-c
venue: runpod (H100)
---

# EM redo — TXC-tracking at the map-chosen layers + the onset pilot

**STATUS AMENDMENT (2026-07-24, mac-local):** this briefing was briefly
superseded, then REINSTATED — run it to completion. The task hunt runs
on the NEW pods (`briefings/task-hunt.md` / `task-hunt-b.md` are NOT
yours). Context shift to carry into the record: per the team decision,
EM stays in the paper as a negative case — Phase A's outcome is scored
exactly as frozen, but its *use* differs by branch: a temporal-arch win
in the paper currency is a reportable new-experiment result; a loss is
the third weak-realization datum for the archival version. Phase B
remains the stretch. Volumes are per-pod and independent (the hunt pods
cannot reach yours — different datacenters); your volume stays entirely
yours.

**You are `runpod-c`** (H100, the 700 GB volume with the preserved EM +
Ward caches). Governing context: `experiments/explorations/conversion_depth/RECORD.md`
(§ 4 + § 7 review) and `docs/ideas/em_onset_anticipation.md`. Parallel
agents: `runpod`, `runpod-b`, `runpod-d`, `runpod-e`; shared-branch
rules + commit-citation rule in `agents/README.md`. Prime directive:
**a sound verdict, never a win** — frozen predictions before every
result; an honest negative is a deliverable here (it becomes the
scoped-claims story). **Deadline: results by 2026-07-26 morning PT.
Phase A is the must-have; Phase B is the stretch. Do A completely
rather than both halfway.**

## Phase A (must-have) — does a trained temporal code claim the measured EM headroom?

The depth ablation found raw window-over-token access for the
misalignment label: g ≈ +0.12/+0.13 at resid_post L9–L13, +0.097 at the
paper's L15, with a position-sensitive slice g_order +0.108 at L13
(`RECORD.md` § 4). The § 5.3 paper result (TXC underperforms T-SAE on
Wang-style steering + sparse-probe PR-AUC at L15) is the comparison to
beat or explain. Test: **train the dictionary panel at three layers and
see whether any temporal architecture's advantage tracks the g(ℓ) map.**

1. **Freeze predictions first** (commit before any training):
   per-layer, per-arch directions. Mac-local's priors, sharpen but do
   not redirect: (i) per-token SAE detection is layer-flat relative to
   the map; (ii) temporal-arch advantage over per-token, if any, is
   LARGEST at L13 and smaller at L15 (tracks g, esp. g_order); (iii) if
   no temporal arch beats per-token anywhere despite the measured
   headroom, that is the third weak-realization datum — state it as a
   pre-registered branch, not a failure.
2. **Training caches**: the paper's dictionary-training stream
   (`medical_em_prompts` = cfierro/personality-qs-bad-medical-advice,
   6k × 128, ORGANISM/LoRA-merged forward — match the paper's § 5.3
   convention, verify against `origin/final:purified/configs/datasources.yaml`)
   cached at resid_post **L9, L13, L15** (hs 10/14/16). GPU-cheap.
3. **Panel**: per-token BatchTopK SAE, T-SAE, TXC-pre, TXC-post
   (+ spectral if budget allows), matched realized l0_per_token per the
   Part II convention, canonical d_sae/k slice + 3 seeds — through the
   canonical runner with new real_lm datasource entries (append-only);
   leaderboard rows ARE wanted here (this is a real-side head-to-head).
4. **Two readout currencies, both mandatory**:
   - probe currency: linear probe on codes (window-pooled vs per-token
     per convention) for the rollout label on the stage-4 cohort cache
     (GroupKFold by prompt, as `phase4_em_depth.py`);
   - **paper currency: the Wang-stage sparse-probe PR-AUC** — port the
     exact procedure from
     `origin/final:purified/experiments/c6_em_detection/run.py`
     (protocol 3.0.0) so numbers are comparable to the paper's Fig
     (S=16). A TXC win that doesn't appear in the paper's own metric is
     not a rebuttal result.
5. **Blind verdict vs the frozen predictions**, per layer × arch ×
   currency, with the g(ℓ) overlay figure (trained advantage vs raw
   headroom). Record: `experiments/explorations/conversion_depth/TRACKING.md`
   (new; RECORD.md § 6.6 carries the predictions context).

## Phase B (stretch) — the onset pilot

Per `docs/ideas/em_onset_anticipation.md`, steps 1–4 ONLY (no
head-to-head): extract the α=0 native subset of the stage-4 cohort
(report its size + misaligned count honestly — if < ~40 misaligned
traces, STOP and report; fresh generation is a follow-up, not this
session); onset labeler (Sonnet, prereg + κ on 30 traces; ≤ $30 API);
full-span re-cache of the subset; the **ambience gate**: g(ℓ) of the
onset label at frozen D+ offsets with the g_agg/g_order decomposition.
Deliverable: the gate verdict (discriminating or not) — that alone is a
rebuttal-relevant sentence either way.

## Acceptance gate — stop for review

Phase-A verdicts + figures + records pushed (leaderboard rows via
canonical runner; 0 dup keys); Phase-B gate verdict if reached; STATUS
rewritten; spend logged. Do NOT quote or cite reviewer text in any
tracked file. Briefing stays until mac-local review.
