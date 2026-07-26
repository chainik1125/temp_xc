---
status: active
created: 2026-07-26 ~20:30 London
for: mac-c (NEW agent — archaeology) — ACTMIX W3: what did the PAPER actually run?
read-first: briefings/actmix-shared.md
---

# ACTMIX W3 — branch/commit archaeology + HF checkpoint hunt (ASAP)

**Workspace:** `~/research/projects/agents/mac-c/temp_xc` (created
by mac-local 20:25, origin = GitHub, branch arxiv). You are
**mac-c**; create `agents/mac-c/STATUS.md` on first commit and add
yourself to the roster table in `agents/README.md`.

**Mission.** The paper's numbers came from THREE branches, not
`final` (which is a lossy consolidation): sparse probing + RLHF +
synthetic ← `han-phase7-unification`; EM ← `dmitry-em-repl` (NOT
final's EM); backtracking ← `aniket-runpod-ward-stage-a` (some
commit mid-history). Forgotten result-bearing branches may exist
(the remote has dozens: andre*, bill*, dmitry*, aniket*, …).

**Per PAPER task, deliver:** source branch → result-producing
commit(s) (locate via committed figures/results artifacts, run
configs, W&B/log references, commit messages, file mtimes) → the
arch file AT that commit → the exact activation composition
(ReLU/TopK order, k scaling with T, per-arm: TXC, SAE, TSAE) →
whether trained checkpoints exist anywhere. **Honesty rule: where
evidence is inconclusive, verdict = AMBIGUOUS + what artifact
would disambiguate.** Han expects ambiguity; a well-organized
map of what is knowable beats a confident guess.

**Priority order:** 1. sparse probing, 2. EM (the two H100 pods
are BLOCKED on these two for their paper-match arms), 3. RLHF,
4. backtracking (informational — Aniket's lane), 5. synthetic.

**HF inventory (token at `~/.tokens/hf_token_datasets`, account
han1823123123 — NEVER print or commit the token):** three
paper-era datasets exist: `txcdr-data` (04-25),
`txcdr-base-data` (05-02), `temp-bench-data` (05-06). Enumerate
each repo's file tree (HF API / huggingface_hub, metadata-first —
do NOT bulk-download; pull individual small files only as needed
to identify contents), classify contents (training data vs
activation caches vs CHECKPOINTS vs results), and map anything
checkpoint-like to its paper task + branch/commit if determinable
(config files inside dumps often pin this). Checkpoints for
probing/EM are the jackpot (they make the pods' paper-match
shuffle arm eval-only).

**Deliverable:** `experiments/explorations/task_hunt/COMPOSITION_AUDIT.md`
— per-task table + per-dataset inventory + AMBIGUOUS list with
disambiguators + a short "what this means for paper-match arms"
section. LOG entry PTR. Read-only otherwise (no compute spend, no
files modified outside the audit doc, your STATUS, and the roster
line). Timeline: first push of partial findings within ~2 h;
probing+EM composition calls ASAP (pods wait on you).
