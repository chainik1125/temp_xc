---
status: active
created: 2026-07-26 ~20:30 London
for: SHARED — mac-a, mac-b, mac-c (Modal/local), runpod-1, runpod-2 (H100 pods), mac-local (orchestrator)
---

# ACTMIX — activation-mixing recovery (post-meeting phase, Han's allocation)

**The finding.** Three ReLU/TopK compositions coexist:

| where | composition | failure mode | harm vs T |
|---|---|---|---|
| `txc_base` (PAPER TXC; probing/em/backtracking/rlhf runners' default) | TopK then ReLU on selected values; k_win = 8·T | selected-negative slots zeroed AFTER selection | grows with T (selection digs deeper) ⇒ paper d(perf)/dT biased DOWN |
| `topk_sae` (paper per-token baseline) | TopK then ReLU | same family, but T = 1 ⇒ shallow selection | ≈ unharmed — so paper comparison is composition-consistent but harm-INconsistent |
| v2 hunt backbone (`txc_batchtopk_{pre,post}`, `batchtopk_sae`, `tsae`, `stacked_batchtopk`) | ReLU then BatchTopK | zero-picks when positive pool is thin ⇒ realized l0 < nominal | worst at SMALL pools: sae@T1 realizes 4.4/8; pre/stacked 5.9→7.9 and post 5.6→8.0 per window as T grows |
| `sae_arditi` | ReLU then TopK (`use_relu=True`, "matches dictionary_learning") | third variant, present in tree | — |

**Pre-registered directional expectations (stated BEFORE any fix
run; verdicts must quote these):** under `btk-only` the per-token
sae baseline improves MOST ⇒ hunt TXC-vs-sae margins likely
shrink; tsae margins move least (6.7/8 realized — already our
licensed lead comparator); hunt T-slopes may soften (low-T cells
recover); the PAPER arch's T-curves should improve (that is
Dmitry's re-run gate: does d(perf)/dT improve).

**Arm labels (mandatory in every card, ledger line, leaderboard
row, figure legend):**
- `relu-mix` — the current v2 backbone behavior (ReLU→BatchTopK).
- `btk-only` — BatchTopK with NO ReLU anywhere in the sparsity
  path (the theoretically-clean target).
- `paper-match` — the composition a given PAPER task actually used,
  pinned per task by mac-c's COMPOSITION_AUDIT. Never assumed.

**Implementation rule.** relu_mode enters as plugin-compliant
variants with an arch_version bump (single file drop + YAML entry
per the hard rules) — never an in-place behavior change to a frozen
arch. Historical rows must stay reproducible against their stamped
code_version. Unit tests: no-ReLU path correctness + realized-l0
sanity. **Single-source rule: mac-a's Stage-1 implementation is
CANONICAL for the btk-only convention** (registry names, hparam,
negative-selection handling, threshold path) — mac-a posts the
convention note early (LOG or briefings/); every other agent
(runpod-1/2 especially) FOLLOWS it and never forks an independent
convention. Cross-venue comparability depends on this.

**Work split.** mac-a → `actmix-mac-a.md` (relu_mode impl +
calibration + KEEP-set survives-the-fix). mac-b →
`actmix-mac-b.md` (forensics + salvage shortlist). mac-c →
`actmix-mac-c.md` (branch archaeology + HF inventory, ASAP).
runpod-1 → `actmix-runpod-1.md` (sparse probing shuffle+T-sweep).
runpod-2 → `actmix-runpod-2.md` (EM shuffle+T-sweep).
Backtracking is Aniket's — hands off everywhere; their ablation
harness lives on `origin/neurips-aniket`
(`purified/experiments/backtracking_window_sweep/` + a
`shuffles.py` utility) and is the cross-task convention reference
for shuffle semantics and the rebuttal table format — read, never
modify, flag divergences.

**Tokens (paths only — values NEVER in git, logs, or cards):**
`~/.tokens/hf_token` (Modal secret `hf-token`; gated-model pulls)
and `~/.tokens/hf_token_datasets` (Han's account han1823123123 —
dataset/checkpoint archaeology; verified working 20:21 London;
3 paper-era datasets visible: txcdr-data, txcdr-base-data,
temp-bench-data). Modal tokens per existing setup. ALL tokens
rotate after the weekend.

**Budget.** $150/day/person cap (team-meeting decision). Modal for
mac-a/b/c (ledger `briefings/MODAL_SPEND.md`, read-before/
append-after, actuals corrections). Pods are MATS runpod spend —
still ledgered (new section header `RUNPOD` in the same file).
Program Modal actuals ≈ $96 of the original $500 word.

**Deadlines.** Rebuttal-grade numbers ideally before the 9am PT
(17:00 London) team meeting 2026-07-27; the AoE deadline is
~36–40 h out. Draft-response support beats completeness.

**Discipline (all standing rules carry):** commit-then-run freezes
with rev-parse pins from ORIGIN's history + `_assert_pinned`;
detach-at-launch; containers never push; repatriate-merge-locally
with dup-key checks; paired v1+v2 probe columns where the probe
spec applies; numeric realized-l0 bands IN cards (now doubly
load-bearing — they are the mixing fingerprint); evidence lines
pre-measured as KILL clauses; no max-over-arms; receipts direct-add
+ mac-local ratification; ALL verdicts PENDING TEAM REVIEW;
liveness rule: a quiet clone is UNKNOWN, not dead — check before
re-dispatching an agent id; pull-rebase before every push.
