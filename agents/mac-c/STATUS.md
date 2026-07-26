# mac-c — STATUS

**Agent:** mac-c (macOS, `~/research/projects/agents/mac-c/temp_xc`; read-only archaeology lane, $0 compute)
**Briefings:** `briefings/actmix-shared.md` → `briefings/actmix-mac-c.md` (ACTMIX W3)
**Last update:** 2026-07-26 ~21:45 London (first deliverable push)

## Supersession note

This workstream **supersedes mac-local's ~20:40 subagent dispatch** (recalled cleanly at Han's direction before anything was frozen/launched/pushed — see be755651a's commit message). mac-c is a standalone CC instance; no state from the recalled agents was reused.

## What's done (this push)

- **`experiments/explorations/task_hunt/COMPOSITION_AUDIT.md`** — the W3 deliverable. Pinned per-arm activation compositions for: sparse probing (§3, incl. the T-SAE ReLU→BatchTopK/threshold vs everything-else TopK→ReLU split), backtracking (§5, headline = `aniket-ward-stage-b:a62175ee7`, ported via ATTRIBUTION.md; A4 flags final's contradictory "NOT paper data" branding), RLHF (§6, produced on `han-phase7-agent-c`, byte-identity chain to the paper figs; TXC arm = `agentic_txc_02`), synthetic (§7, REVISED: runs on `origin/final` purified, line-identical ports), EM part 1 (§4, dmitry-em-repl frozen at `de0c8ea8f`, external `fra_proj` code, no TXC arms there).
- **HF inventory** (§8): 8 repos on han1823123123 (5 model + 3 dataset; briefing's "3 datasets" undercounted). Jackpots: `txcdr-it` = the 12 IT probing ckpts; `temp-bench-models` = 1 283 purified paper-cell ckpts (c6 finance cells confirmed by config sample); `temp-bench-data:runs/` = 430 purified eval artifacts. Token used read-only, never printed; NOTE `temp_xc_a40_checkpoints` (private, created 07-25) is CURRENT stage-2 ops, not paper-era.
- **Forgotten-branch sweep** (§9): all 43 remotes classified; top recoveries: `em-nanda` (parallel EM steering, 479/480 blobs absent), `aniket-ward-stage-b`, `dmitry-backtracking`, `dmitry-rlhf` (raw separation_scaling), `han-phase6`, `andre-steering`, `300k-tfa` (48 orphaned final-night c1/c2 rows).
- LOG.md PTR entry appended.

## Second push (same evening) — A4 + A6 forensics landed

- **A4 RESOLVED:** shipped c7 numbers = purified locked-arch bs-sweep (TXC-base bs1024 lift +0.541/+83%; TXC-pro PR-AUC 0.242; auto-macros committed 05-06) — NOT Aniket's +1.574. neurips-aniket's harness matches the shipped convention exactly.
- **A6 NARROWED + integrity flag:** two generations of the c6 7bmed figs. Committed camera-ready = 2 bars (sae_arditi 16.39 vs txc_base **19.20 — TXC wins steering**), exactly reproducible from in-git artifacts; published arXiv = 5 arms (T-SAE +25.9 winner, matches the caption), **producing runs in NO branch**. The camera-ready caption contradicts its own committed figure. Needs mac-local/team ruling: which generation is runpod-2's paper-match target. Disambiguators listed in audit §10-A6.

## Third push — part-2 ruling executed (config census)

- Per mac-local's ruling (637cc656d): enumerated ALL 1 283 `temp-bench-models` config.jsons. **A1 CLOSED** — the shipped c3 campaign found (six-arch IT panel, 20K steps × seeds 1/2/42, k_pos=20 defaults, saved 05-04→05-06; + TXC-base T∈{10,20} cells + BASE panel). **A2 closed for c3** (L0=20 real; RLHF stays the k500 exception). **A6 strengthened**: 7B-med datasource holds ONLY the 4 Generation-1 cells — the 5-arm run's ckpts were never uploaded. Shipped c7 locked cells also found (llama L10 nousmirror, seed-42). Census committed: `experiments/explorations/task_hunt/tbm_census.jsonl` (train_key index for pods).
- Git identity set to `mac-c-agent` per mac-local housekeeping (first two pushes went out under the default identity).

## Post-midnight wakes (2026-07-27)

- **T5-artifact escalation from runpod-1 CONFIRMED in git** (audit §3 ⚠ + A12, third integrity flag; pushed 6c4262df9): shipped c3 "T10/T20" cells are T=5 replicas (fix-commit postmortem `1ed4fde5f` + bit-identical seed-pair leaderboard rows); appendix T-sweep claim unfulfilled; Dmitry's d(perf)/dT gate must not cite the shipped c3 T-sweep.
- **Scope re-cut by Han's EM FULL STOP** (LOG ~00:15): A3 + A6 residue → Dmitry (annotated in audit §10, no further mac-c cycles). Remaining mac-c scope = A5 + probing/RLHF-relevant + "part 3".
- **A5 CLOSED both venues** (this push): TFA = ReLU→TopK train+eval (dev `tfa_big` kval=500 via `train_phase7.py:345`/`run_probing_phase7.py:216`; shipped c3 via purified `tfa.py`).
- **"part 3" is UNDEFINED in the record** (appears once in af2247d43, no definition anywhere in LOG/briefings) — awaiting mac-local's definition; flagged in my LOG entry.

## Overnight queue (briefings/actmix-overnight.md §3) — status ~01:15

1. **Phase-B recipe** ✓ (audit §3, pushed d6e992db9) — turnkey shipped-eval pin for runpod-1.
2. **HF mirrors** — PARTIAL: λ̂ tsae top-up trio mirrored to `temp_xc_a40_checkpoints/hunt_lambda_tsae_topup_checkpoints/` (manifest w/ sha256 + training_cfg; source `temp-xc-ward-caches:checkpoints_topup/`). **Open question for mac-a/mac-local: dialogue-panel + salvage/calib cells appear to persist result-payload JSONs only (btk_rerun*/`*_results` volumes), no on-volume checkpoints found — confirm whether ckpts exist elsewhere or the payload JSONs are the mirror target.**
3. **A2 paragraph** ✓ (audit §10-A2, quotable for the paper team).
4. **EM archaeology (scope-corrected back in)** ✓ evidence-level: A3 order RESOLVED (Nura SAE = ReLU-first dictionary_learning family; state-dict keys via ranged read; residual = wrapper path, one file in fra_proj); A6 public-artifact search EXHAUSTED (runs/ full inventory + census + snapshot check — Gen-2 is definitively pod-local/private).

## Remaining (low priority, on request)

- A9 threshold-buffer read on txcdr-base/txcdr-it ckpt headers (KB-scale) — probing/RLHF-relevant (tsae arms), so in-scope under the priority test if wanted.
- A12 disambiguator (c3 headline-fig render inputs on origin/final) — probing-relevant, cheap.
- §6 backlog items per overnight briefing if the above closes.

## Git position

Branch `arxiv`, up to date with origin at dispatch (`fd4cc10f9`); this push adds COMPOSITION_AUDIT.md + this STATUS + LOG PTR only (read-only rule otherwise). Roster row already existed (be755651a) — not duplicated.

## Listening

Per shared briefing: watching `experiments/explorations/task_hunt/LOG.md` + `briefings/actmix-*` for scope amendments (poll ~150 s when active). Pods watch my COMPOSITION_AUDIT.md path — pushes to it are the unblock signal for their paper-match arms.

## If resuming from compact

Read COMPOSITION_AUDIT.md §0 + §10 first (verdicts + open ambiguities), then this file. The five subagent forensic reports (EM ×2, backtracking, RLHF+synthetic, branch sweep) live in the session transcript; their substance is fully merged into the audit. Tokens: `~/.tokens/hf_token_datasets` (Han's account) — never print/commit; rotates post-weekend.
