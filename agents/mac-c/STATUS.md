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

## Remaining (low priority, on request)

- Enumerate `temp-bench-models` 1 283 config.jsons (public, KB-scale) to close A1 (shipped c3 cell identity) and A6a (5-arm run widths) — deferred to avoid API hammering; the pods can also do this on-box.
- A9 threshold-buffer read on txcdr-base/txcdr-it ckpt headers (KB-scale) if the team wants the T-SAE degenerate-gate risk closed.

## Git position

Branch `arxiv`, up to date with origin at dispatch (`fd4cc10f9`); this push adds COMPOSITION_AUDIT.md + this STATUS + LOG PTR only (read-only rule otherwise). Roster row already existed (be755651a) — not duplicated.

## Listening

Per shared briefing: watching `experiments/explorations/task_hunt/LOG.md` + `briefings/actmix-*` for scope amendments (poll ~150 s when active). Pods watch my COMPOSITION_AUDIT.md path — pushes to it are the unblock signal for their paper-match arms.

## If resuming from compact

Read COMPOSITION_AUDIT.md §0 + §10 first (verdicts + open ambiguities), then this file. The five subagent forensic reports (EM ×2, backtracking, RLHF+synthetic, branch sweep) live in the session transcript; their substance is fully merged into the audit. Tokens: `~/.tokens/hf_token_datasets` (Han's account) — never print/commit; rotates post-weekend.
