---
status: active
created: 2026-07-28 00:50 London (re-opened; v1 delivered+closed by mac-c)
for: mac-c
---

# RE-OPENED: ELICITATION HARNESS BUILD — you are the owner (LOG authorization entry of this push)

Build the shared elicitation harness (your TIERC_PIPELINE_DESIGNS.md
§ 3) and produce the first two corpora: (1) sycgen_age completion,
(2) your pick of the safety-strongest Tier-C entry, bars-first.
Backend RE-AMENDED (00:1x 07-28): **the ANTHROPIC key is the ACTIVE
API backend for now** — keychain `anthropic-api-key`, verified live;
env-inject `ANTHROPIC_API_KEY` per actmix-shared governance; model
guidance: claude-haiku-4-5 for bulk rollouts, claude-sonnet-5 where
the behavior needs frontier realism (justify per-card). The OpenAI
key stays staged (currently 401) — keychain
`dmitry-mats-openai-key` per the governance block in
actmix-shared.md (env-inject only, mac-only, never on pods),
**$300 generation cap (Han)**, ledger every batch (OPENAI section
in MODAL_SPEND.md). You pick models per behavior (mini-class for
bulk, frontier-class where the phenomenon needs it — justify in
the card). Pod-hosted open-weights stays available as
fallback/comparison arm. GPU-pod budget ≤ $100 slice unchanged. Cards frozen before generation. Per-token baseline
binding; geometry can kill, not clear. Every kill-lesson from
tonight is a DESIGN INPUT: choose event spacing for the clock,
control vocabulary across event/non-event spans, no sentence-scale
kernels. Delete this briefing when the harness + first corpus cards
are frozen and generation is running (report state in LOG as usual).
