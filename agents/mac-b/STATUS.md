# Working state — agent `mac-b`

**2026-07-26 ~16:50 London (SALVAGE sprint, `salvage-mac-b.md` — txcwin
novelty cross-ratification).** Salvage cap $60; spent ≈ $4 est (GAP-B
Modal line). Report state by ~22:00 London; rebuttal deadline
tomorrow (07-27).

## Salvage W2 state (IN FLIGHT — resume here)

- **Audit DONE** (read-only): their audit.py + my independent recompute
  agree. gpt2@T8 c1/c2/c3 REPRODUCED (15/21.9/11.3σ, strict); c4
  reproduces exactly. 8B: c1/c2 2.6/2.7σ; **c3@T8 NOT-REPRODUCED**
  (their own W3/W8 fail; post s1 collapsed +0.198), T=16 strict 12.4σ;
  claims/report T-mismatch → amendment proposed (Andrii's call).
- **Freeze `fedf75aa9`**: `txcwin/crossratify/MINI_CARD.md` +
  `visible_cue.py` (GAP-A) + `rawgate_fill.py` (GAP-B). Driver
  `scripts/modal_txcwin_rawgate.py` pinned to it. NOTHING under
  `txcwin/` outside `crossratify/` + `CROSSRATIFY.md` may be modified.
- **GAP-A DONE ($0, local CPU), results committed**
  (`crossratify/results/visible_cue_{gpt2,llama31}.json`): T=8
  surface-quiet CONFIRMED both models (V-rep +0.058/+0.060, V-uni
  +0.044/+0.084 ≪ per-token 0.215/0.129 ≪ post 0.463/0.393). Riders:
  T=16 repetition floor ~+0.21; V-pos≈0 prediction FAILED (nov_resid
  position residual r≈+0.21/+0.17 — instrument disclosure).
- **GAP-B IN FLIGHT**: Modal app `ap-drsJemgQC9kq7iyNnVvE8A`
  (mac-b-txcwin-rawgate, detached, L40S ×2 sequential: gpt2_L6 T8,
  then 8b_L12 T4/8/16 with in-container 8B cache build → Volume
  `temp-xc-replag-caches` at /workspace/txcwin_caches +
  /workspace/txcwin_crossratify_results). Local launch shells
  backgrounded; if repatriation didn't land, pull payloads:
  `uvx modal volume get temp-xc-replag-caches txcwin_crossratify_results/<f> .`
  → copy into `txcwin/crossratify/results/`. Gate criterion verbatim
  theirs: CANDIDATE iff max(gap_window, gap_mean) > 0.03. Pre-stated
  outcomes in card § 2 GAP-B.
- **Memo `txcwin/CROSSRATIFY.md` DRAFTED + committed** — verdict table
  final for gpt2; two [PENDING] slots await GAP-B numbers (§ 2 G-1 and
  R-X4). After numbers land: fill slots, set final 8B gate sentence in
  § 4, receipts R-X1..X4 → receipts_check.py ONLY after mac-local
  ratification + Andrii ack (both pending).
- Ledger line appended (est ~$4, program ~$101); append actuals
  correction when the app completes.

## Standing state

- Day-2 record (all ratified, sprint closed): W1 ladder MIXED 3/3
  (R25), panel-2 support R27, fig3 + negatives table adopted. Day-2
  actuals ≈ $2.
- `uvx modal …` (plain `modal` NOT on PATH). `source
  scripts/set_agent_env.sh mac-b` each shell (user instruction).
- hf-token + Modal token rotate after the weekend (Han). Containers
  never push. Andrii is a HUMAN collaborator pushing to this branch —
  rebase + reconcile, never modify their txcwin files.
- Post-deadline queue (mac-local gate): gemma overnight-card fills from
  Volume partials (~$4).
