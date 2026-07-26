# FROZEN panel card — diafaces mini-panel: `ttrend` on gpt2/hs7 (`dial_real_ttrend_gpt2_l7`)

**Status: FROZEN at commit (commit-then-run; NO panel cell has been
executed when this card is committed — the Modal container is pinned
to a commit containing it).** Agent: **mac-a (executor)**. Gate:
**FIRED IN WRITING** by mac-local (LOG 2026-07-26 `dce8d085d` § 2,
clauses (i)–(v), face = tt with the tt-not-dq rationale recorded;
re-evaluation trigger — llama tt-KILL before freeze — did NOT fire:
llama landed tt-KEEP). This card executes those binding terms; my
earlier dq proposal (LOG, same date) is SUPERSEDED by that written
decision and noted as history. PENDING TEAM REVIEW ships on the
verdict as always. Screen provenance: `CARD.md` freeze `073611113`,
verdict LOG 2026-07-26 (tt KEEP 3/3; order carriage T32 3/3).

## 1. Question (Stage 2)

Do position-mixing dictionary architectures RECOVER the trailing
turn-length trend from the residual stream better than order-free
ones, on the substrate whose order mechanism (R25: turn layout,
MIXED within+across turns, near-concentrated) is literally this
face's state variable?

## 2. Model pin — gpt2/hs7, rationale stated (the draft's "stronger screen model" rule made concrete)

tt's over-visible-floor margins at the best cell: **gpt2 +0.122
(T16)**, llama +0.005 (T32), gemma +0.019 (T32). The only
UNAMBIGUOUSLY over-floor tt cells live on gpt2; a panel on
llama/gemma would target a state their own screens could not
distinguish from boundary-counting at the claiming T. gpt2 also
independently satisfies the pinned clause (ii) (wd_sc +0.037/T32,
+0.018/T16) and is in the gate's core set. **Flagged alternative
reading** (for mac-local's parallel freeze review): "stronger" =
largest raw gain/order cost → llama31_8b (+0.149 gain, wd_sc
+0.049/T32) — REJECTED here because its best cell sits +0.005 over
its floor; the panel's P-claims are floor-relative. Cost is not the
criterion but favors the same choice (d 768).

## 3. Grid (frozen; enumerated by `run_panel.py`, verified pre-commit)

`dial_real_ttrend_gpt2_l7` (YAML entry in this commit; plugin
`real_dialogue.py` committed `6e651c1c4`); 5 archs
(`batchtopk_sae`, `tsae` at T = 1; `txc_batchtopk_pre`,
`txc_batchtopk_post`, `stacked_batchtopk` at **T ∈ {2,4,8,16,32}**
per the pinned T32 ladder requirement); k_pos 8 (post: 8 per token ⇒
8·T per window from cell one, the λ̂ convention; pooled families
d_sae ≥ 8·32 = 256 ≤ 2048 ✓); d_sae 2048; seeds {1,2,42};
trained (8000 steps) + untrained (0); eval_window_L 32;
**buffer_tokens 524288 UNCHANGED** — corpus 4111 × 128 = **526,208
tokens ≥ 524,288: complete fill, no wrap** (measured, not assumed).
**102 cells** (= the λ̂ 84-cell shape + the T32 column), every one
through `temp_bench.core.runner.run_experiment`; leaderboard merged
locally, 0 dup keys; panel file `stage2_dial_real_ttrend_gpt2_l7.json`.

## 4. Receipts (frozen shape)

Paired **v1 + v2** λ-probe columns on every row (claim on v1, v2
reported — METHODS DECISION 2026-07-25); v2's split groups by
**conversation** (`trace_ids` = dialogue index) — the identity
receipt on the substrate whose label-side conversation-mean floor is
**r-analog of doc_mean_only 0.761**; variance receipts via
`support_stats/stage2_variance --row-layout auto --post-k-rule
times-T`; **realized-l0 band** per arm with under-band cells
disclosed (R22 lesson); **evidence line per T**: label-side Pearson
r of the tt visible floor (kernel-WLS slope over complete previous
turns within T) vs ttrend on the eval rows, computed read-only at
verdict time and drawn under every recovery curve.

## 5. Pre-registered predictions + KEEP/KILL (frozen bars)

- **P1**: best trained pooled arm (pre or stacked) beats trained
  `batchtopk_sae` on v1 lambda_recovery by **Δr ≥ +0.02** at some
  T ∈ {8,16,32}, variance-receipt CI clear of 0.
- **P2**: that margin is larger at T ∈ {16,32} than at T ∈ {2,4}
  (the screen's reach: slope material enters late).
- **P3**: tsae (T=1) lands between sae and the best pooled arm
  (the λ̂/R22 ordering).
- **P4**: untrained arms recover **≤ 0.5×** the winning trained
  arm's v1 r at its T.
- **P5**: the winning arm's **v2 (conversation-grouped) r > 0** at
  its T — else the headline is conversation identity and says so.

**KEEP** iff P1 AND P5. **KILL** if best trained pooled ≤ sae + CI
noise at every T, or P5 fails. Else **WEAK — no rule fires as
written**. No max-over-arms: the claiming arm is named by identity;
every quoted number carries its v2 pair, l0 band, and the evidence
line at its T.

## 6. Venue, economics, discipline (Han's GPU amendment + mac-local's terms)

`run_panel.py --block main` on **H100** (the GPU-bound non-tsae
pools; workers 6) + `--block tsae --only-seed N` ×3 on **high-CPU
cheap-GPU containers** (tsae measured GPU-idle; d 768 shrinks the
buffer copy ~5×, expect ≪ the 62–77 min d-4096 precedent); caches
already on the Volume (screen stage; idempotent rebuild guard in the
driver). Est **≤ $12** (envelope ≤ $60, gate clause iv ledger ≈ $41
≪ $250; mac-a day actuals ≈ $5 of $120). Commit-then-run; PIN via
rev-parse; `_assert_pinned()`; `--detach`; leaderboard delta +
panel JSON persisted to Volume in-container; containers never push;
repatriate-merge-locally with dup check; ledger
read-before/append-after. **Cells done + repatriated by 16:15
London, no exceptions** (mac-local's binding term); nothing new
after 15:30; everything pushed by 16:30. Deliverable: LOG verdict
`mac-a (executor)` PENDING TEAM REVIEW scoring P1–P5 + KEEP/KILL,
leaderboard rows, panel JSON, receipts proposal, checkpoints to
Volume (HF mirror = Han follow-up rule).

## AMENDMENT (2026-07-26, mac-a) — v2-columns defect and re-run

The first tt run (freeze `7ba2e10fd`, completed + merged) landed
**v1-only rows**: the enumeration was cloned from the λ̂ runner,
which PREDATES PROBE_V2_SPEC and carries no `eval_extra` — § 4's
paired-columns term was breached by the executor, caught at scoring
(P5 unscorable). Fix: `run_panel.py` now attaches the oprate § 2 V2
block verbatim to every cell; the panel is RE-RUN at the amendment
freeze. The first run's 102 leaderboard rows keep their (clean-pin,
disclosed-dirty-stamp) places — v2 keys hash into eval_key so the
re-run adds new rows, no collisions; the panel FILE is rebuilt from
the paired re-run (same cell identities). Nothing from the first run
is quotable. Realized-l0 note from the first run carries forward
unchanged expectations (post = k-per-window ⇒ l0 8/T per token,
byte-matching the λ̂ panel's signature).
