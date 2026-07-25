# Stage-2 card — oprate `rate_case`: the CASE STUDY #2 panel

**Status: FROZEN at commit (commit-then-run; NO panel cell executed
when this card is committed — git order is the evidence).** Agent:
runpod-d. Briefing: `briefings/stage2-oprate.md` + its addendum + the
A40 addendum + `briefings/a40-bootstrap.md` (force-majeure restart,
interim 6×A40 pod, ~12 funded hours from session start, EPHEMERAL
storage — push after every completed batch). Stage-1 screen: `CARD.md`
(B2) → KEEP × 2, reviewed & approved. Runner: `run_stage2.py` (this
directory). Datasource plugin: `src/explorations/task_hunt/
real_oprate.py` over the COMMITTED `../labels/oprate.npz`;
`configs/data.yaml` entries `ward_real_oprate_{case,ver}_base_l12`.

## 0. Force-majeure provenance (box facts changed, science unchanged)

All old volumes are lost; the caches this panel reads were REBUILT this
session from the committed builders: Stage-A `traces.json` re-ported
per `results/c7_backtracking/stage_a/ATTRIBUTION.md`; the Ward stream
rebuilt by `conversion_depth/build_ward_stream.py` and verified
**byte-identical** to the committed receipt (`ward_stream_stats.json`
unchanged under git: 4044×128, map_ok_rate 0.99971, 149 round-trip
mismatch tokens, 2805 keyword events); the base 17-point cache rebuilt
by `cache_depth.py` on this pod (A40, GPU 0). The label bundle, its
stats, and its builder never left git. A40 timings are ~2–3× H100 per
GPU cell; the schedule below reflects that.

## 1. Scope — ONE target to a full panel

**`rate_case` is the primary** (position-blind at 0.496 in the screen
bundle — the cleanest triage in the factory). `rate_ver` runs ONLY if
the case panel completes its full acceptance gate with real headroom on
the funding clock (> 2 h remaining per the bootstrap). A complete
single-target panel beats two partials.

**Anchor (single, scarce, disclosed):** `base/hs13` (resid_post L12),
`d_sae = 2048 = d_in/2`, per-token code rate 8. Why this face: hs13 is
the screen protocol's PRIMARY layer; it is the cleanest `g_agg ≈ g`
face for `case` at T32 (+0.063 agg vs +0.067 flat, g_order +0.003); and
it is the λ̂ panel's exact anchor, so the two Ward case studies share a
reader cache and operating point. `distill/hs13` is within 0.004 on g
and is NOT run — disclosed narrowing, one labeled slice, same
convention as the λ̂ panel.

## 2. Design (the λ̂ Stage-2 pattern; nothing invented)

5 archs × T ∈ {2, 4, 8, 16} (token archs at T = 1) × seeds {1, 2, 42},
trained (n_steps 8000) + untrained control per (arch, T) per seed;
`eval_window_L = 32`; batch throughput-normalised (`grid.batch_size`:
1024 // T); canonical runner only, via `grid.run_pool`.

**Row decomposition (pre-stated): 84 cells = 42 trained + 42
untrained; per kind: batchtopk_sae 3, tsae 3, stacked_batchtopk 12,
txc_batchtopk_pre 12, txc_batchtopk_post 12.** Results land in
`results/leaderboard.jsonl` (canonical; receipts recompute from it) and
per-pool JSONs `results/stage2_ward_real_oprate_case_base_l12*.json`.

**Bindings, all paid-for lessons:**

1. **TXC-post at per-T nominal k = 8·T from the first cell**
   (16/32/64/128 at T = 2/4/8/16) — the postmatched correction
   (`../lambda_intensity/card_stage2_postmatched.md`): post spends its
   BatchTopK budget per window, so nominal 8 would be a sparsity ramp,
   not a panel. Code-rate convention ("under the code-readout
   convention", with the code-rate defense) as adopted there. Every
   other arch at nominal k_pos = 8. Dict-feasible everywhere
   (2048 ≥ 128).
2. **Budget-match is scored on REALIZED `l0_per_token`, never nominal
   k.** Pre-registered: untrained post cells realize **exactly 8.00
   (± 0.02) at every T** — if not, the k = 8·T mechanism is wrong and
   the post arm is VOID (reported as failed, not reinterpreted;
   postmatched § 6). Predicted trained band, ALL archs: **[5.0, 8.25]**
   per token, rising toward nominal with T (λ̂ precedent: pre
   5.81–7.84, matched post 5.7–8.12). **Any trained cell outside the
   band is recorded as a residual mismatch in the LOG verdict — never
   called in-band** (the correction I was caught on once).
3. **Paired probe columns on every row:** `eval_extra` =
   `PROBE_V2_SPEC.md` § 2 verbatim (`lambda_probe_v2: true`, ridge,
   pinned alpha grid logspace(−2, 4, 13), n_windows 8192, split:
   trace via the bundle's own `trace_idx`). **Claim on v1** (the
   2026-07-25 taken methods decision: v1 canonical through the
   deadline; levels are conservative for dense codes — receipted
   limitation; ordering is robust). v2 is reported beside v1, never
   quoted as canonical.
4. **No max-over-arms scoring.** Probe class and control width fixed
   above; nothing selected after seeing cells.
5. **buffer_tokens = 524,288, uniform across every arch** (the
   fresh-panel unlock, value frozen here). Rationale, measured not
   assumed: this equals the λ̂ panel value and ≈ the corpus (4044 × 128
   = 517,632 tokens), so the token/window buffers hold the whole corpus
   in their single fill (the buffers sample with replacement and never
   re-drain, so first-fill coverage IS training coverage — shrinking
   below ~4044 sequences would silently train window archs on a corpus
   subset). Shrinking it would ALSO not touch the tsae long pole at
   all: tsae is `consumes='sequence'` (v1 port) → `SequenceBuffer`
   ignores capacity entirely and clones a full `(1024, 128, 4096)`
   fp32 batch (≈ 2.1 GB) EVERY step — the multi-hour, GPU-idle cost my
   top-up measured. The mitigation is scheduling, not the buffer:
6. **tsae trained cells are FIRST in the cell list** and run as their
   own 3-worker pool on a dedicated GPU, launched before everything
   else (A40 addendum item 1). If the tsae arm is still running at
   review time the panel is reportable as PARTIAL with tsae pending; a
   panel that never scheduled it would not be reportable at all. The
   runner's `sel` argument (only-tsae / skip-tsae) is scheduling only —
   cell content is byte-identical either way, and each pool writes its
   own results JSON so concurrent pools cannot clobber each other.
7. **Leading-edge drops reported per T:** label coverage is ≈ 0.90 of
   valid positions (NaN where any kernel-lag sentence is unlabeled or
   the current sentence is itself the event class), so the non-finite
   guard in `lambda_recovery` is LIVE on this datasource for the first
   time. The verdict reports how many sampled windows drop, per T.

## 3. The evidence line (Stage-2 vocabulary — addendum item 3)

Screen-side visible-evidence AUCs do not transplant to Stage 2. They
are quoted for context only: `case` label-side count AUC = 0.572 /
0.648 / **0.783** at T8/T16/T32; screen per-token AUC at this face
0.738, flat 0.804, mean 0.801 at T32. **The binding Stage-2 comparator
is the REGRESSION analog**, computed label-side at panel time: OLS from
the in-window count of case-class tokens (`op == 2` over labeled
positions; the bundle's class map) to the `rate_case` target, on the
SAME sampled windows and the same probe convention as the panel eval,
its held-out r reported **beside every window cell, per T**. A window
cell that does not beat the analog at matched T is counting visible
event sentences, and no latent-state language may be used for it.

## 4. Pre-registered predictions (scoreable, each falsifiable)

- **P1 (T-pattern — the λ̂ shape, regime-2 aggregation):** v1
  `lambda_recovery` for TXC-pre and Stacked rises monotonically
  T2 → T8. The T16 v1 point may sag — pre-registered as the RECEIPTED
  probe-capacity limitation, NOT representation decline; the paired v2
  column is the discriminator (v2 restoring T16 monotonicity ⇒
  probe-side sag; v2 sagging too ⇒ real, and reported as real).
- **P2 (the money margin):** at T ≥ 8, TXC-pre (v1) > BOTH per-token
  decoded baselines (batchtopk_sae, tsae) at matched realized budget,
  and the margin does not shrink under v2.
- **P3 (matched post):** post tracks the pre/stacked band across the
  ladder (screen fact `g_agg ≈ g`: a LINEAR pool carries the gain). No
  starved-budget artifact should appear (budget is matched from cell
  one).
- **P4 (evidence line):** TXC-pre beats the § 3 regression analog at
  T = 8 and T = 16. This is also KILL clause (iii).
- **P5 (controls):** every untrained cell sits at chance (|r| ≤ 0.05)
  and post untrained realizes exactly 8.00/token (binding 2).

**Explicit NEGATIVE statement (we want a sound verdict, never a win):**
flat-or-falling v1 AND v2 across the whole ladder for every window
arch, or window ≤ per-token baselines, or window ≤ the evidence-line
analog, is a sound, publishable outcome: it would say this
independent target's trailing rate is linearised into the current
token and window codes add nothing — a real constraint on the TXC
story, reported with the same prominence a win would get.

## 5. KEEP / KILL (the acceptance gate is the briefing's)

**KEEP (CASE STUDY #2)** requires ALL of: (i) P2 holds with variance
receipts (per-seed cells, 95% CIs, exact within-seed trend permutation
p over the T ladder, paired pre − tsae / pre − batchtopk margins with
sign-flip p; **state plainly what is bounded at n = 3 and what is
not** — the λ̂ panel's pre-vs-tsae margin is formally unbounded and
says so); (ii) P4 holds (beats the evidence line at matched T);
(iii) panel complete-or-partial-with-tsae-pending, 0 failures, 0 dup
eval_keys, 0 null metrics, row decomposition as stated; (iv) binding-2
bookkeeping discharged (every realized-l0 mismatch listed).

**KILL / NEGATIVE** if any § 4 NEGATIVE branch holds — written to the
LOG with the scorecard (which prediction held, which was falsified),
the same way a KEEP would be.

Variance receipts run via the probe-agnostic harness
(`../support_stats/stage2_variance.py`, CLI params, k_pos = 8·T row
shape for post per runpod-b's pre-flight). Figure = the money plot
(recovery vs T, one line per arch, v1 solid / v2 beside, realized-l0
annotation on every post point, evidence-line analog drawn per T).

## 6. The 12-hour queue (A40 addendum item 4)

1. This card committed → 2. tsae pool (GPU 1) + main pool (GPU 0)
launched, push per completed batch → 3. variance receipts →
4. LOG verdict + scorecard + figure + RECORD section + leaderboard
hygiene + STATUS rewrite → 5. `rate_ver` ONLY if 1–4 are done with
> 2 h on the funding clock. Stopping early at any gate is fine;
anything not pushed does not exist.
