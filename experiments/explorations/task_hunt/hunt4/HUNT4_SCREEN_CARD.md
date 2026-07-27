# HUNT4_SCREEN_CARD — fourth-generation candidates: cross-speaker adoption (`xnov`), topic-return intensity (`tret`), speaker dominance (`sdom`), adoption trend (`xtrend`), return depth (`tretd`)

**Pre-registration. Frozen BEFORE any screen cell; pin in
`scripts/modal_hunt4_screen.py` from ORIGIN-history `git rev-parse`
post-push, asserted in-container.** Gen-4 directive 59ad15f38 scaled
by c1c5c949e (Han, $200/10h hunt envelope). Owner mac-a; mac-local
reviews on push; mac-b stages adversarial replication on any KEEP.
ALL verdicts PENDING TEAM REVIEW.

## § 1 The slate (7 designed → 5 screened; 1 killed label-side $0, 1 on its own Ward lane)

Template (the measured hill-climb gradient, cnov's recipe):
offset-weighted trailing functionals of SPARSE per-token-SILENT
events, out-of-window / cross-distance structure preferred, on the
order-carried dialogue substrate. Speaker attribution = turn parity
(DailyDialog strict alternation — the tempo kill's premise).

| candidate | construction | disposition |
|---|---|---|
| **xnov** | kernel trailing rate (support 64 tok, HL 16) of ADOPTION events: type seen before in conversation but never by the current speaker (lexical entrainment; event rate 10.5–11.7%) | **SCREEN** — speaker-resolved memory: the window can see recent cross-use but "never by me" is unbounded-history |
| **tret** | kernel trailing rate of LONG-RETURN events: gap = idx − last_occ > 64 (event rate 7.7–7.8%) | **SCREEN** — out-of-window at EVERY ladder T by construction (gap > 64 ≥ T): a window sees only "novel-in-window"; tret and cnov partition that guarantee (resumed vs new) |
| **sdom** | signed dominance D = K_cur − K_oth of per-speaker kernel novelty rates (mass guard 0.15, NaNs 7.4–7.8%) | **SCREEN** — cross-distance comparison of two trailing states, sign attached to the current speaker |
| **xtrend** | ttrend's kernel-WLS machinery (5 turns, HL 2) on PER-TURN adoption rates | **SCREEN** — the Δ-face of xnov; floor-free through T32 (nvtrend's winning profile) |
| **tretd** | kernel-weighted mean log2(gap) over trailing long-return events (mass guard: trailing return rate ≥ 0.02; labeled 45–46%) | **SCREEN** — a cross-distance VALUE: the gap of an out-of-window return is uncomputable from any window |
| xret | long returns split by attribution: most recent prior use = OTHER speaker's (rate 3.4%) | **KILLED label-side, $0**: Spearman vs tret = 0.809 / 0.812 / 0.800 (gpt2/gemma2/llama31) — at/above the 0.8 anti-dup bar (tempo precedent); the attribution twist does not decorrelate the trailing rate from its parent. tret carries |
| rdens | referential-density trend on Ward (gen-4 seed 3) | **SEPARATE LANE** — Ward substrate ≠ this screen's harness; token stream fetched (llama31 tokenizer confirmed via `base/meta.json`), builds during this screen's wall-time behind its OWN chaz-style card + the window-MEAN control that killed chaz; NOT part of this freeze |

Pre-registered near-dup rule (applied above, stated for the record):
on any new-face pair |ρ| > 0.8, screen the simpler construction.

## § 2 Design

Substrate/caches/layers: dialevel's verbatim (committed stream
`labels/dialevel_dailydialog_<tok>.npz`, caches rebuilt in-container
by the committed builder). **Models: gpt2/hs7 + gemma2_2b/hs14 FIRST
(both panel substrates from the start, c1c5c949e item 2); llama31_8b
third leg launches IMMEDIATELY for every face not already 2/2 KILL
(item 4) — same driver, `--models llama31_8b`, same pin.** Probe
grid = hunt3 convention-of-record clone with the item-3 extension:
tok linear+MLP first; position floor; visible floor per
T ∈ {4,8,16,32,64}; actxmean ± foreign at those T; **win + win_shuf
linear at T ∈ {4,8,16,32} (shuffle twins on the full ladder)**;
win_foreign at T ∈ {16,32} (hunt3 parity); MLP order triple at T32;
permutation nulls at T16; within-dialogue arms BINDING with the same
extended wd order ladder {4,8,16,32} (foreign {16,32}). Manifests:
position-matched stratified balanced, 3-class, CAP 4000/1500,
MIN_ROWS 300, pos ≥ 64; wd pairs need ≥ 30 rows/doc.

Reach: turn ≈ 14.5–15.7 tok; every face's support = 64 tok ≈ 4 turns
(xtrend 5-turn ≈ 75 tok); tret/tretd events cite occurrences > 64
tokens back — beyond the whole ladder.

## § 3 Label-side pre-measures (measured BEFORE this freeze; builder
`labels/build_hunt4.py`, artifact `labels/hunt4_stats.json`; tests
`tests/test_hunt4_labels.py` 9 green)

Overlap |Spearman| vs confirmed/kept faces — ALL under the 0.8 bar
(max: tret vs cnov −0.60/−0.60/−0.64; tretd vs tret 0.64–0.66;
xtrend vs nvtrend −0.30/−0.30/−0.42; full matrix in the artifact).

Triage AUCs (test rows, pos ≥ 64; gpt2 / gemma2 — llama31 in artifact):

| face | unigram | position | doc-mean | wd docs |
|---|---|---|---|---|
| xnov | 0.538 / 0.538 | 0.435 / 0.425 | **0.820 / 0.817** | 902 / 911 |
| tret | 0.556 / 0.555 | **0.983 / 0.984** | **0.853 / 0.851** | 783 / 803 |
| sdom | 0.576 / 0.577 | 0.474 / 0.477 | 0.671 / 0.671 | 899 / 908 |
| xtrend | 0.523 / 0.519 | 0.523 / 0.523 | 0.772 / 0.769 | 824 / 838 |
| tretd | 0.546 / 0.547 | **0.978 / 0.979** | **0.891 / 0.888** | 643 / 667 |

Named-trap disclosures, instruments stated NOW: tret/tretd position
0.98 is MECHANICAL (returns require history; depth grows with it) —
the position-matched manifest + the position-floor arm + BINDING wd
arms are the instruments; a result that does not survive them is
dead on those clauses. tretd doc-mean 0.89 is the slate's hottest
identity trap — wd arms binding. xnov doc-mean 0.82 is cnov's known
vocabulary-breadth trap — same instruments.

**Visible-floor evidence lines (per-T KILL instruments; AUC vs face
terciles, test rows, gpt2 / gemma2):**

| T | xnov | tret | sdom | xtrend | tretd |
|---|---|---|---|---|---|
| 4 | 0.498 / 0.498 | 0.509 / 0.509 | 0.523 / 0.519 | 0.501 / 0.501 | 0.501 / 0.499 |
| 8 | 0.510 / 0.508 | 0.554 / 0.554 | 0.574 / 0.570 | 0.498 / 0.497 | 0.525 / 0.521 |
| 16 | 0.622 / 0.620 | 0.609 / 0.613 | 0.667 / 0.660 | 0.487 / 0.488 | 0.550 / 0.544 |
| 32 | 0.783 / 0.783 | 0.671 / 0.674 | 0.803 / 0.801 | 0.504 / 0.501 | 0.576 / 0.570 |
| 64 | 0.906 / 0.908 | 0.610 / 0.611 | 0.949 / 0.946 | 0.594 / 0.591 | 0.543 / 0.540 |

Pre-registered readings: **xnov claimable zone T ≤ 16** (0.78 floor
at T32 eats it, cnov-ruling logic); **sdom claimable zone T ≤ 16**
(0.80 at T32); **tret, tretd, xtrend are floor-free ACROSS THE
LADDER** (≤ 0.67 / ≤ 0.58 / ≈ chance) — their screen question is
purely whether activations carry them. In-screen floor arms get the
multi-feature versions (xnov: cheat-rate + wnov rate; sdom: D + both
K's; xtrend: cheat slope + rate) — strictly STRONGER than these
lines; strengthening the kill instrument is the conservative
direction. tret/tretd share the wnov-rate floor (the window's ONLY
correlate of an out-of-window return — disclosed, § 1).

## § 4 KEEP / KILL (frozen; the diafaces § 7 rules verbatim, existential form per ruling bed236f1d)

**KEEP** iff: SOME matched-class window arm beats tok by ≥ +0.05
with width null cleared ≥ +0.02 AND beats the visible-evidence floor
at ITS OWN T (all three simultaneously on that arm), AND the
within-dialogue arm shows a same-direction window gain on supported
rows. **KILL** if ANY of: (1) tok within 0.02 of every window arm at
every T; (2) every window gain fails its width null; (3) every
window gain fails the visible floor; (4) the within-dialogue arm
erases the gain. Else **WEAK — no rule fires as written**, numbers
only. Order sensitivity (wd win − win_shuf ≥ +0.03 at ANY
T ∈ {4,8,16,32} where the wd gain is positive) KEEPs/KILLs nothing
by itself; it decides panel gate vs breadth. Bundle verdict:
majority over screened models; 2/2 agreement stands, splits are
PENDING-THIRD-LEG (llama31 decides; it runs regardless unless 2/2
KILL). Scorer `hunt4/verdict.py` committed in THIS freeze — before
any deciding model result exists.

## § 5 Venue, economics, discipline

Modal **L40S**, one model per container in parallel, `--detach`,
retries 1, 4 h timeout; Volume `temp-xc-replag-caches`; dialevel
caches expected cache-hit (hunt3/panel lanes warmed them).
Containers never push; per-model result JSONs persist to Volume
`/workspace/hunt4_screen` after every cell + repatriate locally.
Est ≈ $6–9/model → first wave ~$12–18, llama31 leg ~$6–9 — inside
the $200/10h hunt envelope (c1c5c949e; ledger line per launch,
actuals corrections after). Deliverables:
`results/screen_{gpt2,gemma2_2b[,llama31_8b]}.json` + ONE bundle
verdict in the LOG (PTR) + DRAFT panel card(s) for any § 4 KEEP with
order receipts (drafts are NOT freezes; the cnov panel remains
17:00-pick-gated and untouched by this card).
