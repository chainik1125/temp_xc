# HUNT3_SCREEN_CARD — third-generation candidates: conversation novelty (`cnov`) + novelty-rate trend (`nvtrend`)

**Pre-registration. Frozen BEFORE any screen cell; pin in
`scripts/modal_hunt3_screen.py` from `git rev-parse`, asserted
in-container.** Overnight allocation `briefings/actmix-overnight.md`
§ 1 (Han); read-first actmix-shared.md. Owner mac-a; mac-local
freeze-reviews in parallel. ALL verdicts PENDING TEAM REVIEW.

## § 1 The slate (4 designed → 2 screened; 2 killed label-side, $0)

Template (briefing § 1, the measured hill-climb gradient):
offset-weighted trailing functionals of SPARSE per-token-SILENT
events, no surface marker at any T, on the order-carried substrate.

| candidate | construction | disposition |
|---|---|---|
| **cnov** | kernel trailing rate (support 64 tok, HL 16) of FIRST-IN-CONVERSATION token types; full-support rows only (pos ≥ 64, stated deviation) | **SCREEN** — the txcwin out-of-window definition on DailyDialog: pre-window occurrences are invisible to any T-window by construction |
| **nvtrend** | ttrend's exact kernel-WLS machinery (5 turns, HL 2) on PER-TURN novelty rates | **SCREEN** — the Δ-face; regime-3-shaped; independent of ttrend (ρ ≤ 0.09) |
| tempo | ttrend kernel-WLS slope of 1/turn_size (the briefing's "turn-taking rate trend") | **KILLED label-side, $0**: Spearman vs the confirmed ttrend face = −0.832/−0.834/−0.806 (gpt2/gemma2/llama31) — strict-alternation corpus makes tempo (anti-)ttrend in a hat; screening it would near-duplicate a confirmed KEEP. The briefing's improve/replace clause exercised: nvtrend is the replacement instance of the marker-free cross-distance template |
| qres | turns since most recent OPEN question, resolved at first ?-free turn (the briefing's q→a latency) | **KILLED label-side, $0** per the briefing's own gate: P(latency = 1) = 0.84 on all three tokenizers (nearly no variance to probe) AND the anchor turn carries a visible "?" token — dq's demotion marker one step removed |

Correction-hazard on Ward (briefing seed 4) is NOT in this card —
queued behind these screens if the night allows; its cue-free-window
design note is in the LOG entry accompanying this freeze.

## § 2 Design

Substrate/caches/layers: dialevel's verbatim (committed stream
`labels/dialevel_dailydialog_<tok>.npz`, caches rebuilt in-container
by the committed builder). Models: gpt2 + llama31_8b + gemma2_2b
(majority-of-3 verdict, diafaces convention). Probe grid = the
convention-of-record clone (diafaces/screen.py): tok linear+MLP
first (per-token-first triage); position floor; visible floor per T;
actxmean ± foreign T ∈ {4,8,16,32,64}; order arms linear T ∈ {16,32}
+ MLP triple T32; permutation nulls at T16; within-dialogue arms
BINDING. Manifests: position-matched stratified balanced, 3-class,
CAP 4000/1500, MIN_ROWS 300, pos ≥ 64.

Reach: turn ≈ 14.5–15.7 tok; cnov kernel support 64 tok ≈ 4 turns;
nvtrend 5-turn support ≈ 75 tok — both inside the T ≤ 64 ladder.

## § 3 Label-side pre-measures (measured BEFORE this freeze; builder
`labels/build_hunt3.py`, artifact `labels/hunt3_stats.json`)

Triage AUCs (test rows, pos ≥ 64):

| face | model | unigram | position | doc-mean | wd docs |
|---|---|---|---|---|---|
| cnov | gpt2 | 0.584 | 0.143 (inverted ⇒ leak 0.857) | 0.856 | 902 |
| cnov | llama31 | 0.560 | 0.115 (⇒ 0.885) | 0.865 | 836 |
| nvtrend | gpt2 | 0.537 | 0.638 | 0.780 | 821 |
| nvtrend | llama31 | 0.535 | 0.625 | 0.795 | 766 |

Both known traps are HOT for cnov (position mechanically anti-drives
novelty; conversations differ in vocabulary breadth) — the
position-matched manifest and the BINDING wd arms are the
instruments; a cnov result that does not survive them is dead on
those clauses, and we say so now.

**Visible-floor evidence lines (the per-T KILL instruments;
first-in-WINDOW novelty rate for cnov / its token-kernel slope for
nvtrend; AUC vs face terciles, test rows):**

| T | cnov floor (gpt2 / llama31) | nvtrend floor (gpt2 / llama31) |
|---|---|---|
| 4 | 0.513 / 0.515 | 0.499 / 0.499 |
| 8 | 0.572 / 0.579 | 0.491 / 0.491 |
| 16 | 0.661 / 0.663 | 0.473 / 0.479 |
| 32 | 0.738 / 0.742 | 0.487 / 0.502 |
| 64 | 0.881 / 0.885 | 0.595 / 0.622 |

Pre-registered readings: **cnov's claimable zone is T ≤ 32** (at T64
the window sees the whole kernel and the floor eats the face —
0.88); any cnov quote must beat 0.66–0.74 at T ∈ {16,32}. nvtrend is
floor-FREE through T32 (floor ≈ chance, ttrend's winning profile) —
its screen question is purely whether activations carry it. The
in-screen floor arm for nvtrend gets BOTH floor features (slope +
rate) — strictly stronger than the line above; strengthening the
kill instrument is the conservative direction.

## § 4 KEEP / KILL (frozen; majority of the 3 screened models — the
diafaces § 7 rules verbatim)

**KEEP** iff: some matched-class window arm beats tok by ≥ +0.05
with width null cleared ≥ +0.02, AND beats the visible-evidence
floor at the same T, AND the within-dialogue arm shows a
same-direction window gain on supported rows. **KILL** if ANY of:
(1) tok within 0.02 of every window arm at every T; (2) every window
gain fails its width null; (3) every window gain fails the visible
floor; (4) the within-dialogue arm erases the gain. Else **WEAK — no
rule fires as written**, numbers only. Order sensitivity
(win − win_shuf ≥ +0.03 at T ∈ {16,32} on wd arms where the gain is
positive) KEEPs/KILLs nothing by itself; it decides panel gate vs
breadth table.

## § 5 Venue, economics, discipline

Modal **L40S**, one model per container, 3 in parallel (diafaces
precedent), `--detach`, retries 1; Volume `temp-xc-replag-caches`;
dialevel caches (re)built in-container by the committed builder
(idempotent, likely cache-hit from day-2). Containers never push;
per-model result JSONs persist to Volume after every cell +
repatriate locally. Est ≈ $8–12 of the ≤ $30 § 1 envelope ($60 cap).
Ledger read-before/append-after. Deliverable:
`results/screen_{gpt2,llama31_8b,gemma2_2b}.json` + ONE bundle
verdict in the LOG + DRAFT panel card(s) for the morning if § 4
KEEPs anything (team picks at 17:00; drafts are NOT freezes).
