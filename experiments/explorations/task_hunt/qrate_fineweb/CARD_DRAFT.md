# DRAFT mini-card — question-rate intensity (fineweb)

**Status: DRAFT (runpod, `briefings/candidate-factory-broad.md`,
ledger `../CANDIDATES.md` B4 — BUILD-with-gate: the ledger flagged
this face as between-doc-heavy, so the frozen bars below carry the
ship/kill decision).** Committed BEFORE `../labels/build_punctint.py`
runs. Disjointness note: runpod-b's `qrate` bundle is question-rate
on the WARD grid (traces briefing item 2); this is the fineweb
cousin — different corpus, different builder, no shared numbers.

Data side: same builder/npz/stats as `../list_density/CARD_DRAFT.md`
(the B4 face is `lam_q` / `q_bin` / `is_q` / `man_q_*` in
`../labels/punctint_fineweb_<tok>.npz`). Same zero-caching economics.

## The candidate logic

Events = "?"-terminated sentences (exact). Primary `lam_q` =
8-sentence-lag, half-life-2 kernel rate over PREVIOUS sentences;
tokens inherit sentence λ̂; **masking rule: question-sentence tokens
excluded from this face's manifests** (`is_q` = the ambient anchor).
Axis a: interrogative mode is weakly generative (wh-words, inversion)
— ambient face expected converted; the trailing rate is the bet.
Axis b, the known risk: scan rate mean 0.038 with variance
between-doc-heavy (FAQ/forum vs prose) — the tercile label may read
as document identity through topic vocabulary; that is precisely what
the unigram bar tests. Axis c: identical kernel/clock to B3. Axis d:
regime-2 rise predicted; zero_split expected (≈ 73 % of 8-sentence
windows carry no question at the mean rate).

## Label-side triage — FROZEN BARS (kill authority)

Identical to B3, frozen here for this face: test-doc rows, masked,
top vs bottom class, direction-agnostic: unigram type-mean AUC
**≥ 0.65 ⇒ KILL** (a free kill is a win — expected outcome if the
FAQ-vocabulary route dominates); position AUC **≥ 0.65 ⇒ KILL**;
0.55–0.65 ships with disclosure.

## Triage RESULT (builder-derived; appended after the frozen bars ran)

**PASS — the face ships clean.** On the shipped (position-matched)
manifest rows, test docs, direction-agnostic: unigram type-mean AUC
**0.520 / 0.533 / 0.521** (gpt2/gemma2/llama31), position AUC
**0.528 / 0.522 / 0.529** — both under the 0.55 disclosure band; the
between-doc/FAQ-vocabulary fear did not materialize once
question-sentence tokens are masked. All-eligible-row numbers agree
(unigram 0.523–0.530, position ≈ 0.52 direction-agnostic).
zero_split fired as predicted (λ̂ = 0 on 80.6 % of labeled rows;
event-sentence rate 0.036). Same npz/economics as B3.

## Predicted T-pattern + draft kill rule

Same shape as B3 (regime-2 rise, order-free, anchor face disclosed);
same three screen kill clauses. If the unigram bar kills this face
label-side, the kill is recorded in the LOG and the B3 face of the
same npz ships alone.
