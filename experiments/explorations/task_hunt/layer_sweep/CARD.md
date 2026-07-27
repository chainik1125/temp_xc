# LAYER_SWEEP CARD — depth profiles of the order-carried dialogue faces (cnov + ttrend)

Pre-registration. Frozen with the runner (`sweep.py` + `extract.py`)
and the mechanical scorer (`score_sweep.py`) in ONE commit before any
cell runs (hunt3 discipline). Directive: 059a66239 P2 sweep (a),
claimed in LOG 5f21474c3; layer-semantics flag + default posted LOG
6307ce5a3. **Screens only — no panel/claiming cells, no KEEP/KILL
verdicts; the cnov panel stays 17:00-pick-gated.**

## 1. Question (screen class)

How does decodability of the two order-carried dialogue trailing
faces — `cnov` (conversation novelty, hunt3) and `ttrend`
(trailing-turn-size slope, diafaces `tt`) — vary with DEPTH in the
two larger substrate models, relative to the label-side floors that
are constant across layers? One depth profile per (face, model);
descriptive, feeding mac-a's slate, claiming nothing.

## 2. Design

- **Substrate**: dialevel DailyDialog streams VERBATIM (committed
  `labels/dialevel_dailydialog_{tag}.npz`; never re-tokenized;
  `chunk_stream`/`verify_mapping` contracts intact). Canonical
  dialevel caches do not exist on this pod — pass 1 of `extract.py`
  builds them by calling `dialevel.cache_acts.main` unmodified (its
  root, its meta contract); pass 2 writes ONLY the sweep's extra
  layers to `/workspace/layer_sweep_caches` (own meta, same format),
  geometry asserted equal to the canonical `tokens.npz`.
- **Models**: `llama31_8b` (NousResearch/Meta-Llama-3.1-8B, 32
  layers) and `gemma2_2b` (google/gemma-2-2b, 26 layers). gpt2 is
  NOT in the directive's sweep.
- **Layers — semantics DEFAULT (LOG 6307ce5a3, overrulable to
  probe-time)**: directive lists resid_post L; cache convention hs =
  L+1. Probe set: llama31 hs{8,15,22,29} (L{7,14,21,28}), gemma2
  hs{7,14,21} (L{6,13,20}). CAPTURE is the UNION of both readings
  (llama +hs14) so a reversal costs no re-extraction; the scorer's
  `PROBE_SET` is the single amendment point if overruled.
- **Instrument**: the parent screens VERBATIM by import —
  `hunt3/screen.py` machinery for cnov, `diafaces/screen.py` for tt;
  manifests, caps, seeds, floor features, probe grid (tok
  linear+MLP; position floor; visible floor per T; actxmean ±
  foreign at T{4,8,16,32,64}; win/shuf/foreign linear at T{16,32},
  MLP triple at T32; permutation null at T16; within-dialogue arms
  BINDING per ops rule 7) all imported constants, never re-declared.
  Deviations (2, both structural): hs parameterised with layer in
  BOTH filename and cell key (parents' resume contract would
  silently clobber their committed results otherwise); acts resolved
  via `extract.acts_path`. Faces limited to cnov + tt (nvtrend, dq
  not run — not in the directive).
- **Results**: `results/screen_{key}_hs{hs}.json` (committed);
  scorer emits `results/depth_profile.{json,md}`.

## 3. Committed evidence lines (label-side ⇒ layer-independent; cited, not recomputed)

- **cnov**: visible-floor AUC-by-T per tokenizer in
  `labels/hunt3_stats.json` (`visible_floor_auc_by_T`), incl. the
  gemma2 values the hunt3 card's §3 table did not print (cnov gemma2:
  0.515/0.578/0.660/0.736/0.881 at T=4/8/16/32/64; triage unigram
  0.583, position 0.140, doc-mean 0.854). llama31 + gpt2 values as
  printed in `hunt3/HUNT3_SCREEN_CARD.md` §3.
- **ttrend — ASYMMETRY DISCLOSED**: no per-T floor AUC table in
  `labels/diafaces_stats.json` (triage only: unigram ≈0.548,
  position ≈0.453, doc_mean_only 0.761/0.764/0.768); the per-T
  evidence line is the gpt2-only Pearson artifact
  `diafaces/results/panel_evidence_line_tt.json` + each model's
  in-screen `tt/T*/visible_evidence_floor` cells (label-side, so
  layer-independent — the sweep re-emits them per hs as a
  consistency check, and they must agree across hs within noise;
  disagreement = manifest bug, halt).
- **In-screen floors are recomputed per (model, hs) file by
  construction** (they ride the shared cell loop) — this is the
  cross-layer consistency instrument, not new evidence.

## 4. Pre-registered readings (directional, stated before any cell)

R1. tok-linear acc vs depth is single-peaked per (face, model), with
    the max at or adjacent to the parents' frozen screen layer
    (hs14) — that layer was picked for a reason; a monotone-to-deep
    profile would be news.
R2. The earliest layers (llama hs8, gemma hs7) sit closest to the
    label-side floors (visible/position) — trailing-state structure
    is accumulated, not lexical.
R3. The actxmean−visible-floor gap varies with depth while the
    visible floor itself is hs-invariant (§ 3 consistency check).
R4. Within-dialogue arms track the main arms' depth profile; a layer
    where the main arm is high but wd collapses to ~0.5 is serving
    dialogue identity, not the face (ops rule 7 reading, reported
    per layer; screens issue no KEEP/KILL either way).
R5. Order sensitivity (win ordered−shuffled at T32) is largest at
    mid-depth where the faces peak, smaller at hs≈8 and at the
    deepest layer — the paper-side analogy (probing btk-only
    inverted-U) motivates but does not constrain this; descriptive.

Scored mechanically by `score_sweep.py` (staged this commit); any
missing cell hard-fails the scorer — no partial tables.

## 5. Venue, economics, discipline

- **Venue**: this pod, MY GPUs (0,1) only, at P1 slack (post-42 pass
  drains first; post-1/2 may be cut only per its own card's clause).
  GPU 2 untouched. nohup + logs under /workspace/logs/.
- **Cost est**: extraction ≈ 2 passes/model ≈ 20–30 GPU-min total on
  H100 (dialevel actuals: 6–10 min/model on L40S) + probes ≈ 7
  (model,hs) pairs × ≈2–8 min ≈ 30–60 GPU-min → ~1–1.5 GPU-h ≈
  **$3–5** at the RUNPOD $3/H100-h basis. Disk ≈ 32 GB (union
  capture + canonical caches), 1.6 TB free. Ledger line at launch +
  actuals at report; $150/day cap unaffected.
- **Discipline**: card+runner+scorer this commit; PIN =
  `git rev-parse HEAD` recorded in the LOG launch line;
  `results/*.json` committed; scorer output PTR; report by ~21:30
  London (059a66239). Tokens by path only. Containers never push.
