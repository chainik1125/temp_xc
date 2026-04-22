---
author: Han
date: 2026-04-22
tags:
  - design
  - in-progress
---

## Phase 6 encoding protocol for the 4 archs

Notes pinning down the per-token encoding convention for each arch, so
downstream UMAP + autointerp compare like-for-like. Complements
[[brief]] and [[plan]].

### Why this matters

Each arch has a different native input shape, so "per-token
activation pattern on the same concat sequence" is not uniquely defined
without a choice. This doc records the choice so reviewers can verify
we compared apples-to-apples.

### Per-arch conventions

**`agentic_txc_02`** — window SAE, `T=5`, input shape `(B, T, d_in)`.
For each position $t$ in the concat we construct a window
$[t-2, t-1, t, t+1, t+2]$ of L13 activations and edge-replicate-pad at
the boundaries (position 0 sees `[0,0,0,1,2]`, etc.). Encode returns a
`(1, d_sae)` vector, which we assign to token $t$. Rationale: the
T=5 TXC was trained with each window's latent jointly representing
the central token plus its context — the central token is a natural
assignee of that window's z.

**`agentic_mlc_08`** — multi-layer crosscoder, input shape
`(B, L, d_in)` with `L=5` (residual layers L11-L15). For each position
$t$ we stack `[resid_L11[t], ..., resid_L15[t]]`, encode, and emit
one `(1, d_sae)` vector. Gemma is forward-passed once per concat with
hooks on all 5 layers; no resid is reused across positions (unlike
TXC's window overlap).

**`tfa_big`** *(optional this phase)* — temporal-attention SAE with a
novel/pred decomposition, input shape `(B, T, d_in)` where `T` is
the training seq_len (128). We chunk the concat into non-overlapping
128-token blocks, pad the final chunk by edge-replication, forward
through the SAE, take `results_dict["novel_codes"]` — the sparse
TopK-selected component (the "pred" component is dense and not
comparable to the other archs' TopK latents). Per-token z drops the
right-pad. Rationale follows the Phase 5.7 TFA probing convention
(see `2026-04-22-handover-batchtopk-tsweep.md`).

**`tsae_ours`** — paper-literal T-SAE port, input shape `(B, d_in)`.
Encode is token-at-a-time on the L13 residual. The InfoNCE temporal
signal is used only at training; inference is single-token, matching
the other archs' per-position output convention.

### Shared pre-processing

All four archs consume the *same* Gemma-2-2b-IT residual-stream
activations at layer 13 (and L11-L15 for MLC). The concat-set token
IDs (from `build_concat_corpora.py`) are forwarded through Gemma with
`transformers` PyTorch hooks on the decoder layer output — this is
the same hook path used by the Phase 5 `cache_activations.py`.

The per-arch z tensors are saved as `fp16` .npy under
`experiments/phase6_qualitative_latents/z_cache/<concat>/<arch>__z.npy`.

### High-level / low-level prefix splits for UMAP

For the UMAP "high" vs "low" panels (see [[plan]] Figure 1):

- `agentic_txc_02`: scale-1 prefix = first `d_sae // T = 3686` latents
  (high) and scales 2..T combined = remaining 14 746 latents (low).
- `agentic_mlc_08`: `d_sae // 2 = 9216` high prefix + 9216 low suffix.
- `tsae_ours`: same 9216 + 9216 split.
- `tfa_big`: no native high/low split. We rank features by variance on
  concat-set C and take the top 50% as "high", bottom 50% as "low"
  — a pragmatic analogue of the paper's variance-ranked baseline.

### Boundary tokens

For `agentic_txc_02` on short concat-C sequences (20 tokens), the
first two and last two positions of each sequence see partial windows
(edge-replicated). We include them in the UMAP so each arch contributes
the same number of points (n_seq × 20), but flag this in the writeup.
