"""Real-activation punctint-q datasource for the task hunt's Stage-2 panel.

Presents the replag fineweb activation caches
(`/workspace/replag_caches/<model>/hs<k>.npy`, one forward sweep per
model over the pinned 400-doc fineweb sample —
`task_hunt/replag/cache_acts.py`) plus the frozen question-rate
intensity labels (`task_hunt/labels/build_punctint.py`, face ``lam_q``)
as a :class:`~temp_bench.data.synthetic.SyntheticData`, so the
canonical runner + the existing :mod:`temp_bench.evals.lambda_recovery`
block can panel a REAL fineweb task with no ``temp_bench/core/`` edits.
Reached through the ``module:fn`` generator path (`configs/data.yaml`).
The `real_lambda.py` (Ward λ̂) pattern, applied to a second corpus.

**What is and is not ground truth here.** ``lam_q`` is exact — a
deterministic, frozen function of the corpus' sentence stream
(exponential kernel over the PREVIOUS 8 sentences, half-life 2; the
current sentence never contributes to its own label). So
``lambda_recovery`` (held-out Pearson r of a linear probe on the tile
code) is a sound recovery metric and **is the only headline**.

**Label masking (mirrors the frozen screen manifests):** positions
carry ``NaN`` — and are dropped by the probe's non-finite guard — where

- the kernel does not fully fit (sentence index < 8, per-doc warm-up);
- the token belongs to a QUESTION sentence (``is_q``): the builder's
  ambient-anchor rule — event-sentence tokens read the event ambiently
  and are excluded from the very face they anchor;
- the position is a BOS prefix slot (gemma/llama; gpt2 has none).

No additional position floor is applied (the screen's ``pos ≥ 32``
manifest floor is a screen-manifest convention; position is not a
strong route for q — measured position AUC 0.47–0.53 ≈ chance,
`punctint_stats.json`). Tokens with ``in_span == 0`` (inter-sentence
whitespace) keep their nearest-sentence label, as in the builder.

**Alignment is asserted, not assumed.** The labels live on the flat
no-special-tokens token stream; the cache rows are non-overlapping
content chunks of that stream (`replag/build_labels.py`: chunk c of doc
d covers flat positions [c·content, (c+1)·content), document tails
dropped). At load, every cache row's token ids are re-derived from the
label npz' flat ``token_ids`` and asserted equal before any label is
attached — the dialevel `verify_mapping` discipline.

``trace_ids`` = the per-row document index (`tokens.npz` ``doc_idx``,
asserted non-decreasing), read ONLY by the v2 λ probe's trace split
(`PROBE_V2_SPEC.md` § 1 knob 3) so no document's rows straddle the
probe halves — which matters more here than anywhere: the screen
measured ``doc_mean_only_auc`` = 0.901 on this face.

There are **no ground-truth feature directions** in a real residual
stream; ``emission_features`` carries the same **reference basis, not
ground truth** as the Ward datasource (DC direction + top principal
directions of a fixed subsample), so ``eauc`` stays finite but answers
only "does the dictionary span the stream's dominant variance
directions?" — never feature recovery. ``support`` remains None.

Normalization: fp16 → fp32, one global RMS constant over a 64-row
sample — the Ward convention. No per-position or per-feature whitening.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/replag_caches")
REPO_ROOT = Path(__file__).resolve().parents[3]
LABEL_DIR = (REPO_ROOT / "experiments/explorations/task_hunt/labels")

# cache key → label-npz tokenizer key (build_punctint.TOKENIZERS)
_TOK_KEY = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}
KERNEL_SUPPORT_S = 8      # sentences; label NaN while sent_idx < support


def _row_label_grid(key: str, face: str = "q"):
    """(N, 128) float32 label grid + (N,) doc ids, alignment-asserted.

    Re-derives every cache row from the flat label stream and asserts
    byte equality of the token ids before attaching any label.
    """
    tok = np.load(CACHE_ROOT / key / "tokens.npz")
    ids, doc_idx = tok["ids"], tok["doc_idx"].astype(np.int64)
    n_prefix = int(tok["n_prefix"])
    N, seq_len = ids.shape
    content = seq_len - n_prefix
    assert np.all(np.diff(doc_idx) >= 0), f"{key}: doc_idx not contiguous"

    z = np.load(LABEL_DIR / f"punctint_fineweb_{_TOK_KEY[key]}.npz")
    flat_ids = z["token_ids"]
    doc_off = z["doc_off"]
    lam = z[f"lam_{face}"].astype(np.float32)
    evt = z[f"is_{face}"].astype(bool)
    sent = z["sent_idx"]

    lam = lam.copy()
    lam[evt] = np.nan                      # ambient-anchor masking
    lam[sent < KERNEL_SUPPORT_S] = np.nan  # kernel warm-up (belt & braces:
    # sentence_lambda already emits NaN there; asserted cheap, kept explicit)

    grid = np.full((N, seq_len), np.nan, dtype=np.float32)
    chunk_of = np.zeros(N, dtype=np.int64)
    seen: dict[int, int] = {}
    for i in range(N):
        d = int(doc_idx[i])
        c = seen.get(d, 0)
        seen[d] = c + 1
        chunk_of[i] = c
    starts = doc_off[doc_idx] + chunk_of * content
    for i in range(N):
        s = int(starts[i])
        want = flat_ids[s: s + content]
        got = ids[i, n_prefix:]
        assert np.array_equal(want, got), (
            f"{key} row {i} (doc {doc_idx[i]} chunk {chunk_of[i]}): cache "
            "tokens diverge from the label stream — label reuse would break")
        grid[i, n_prefix:] = lam[s: s + content]
    return grid, doc_idx


def fineweb_punctint_q(
    *,
    model_key: str = "gemma2_2b",
    hs: int = 14,
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    d_in: int | None = None,
    n_ref: int = 8,
    seed: int = 0,
) -> SyntheticData:
    """Real fineweb activations + frozen lam_q labels as a SyntheticData.

    ``model_key`` selects the replag cache (gpt2 | gemma2_2b |
    llama31_8b), ``hs`` the hidden-state capture point (the screen
    layers: gpt2 7, gemma2_2b 14, llama31_8b 14). ``d_in`` is declared
    in the datasource params because the trainer infers the input width
    from the spec BEFORE materializing; it is checked against the cache
    here rather than trusted. ``seed`` follows the Ward precedent: the
    runner materialises per cell seed, so the RMS constant and the
    reference basis carry per-seed sampling jitter (part of seed
    variance, disclosed).
    """
    acts_path = CACHE_ROOT / model_key / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — rebuild via "
            "experiments.explorations.task_hunt.replag.cache_acts")
    arr = np.load(acts_path, mmap_mode="r")
    grid, doc_idx = _row_label_grid(model_key)
    if arr.shape[:2] != grid.shape:
        raise ValueError(f"cache {arr.shape[:2]} vs labels {grid.shape}")
    N = arr.shape[0] if n_seqs is None else min(int(n_seqs), arr.shape[0])
    if arr.shape[1] != seq_len:
        raise ValueError(f"seq_len {seq_len} != cache {arr.shape[1]}")
    if d_in is not None and int(d_in) != arr.shape[-1]:
        raise ValueError(
            f"datasource declares d_in={d_in} but cache is {arr.shape[-1]} — "
            "the trainer sizes the dictionary from the declared value")

    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).float()
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(rms_sample, N), replace=False)
    rms = float(x[idx].pow(2).mean().sqrt().clamp(min=1e-6))
    x = x / rms

    # Reference basis (NOT ground truth — see the module docstring).
    flat = x[idx].reshape(-1, x.shape[-1])
    dc = flat.mean(0)
    dc = dc / dc.norm().clamp(min=1e-8)
    centred = flat - flat.mean(0, keepdim=True)
    step = max(1, centred.shape[0] // 8192)
    _, _, V = torch.pca_lowrank(centred[::step], q=max(1, n_ref - 1))
    ref = torch.cat([dc.unsqueeze(0), V.T[: n_ref - 1]], dim=0)
    ref = ref / ref.norm(dim=1, keepdim=True).clamp(min=1e-8)

    return SyntheticData(
        x=x,
        emission_features=ref.contiguous().float(),
        hidden_features=None,
        support=None,
        hidden_support=None,
        seq_len=int(x.shape[1]),
        d_in=int(x.shape[-1]),
        extra={
            "lambda_labels": torch.from_numpy(grid[:N]),
            "trace_ids": doc_idx[:N].copy(),
            "real_activations": True,
            "model_key": model_key,
            "hs": hs,
            "label": "lam_q",
            "corpus": "fineweb 400-doc pinned sample (replag caches)",
            "rms_scale": rms,
            "no_ground_truth_directions": True,
            "emission_features_are_reference_basis_not_ground_truth": True,
            "n_ref": int(ref.shape[0]),
        },
    )
