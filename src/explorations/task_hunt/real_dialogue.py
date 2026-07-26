"""Real-activation dialogue-face datasource for the day-2 W2 gated
mini-panel (`diafaces/PANEL_CARD_DRAFT.md`).

Presents the dialevel activation caches (DailyDialog forward passes,
`dialevel/cache_acts.py`) plus a committed diafaces label grid
(`labels/build_diafaces.py`: `ttrend` or `dqgap`) as a
:class:`~temp_bench.data.synthetic.SyntheticData` — the
`real_lambda`/`real_oprate` pattern verbatim, so the canonical runner
+ `lambda_recovery` panel a REAL dialogue task with no
``temp_bench/core/`` edits. Reached through the ``module:fn``
generator path; the ONE winning-face datasource entry is added to
`configs/data.yaml` in the panel FREEZE commit, never before the
shared-doc gate fires.

**What is and is not ground truth.** The face labels are exact —
deterministic frozen functions of the committed turn segmentation —
so ``lambda_recovery`` (held-out Pearson r) is a sound recovery
metric and the only headline. There are NO ground-truth feature
directions in a real residual stream: ``emission_features`` carries
the same reference basis as `real_lambda` (DC + top principal
directions of a fixed subsample; a spanning sanity check, NEVER
feature recovery). ``trace_ids`` carry the CONVERSATION index per
cache row, so the v2 probe's doc split groups by dialogue — the
substrate whose doc-identity route is 0.76/0.85 label-side.

Labels are NaN at BOS prefix slots, boundary (newline-marker) tokens,
and positions where the face is undefined (< 5 previous turns for
`ttrend`; before the first question turn for `dqgap`); the probe fits
on finite targets only. Corpus licence: DailyDialog CC BY-NC-SA 4.0
(research use) — the note travels in ``extra``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from temp_bench.data.synthetic import SyntheticData

CACHE_ROOT = Path("/workspace/dialevel_caches")
REPO_ROOT = Path(__file__).resolve().parents[3]
LABELS_DIR = REPO_ROOT / "experiments/explorations/task_hunt/labels"
TOK_TAG = {"gpt2": "gpt2", "gemma2_2b": "gemma2", "llama31_8b": "llama31"}
FACES = ("ttrend", "dqgap")


def dialogue_face_real(
    *,
    face: str = "ttrend",
    model_key: str = "gpt2",
    hs: int = 7,
    seq_len: int = 128,
    n_seqs: int | None = None,
    rms_sample: int = 64,
    d_in: int | None = None,
    n_ref: int = 8,
    seed: int = 0,
) -> SyntheticData:
    """Dialevel activations + a frozen diafaces label grid.

    ``face`` selects the committed target (`ttrend` | `dqgap`); ``hs``
    is the anchor layer DECLARED by the panel card (the screen layer of
    the winning model — never inferred here). ``d_in`` is declared in
    the datasource params because the trainer sizes the dictionary
    from the spec before materializing; it is checked, not trusted.
    """
    if face not in FACES:
        raise ValueError(f"face must be one of {FACES}, got {face!r}")
    tag = TOK_TAG[model_key]
    acts_path = CACHE_ROOT / model_key / f"hs{hs}.npy"
    if not acts_path.exists():
        raise FileNotFoundError(
            f"activation cache missing: {acts_path} — rebuild via "
            "experiments.explorations.task_hunt.dialevel.cache_acts")
    arr = np.load(acts_path, mmap_mode="r")
    tok = np.load(CACHE_ROOT / model_key / "tokens.npz")
    ids, doc_idx = tok["ids"], tok["doc_idx"]
    n_prefix = int(tok["n_prefix"])
    content = ids.shape[1] - n_prefix

    zd = np.load(LABELS_DIR / f"dialevel_dailydialog_{tag}.npz")
    zf = np.load(LABELS_DIR / f"diafaces_dailydialog_{tag}.npz")
    flat, off = zd["token_ids"], zd["doc_off"]
    val = zf[face].astype(np.float32)
    val = np.where(zd["is_boundary"] == 1, np.float32(np.nan), val)

    if arr.shape[0] != len(doc_idx):
        raise ValueError(f"cache rows {arr.shape[0]} vs tokens {len(doc_idx)}")
    if arr.shape[1] != seq_len or ids.shape[1] != seq_len:
        raise ValueError(f"seq_len {seq_len} != cache {arr.shape[1]}")
    if d_in is not None and int(d_in) != arr.shape[-1]:
        raise ValueError(
            f"datasource declares d_in={d_in} but cache is {arr.shape[-1]} — "
            "the trainer sizes the dictionary from the declared value")

    lam = np.full((len(doc_idx), seq_len), np.nan, dtype=np.float32)
    seen: dict = {}
    for i, d in enumerate(doc_idx.tolist()):
        k = seen.get(d, 0)
        seen[d] = k + 1
        s = off[d] + k * content
        lam[i, n_prefix:] = val[s:s + content]
        if i < 50:  # alignment contract, re-asserted at materialize time
            assert np.array_equal(flat[s:s + content], ids[i, n_prefix:]), \
                f"flat/window mismatch at cache row {i} (doc {d} chunk {k})"

    N = arr.shape[0] if n_seqs is None else min(int(n_seqs), arr.shape[0])
    x = torch.from_numpy(np.ascontiguousarray(arr[:N])).float()
    rng = np.random.default_rng(seed)
    idx = rng.choice(N, size=min(rms_sample, N), replace=False)
    rms = float(x[idx].pow(2).mean().sqrt().clamp(min=1e-6))
    x = x / rms

    # Reference basis (NOT ground truth — module docstring):
    flat_x = x[idx].reshape(-1, x.shape[-1])
    dc = flat_x.mean(0)
    dc = dc / dc.norm().clamp(min=1e-8)
    centred = flat_x - flat_x.mean(0, keepdim=True)
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
            "lambda_labels": torch.from_numpy(lam[:N]),
            # Conversation identity per cache row — read ONLY by the v2
            # probe's grouped split; v1 never touches this key.
            "trace_ids": doc_idx[:N].astype(np.int64).copy(),
            "real_activations": True,
            "face": face,
            "model_key": model_key,
            "hs": hs,
            "rms_scale": rms,
            "no_ground_truth_directions": True,
            "emission_features_are_reference_basis_not_ground_truth": True,
            "n_ref": int(ref.shape[0]),
            "corpus_licence": "DailyDialog CC BY-NC-SA 4.0 (research use)",
        },
    )
