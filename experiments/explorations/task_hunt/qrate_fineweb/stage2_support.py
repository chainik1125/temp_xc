"""Stage-2 support receipts — doc-identity FLOOR + evidence line + NaN receipt.

Pre-registered in `CARD_STAGE2.md` § 6a and § 7 (frozen `b8f2f0bd`,
BEFORE any panel cell ran; rewritten at the A40 restart from the card's
spec — the original was lost unpushed, see APPENDIX A). Off-leaderboard:
nothing here appends a row; it produces the numbers the card orders
printed BESIDE every panel cell.

Three receipts, all label-side (no checkpoints, no training):

1. **Doc-identity FLOOR (§ 6a)** — a doc-mean-only predictor's Pearson r
   on the SAME v1 eval windows (`_sample_windows` imported, eval pool =
   second half, seed 1, nw 1024 — the exact eval convention). Doc mean =
   mean of the doc's finite lam_q over the WHOLE stream (the identity
   route's ceiling as one number per doc). Outcome rule (card P4): floor
   r ≥ 0.5 expected; printed beside every window cell either way.
2. **Evidence line (§ 7, binding 5 clarified)** — the regression analog:
   in-window question-token count on the CURRENT tile → target, same
   probe convention (train pool seed 0 / eval pool seed 1, held-out r,
   non-finite guard). A window cell that does not beat it at matched T
   is counting visible question marks. Card prediction: small at T ≤ 16.
3. **NaN receipt (§ 7)** — per T: fraction of sampled tile targets
   dropped by the non-finite guard (card: 8.7–10.8 % at load) + row
   counts after the drop, train and eval.

Run: .venv/bin/python -m \
       experiments.explorations.task_hunt.qrate_fineweb.stage2_support [ds]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

from temp_bench.core.config import load_datasource
from temp_bench.data.synthetic import materialise
from temp_bench.evals.synthetic_recovery import _sample_windows

HERE = Path(__file__).resolve().parent
LABEL_DIR = HERE.parent / "labels"
CACHE_ROOT = Path("/workspace/replag_caches")

DS_DEFAULT = "fineweb_punctint_q_gemma2_l14"
DS_MODEL = {"fineweb_punctint_q_gemma2_l14": ("gemma2_2b", "gemma2"),
            "fineweb_punctint_q_gpt2_l7": ("gpt2", "gpt2"),
            "fineweb_punctint_q_llama31_l14": ("llama31_8b", "llama31")}
EVAL_L = 32
N_WINDOWS = 1024          # the v1 eval convention
TS = (1, 2, 4, 8, 16)
SEED = 0                  # datasource materialise seed (labels are seed-free)


def _event_grid(model_key: str, tok_key: str) -> np.ndarray:
    """(N, seq_len) bool grid of question-sentence tokens, cache-aligned.

    The same flat-stream → cache-row mapping `real_punctint._row_label_grid`
    asserts byte-exactly at materialise (that assertion has already run by
    the time this is called; the mapping here mirrors it for `is_q`).
    """
    tok = np.load(CACHE_ROOT / model_key / "tokens.npz")
    ids, doc_idx = tok["ids"], tok["doc_idx"].astype(np.int64)
    n_prefix = int(tok["n_prefix"])
    N, seq_len = ids.shape
    content = seq_len - n_prefix
    z = np.load(LABEL_DIR / f"punctint_fineweb_{tok_key}.npz")
    evt = z["is_q"].astype(bool)
    doc_off = z["doc_off"]
    grid = np.zeros((N, seq_len), dtype=bool)
    seen: dict[int, int] = {}
    for i in range(N):
        d = int(doc_idx[i])
        c = seen.get(d, 0)
        seen[d] = c + 1
        s = int(doc_off[d]) + c * content
        grid[i, n_prefix:] = evt[s: s + content]
    return grid


def _tile_targets(win_l: torch.Tensor, T: int) -> np.ndarray:
    """(W, L, 1) λ windows → (W·L/T,) leading-edge targets (the eval tiling)."""
    W, L, _ = win_l.shape
    return (win_l.reshape(W, L // T, T)[:, :, T - 1]
            .reshape(-1).detach().float().cpu().numpy())


def _tile_counts(win_e: torch.Tensor, T: int) -> np.ndarray:
    """(W, L, 1) event windows → (W·L/T,) in-tile event counts (current tile)."""
    W, L, _ = win_e.shape
    return (win_e.reshape(W, L // T, T).sum(dim=2)
            .reshape(-1).detach().float().cpu().numpy())


def main():
    from sklearn.linear_model import LinearRegression

    ds = sys.argv[1] if len(sys.argv) > 1 else DS_DEFAULT
    model_key, tok_key = DS_MODEL[ds]
    data = materialise(load_datasource(ds), seed=SEED)
    lam = torch.as_tensor(data.extra["lambda_labels"]).float()
    trace = np.asarray(data.extra["trace_ids"])
    n = data.x.shape[0]
    split = n // 2
    lam3 = lam.reshape(lam.shape[0], lam.shape[1], 1)
    evt3 = torch.from_numpy(
        _event_grid(model_key, tok_key).astype(np.float32)).unsqueeze(-1)

    lam_np = lam.numpy()
    fin = np.isfinite(lam_np)
    # Doc mean over the WHOLE stream (identity ceiling), one number per doc.
    doc_mean = {int(d): float(lam_np[trace == d][fin[trace == d]].mean())
                for d in np.unique(trace)}

    out = {"ds": ds, "n_seqs": int(n), "split": int(split),
           "n_windows": N_WINDOWS, "eval_L": EVAL_L,
           "corpus": data.extra.get("corpus"), "per_T": {}}
    print(f"[stage2_support] {ds}  ({out['corpus']})", flush=True)
    for T in TS:
        # identical sampler calls to the v1 eval (train seed 0 / eval seed 1)
        win_l_tr, _ = _sample_windows(lam3[:split], L=EVAL_L,
                                      n_windows=N_WINDOWS, seed=0)
        win_e_tr, _ = _sample_windows(evt3[:split], L=EVAL_L,
                                      n_windows=N_WINDOWS, seed=0)
        win_l_ev, ev_idx = _sample_windows(lam3[split:], L=EVAL_L,
                                           n_windows=N_WINDOWS, seed=1)
        win_e_ev, _ = _sample_windows(evt3[split:], L=EVAL_L,
                                      n_windows=N_WINDOWS, seed=1)

        t_tr, t_ev = _tile_targets(win_l_tr, T), _tile_targets(win_l_ev, T)
        c_tr, c_ev = _tile_counts(win_e_tr, T), _tile_counts(win_e_ev, T)
        n_tiles = EVAL_L // T
        ev_docs = np.repeat(trace[split + ev_idx], n_tiles)

        tr_m, ev_m = np.isfinite(t_tr), np.isfinite(t_ev)
        nan_tr = 1.0 - tr_m.mean()
        nan_ev = 1.0 - ev_m.mean()

        # (1) doc-identity floor on the eval windows
        pred_floor = np.array([doc_mean[int(d)] for d in ev_docs[ev_m]])
        floor_r = float(np.corrcoef(pred_floor, t_ev[ev_m])[0, 1]) \
            if np.std(pred_floor) > 1e-12 else 0.0

        # (2) evidence line: in-tile q-token count → target, held-out
        reg = LinearRegression().fit(c_tr[tr_m, None], t_tr[tr_m])
        pred = reg.predict(c_ev[ev_m, None])
        ev_r = float(np.corrcoef(pred, t_ev[ev_m])[0, 1]) \
            if np.std(pred) > 1e-12 else 0.0

        out["per_T"][T] = {
            "doc_floor_r": floor_r, "evidence_count_r": ev_r,
            "nan_drop_train": float(nan_tr), "nan_drop_eval": float(nan_ev),
            "rows_train": int(tr_m.sum()), "rows_eval": int(ev_m.sum()),
        }
        print(f"  T={T:<3} floor_r={floor_r:+.4f}  evidence_r={ev_r:+.4f}  "
              f"nan tr/ev={nan_tr:.3f}/{nan_ev:.3f}  "
              f"rows tr/ev={tr_m.sum()}/{ev_m.sum()}", flush=True)

    res_dir = HERE / "results"
    res_dir.mkdir(exist_ok=True)
    path = res_dir / f"stage2_support_{ds}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"-> {path}", flush=True)


if __name__ == "__main__":
    main()
