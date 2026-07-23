"""Permuted-tone schedule recovery — FreqBench FB-5 add-on.

Like ``frequency_recovery``, this is NOT a routed evaluator:
:class:`SyntheticRecovery` calls :func:`permuted_metrics` whenever the
materialised data carries ``extra['schedule_table']`` (the
``toy_permuted_*`` datasources). No-op for every other bench → the § 4
metrics stay byte-identical and ``protocol_version`` does not move.

Conventions are the frequency bench's exactly (frozen card
``freqbench/cards/FB-5.md`` § 2): per-tile leading-edge probes, leak-free
sequence split, LINEAR / multinomial-logistic probes only, normalization to
[chance = 1/K, 1] with the oracle as a separate reference.

Metrics:

- ``schedule_recovery`` (**headline**): multinomial-logistic probe on the
  code → the schedule class Y, balanced accuracy normalized from chance.
- ``schedule_oracle``: the **matched filter** over (schedule, offset) —
  score(k, s) = Σ_t ⟨x_t, u_{π_k((s+t) mod M)}⟩, max over s, argmax over k —
  the exact-template ML decoder in white Gaussian noise, computed on the
  same held-out raw tiles (the card's ceiling obligation).
- per-class recalls (``sched_recall_c{c}`` / ``sched_oracle_c{c}``) and the
  shuffled-label chance floor; band decomposition for spectral archs.
"""

from __future__ import annotations

import numpy as np

from temp_bench.interfaces.architecture import TempBenchArch


def _matched_filter_pred(x_tiles, U, perms, M):
    """Matched-filter ML over (schedule k, offset s) on raw tiles."""
    N, T, _d = x_tiles.shape
    S = np.einsum("ntd,md->ntm", x_tiles, U)                # (N, T, M) symbol scores
    t = np.arange(T)
    K = perms.shape[0]
    best = np.full(N, -np.inf)
    pred = np.zeros(N, dtype=np.int64)
    for k in range(K):
        idx = perms[k][(np.arange(M)[:, None] + t[None, :]) % M]   # (M_s, T)
        sc = np.zeros((N, M))
        for tt in range(T):
            sc += S[:, tt, idx[:, tt]]
        bestk = sc.max(axis=1)
        better = bestk > best
        pred[better] = k
        best[better] = bestk[better]
    return pred


def permuted_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0, max_rows: int = 30_000,
) -> dict[str, float]:
    """Schedule recovery + matched-filter oracle for permuted-tone data."""
    from temp_bench.evals.frequency_recovery import (
        _logistic_probe,
        _per_class_recall,
        _tile_examples,
    )
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    M = int(data.extra["M"])
    K = int(data.extra["K"])
    chance = 1.0 / K
    norm = 1.0 - chance
    U = data.emission_features.detach().cpu().numpy().astype(np.float64)
    perms = data.extra["schedule_table"].detach().cpu().numpy().astype(np.int64)

    lab = data.extra["schedule_labels"].float().unsqueeze(-1)   # (N, seq_len, 1)
    n = x.shape[0]
    split = n // 2
    L = eval_window_L
    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(lab[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(lab[split:], L=L, n_windows=n_windows, seed=seed + 1)

    z_tr, _, y_tr = _tile_examples(model, win_x_tr, win_y_tr, T)
    z_ev, raw_ev, y_ev = _tile_examples(model, win_x_ev, win_y_ev, T)

    rng = np.random.default_rng(seed)
    if z_tr.shape[0] > max_rows:
        i = rng.choice(z_tr.shape[0], max_rows, replace=False)
        z_tr, y_tr = z_tr[i], y_tr[i]
    if z_ev.shape[0] > max_rows:
        j = rng.choice(z_ev.shape[0], max_rows, replace=False)
        z_ev, y_ev, raw_ev = z_ev[j], y_ev[j], raw_ev[j]

    out: dict[str, float] = {}
    bal, recall, floor = _logistic_probe(z_tr, y_tr, z_ev, y_ev,
                                         n_classes=K, seed=seed)
    opred = _matched_filter_pred(raw_ev, U, perms, M)
    oracle = float((opred == y_ev).mean())
    oracle_recall = _per_class_recall(y_ev, opred, K)

    out["schedule_recovery"] = float((bal - chance) / norm)
    out["schedule_balacc"] = bal
    out["schedule_oracle"] = oracle
    out["schedule_chance"] = floor
    for c in range(K):
        out[f"sched_recall_c{c}"] = recall[c]
        out[f"sched_oracle_c{c}"] = oracle_recall[c]

    if hasattr(model, "band_of_features"):
        for b, (s, e) in enumerate(model.band_of_features()):
            bb, _, _ = _logistic_probe(z_tr[:, s:e], y_tr, z_ev[:, s:e], y_ev,
                                       n_classes=K, seed=seed + b)
            out[f"band{b}_schedule_recovery"] = float((bb - chance) / norm)

    return out
