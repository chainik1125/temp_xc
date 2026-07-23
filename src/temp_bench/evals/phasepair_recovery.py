"""Phasepair (± velocity pairs) sign/pair decomposition — FreqBench FB-1 add-on.

Fires from :class:`SyntheticRecovery` when the materialised data carries
``extra['velocity_labels']`` AND its Ω contains ± pairs (``y + y' ≡ 0 mod
M``). The frequency bench's Ω has none, so this is a no-op there (no keys
added — old rows stay byte-identical). Protocol stays 1.3.0.

The FB-1 card (freqbench/cards/FB-1.md) decomposes the 6-class velocity
into **pair id** |Y| (power-readable — plain tone detection) and **sign**
(phase-only: within a pair the per-channel power spectra and the bag-of-
symbols distribution are exactly identical; only the cross-channel
quadrature carries it). Metrics:

- ``pair_recovery`` — logistic probe on the per-tile code → pair id,
  balanced accuracy normalized to [chance = 1/n_pairs, 1].
- ``sign_recovery`` (**headline**) — mean over pairs of a per-pair binary
  probe (tiles restricted to that true pair) → sign, balanced accuracy
  normalized to [½, 1]. Conditioning on the true pair isolates the
  phase-only question from pair-detection errors.
- ``sign_oracle`` — same restriction, signed periodogram matched filter on
  the raw tiles (the ML reference); ``sign_chance`` — shuffled-label floor.

Probe conventions identical to ``frequency_recovery`` (per-tile leading
edge, single-tile codes, leak-free split, linear probes).
"""

from __future__ import annotations

import numpy as np

from temp_bench.interfaces.architecture import TempBenchArch


def find_pairs(omega: list[int], M: int) -> list[tuple[int, int]]:
    """Class-index pairs (i, j), i < j, with Ω_i + Ω_j ≡ 0 (mod M)."""
    return [(i, j) for i in range(len(omega)) for j in range(i + 1, len(omega))
            if (omega[i] + omega[j]) % M == 0 and omega[i] != omega[j]]


def phasepair_metrics(
    model: TempBenchArch, data, *, eval_window_L: int,
    n_windows: int = 1024, seed: int = 0, max_rows: int = 30_000,
) -> dict[str, float]:
    from temp_bench.evals.frequency_recovery import (
        _logistic_probe,
        _tile_examples,
    )
    from temp_bench.evals.synthetic_recovery import _check_tileable, _sample_windows

    omega = [int(y) for y in data.extra["omega"]]
    M = int(data.extra["M"])
    pairs = find_pairs(omega, M)
    if not pairs:
        return {}

    T = _check_tileable(model, eval_window_L)
    model.eval()
    x = data.x
    n = x.shape[0]
    split = n // 2
    L = eval_window_L
    vel = data.extra["velocity_labels"].float().unsqueeze(-1)

    win_x_tr, _ = _sample_windows(x[:split], L=L, n_windows=n_windows, seed=seed)
    win_y_tr, _ = _sample_windows(vel[:split], L=L, n_windows=n_windows, seed=seed)
    win_x_ev, _ = _sample_windows(x[split:], L=L, n_windows=n_windows, seed=seed + 1)
    win_y_ev, _ = _sample_windows(vel[split:], L=L, n_windows=n_windows, seed=seed + 1)
    z_tr, _, y_tr = _tile_examples(model, win_x_tr, win_y_tr, T)
    z_ev, raw_ev, y_ev = _tile_examples(model, win_x_ev, win_y_ev, T)

    rng = np.random.default_rng(seed)
    if z_tr.shape[0] > max_rows:
        i = rng.choice(z_tr.shape[0], max_rows, replace=False)
        z_tr, y_tr = z_tr[i], y_tr[i]
    if z_ev.shape[0] > max_rows:
        j = rng.choice(z_ev.shape[0], max_rows, replace=False)
        z_ev, y_ev, raw_ev = z_ev[j], y_ev[j], raw_ev[j]

    # class index → pair index / sign
    pair_of = np.full(len(omega), -1)
    sign_of = np.zeros(len(omega))
    for p, (i, j) in enumerate(pairs):
        pair_of[i], pair_of[j] = p, p
        sign_of[i], sign_of[j] = +1.0, -1.0

    out: dict[str, float] = {}
    n_pairs = len(pairs)

    # pair id probe (power-readable component)
    p_tr, p_ev = pair_of[y_tr], pair_of[y_ev]
    m_tr, m_ev = p_tr >= 0, p_ev >= 0
    bal, _, floor = _logistic_probe(z_tr[m_tr], p_tr[m_tr], z_ev[m_ev],
                                    p_ev[m_ev], n_classes=n_pairs, seed=seed)
    chance_p = 1.0 / n_pairs
    out["pair_recovery"] = float((bal - chance_p) / (1 - chance_p))
    out["pair_balacc"] = bal
    out["pair_chance"] = floor

    # per-pair sign probes (the phase-only headline) + signed oracle
    R = (data.extra["circle_plane"].detach().cpu().numpy().astype(np.float64)
         if "circle_plane" in data.extra else None)
    t = np.arange(raw_ev.shape[1])
    sign_bals, sign_oracles, sign_floors = [], [], []
    for p, (i, j) in enumerate(pairs):
        tr_m = (y_tr == i) | (y_tr == j)
        ev_m = (y_ev == i) | (y_ev == j)
        s_tr = (y_tr[tr_m] == i).astype(np.int64)
        s_ev = (y_ev[ev_m] == i).astype(np.int64)
        if s_tr.sum() in (0, len(s_tr)) or s_ev.sum() in (0, len(s_ev)):
            continue
        bal_s, _, floor_s = _logistic_probe(z_tr[tr_m], s_tr, z_ev[ev_m], s_ev,
                                            n_classes=2, seed=seed + 31 * p)
        sign_bals.append(bal_s)
        sign_floors.append(floor_s)
        out[f"sign_balacc_pair{p}"] = bal_s
        if R is not None:
            proj = raw_ev[ev_m].astype(np.float64) @ R
            c = proj[..., 0] + 1j * proj[..., 1]
            basis = np.exp(-2j * np.pi
                           * np.asarray([omega[i], omega[j]], dtype=np.float64)[:, None]
                           * t[None, :] / M)
            opred = (np.abs(c @ basis.T).argmax(axis=1) == 0).astype(np.int64)
            o = float(((opred == s_ev)[s_ev == 1].mean()
                       + (opred == s_ev)[s_ev == 0].mean()) / 2)
            sign_oracles.append(o)
            out[f"sign_oracle_pair{p}"] = o

    if sign_bals:
        mb = float(np.mean(sign_bals))
        out["sign_balacc"] = mb
        out["sign_recovery"] = float((mb - 0.5) / 0.5)
        out["sign_chance"] = float(np.mean(sign_floors))
    if sign_oracles:
        out["sign_oracle"] = float(np.mean(sign_oracles))
    return out
