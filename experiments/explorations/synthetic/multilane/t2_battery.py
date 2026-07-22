"""Multilane (FB-2) — T2 non-triviality battery (LOOP.md gate T2).

The four committed controls, run on the ACTUAL generator at the frozen card
parameters. Complements ``gating.py`` (which carries the raw-readout floors
and the info-presence checks):

1. **Symmetry/relabeling audit** — analytic, recorded: the circle embedding
   carries metric geometry on Z_M; no relabeling group action maps velocity
   classes onto each other while preserving the data law (P3's
   exchangeability premise fails by construction — that is WHY the circle
   embedding exists; the random-frame variant is the frequency bench's null).
2. **Bag-of-symbols control** — a short-trained canonical token SAE
   (BatchTopKSAE, the panel's own token arch), codes mean-pooled over the
   window, MLP readout per lane: the strongest order-destroyed code route.
   Claim: sits FAR below the periodogram oracle (order is required); it may
   sit above chance (the circle-spread cue survives pooling — same cue that
   gives txc-pre its bag-level number on the frequency bench).
3. **P2 empirical (additive linear)** — LINEAR probe on the same token
   SAE's stacked (concatenated) window codes: the additive-over-time
   readout the theorem covers ⇒ ≈ chance.
4. **Shuffle semantics** (stated + measured): per-window INDEPENDENT
   permutations. For a tone task the shuffle destroys the phase
   progression but preserves the window's symbol multiset ⇒ NOT a full
   null: the periodogram on shuffled tiles degrades toward the bag level,
   not to chance. Measured here so the record states exactly what the
   shuffle destroys.

Memorization budget (P6): analytic — |Ω|³M³ ≈ 1.03e9 whole-window templates
vs d_sae ≤ 202; probe budget: probe rows ≥ 100× code dim at every cell
(30k rows vs d_sae ≤ 202). Recorded in the JSON.

    .venv/bin/python -m experiments.explorations.synthetic.multilane.t2_battery
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

SEED = 0
M, OMEGA = 101, [0, 1, 2, 4, 8, 16, 24, 32, 40, 50]
D_IN, SIGMA, SEQ_LEN, N_LANES = 24, 0.25, 64, 3
T_BAG = 8                        # window for the bag/stacked controls
N_SEQS = 4000
TRAIN_STEPS = 5000               # short train — a control, not a grid cell
CHANCE = 1.0 / len(OMEGA)

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "multilane_t2_stats.json"


def _mlp_acc(z_tr, y_tr, z_ev, y_ev, seed=0):
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                            random_state=seed).fit(z_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(z_ev)))


def _lin_acc(z_tr, y_tr, z_ev, y_ev):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = LogisticRegression(max_iter=200).fit(z_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(z_ev)))


def main() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.data.synthetic import multilane_tones

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    data_tr = multilane_tones(M=M, omega=tuple(OMEGA), d_in=D_IN, sigma=SIGMA,
                              seq_len=SEQ_LEN, n_seqs=N_SEQS, seed=SEED)
    data_ev = multilane_tones(M=M, omega=tuple(OMEGA), d_in=D_IN, sigma=SIGMA,
                              seq_len=SEQ_LEN, n_seqs=N_SEQS, seed=SEED + 1)

    # ── short-train the canonical token arch on the training stream ──
    sae = BatchTopKSAE(d_in=D_IN, d_sae=M, k_pos=2)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    xpool = data_tr.x.reshape(-1, D_IN)
    for step in range(TRAIN_STEPS):
        idx = torch.randint(0, len(xpool), (1024,))
        out = sae.train_step(xpool[idx])
        opt.zero_grad(); out["loss"].backward(); opt.step(); sae.post_step()
    sae.eval()
    print(f"[t2] token SAE trained {TRAIN_STEPS} steps  "
          f"final mse {float(out['mse']):.4f}", flush=True)

    def tiles_and_labels(data, T, n_max=20_000, seed=0):
        r = np.random.default_rng(seed)
        x = data.x.numpy()
        lab = data.extra["lane_velocity_labels"].numpy()
        k = SEQ_LEN // T
        tiles = x[:, : k * T].reshape(-1, T, D_IN)
        y = lab[:, : k * T].reshape(len(x), k, T, N_LANES)[:, :, T - 1, :]
        y = y.reshape(-1, N_LANES)
        idx = r.choice(len(tiles), min(n_max, len(tiles)), replace=False)
        return tiles[idx], y[idx]

    tr_tiles, tr_y = tiles_and_labels(data_tr, T_BAG, seed=1)
    ev_tiles, ev_y = tiles_and_labels(data_ev, T_BAG, seed=2)

    @torch.no_grad()
    def codes(tiles):
        z = sae.encode(torch.from_numpy(tiles).reshape(-1, 1, D_IN).float())
        return z.reshape(len(tiles), T_BAG, -1).numpy()

    z_tr, z_ev = codes(tr_tiles), codes(ev_tiles)
    bag_tr, bag_ev = z_tr.mean(axis=1), z_ev.mean(axis=1)
    stk_tr = z_tr.reshape(len(z_tr), -1)
    stk_ev = z_ev.reshape(len(z_ev), -1)

    out_j: dict = {"card": "freqbench/cards/FB-2.md", "T": T_BAG,
                   "train_steps": TRAIN_STEPS, "chance": CHANCE,
                   "symmetry_audit": (
                       "analytic: circle embedding fixes a metric on Z_M; no "
                       "symbol relabeling preserves the data law while permuting "
                       "velocity classes (P3 exchangeability fails by design — "
                       "the random-frame null lives in the frequency bench)."),
                   "memorization_audit": {
                       "template_count": float(len(OMEGA)) ** 3 * float(M) ** 3,
                       "max_d_sae": 202,
                       "probe_rows": 20000,
                       "note": "templates ~1.03e9 >> every capacity; route dead."}}

    # per-lane oracle on the same eval tiles (reference ceiling)
    planes = data_ev.extra["lane_planes"].numpy().astype(np.float64)
    t = np.arange(T_BAG)
    basis = np.exp(-2j * np.pi * np.asarray(OMEGA, dtype=np.float64)[:, None]
                   * t[None, :] / M)

    def oracle_acc(tiles, y):
        accs = []
        for k in range(N_LANES):
            proj = tiles.astype(np.float64) @ planes[k]
            c = proj[..., 0] + 1j * proj[..., 1]
            pred = np.abs(c @ basis.T).argmax(axis=1)
            accs.append(float((pred == y[:, k]).mean()))
        return accs

    out_j["oracle"] = oracle_acc(ev_tiles, ev_y)

    # 2. bag-of-symbols (pooled codes + MLP)
    out_j["bag_mlp"] = [_mlp_acc(bag_tr, tr_y[:, k], bag_ev, ev_y[:, k],
                                 seed=SEED + k) for k in range(N_LANES)]
    # 3. additive linear (stacked codes + linear) — the P2 readout
    out_j["stacked_linear"] = [_lin_acc(stk_tr, tr_y[:, k], stk_ev, ev_y[:, k])
                               for k in range(N_LANES)]
    # 4. shuffle semantics: periodogram on independently permuted tiles
    perm = np.argsort(rng.random((len(ev_tiles), T_BAG)), axis=1)
    shuf = np.take_along_axis(ev_tiles, perm[..., None], axis=1)
    out_j["oracle_shuffled"] = oracle_acc(shuf, ev_y)
    out_j["shuffle_semantics"] = (
        "per-window independent permutations destroy the phase progression "
        "(order) but preserve the symbol multiset — the circle-spread cue "
        "survives, so the shuffle is NOT a full null for velocity; it "
        "degrades the oracle toward the bag level.")

    o, b, s, sh = (np.mean(out_j["oracle"]), np.mean(out_j["bag_mlp"]),
                   np.mean(out_j["stacked_linear"]),
                   np.mean(out_j["oracle_shuffled"]))
    out_j["verdict"] = {
        "oracle_mean": round(float(o), 4),
        "bag_mlp_mean": round(float(b), 4),
        "stacked_linear_mean": round(float(s), 4),
        "oracle_shuffled_mean": round(float(sh), 4),
        "checks": {
            "bag_fails_vs_oracle": bool(b < o - 0.15),
            "additive_linear_at_chance": bool(abs(s - CHANCE) < 0.05),
            "shuffle_degrades_oracle": bool(sh < o - 0.15),
        },
    }
    out_j["verdict"]["passes_t2"] = bool(all(out_j["verdict"]["checks"].values()))
    print(json.dumps(out_j["verdict"], indent=1), flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_j, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
