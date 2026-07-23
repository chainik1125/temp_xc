"""Phasepair (FB-1) — T2 non-triviality battery (LOOP.md gate T2).

1. **Symmetry/relabeling audit (recorded finding).** The reflection
   ψ(a) = −a (mod M) composed with the fixed orthogonal map that flips the
   second column of R maps class-y data EXACTLY onto class-(−y) data: the
   sign classes are exchangeable under one global orthogonal
   transformation. Consequence: the sign latent is **chirality relative to
   the realized embedding R** — well-defined within a seed (eval
   re-materializes with the training seed), distributionally separable for
   fixed R, but NOT poolable across seeds and carrying no seed-free
   "preferred direction". This is not the P3 trivialization (the classes
   are distinct for fixed R; pair id is reflection-invariant) — recorded as
   the audit outcome, with the cross-seed caution.
2. **Bag-of-symbols control on trained token CODES** — 5k-step canonical
   token SAE, codes mean-pooled over T=8 windows, MLP → per-pair sign:
   must sit at ½ (the exact null transfers to any per-token feature map —
   the bag of φ(x_t) inherits the identical-set distribution).
3. **P2 empirical** — linear probe on the stacked (concatenated) token
   codes → per-pair sign ≈ ½.
4. **Shuffle semantics:** identical to the bag here — a within-window
   permutation destroys exactly the order; for SIGN the shuffle IS a full
   null (unlike frequency/multilane magnitude): shuffled-tile signed
   oracle must fall to ½. Measured.

Memorization audit: |Ω|·M = 606 clean windows vs d_sae ≤ 202 and probe
rows 20k ≥ 100× code dim — recorded.

    .venv/bin/python -m experiments.explorations.synthetic.phasepair.t2_battery
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import torch

SEED = 0
M = 101
OMEGA = (3, 98, 12, 89, 30, 71)
PAIRS = [(0, 1), (2, 3), (4, 5)]
D_IN, SIGMA, SEQ_LEN = 24, 0.10, 64
T = 8
N_SEQS = 4000
TRAIN_STEPS = 5000

HERE = Path(__file__).resolve().parent
OUT_JSON = HERE / "results" / "phasepair_t2_stats.json"


def _probe(z_tr, y_tr, z_ev, y_ev, mlp=False, seed=0):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neural_network import MLPClassifier
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf = (MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                             random_state=seed) if mlp
               else LogisticRegression(max_iter=200))
        clf.fit(z_tr, y_tr)
        return float(balanced_accuracy_score(y_ev, clf.predict(z_ev)))


def main() -> None:
    from temp_bench.archs.batchtopk_sae import BatchTopKSAE
    from temp_bench.data.synthetic import cyclic_tones

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    mk = lambda s: cyclic_tones(M=M, omega=OMEGA, embedding="circle",
                                d_in=D_IN, sigma=SIGMA, seq_len=SEQ_LEN,
                                n_seqs=N_SEQS, seed=s)
    dtr, dev = mk(SEED), mk(SEED + 1)

    sae = BatchTopKSAE(d_in=D_IN, d_sae=M, k_pos=2)
    opt = torch.optim.Adam(sae.parameters(), lr=1e-3)
    xpool = dtr.x.reshape(-1, D_IN)
    for _ in range(TRAIN_STEPS):
        idx = torch.randint(0, len(xpool), (1024,))
        res = sae.train_step(xpool[idx])
        opt.zero_grad(); res["loss"].backward(); opt.step(); sae.post_step()
    sae.eval()

    def tiles(data, seed):
        r = np.random.default_rng(seed)
        x = data.x.numpy()
        lab = data.extra["velocity_labels"].numpy()
        k = SEQ_LEN // T
        t_ = x[:, : k * T].reshape(-1, T, D_IN)
        y_ = lab[:, : k * T].reshape(len(x), k, T)[:, :, T - 1].reshape(-1)
        idx = r.choice(len(t_), min(20_000, len(t_)), replace=False)
        return t_[idx], y_[idx]

    tr_t, tr_y = tiles(dtr, 1)
    ev_t, ev_y = tiles(dev, 2)

    @torch.no_grad()
    def codes(tiles_):
        z = sae.encode(torch.from_numpy(tiles_).reshape(-1, 1, D_IN).float())
        return z.reshape(len(tiles_), T, -1).numpy()

    z_tr, z_ev = codes(tr_t), codes(ev_t)
    out: dict = {
        "card": "freqbench/cards/FB-1.md", "T": T,
        "symmetry_audit": (
            "reflection a->-a + orthogonal column-flip of R exchanges the "
            "sign classes exactly: sign = chirality w.r.t. the realized R. "
            "Well-defined per seed (eval rematerializes with the training "
            "seed); NOT poolable across seeds; pair id is "
            "reflection-invariant. Not a P3 trivialization."),
        "memorization_audit": {"templates": 6 * M, "max_d_sae": 202,
                               "probe_rows": 20_000},
    }

    bag_tr, bag_ev = z_tr.mean(axis=1), z_ev.mean(axis=1)
    stk_tr = z_tr.reshape(len(z_tr), -1)
    stk_ev = z_ev.reshape(len(z_ev), -1)
    bag_sign, stk_sign = [], []
    for p, (i, j) in enumerate(PAIRS):
        mtr = (tr_y == i) | (tr_y == j)
        mev = (ev_y == i) | (ev_y == j)
        s_tr = (tr_y[mtr] == i).astype(int)
        s_ev = (ev_y[mev] == i).astype(int)
        bag_sign.append(_probe(bag_tr[mtr], s_tr, bag_ev[mev], s_ev,
                               mlp=True, seed=SEED + p))
        stk_sign.append(_probe(stk_tr[mtr], s_tr, stk_ev[mev], s_ev))
    out["bag_codes_mlp_sign"] = [round(v, 4) for v in bag_sign]
    out["stacked_codes_linear_sign"] = [round(v, 4) for v in stk_sign]

    # shuffle = full null for sign: shuffled-tile signed oracle
    R = dev.extra["circle_plane"].numpy().astype(np.float64)
    perm = np.argsort(rng.random((len(ev_t), T)), axis=1)
    shuf = np.take_along_axis(ev_t, perm[..., None], axis=1)
    t_ = np.arange(T)
    sh_or = []
    for p, (i, j) in enumerate(PAIRS):
        m = (ev_y == i) | (ev_y == j)
        proj = shuf[m].astype(np.float64) @ R
        c = proj[..., 0] + 1j * proj[..., 1]
        b2 = np.exp(-2j * np.pi
                    * np.asarray([OMEGA[i], OMEGA[j]], dtype=np.float64)[:, None]
                    * t_[None, :] / M)
        sp = (np.abs(c @ b2.T).argmax(axis=1) == 0).astype(int)
        st = (ev_y[m] == i).astype(int)
        sh_or.append(float(((sp == st)[st == 1].mean()
                            + (sp == st)[st == 0].mean()) / 2))
    out["shuffled_sign_oracle"] = [round(v, 4) for v in sh_or]

    checks = {
        "bag_codes_sign_at_half": bool(
            max(abs(v - 0.5) for v in bag_sign) <= 0.04),
        "stacked_linear_sign_at_half": bool(
            max(abs(v - 0.5) for v in stk_sign) <= 0.05),
        "shuffle_full_null_for_sign": bool(
            max(abs(v - 0.5) for v in sh_or) <= 0.04),
    }
    out["verdict"] = {"checks": checks, "passes_t2": bool(all(checks.values()))}
    print(json.dumps(out, indent=1), flush=True)

    OUT_JSON.parent.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT_JSON}", flush=True)


if __name__ == "__main__":
    main()
