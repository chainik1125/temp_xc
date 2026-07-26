"""Does true-feature recovery improve systematically with window length T?

Everything else in this sprint trains on data whose segments are independent, where a
window code can only lose: there is no cross-segment structure to recover, so sharing one
code across T segments is pure cost. That measures the penalty but says nothing about the
payoff. This asks the complementary question on data built to have a payoff.

THE GENERATIVE MODEL. A dictionary of ground-truth features, each with a direction u in
R^d and a temporal profile p over L consecutive segments. When feature i fires at segment
s it contributes p[j] * u_i to segment s+j for j in 0..L-1. Features with L=1 are ordinary
per-token features; features with L>1 are genuinely temporal and cannot be represented by
any per-segment dictionary without splitting them across positions.

THE ANALYTIC PREDICTION, which is what makes this worth running. A window of length T can
cover at most T consecutive entries of a length-L profile, so the best achievable cosine
between any T-slab and the true feature's spatiotemporal atom p (x) u is

    ceiling(T, p) = sqrt( max over contiguous T-windows of sum p_j^2 ) / ||p||

For a flat profile this is sqrt(min(T, L) / L). So:

  * a per-token dictionary (T=1) can recover at most 1/sqrt(L) of an extent-L feature --
    for L=8 that is 0.354, no matter how good the training;
  * recovery should climb along that ceiling as T grows, and saturate once T >= L;
  * for L=1 features, T buys nothing and may cost.

If measured recovery tracks the ceiling, the practical rule follows directly and is
testable on real tasks: **set T to the temporal extent of the behaviour you want to
capture; below that, recovery is capped by geometry rather than by optimisation.**

REGISTERED PREDICTIONS (written before the run):
  V1  For L=1 features, recovery is high at every T and flat-to-declining in T.
  V2  For L=8 features, recovery at T=1 is near 1/sqrt(8) = 0.354 and rises monotonically
      with T, saturating at T >= 8.
  V3  The T at which each extent group saturates equals its L. This is the transferable
      principle; if recovery instead saturates at the same T for every L, the effect is
      about window size in general and not about matching the feature's timescale.
  V4  Recovery stays below the analytic ceiling everywhere (it is an upper bound), and the
      gap to it widens with L, since longer features are rarer per unit of data.

Runs locally -- pure synthetic, no language model, small dimensions.
"""
import argparse
import json
import pathlib

import numpy as np
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "dict_bench" / "recovery.json"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_features(n_per_extent, extents, d, gen):
    """Ground-truth features: unit direction u, unit-norm profile p of length L."""
    feats = []
    for L in extents:
        for _ in range(n_per_extent):
            u = torch.randn(d, generator=gen)
            u = u / u.norm()
            # Positive, non-degenerate profiles so extent is real rather than nominal:
            # a profile with all its mass in one entry is an L=1 feature in disguise.
            p = 0.5 + torch.rand(L, generator=gen)
            p = p / p.norm()
            feats.append({"u": u, "p": p, "L": L})
    return feats


def generate(feats, n_seq, seq_len, d, fire_p, noise, gen):
    """(n_seq, seq_len, d) activations from the generative model above."""
    X = torch.zeros(n_seq, seq_len, d)
    for f in feats:
        L, u, p = f["L"], f["u"], f["p"]
        # Fire only where the whole profile fits, so every firing has its full extent.
        fires = torch.rand(n_seq, seq_len - L + 1, generator=gen) < fire_p
        amp = 1.0 + torch.rand(n_seq, seq_len - L + 1, generator=gen)
        w = (fires.float() * amp)
        for j in range(L):
            X[:, j:j + (seq_len - L + 1), :] += (w * p[j]).unsqueeze(-1) * u
    X += noise * torch.randn(n_seq, seq_len, d, generator=gen)
    return X


def true_slabs(f, T):
    """Every placement of a feature's atom inside a T-window, as (n_off, T, d).

    Includes partial overlaps at the window edges, because a real window sees features
    that start before it and end after it.

    Deliberately NOT renormalised. The full atom p (x) u is unit-norm, and a T-window sees
    only the part of it that fits; the norm of what is left is exactly how much of the
    feature a T-slab can account for. Renormalising each truncation would ask "does the
    learned atom match the visible fragment", which is a strictly easier question and
    would let measured recovery exceed the analytic ceiling -- as it did before this fix,
    scoring 0.724 against a ceiling of 0.571 because at T=1 every truncation renormalises
    to the same +/-u.
    """
    L, u, p = f["L"], f["u"], f["p"]
    slabs = []
    for off in range(-(L - 1), T):
        s = torch.zeros(T, u.shape[0])
        lo, hi = max(0, off), min(T, off + L)
        if hi <= lo:
            continue
        s[lo:hi] = p[lo - off:hi - off].unsqueeze(-1) * u
        if s.norm() > 1e-8:
            slabs.append(s)
    return torch.stack(slabs)


def ceiling(f, T):
    """Best cosine any T-slab can achieve against this feature: the norm of the largest
    contiguous T-chunk of the profile, over the whole profile norm."""
    p, L = f["p"], f["L"]
    best = 0.0
    for off in range(-(L - 1), T):
        lo, hi = max(0, off), min(T, off + L)
        if hi <= lo:
            continue
        best = max(best, float(p[lo - off:hi - off].norm() / p.norm()))
    return best


def recovery(W_dec, feats, T, device):
    """Max cosine between each true feature's atom and any learned atom, at any offset.

    W_dec: (d_sae, T, d), rows already unit-norm over (1, 2).
    """
    A = W_dec.reshape(W_dec.shape[0], -1)
    A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-8)
    out = []
    for f in feats:
        S = true_slabs(f, T).reshape(-1, T * W_dec.shape[2]).to(device)
        cos = (S @ A.T).abs()          # signed features: sign is a decoder convention
        out.append(float(cos.max()))
    return out


def train(T, X, d, d_sae, k_per, steps, batch, lr, device, activation, log):
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    n_seq, seq_len, _ = X.shape
    n_win = seq_len // T
    W_all = X[:, :n_win * T, :].reshape(-1, T, d).to(device)
    # Hold out whole sequences' worth of windows, so no training window shares a
    # generative firing with an evaluation one.
    n_hold = max(int(0.15 * W_all.shape[0]), 64)
    W, W_ho = W_all[:-n_hold], W_all[-n_hold:]

    torch.manual_seed(0)
    m = TemporalCrosscoder(d, d_sae, T, k_per, activation=activation).to(device)
    with torch.no_grad():
        m._normalize_decoder()
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for s in range(steps):
        xb = W[torch.randint(0, W.shape[0], (batch,), device=device)]
        loss, _, _ = m(xb)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); m._normalize_decoder()
        if log and (s % max(1, steps // 3) == 0 or s == steps - 1):
            print(f"      [T={T}] {s}/{steps} loss={float(loss.detach()):.4f}", flush=True)

    m.eval()
    with torch.no_grad():
        z = m.encode(W_ho)
        l0 = float((z != 0).float().sum(-1).mean())
        xh = m.decode(z)
        denom = float(W_ho.reshape(-1, d).var(0).sum())
        fvu = float(((xh - W_ho) ** 2).sum(-1).mean() / denom)
    return m, l0, fvu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--d_sae", type=int, default=512)
    ap.add_argument("--n_per_extent", type=int, default=16)
    ap.add_argument("--extents", type=str, default="1,2,4,8")
    ap.add_argument("--Ts", type=str, default="1,2,4,8,16")
    ap.add_argument("--n_seq", type=int, default=400)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--fire_p", type=float, default=0.02)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--k_per", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--activation", type=str, default="batchtopk")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))

    extents = [int(x) for x in a.extents.split(",")]
    Ts = [int(x) for x in a.Ts.split(",")]
    dev = pick_device()
    print(f"[device] {dev}  extents={extents}  T={Ts}  activation={a.activation}",
          flush=True)

    gen = torch.Generator().manual_seed(20260725)
    feats = make_features(a.n_per_extent, extents, a.d, gen)
    X = generate(feats, a.n_seq, a.seq_len, a.d, a.fire_p, a.noise, gen)
    print(f"[data] {tuple(X.shape)}  {len(feats)} true features", flush=True)

    print("\nanalytic ceiling on recovery, by feature extent (max cosine any T-slab "
          "can reach):")
    hdr = "  extent " + "".join(f"{f'T={T}':>9}" for T in Ts)
    print(hdr, flush=True)
    for L in extents:
        f0 = next(f for f in feats if f["L"] == L)
        print(f"  L={L:<5}" + "".join(f"{ceiling(f0, T):>9.3f}" for T in Ts), flush=True)

    rows = []
    for T in Ts:
        m, l0, fvu = train(T, X, a.d, a.d_sae, a.k_per, a.steps, a.batch, a.lr, dev,
                           a.activation, log=True)
        rec = recovery(m.W_dec.data, feats, T, dev)
        by_L = {}
        for f, r in zip(feats, rec):
            by_L.setdefault(f["L"], []).append(r)
        row = {"T": T, "realised_l0_per_window": l0, "coeff_per_segment": l0 / T,
               "fvu": fvu,
               "recovery": {str(L): float(np.mean(v)) for L, v in sorted(by_L.items())},
               "ceiling": {str(L): ceiling(next(f for f in feats if f["L"] == L), T)
                           for L in extents}}
        rows.append(row)
        print(f"  [T={T}] coeff/seg {l0/T:5.2f}  FVU {fvu:.4f}  recovery " +
              "  ".join(f"L={L}:{np.mean(v):.3f}" for L, v in sorted(by_L.items())),
              flush=True)

    print("\n===== recovery by feature extent, against the analytic ceiling =====",
          flush=True)
    print("  (each cell: measured / ceiling)", flush=True)
    print("  extent " + "".join(f"{f'T={T}':>16}" for T in Ts), flush=True)
    for L in extents:
        cells = []
        for r in rows:
            cells.append(f"{r['recovery'][str(L)]:.3f} / {r['ceiling'][str(L)]:.3f}")
        print(f"  L={L:<5}" + "".join(f"{c:>16}" for c in cells), flush=True)

    print("\n===== V3: does each extent saturate at T = its own L? =====", flush=True)
    for L in extents:
        series = [(r["T"], r["recovery"][str(L)]) for r in rows]
        best_T, best_v = max(series, key=lambda t: t[1])
        # First T reaching 95% of this extent's best -- the practical saturation point.
        sat = next(T for T, v in series if v >= 0.95 * best_v)
        verdict = "matches L" if sat == L else f"does NOT match L (saturates at {sat})"
        print(f"  L={L:<3} peak {best_v:.3f} at T={best_T:<3} "
              f"saturates at T={sat:<3} -> {verdict}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "extents": extents, "Ts": Ts,
                               "config": vars(a)}, indent=2))
    print("\n[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
