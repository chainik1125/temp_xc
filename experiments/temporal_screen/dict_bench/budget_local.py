"""Is the degradation with excess T a sparsity-scaling problem?

The desideratum is that performance rises with T while T buys something and flattens once
it does not. The crosscoder instead rises and falls. Two accounts are on the table.

  (a) POSITION-TYING. `W_dec` is (d_sae, T, d), so an atom specifies its contribution at
      each absolute position in the window. A feature at offset 3 and the same feature at
      offset 7 are different atoms, so the dictionary needed grows with T and a fixed
      d_sae is progressively starved. Tested in `saturation_local.py`.

  (b) SPARSITY SCALING. The per-segment scalar budget is already held fixed here --
      `TemporalCrosscoder` sets `self.k = kper * T`, and realised coefficients per segment
      measured 3.95 -> 5.01 across T=1..16, so that is matched. What is NOT matched is the
      per-token *density*: every one of the kper*T selected latents contributes to every
      position through its (T, d) slab, so the number of atoms touching a given token
      grows like kper*T while the coefficients multiplying them stay shared across
      positions. If that is what degrades, then some other kper should rescue large T, and
      the required kper should scale with T in a readable way.

This sweeps T against kper so (b) is answered directly: for each T, is there ANY per-segment
budget that recovers the performance seen at T=1?

REGISTERED PREDICTIONS (written before the run):
  B1  If (b) is the story, every T has a kper reaching within 5% of the best T=1 recovery,
      and the arg-max kper moves monotonically with T. The degradation is then a tuning
      artefact and the desideratum is satisfiable by scaling the budget.
  B2  If (a) is the story, large T is not rescued at ANY kper -- the best-over-kper
      recovery still falls with T -- because the shortfall is dictionary entries, not
      coefficients.
  B3  Under (b), raising kper at fixed T should help monotonically up to a point. Under
      (a), raising kper at large T should do very little, since extra coefficients cannot
      substitute for absent atoms.

Reported per cell: recovery against the true features, FVU, realised coefficients per
segment, and the fraction of the dictionary that is alive.

Runs locally -- pure synthetic, no language model.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "dict_bench" / "budget.json"


def windows(X, T, stride):
    """(n_seq, seq_len, d) -> (n_win, T, d). stride=1 keeps the window count roughly
    constant in T; the disjoint reshape makes it fall as seq_len/T, starving large T."""
    n_seq, seq_len, d = X.shape
    if stride >= T:
        n_win = seq_len // T
        return X[:, :n_win * T, :].reshape(-1, T, d)
    return X.unfold(1, T, stride).permute(0, 1, 3, 2).reshape(-1, T, d).contiguous()


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_features(n_per_extent, extents, d, gen):
    feats = []
    for L in extents:
        for _ in range(n_per_extent):
            u = torch.randn(d, generator=gen); u = u / u.norm()
            p = 0.5 + torch.rand(L, generator=gen); p = p / p.norm()
            feats.append({"u": u, "p": p, "L": L})
    return feats


def generate(feats, n_seq, seq_len, d, fire_p, noise, gen):
    X = torch.zeros(n_seq, seq_len, d)
    for f in feats:
        L, u, p = f["L"], f["u"], f["p"]
        fires = torch.rand(n_seq, seq_len - L + 1, generator=gen) < fire_p
        amp = 1.0 + torch.rand(n_seq, seq_len - L + 1, generator=gen)
        w = fires.float() * amp
        for j in range(L):
            X[:, j:j + (seq_len - L + 1), :] += (w * p[j]).unsqueeze(-1) * u
    return X + noise * torch.randn(n_seq, seq_len, d, generator=gen)


def true_slabs(f, T):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--d_sae", type=int, default=512)
    ap.add_argument("--Ts", type=str, default="1,2,4,8,16")
    ap.add_argument("--kpers", type=str, default="1,2,4,8,16")
    ap.add_argument("--n_per_extent", type=int, default=12)
    ap.add_argument("--extents", type=str, default="1,2,4")
    ap.add_argument("--n_seq", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--fire_p", type=float, default=0.02)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--activation", type=str, default="batchtopk")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    Ts = [int(x) for x in a.Ts.split(",")]
    kpers = [int(x) for x in a.kpers.split(",")]
    extents = [int(x) for x in a.extents.split(",")]
    dev = pick_device()
    print(f"[device] {dev}  T={Ts}  kper={kpers}  d_sae={a.d_sae}  "
          f"extents={extents} (longest {max(extents)})", flush=True)

    gen = torch.Generator().manual_seed(20260726)
    feats = make_features(a.n_per_extent, extents, a.d, gen)
    X = generate(feats, a.n_seq, a.seq_len, a.d, a.fire_p, a.noise, gen)
    print(f"[data] {tuple(X.shape)}  {len(feats)} true features", flush=True)

    rows = []
    print(f"\n{'T':>4}{'kper':>6}{'nom k/win':>11}{'coeff/seg':>11}{'alive':>8}"
          f"{'FVU':>9}{'recovery':>10}", flush=True)
    for T in Ts:
        W_all = windows(X, T, a.stride).to(dev)
        n_hold = max(int(0.15 * W_all.shape[0]), 64)
        Wtr, Who = W_all[:-n_hold], W_all[-n_hold:]
        S_all = [true_slabs(f, T).reshape(-1, T * a.d).to(dev) for f in feats]

        for kp in kpers:
            if kp * T > a.d_sae:
                continue
            torch.manual_seed(0)
            m = TemporalCrosscoder(a.d, a.d_sae, T, kp,
                                   activation=a.activation).to(dev)
            with torch.no_grad():
                m._normalize_decoder()
            m.train()
            opt = torch.optim.Adam(m.parameters(), lr=a.lr)
            for s in range(a.steps):
                xb = Wtr[torch.randint(0, Wtr.shape[0], (a.batch,), device=dev)]
                loss, _, _ = m(xb)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step(); m._normalize_decoder()

            m.eval()
            with torch.no_grad():
                z = m.encode(Who)
                l0 = float((z != 0).float().sum(-1).mean())
                alive = float(((z != 0).float().mean(0) >= 0.001).float().mean())
                xh = m.decode(z)
                fvu = float(((xh - Who) ** 2).sum(-1).mean()
                            / Who.reshape(-1, a.d).var(0).sum())
                A = m.W_dec.data.reshape(a.d_sae, -1)
                A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-8)
                rec = float(np.mean([float((S @ A.T).abs().max()) for S in S_all]))

            rows.append({"T": T, "kper": kp, "nominal_k_window": m.k,
                         "coeff_per_segment": l0 / T, "alive_frac": alive,
                         "fvu": fvu, "recovery": rec})
            print(f"{T:>4}{kp:>6}{m.k:>11}{l0/T:>11.2f}{alive:>8.3f}{fvu:>9.4f}"
                  f"{rec:>10.3f}", flush=True)

    base = max((r["recovery"] for r in rows if r["T"] == Ts[0]), default=0.0)
    print(f"\n===== B1/B2: can ANY kper rescue each T?  (best at T={Ts[0]} is "
          f"{base:.3f}) =====", flush=True)
    for T in Ts:
        sub = [r for r in rows if r["T"] == T]
        if not sub:
            continue
        best = max(sub, key=lambda r: r["recovery"])
        gap = best["recovery"] - base
        verdict = "RESCUED" if best["recovery"] >= 0.95 * base else "not rescued"
        print(f"  T={T:<3} best recovery {best['recovery']:.3f} at kper={best['kper']:<3}"
              f"  ({gap:+.3f} vs T={Ts[0]})  -> {verdict}", flush=True)

    print("\n===== B3: does raising kper help at each T? =====", flush=True)
    for T in Ts:
        sub = sorted([r for r in rows if r["T"] == T], key=lambda r: r["kper"])
        if len(sub) < 2:
            continue
        span = max(r["recovery"] for r in sub) - min(r["recovery"] for r in sub)
        print(f"  T={T:<3} span {span:.3f}  " +
              "  ".join(f"k{r['kper']}:{r['recovery']:.2f}" for r in sub), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "Ts": Ts, "kpers": kpers,
                               "extents": extents, "config": vars(a)}, indent=2))
    print("\n[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
