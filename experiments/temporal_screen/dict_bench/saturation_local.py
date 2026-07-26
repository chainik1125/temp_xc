"""Why does performance degrade with unnecessary T instead of saturating?

The property we want from a window dictionary is: performance rises with T while T is
buying something, and *flattens* once it is not. What the crosscoder actually does is
rise then fall. `recovery_local.py` peaked at T=8 and collapsed at T=16 (FVU 0.116 -> 0.832,
recovery 0.83 -> 0.46) on data whose longest feature had extent 8, and the language-corpus
sweep in `tsweep_modal.py` rose from 1.0x to 6.8x the SAE's FVU on a corpus whose features
are all effectively extent 1. Excess window length is not free; it is actively harmful.

THE PROPOSED CAUSE. This crosscoder's atoms are tied to ABSOLUTE POSITION within the
window: `W_dec` is (d_sae, T, d), so an atom specifies what it contributes at position 0,
at position 1, and so on. A feature occurring at offset 3 and the same feature at offset 7
are therefore two different atoms. Nothing ties them together, so representing a
shift-invariant world costs a factor of T in dictionary entries, and a fixed d_sae is
progressively starved as T grows. Under that account the collapse is not about T at all --
it is about d_sae/T.

THE TEST. Sweep T against d_sae. If the account is right:

  S1  For fixed d_sae, recovery peaks and then falls, and the peak moves to larger T as
      d_sae grows -- the collapse tracks d_sae/T, not T.
  S2  Configurations with equal d_sae/T have similar recovery, so plotting against d_sae/T
      collapses the family onto one curve.
  S3  Learned dictionaries contain near-duplicate atoms related by a time shift, and the
      duplicate fraction grows with T. This is the mechanism made visible: capacity spent
      on storing the same feature at several offsets.
  S4  If instead recovery collapses at the same T for every d_sae, the position-tying
      account is wrong and something else limits large windows.

If S1-S3 hold, two fixes follow and both are worth stating: scale d_sae with T, or remove
the position-tying by making the decoder convolutional in time so one atom covers all
offsets. Only the second gives genuine saturation at constant cost.

Runs locally -- pure synthetic, no language model.
"""
import argparse
import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "dict_bench" / "saturation.json"


def windows(X, T, stride):
    """(n_seq, seq_len, d) -> (n_win, T, d).

    stride=T is the disjoint reshape used originally, which makes the number of training
    windows fall as seq_len/T -- so larger T sees proportionally less data while carrying
    T times more decoder parameters. stride=1 keeps the window count roughly constant in
    T, which is what any comparison across T needs.
    """
    n_seq, seq_len, d = X.shape
    if stride >= T:
        n_win = seq_len // T
        return X[:, :n_win * T, :].reshape(-1, T, d)
    W = X.unfold(1, T, stride)                 # (n_seq, n_win, d, T)
    return W.permute(0, 1, 3, 2).reshape(-1, T, d).contiguous()


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


def shift_redundancy(W_dec, n_sample=192, thresh=0.7):
    """Fraction of atoms that have a near-duplicate at a NON-ZERO time shift.

    Capacity spent storing the same feature at several offsets shows up here directly.
    """
    d_sae, T, d = W_dec.shape
    if T < 2:
        return 0.0
    idx = torch.randperm(d_sae)[:min(n_sample, d_sae)]
    A = W_dec[idx]
    A = A / A.reshape(A.shape[0], -1).norm(dim=1).clamp(min=1e-8).view(-1, 1, 1)
    n = A.shape[0]
    hit = torch.zeros(n, dtype=torch.bool)
    for s in range(1, T):
        # Compare atom i against atom j rolled by s, keeping only the overlapping span so
        # a shift is a genuine translation rather than a wrap-around artefact.
        a = A[:, s:, :].reshape(n, -1)
        b = A[:, :T - s, :].reshape(n, -1)
        a = a / a.norm(dim=1, keepdim=True).clamp(min=1e-8)
        b = b / b.norm(dim=1, keepdim=True).clamp(min=1e-8)
        c = (a @ b.T).abs()
        c.fill_diagonal_(0)
        hit |= (c.max(dim=1).values > thresh).cpu()
    return float(hit.float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=96)
    ap.add_argument("--d_saes", type=str, default="128,256,512,1024")
    ap.add_argument("--Ts", type=str, default="2,4,8,16,32")
    ap.add_argument("--n_per_extent", type=int, default=12)
    ap.add_argument("--extents", type=str, default="1,2,4")
    ap.add_argument("--n_seq", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=96)
    ap.add_argument("--fire_p", type=float, default=0.02)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--k_per", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--activation", type=str, default="batchtopk")
    ap.add_argument("--stride", type=int, default=1)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    d_saes = [int(x) for x in a.d_saes.split(",")]
    Ts = [int(x) for x in a.Ts.split(",")]
    extents = [int(x) for x in a.extents.split(",")]
    dev = pick_device()
    print(f"[device] {dev}  d_sae={d_saes}  T={Ts}  extents={extents} "
          f"(longest feature = {max(extents)})", flush=True)

    gen = torch.Generator().manual_seed(20260726)
    feats = make_features(a.n_per_extent, extents, a.d, gen)
    X = generate(feats, a.n_seq, a.seq_len, a.d, a.fire_p, a.noise, gen)
    print(f"[data] {tuple(X.shape)}  {len(feats)} true features  "
          f"stride={a.stride}", flush=True)
    for T in Ts:
        print(f"   T={T:>3} -> {windows(X, T, a.stride).shape[0]} windows",
              flush=True)

    rows = []
    print(f"\n{'d_sae':>7}{'T':>5}{'d_sae/T':>9}{'coeff/seg':>11}{'FVU':>9}"
          f"{'recovery':>10}{'shift-dup':>11}", flush=True)
    for d_sae in d_saes:
        for T in Ts:
            W_all = windows(X, T, a.stride).to(dev)
            n_hold = max(int(0.15 * W_all.shape[0]), 64)
            Wtr, Who = W_all[:-n_hold], W_all[-n_hold:]

            torch.manual_seed(0)
            m = TemporalCrosscoder(a.d, d_sae, T, a.k_per,
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
                xh = m.decode(z)
                fvu = float(((xh - Who) ** 2).sum(-1).mean()
                            / Who.reshape(-1, a.d).var(0).sum())
                A = m.W_dec.data.reshape(d_sae, -1)
                A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-8)
                rec = []
                for f in feats:
                    S = true_slabs(f, T).reshape(-1, T * a.d).to(dev)
                    rec.append(float((S @ A.T).abs().max()))
                dup = shift_redundancy(m.W_dec.data)

            row = {"d_sae": d_sae, "T": T, "ratio": d_sae / T,
                   "coeff_per_segment": l0 / T, "fvu": fvu,
                   "recovery": float(np.mean(rec)), "shift_dup": dup}
            rows.append(row)
            print(f"{d_sae:>7}{T:>5}{d_sae/T:>9.1f}{l0/T:>11.2f}{fvu:>9.4f}"
                  f"{np.mean(rec):>10.3f}{dup:>11.3f}", flush=True)

    print(f"\n===== S1: for fixed d_sae, where does recovery peak? "
          f"(longest true feature has extent {max(extents)}) =====", flush=True)
    for d_sae in d_saes:
        sub = [r for r in rows if r["d_sae"] == d_sae]
        pk = max(sub, key=lambda r: r["recovery"])
        print(f"  d_sae={d_sae:>5}  peak recovery {pk['recovery']:.3f} at T={pk['T']:<3} "
              + "  ".join(f"T{r['T']}:{r['recovery']:.2f}" for r in sub), flush=True)

    print("\n===== S2: does d_sae/T collapse the family onto one curve? =====", flush=True)
    print(f"  {'d_sae/T':>9}{'recovery':>10}   (configurations sharing a ratio)",
          flush=True)
    by_ratio = {}
    for r in rows:
        by_ratio.setdefault(round(r["ratio"], 1), []).append(r)
    for ratio, group in sorted(by_ratio.items()):
        if len(group) < 2:
            continue
        vals = [g["recovery"] for g in group]
        tag = ", ".join(f"({g['d_sae']},T={g['T']})" for g in group)
        print(f"  {ratio:>9.1f}{np.mean(vals):>10.3f}   spread "
              f"{max(vals) - min(vals):.3f}  {tag}", flush=True)

    print("\n===== S3: does shift-duplication grow with T? =====", flush=True)
    for d_sae in d_saes:
        sub = [r for r in rows if r["d_sae"] == d_sae]
        print(f"  d_sae={d_sae:>5}  " + "  ".join(
            f"T{r['T']}:{r['shift_dup']:.2f}" for r in sub), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "d_saes": d_saes, "Ts": Ts,
                               "extents": extents, "config": vars(a)}, indent=2))
    print("\n[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
