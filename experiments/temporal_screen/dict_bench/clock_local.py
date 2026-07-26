"""What sets the required window length when the feature has no temporal extent?

`recovery_local.py` established, for features with a profile p of finite support L, that a
T-window's recovery is capped at ||largest contiguous T-chunk of p|| / ||p||, and that
per-token dictionaries sit exactly on that ceiling. The rule it suggested -- "set T to the
temporal extent of the behaviour" -- is underspecified, and a clock is the counterexample
that shows why: a periodic feature never terminates, so ||p|| diverges, L does not exist,
and the ceiling formula is not merely loose but undefined.

Yet the required T is obviously finite and obviously depends on the period. So extent is
the wrong primitive. The proposal here is that the right one is RESOLUTION: T must be long
enough to *separate* the features you care about, in whatever structure distinguishes them.
For compactly supported profiles that reduces to extent. For a clock it is frequency
spacing, and it has a textbook value -- distinguishing angular velocities 2pi/P and 2pi/P'
from a T-sample window needs

    T  >~  1 / | 1/P - 1/P' |

THE SETUP. All clocks live in the SAME 2-D plane (u, v) and differ only in angular
velocity. This matters: if each clock had its own plane, a single segment would identify it
by which subspace the point lies in, and T=1 would already discriminate perfectly --
measuring the directions rather than the dynamics. With a shared plane, one segment is a
point on the same circle whatever the period, so the period is recoverable only from the
rotation observed across segments.

TWO METRICS, and the gap between them is the point.

  atom recovery   max cosine between the true T-window waveform and any learned atom.
                  Expected to be high even at small T, because matching a short arc of a
                  sinusoid is easy -- which is exactly why it is the wrong thing to
                  optimise, and why "recovery" alone would have hidden this.

  period ID       linear probe from the window code z to which clock is running. This is
                  the quantity that actually needs resolution, and it should stay near
                  chance until T approaches the pairwise resolution requirement.

REGISTERED PREDICTIONS (written before the run):
  W1  Atom recovery is high (> 0.8) at every T including T=1, so it does NOT identify the
      required window length for a periodic feature.
  W2  Period-ID accuracy is near chance at T=1 and rises with T.
  W3  Pairwise period-ID between P and P' becomes reliable near T ~ 1/|1/P - 1/P'|,
      not near max(P, P'). With P in {2,4,8,16} the hardest pair is (8,16), needing
      T ~ 16, while (2,4) needs only T ~ 4.
  W4  If instead every pair becomes separable at the same T, resolution is the wrong
      story too and the effect is just "more window is better".

Runs locally -- pure synthetic, no language model.
"""
import argparse
import itertools
import json
import pathlib

import numpy as np
import torch

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "dict_bench" / "clock.json"


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolution_requirement(P, Q):
    """Window length needed to separate periods P and Q by frequency resolution."""
    return 1.0 / abs(1.0 / P - 1.0 / Q)


def generate(periods, n_seq, seq_len, d, n_distract, noise, gen):
    """Each sequence runs exactly one clock, in a plane shared by all of them."""
    u = torch.randn(d, generator=gen); u = u / u.norm()
    v = torch.randn(d, generator=gen)
    v = v - (v @ u) * u
    v = v / v.norm()

    # Static distractors so the dictionary has non-clock structure to spend capacity on.
    D = torch.randn(n_distract, d, generator=gen)
    D = D / D.norm(dim=1, keepdim=True)

    which = torch.randint(0, len(periods), (n_seq,), generator=gen)
    phase = 2 * np.pi * torch.rand(n_seq, generator=gen)
    amp = 1.0 + torch.rand(n_seq, generator=gen)
    t = torch.arange(seq_len, dtype=torch.float32)

    omega = torch.tensor([2 * np.pi / P for P in periods])[which]      # (n_seq,)
    ang = omega.unsqueeze(1) * t.unsqueeze(0) + phase.unsqueeze(1)     # (n_seq, seq_len)
    X = (amp.view(-1, 1, 1) * (torch.cos(ang).unsqueeze(-1) * u
                               + torch.sin(ang).unsqueeze(-1) * v))

    fire = (torch.rand(n_seq, seq_len, n_distract, generator=gen) < 0.03).float()
    X = X + fire @ D
    X = X + noise * torch.randn(n_seq, seq_len, d, generator=gen)
    return X, which, (u, v)


def clock_atoms(P, T, u, v, n_phase=64):
    """Every phase of a period-P clock as a T-window waveform, unit-norm, as (n_phase, T*d)."""
    t = torch.arange(T, dtype=torch.float32)
    out = []
    for i in range(n_phase):
        ang = 2 * np.pi / P * t + 2 * np.pi * i / n_phase
        s = torch.cos(ang).unsqueeze(-1) * u + torch.sin(ang).unsqueeze(-1) * v
        out.append((s / s.norm()).reshape(-1))
    return torch.stack(out)


def probe(Z_tr, y_tr, Z_ho, y_ho, n_cls, device, steps=800, lr=1e-2):
    """Multinomial logistic probe; returns held-out accuracy."""
    torch.manual_seed(0)
    W = torch.zeros(Z_tr.shape[1], n_cls, device=device, requires_grad=True)
    b = torch.zeros(n_cls, device=device, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)
    for _ in range(steps):
        logits = Z_tr @ W + b
        loss = torch.nn.functional.cross_entropy(logits, y_tr)
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        return float(((Z_ho @ W + b).argmax(-1) == y_ho).float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--d_sae", type=int, default=512)
    ap.add_argument("--periods", type=str, default="2,4,8,16")
    ap.add_argument("--Ts", type=str, default="1,2,4,8,16,24")
    ap.add_argument("--n_seq", type=int, default=1200)
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--n_distract", type=int, default=32)
    ap.add_argument("--noise", type=float, default=0.05)
    ap.add_argument("--k_per", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--activation", type=str, default="batchtopk")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT))
    from src.bench.architectures.crosscoder import TemporalCrosscoder

    periods = [int(x) for x in a.periods.split(",")]
    Ts = [int(x) for x in a.Ts.split(",")]
    dev = pick_device()
    print(f"[device] {dev}  periods={periods}  T={Ts}", flush=True)

    print("\nresolution requirement T ~ 1/|1/P - 1/P'| for each pair:")
    for P, Q in itertools.combinations(periods, 2):
        print(f"  P={P:<3} vs P'={Q:<3}  ->  T >~ {resolution_requirement(P, Q):.1f}",
              flush=True)
    print("  (contrast: 'set T to the extent' would say T ~ max(P, P') or be undefined)",
          flush=True)

    gen = torch.Generator().manual_seed(20260726)
    X, which, (u, v) = generate(periods, a.n_seq, a.seq_len, a.d, a.n_distract,
                                a.noise, gen)
    print(f"[data] {tuple(X.shape)}  one clock per sequence, shared plane", flush=True)

    rows = []
    for T in Ts:
        n_win = a.seq_len // T
        W = X[:, :n_win * T, :].reshape(a.n_seq, n_win, T, a.d)
        lab = which.unsqueeze(1).expand(-1, n_win).reshape(-1)
        Wf = W.reshape(-1, T, a.d).to(dev)
        lab = lab.to(dev)

        n_hold = max(int(0.2 * Wf.shape[0]), 64)
        Wtr, Who = Wf[:-n_hold], Wf[-n_hold:]
        ytr, yho = lab[:-n_hold], lab[-n_hold:]

        torch.manual_seed(0)
        m = TemporalCrosscoder(a.d, a.d_sae, T, a.k_per,
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
            Ztr, Zho = m.encode(Wtr), m.encode(Who)
            l0 = float((Zho != 0).float().sum(-1).mean())
            xh = m.decode(Zho)
            fvu = float(((xh - Who) ** 2).sum(-1).mean()
                        / Who.reshape(-1, a.d).var(0).sum())
            A = m.W_dec.data.reshape(a.d_sae, -1)
            A = A / A.norm(dim=1, keepdim=True).clamp(min=1e-8)
            rec = {}
            for P in periods:
                S = clock_atoms(P, T, u, v).to(dev)
                rec[str(P)] = float((S @ A.T).abs().max())

        acc = probe(Ztr, ytr, Zho, yho, len(periods), dev)
        pair = {}
        for P, Q in itertools.combinations(periods, 2):
            i, j = periods.index(P), periods.index(Q)
            mtr = (ytr == i) | (ytr == j)
            mho = (yho == i) | (yho == j)
            if int(mho.sum()) < 16:
                continue
            pair[f"{P}v{Q}"] = probe(Ztr[mtr], (ytr[mtr] == j).long(),
                                     Zho[mho], (yho[mho] == j).long(), 2, dev)

        rows.append({"T": T, "coeff_per_segment": l0 / T, "fvu": fvu,
                     "atom_recovery": rec, "period_id_acc": acc, "pairwise": pair})
        print(f"  [T={T:>2}] coeff/seg {l0/T:5.2f}  FVU {fvu:.4f}  "
              f"atom-recovery {min(rec.values()):.3f}-{max(rec.values()):.3f}  "
              f"period-ID {acc:.3f}", flush=True)

    chance = 1.0 / len(periods)
    print(f"\n===== W1/W2: recovery says one thing, identification says another "
          f"(chance = {chance:.2f}) =====", flush=True)
    print(f"  {'T':>4}{'min atom recovery':>20}{'period-ID acc':>16}", flush=True)
    for r in rows:
        print(f"  {r['T']:>4}{min(r['atom_recovery'].values()):>20.3f}"
              f"{r['period_id_acc']:>16.3f}", flush=True)

    print("\n===== W3: does each pair separate near its own resolution requirement? =====",
          flush=True)
    for P, Q in itertools.combinations(periods, 2):
        key = f"{P}v{Q}"
        series = [(r["T"], r["pairwise"].get(key)) for r in rows
                  if r["pairwise"].get(key) is not None]
        if not series:
            continue
        need = resolution_requirement(P, Q)
        hit = next((T for T, v in series if v >= 0.9), None)
        got = "never" if hit is None else f"T={hit}"
        print(f"  {key:<8} needs T >~ {need:5.1f}   reaches 90% at {got:<8} "
              + "  ".join(f"T{T}:{v:.2f}" for T, v in series), flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows, "periods": periods, "Ts": Ts,
                               "config": vars(a)}, indent=2))
    print("\n[saved]", OUT, flush=True)


if __name__ == "__main__":
    main()
