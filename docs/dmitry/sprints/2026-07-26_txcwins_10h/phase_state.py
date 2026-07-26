"""Why does the phase ladder's r1 RISE toward 1 as alternation gets finer?

Block algebra says the phase ladder is rank 1 at every rung (two pools -> one
content direction with a sign schedule). Measured r1 is 0.921 -> 0.970 as the
block count goes 2 -> 12: close to 1 everywhere, and CLOSER as runs get shorter.

Claim: the residue is the CARRIED STATE -- the second attribute from the
content+state recipe -- and its energy depends on RUN LENGTH. Long runs let an
accumulated state build up; alternating every segment gives it no time. Model the
state as a leaky integral of the content, s(t) = sum_{k<t} lam^(t-k) c(k).

If this is right, the phase ladder is not a failed rank experiment: it is a
MEASUREMENT of how much carried-state energy a run of length L supports.
"""
import numpy as np

rng = np.random.default_rng(0)
d, T = 1536, 12
u_c = rng.standard_normal(d) / np.sqrt(d)          # content direction
u_s = rng.standard_normal(d) / np.sqrt(d)
u_s -= (u_s @ u_c) * u_c / (u_c @ u_c)             # state direction, independent


def phase_slab(n_blocks, lam=0.6, w=0.5):
    """Class A alternates; class B is A rotated by one block."""
    L = T // n_blocks
    cA = np.array([1.0 if (t // L) % 2 == 0 else -1.0 for t in range(T)])
    cB = np.roll(cA, L)

    def leaky(c):
        s, acc = np.zeros(T), 0.0
        for t in range(T):
            acc = lam * acc + c[t]
            s[t] = acc
        return s

    dc, ds = cA - cB, leaky(cA) - leaky(cB)
    return np.outer(dc, u_c) + w * np.outer(ds, u_s)


def screen(P):
    s = np.linalg.svd(P, compute_uv=False)
    f = (P ** 2).sum()
    return s[0] ** 2 / f, (s[1] ** 2 / f if len(s) > 1 else 0.0)


print("measured (implement, real Qwen L14):  m=2 r1=0.921 | m=4 0.934 | "
      "m=6 0.945 | m=12 0.970")
print("\nleaky-integrator model, r1 by block count:")
print(f"{'lam':>5} {'w':>5} " + "".join(f"{f'm={m}':>9}" for m in (2, 4, 6, 12))
      + "   monotone rising?")
for lam in (0.4, 0.6, 0.8):
    for w in (0.35, 0.5):
        r1s = [screen(phase_slab(m, lam, w))[0] for m in (2, 4, 6, 12)]
        mono = all(r1s[i] < r1s[i + 1] for i in range(3))
        print(f"{lam:>5} {w:>5} " + "".join(f"{v:>9.3f}" for v in r1s)
              + f"   {'YES' if mono else 'no'}")

print("\nState energy share (1 - r1) against run length L = 12/m:")
for m in (2, 4, 6, 12):
    r1, r2 = screen(phase_slab(m, 0.6, 0.5))
    print(f"  m={m:>2}  run length {T//m:>2}   1-r1 = {1-r1:.3f}   "
          f"sigma_2^2 share = {r2:.3f}")
print("\n-> the second attribute loses energy as runs shorten, because an")
print("   accumulated state has no time to accumulate. Long runs are the lever.")
