"""Demonstration order: does matching the multiset give rank>=2 AND kill the DC?

Content attribute  : the label at position t          -> schedule dc
State attribute    : the running label balance        -> schedule ds = cumsum(dc)

Matching the label multiset forces sum(dc) = 0 (the ENDPOINTS agree).
Review's claim: that also forces content and state to have different positional
patterns, so A >= 2. TRUE -- cumsum(v) is proportional to v only if v is an
eigenvector of the cumulative-sum matrix L, and L is unipotent lower-triangular
whose only eigenvector is e_T (a single spike at the LAST position).

But does matching the multiset also kill the DC component? NO -- and that is the
part worth knowing, because the state attribute leaves a constant residue:

    sum_t cumsum(dc)(t) = sum_j (T-j+1) dc_j
                        = (T+1) sum_j dc_j  -  sum_j j*dc_j
                        = -sum_j j*dc_j          (since sum dc = 0)

So the DC vanishes iff the FIRST MOMENT of dc vanishes too. Matching the
multiset is the zeroth moment; the state's DC needs the first as well.
"""
import itertools

import numpy as np

T = 8


def schedules(labels_a, labels_b):
    dc = np.array(labels_a, float) - np.array(labels_b, float)
    return dc, np.cumsum(dc)


def dc_share(dc, ds, d=512, rng=None):
    rng = rng or np.random.default_rng(0)
    u_c, u_s = rng.standard_normal((2, d)) / np.sqrt(d)
    P = np.outer(dc, u_c) + 0.5 * np.outer(ds, u_s)
    if np.linalg.norm(P) < 1e-12:
        return None
    s = np.linalg.svd(P, compute_uv=False)
    f = (P ** 2).sum()
    return (T * (P.mean(0) ** 2).sum() / f, s[0] ** 2 / f,
            int((s > 1e-9 * s[0]).sum()))


# The reference ordering must be NON-EXTREMAL. [1,1,1,1,0,0,0,0] uniquely maximises
# the first moment, so no multiset-matched foil can match it and the m1 == 0 cell is
# empty -- the script then prints nan for the very cell it exists to demonstrate.
# Centred and alternating references admit 6-7 valid foils each; this one admits 7.
base = [0, 0, 1, 1, 1, 1, 0, 0]
print("multiset-matched permutations of a balanced 8-shot label sequence")
print(f"reference: {base}  (non-extremal; extremal references admit zero valid foils)")
print(f"{'perm':<26} {'sum dc':>7} {'1st mom':>8} {'c':>7} {'r1':>7} {'rank':>5}")
seen, rows = set(), []
for p in itertools.islice(itertools.permutations(base), 0, 4000):
    if p in seen:
        continue
    seen.add(p)
    dc, ds = schedules(base, p)
    if np.allclose(dc, 0):
        continue
    m1 = float(np.dot(np.arange(1, T + 1), dc))
    out = dc_share(dc, ds)
    if out:
        rows.append((p, dc.sum(), m1, *out))

rows.sort(key=lambda r: abs(r[2]))
for p, s0, m1, c, r1, rk in rows[:4]:
    print(f"{str(p):<26} {s0:>7.1f} {m1:>8.1f} {c:>7.4f} {r1:>7.4f} {rk:>5}")
print("  ...")
for p, s0, m1, c, r1, rk in rows[-3:]:
    print(f"{str(p):<26} {s0:>7.1f} {m1:>8.1f} {c:>7.4f} {r1:>7.4f} {rk:>5}")

zero_m1 = [r for r in rows if abs(r[2]) < 1e-9]
nz_m1 = [r for r in rows if abs(r[2]) > 1e-9]
print(f"\nfirst moment == 0  ({len(zero_m1):>4} perms):  mean c = "
      f"{np.mean([r[3] for r in zero_m1]):.4f}   mean rank = "
      f"{np.mean([r[5] for r in zero_m1]):.2f}")
print(f"first moment != 0  ({len(nz_m1):>4} perms):  mean c = "
      f"{np.mean([r[3] for r in nz_m1]):.4f}   mean rank = "
      f"{np.mean([r[5] for r in nz_m1]):.2f}")
print("\n-> matching the multiset gives rank 2 everywhere, but only the")
print("   first-moment-matched permutations also give c = 0.")

print("\nCONTROL review predicts: all labels identical -> state collapses.")
dc, ds = schedules([1] * T, [1] * T)
print(f"  dc = {dc.astype(int)}  ds = {ds.astype(int)}  -> no attribute at all, "
      f"rank 0 from labels")
