"""Closed form for the rotation ladder.

P has rows b_t - b_{t+1}, i.e. P = C B with C circulant, first row (1,-1,0,...).
Circulant symbol f(w) = 1 - w, eigenvalues 1 - exp(2*pi*i*j/m), so
    sigma_j^2 = 4 sin^2(pi j / m),   j = 0..m-1
    sum_j sigma_j^2 = 2m      (the j=0 mode is exactly 0 -> constant share c = 0)
    max_j sigma_j^2 = 4 sin^2(pi floor(m/2) / m)

=>  r1 = 4 sin^2(pi*floor(m/2)/m) / (2m)
       = 2/m                       for even m
       = 2 cos^2(pi/(2m)) / m      for odd m
Both -> 2/m. NOT 1/(m-1).
"""
import numpy as np

rng = np.random.default_rng(1)
d = 1536


def simplex_blocks(m, d, rng):
    A = rng.standard_normal((m, d))
    Q, _ = np.linalg.qr(A.T)
    B = Q[:, :m].T
    return B - B.mean(0, keepdims=True)


def closed_form(m):
    j = np.arange(m)
    s2 = 4 * np.sin(np.pi * j / m) ** 2
    return s2.max() / s2.sum()


print(f"{'m':>3} {'r1 meas':>9} {'closed':>9} {'2/m':>7} {'1/(m-1)':>9} "
      f"{'sqrt(r1)':>9} {'rank':>5}")
for m in (2, 3, 4, 5, 6, 8, 12):
    B = simplex_blocks(m, d, rng)
    P = np.stack([B[t] - B[(t + 1) % m] for t in range(m)])
    s = np.linalg.svd(P, compute_uv=False)
    r1 = s[0] ** 2 / (s ** 2).sum()
    c = m * np.linalg.norm(P.mean(0)) ** 2 / np.linalg.norm(P) ** 2
    rank = int((s > 1e-9 * s[0]).sum())
    assert c < 1e-20, f"constant share not zero at m={m}: {c}"
    print(f"{m:>3} {r1:>9.4f} {closed_form(m):>9.4f} {2/m:>7.4f} "
          f"{1/(m-1):>9.4f} {np.sqrt(r1):>9.4f} {rank:>5}")

print("\nDegeneracy: sigma^2_j = 4 sin^2(pi j/m) pairs up j <-> m-j.")
for m in (3, 4, 6, 12):
    j = np.arange(m)
    s2 = np.sort(4 * np.sin(np.pi * j / m) ** 2)[::-1]
    print(f"  m={m:>2}  sigma^2 sorted: {np.round(s2, 3)}   "
          f"top-2 share {(s2[:2].sum()/s2.sum()):.3f}")

print("\nUsable ladder at k_seg=12 (m must divide 12), distinct r1:")
for m in (2, 3, 6, 12):
    print(f"  m={m:>2}  block_len={12//m}  r1={closed_form(m):.3f}  "
          f"sqrt(r1)={np.sqrt(closed_form(m)):.3f}")
