"""Rank of each design's difference slab under context-free block algebra.

Checks two claims I am about to send:
  (1) per-item repetition (D2/D2b) leaves r1 unchanged vs one-block-per-type
  (2) the phase ladder is rank 1 at EVERY switch count, so it cannot reach L3
"""
import numpy as np

rng = np.random.default_rng(0)
d = 256


def screen(P):
    T = P.shape[0]
    fro2 = (P ** 2).sum()
    s = np.linalg.svd(P, compute_uv=False)
    return (T * (P.mean(0) ** 2).sum() / fro2, s[0] ** 2 / fro2,
            s[:2].sum() ** 0 * (s[:2] ** 2).sum() / fro2,
            int((s > 1e-9 * s[0]).sum()))


def simplex(m, d):
    A = rng.standard_normal((m, d))
    Q, _ = np.linalg.qr(A.T)
    B = Q[:, :m].T
    return B - B.mean(0, keepdims=True)


print("(1) per-item repetition vs single blocks, m=3 (D2b) and m=2 (D2)")
for m in (2, 3):
    B = simplex(m, d)
    k_seg = 12
    blk = k_seg // m
    # one block per type: rows b_t - b_{t+1}, each repeated blk times
    P_block = np.stack([B[t // blk] - B[((t // blk) + 1) % m] for t in range(k_seg)])
    # per-item repetition: the m rows cycle k_seg/m times
    P_item = np.stack([B[t % m] - B[(t + 1) % m] for t in range(k_seg)])
    for nm, P in (("block ", P_block), ("per-item", P_item)):
        c, r1, r2, rk = screen(P)
        print(f"  m={m} {nm}: c={c:.2e}  r1={r1:.4f}  r2={r2:.4f}  rank={rk}")

print("\n(2) phase ladder -- two pools alternating, rotation by one block")
a, b = rng.standard_normal((2, d)) / np.sqrt(d)
for n_sw in (1, 3, 5, 11):
    n_blocks = n_sw + 1
    blk = 12 // n_blocks
    vecs = [a if i % 2 == 0 else b for i in range(n_blocks)]
    P = np.stack([vecs[t // blk] - vecs[((t // blk) + 1) % n_blocks]
                  for t in range(12)])
    c, r1, r2, rk = screen(P)
    print(f"  {n_sw:>2} switches (blocks of {blk}): c={c:.2e}  r1={r1:.4f}  rank={rk}")

print("\n(3) D1 grouped ladder for comparison -- rank grows with m")
for m in (2, 3, 4, 6, 12):
    B = simplex(m, d)
    blk = 12 // m
    P = np.stack([B[t // blk] - B[((t // blk) + 1) % m] for t in range(12)])
    c, r1, r2, rk = screen(P)
    print(f"  m={m:>2}: c={c:.2e}  r1={r1:.4f}  rank={rk}")
