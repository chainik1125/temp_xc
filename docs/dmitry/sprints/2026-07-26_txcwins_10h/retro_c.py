"""Would the c screen have killed the two failed language steering demos?

PASSPHRASE. k slots, distinct code-words. Target = valid, foil = ONE word corrupted
at a uniformly random slot. So at each position the foil holds the correct word with
prob (k-1)/k and a wrong word with prob 1/k:

    P[t] = v_t - [ (k-1)/k v_t + (1/k) wbar ] = (1/k)(v_t - wbar)

The multiset is NOT matched -- k-1 of k words agree, which LOOKS nearly matched to an
eyeball check, and that is exactly the case a binary "is the multiset matched?" test
gets wrong and a graded c catches.
"""
import numpy as np

rng = np.random.default_rng(0)
d = 1536


def screen(P):
    T = P.shape[0]
    fro2 = (P ** 2).sum()
    s = np.linalg.svd(P, compute_uv=False)
    return T * (P.mean(0) ** 2).sum() / fro2, s[0] ** 2 / fro2


print("PASSPHRASE -- lexical component only (no validity state yet)")
print(f"{'k':>3} {'c':>8} {'1/k':>8} {'r1':>8}   verdict")
for k in (2, 3, 4, 6, 8, 12):
    cs, r1s = [], []
    for _ in range(200):
        V = rng.standard_normal((k, d)) / np.sqrt(d)      # correct words
        wbar = rng.standard_normal(d) / np.sqrt(d) / np.sqrt(k)  # mean wrong word
        P = (V - wbar) / k
        c, r1 = screen(P)
        cs.append(c); r1s.append(r1)
    c, r1 = np.mean(cs), np.mean(r1s)
    verdict = "DISCARD (c > 0.3)" if c > 0.3 else ("marginal" if c > 0.1 else "would pass")
    print(f"{k:>3} {c:>8.3f} {1/k:>8.3f} {r1:>8.3f}   {verdict}")

print("\nPASSPHRASE -- adding the VALIDITY STATE, a pure DC component.")
print("The steering target IS 'authenticated', a scalar the model computes and writes")
print("at every position. Let its norm be a multiple s of the per-slot lexical norm.\n")
print(f"{'k':>3} {'s=0':>7} {'s=0.5':>7} {'s=1':>7} {'s=2':>7}")
for k in (4, 6, 8):
    row = []
    for s in (0.0, 0.5, 1.0, 2.0):
        cs = []
        for _ in range(200):
            V = rng.standard_normal((k, d)) / np.sqrt(d)
            u_dc = rng.standard_normal(d) / np.sqrt(d)
            P = V / k + s * np.tile(u_dc / k, (k, 1))
            cs.append(screen(P)[0])
        row.append(np.mean(cs))
    print(f"{k:>3} " + " ".join(f"{v:>7.3f}" for v in row))

print("\nORDERED GENERATION -- 'mode-dominated' and 'large c' are the same statement.")
print("A mode is a state present at EVERY position, i.e. a constant write, i.e. c -> 1.")
for s in (0.5, 1.0, 2.0, 4.0):
    cs = []
    for _ in range(200):
        k = 6
        V = rng.standard_normal((k, d)) / np.sqrt(d)     # per-slot content
        u_dc = rng.standard_normal(d) / np.sqrt(d)        # the listing/counting mode
        P = V + s * np.tile(u_dc, (k, 1))
        cs.append(screen(P)[0])
    print(f"  mode strength s={s:>4}:  c = {np.mean(cs):.3f}")

print("\nTRAJECTORY TASKS (the four that WON) -- multiset-matched permutation foils.")
print("Profile p in {+1,-1}^k along ONE attribute direction, foil = a permutation.")
for k in (2, 4, 6, 8, 10):
    cs, r1s = [], []
    for _ in range(200):
        u = rng.standard_normal(d) / np.sqrt(d)
        p = rng.permutation([1] * (k // 2) + [-1] * (k - k // 2)).astype(float)
        q = rng.permutation(p)
        P = np.outer(p - q, u)
        if np.linalg.norm(P) < 1e-9:
            continue
        c, r1 = screen(P)
        cs.append(c); r1s.append(r1)
    print(f"  k={k:>2}:  c = {np.mean(cs):.4f}   r1 = {np.mean(r1s):.4f}  "
          f"<- rank 1: ONE direction, sign schedule")
