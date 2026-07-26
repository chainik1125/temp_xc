"""Rank of a difference slab when class B is an arbitrary permutation of class A.

Rows are d_t - d_{sigma(t)} for distinct content vectors d_1..d_k.
Claim: rank = k - (number of cycles of sigma), so a single k-cycle (the rotation
ladder) gives k-1 and is the MAXIMUM; permutations with many cycles give less.
Also: what happens when the content is RESAMPLED per document, so the DoM averages.
"""
import numpy as np

rng = np.random.default_rng(0)
d = 512


def cycles(sigma):
    seen, c = set(), 0
    for i in range(len(sigma)):
        if i not in seen:
            c += 1
            j = i
            while j not in seen:
                seen.add(j); j = sigma[j]
    return c


def rank_of(sigma, D):
    P = np.stack([D[t] - D[sigma[t]] for t in range(len(sigma))])
    s = np.linalg.svd(P, compute_uv=False)
    r = int((s > 1e-9 * s[0]).sum())
    return r, s[0] ** 2 / (s ** 2).sum()


print("k   sigma                       cycles  rank  k-cycles  r1")
for k in (4, 6, 8):
    D = rng.standard_normal((k, d)) / np.sqrt(d)
    sigmas = {
        "rotation (single k-cycle)": [(i + 1) % k for i in range(k)],
        "all transpositions": [i ^ 1 for i in range(k)],
        "reversal": [k - 1 - i for i in range(k)],
    }
    for nm, sig in sigmas.items():
        c = cycles(sig)
        r, r1 = rank_of(sig, D)
        ok = "OK" if r == k - c else "MISMATCH"
        print(f"{k:<3} {nm:<27} {c:>5}  {r:>4}  {k-c:>8}  {r1:.3f}  {ok}")

print("\nSHARED-WRITE CONSTRAINT: what if the content is resampled per document?")
print("A fixed write can only capture what is COMMON across documents, so the")
print("object that matters is the MEAN difference slab over documents.\n")
k, n_docs = 6, 400
sig = [(i + 1) % k for i in range(k)]
for label, resample in (("content FIXED across documents", False),
                        ("content RESAMPLED per document", True)):
    Ds = [rng.standard_normal((k, d)) / np.sqrt(d)] * n_docs if not resample else \
         [rng.standard_normal((k, d)) / np.sqrt(d) for _ in range(n_docs)]
    Ps = [np.stack([D[t] - D[sig[t]] for t in range(k)]) for D in Ds]
    mean_P = np.mean(Ps, axis=0)
    per_doc = np.mean([np.linalg.norm(P) for P in Ps])
    print(f"  {label:<34} ||mean P||={np.linalg.norm(mean_P):8.4f}   "
          f"mean ||P||={per_doc:7.4f}   ratio={np.linalg.norm(mean_P)/per_doc:.4f}")
