"""recency_var: instruction positions drawn PER DOCUMENT. What survives in a fixed write?

A dictionary latent is ONE write reused across documents, so what any fixed-write arm
(dom_slab included) can achieve is set by the MEAN difference slab. When the positions
vary, the per-document slab keeps its shape but SLIDES, so the mean is a SMEARED version
of it -- not zero, but blunt: it writes the right thing at reduced strength in the right
place and wrong things everywhere else.

Retention = ||mean_docs P|| / mean_docs ||P||.  This bounds dom_slab, hence every arm.
"""
import numpy as np

rng = np.random.default_rng(0)
d, T = 1536, 12
D = rng.standard_normal(d) / np.sqrt(d)      # instruction lexical contrast
G = rng.standard_normal(d) / np.sqrt(d)      # governing-state contrast
G -= (G @ D) * D / (D @ D)


def slab(p, q, t=0.4):
    """Per-document difference slab for instructions at positions p < q."""
    P = np.zeros((T, d))
    P[p] += D
    P[q] -= D
    P[p + 1:q] += t * G          # instr at p governs the span up to q
    P[q + 1:] -= t * G           # instr at q governs the tail
    return P


def retention(pairs, t=0.4):
    Ps = [slab(p, q, t) for p, q in pairs]
    mean_P = np.mean(Ps, axis=0)
    return (np.linalg.norm(mean_P) / np.mean([np.linalg.norm(P) for P in Ps]),
            mean_P)


print("FIXED positions (the recency task as run): p=2, q=9")
r, _ = retention([(2, 9)])
print(f"  retention = {r:.3f}   (identically 1 -- every document needs the same write)\n")

print("VARIABLE positions (recency_var). Retention bounds dom_slab and every fixed arm.")
print(f"{'position set':<38} {'n pairs':>8} {'retention':>10} {'1/sqrt(n_p)':>12}")
sets = {
    "p in 1-4, q in 7-10": [(p, q) for p in range(1, 5) for q in range(7, 11)],
    "p in 1-5, q in 6-11": [(p, q) for p in range(1, 6) for q in range(6, 12)],
    "all pairs p<q, gap>=3": [(p, q) for p in range(T) for q in range(p + 3, T)],
    "all pairs p<q": [(p, q) for p in range(T) for q in range(p + 1, T)],
}
for nm, pairs in sets.items():
    r, _ = retention(pairs)
    n_p = len(set(p for p, _ in pairs))
    print(f"{nm:<38} {len(pairs):>8} {r:>10.3f} {1/np.sqrt(n_p):>12.3f}")

print("\nSensitivity to the state/lexical ratio t (all pairs, gap>=3):")
pairs = sets["all pairs p<q, gap>=3"]
for t in (0.0, 0.2, 0.4, 0.7, 1.0):
    r, _ = retention(pairs, t)
    print(f"  t={t:>4}: retention = {r:.3f}")

print("\nWhat the smeared mean write looks like (all pairs, gap>=3, t=0.4):")
_, mean_P = retention(pairs)
prof = np.linalg.norm(mean_P, axis=1)
print("  per-position norm profile:",
      " ".join(f"{v:.3f}" for v in prof / prof.max()))
print("  -> a broad ramp, not two spikes: the right direction, the wrong places.")
