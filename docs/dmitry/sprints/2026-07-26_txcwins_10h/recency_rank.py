"""Rank of the instruction-recency difference slab.

Geometry: 12 segments, instructions at FIXED positions 2 and 9, filler elsewhere.
Class A: instr1@2, instr2@9.   Class B: instr2@2, instr1@9.

Context-free algebra says only positions 2 and 9 differ -> rows +D, -D -> rank 1.
But the recency EFFECT lives in the filler: after position 2 the governing instruction
differs between classes, and after position 9 it differs the other way. So

  P[2]     = D           (instruction lexical content)     D = i1 - i2
  P[3..8]  = +g          (which instruction is GOVERNING)  g = g(1) - g(2)
  P[9]     = -D
  P[10,11] = -g

P = e_lex (x) D + e_state (x) g, with e_lex . e_state = 0 (DISJOINT SUPPORT).
Two orthogonal rank-1 terms => rank exactly 2 whenever g is not parallel to D,
and the SVD separates them cleanly.
"""
import numpy as np

d = 1536
rng = np.random.default_rng(0)

e_lex = np.zeros(12); e_lex[2] = 1.0; e_lex[9] = -1.0
e_state = np.zeros(12); e_state[3:9] = 1.0; e_state[10:12] = -1.0
print(f"e_lex . e_state = {e_lex @ e_state:.1f}   "
      f"||e_lex||={np.linalg.norm(e_lex):.3f} ||e_state||={np.linalg.norm(e_state):.3f}")

D = rng.standard_normal(d); D /= np.linalg.norm(D)
G = rng.standard_normal(d); G -= (G @ D) * D; G /= np.linalg.norm(G)   # orthogonal to D

print(f"\n{'|g|/|D|':>8} {'c(DoM)':>8} {'r1':>7} {'rank':>5} {'sv1 is':>9}  "
      f"{'c(antisym)':>11}")
for ratio in (0.2, 0.3, 0.5, 0.7, 1.0, 1.5):
    P = np.outer(e_lex, D) + ratio * np.outer(e_state, G)
    s = np.linalg.svd(P, compute_uv=False)
    fro2 = (P ** 2).sum()
    c = 12 * (P.mean(0) ** 2).sum() / fro2
    r1 = s[0] ** 2 / fro2
    rank = int((s > 1e-9 * s[0]).sum())
    # closed forms: sigma_lex^2 = 2, sigma_state^2 = 8*ratio^2
    which = "lex" if 2 > 8 * ratio ** 2 else "state"
    # the probe-mode metric cancels the CLASS-SYMMETRIC part of the write's effect;
    # what survives is the antisymmetric slab. Model it by removing the constant row.
    Pa = P - P.mean(0, keepdims=True)
    ca = 12 * (Pa.mean(0) ** 2).sum() / (Pa ** 2).sum()
    print(f"{ratio:>8.1f} {c:>8.4f} {r1:>7.4f} {rank:>5} {which:>9}  {ca:>11.2e}")

print("\nclosed form: sigma_lex^2 = 2|D|^2, sigma_state^2 = 8|g|^2")
print("  r1 = max(2, 8t^2)/(2 + 8t^2), t = |g|/|D|  -> MINIMUM 0.5 at t = 0.5")
print("  c  = (4/3)t^2 / (2 + 8t^2)                 -> maximum 1/6 as t -> inf")
for t in (0.5, 0.25, 1.0):
    print(f"    t={t}: r1={max(2,8*t**2)/(2+8*t**2):.4f}  c={(4/3)*t**2/(2+8*t**2):.4f}")

print("\nSVD separates the two components (t=0.3): "
      "check singular vectors align with e_lex / e_state")
P = np.outer(e_lex, D) + 0.3 * np.outer(e_state, G)
U, S, Vt = np.linalg.svd(P, full_matrices=False)
for j in (0, 1):
    u = U[:, j]
    print(f"  sv{j+1}: |cos(u, e_lex)|={abs(u @ e_lex/np.linalg.norm(e_lex)):.3f}  "
          f"|cos(u, e_state)|={abs(u @ e_state/np.linalg.norm(e_state)):.3f}")
