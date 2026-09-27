"""The locked uniform grid design for the synthetic B×A program.

One source of truth for the clean-room re-grid (briefings/full-rerun-and-purge.md
→ superseded, but the design is locked): the fair-backbone arch set, the
per-family dictionary constraint, the ``k_pos`` sweep, and the ``{F//2, F, 2F}``
capacity fractions. Every bench's ``run_grid.py`` enumerates the SAME uniform
cells through :func:`uniform_cells` — only ``F``, ``n_steps``, and the datasource
differ — so the B benchmarks × A architectures grid is natively comparable.

Mirrors the program registry (``registry.ARCHS`` / ``registry.capacities`` /
``registry.OP``): the arch families and the ``{F//2, F, 2F}`` capacities are the
same design, expressed here for the grid *producers* and there for the report
*consumer*. Kept in the src library so the drivers depend only on it, not on the
experiments package.

Dictionary constraint (why a cell can be dropped):
- ``token`` / ``post`` archs budget ``k_pos`` atoms per *token* → ``d_sae ≥ k_pos``.
- ``pre`` / ``stacked`` / ``spectral`` pool ``k_win = k_pos·T`` atoms per *window*
  (spectral splits that budget across DCT bands, sitting at pre density) →
  ``d_sae ≥ k_pos·T``.
Infeasible ``(d_sae, k_pos)`` combinations at large ``T`` are dropped and logged
(never silently), so a clipped grid never reads as full coverage.
"""

from __future__ import annotations

# (arch, family): family sets the dict constraint. Token archs are pinned T=1.
# Every arch shares the BatchTopK→JumpReLU fair backbone, so the only variable
# across the matrix is decode structure. (spectral_txc runs on all four benches
# to fill its matrix column; on the non-frequency benches it is a plain DCT-band
# window arch.)
FAIR_BACKBONE = (
    ("batchtopk_sae", "token"),
    ("tsae", "token"),
    ("stacked_batchtopk", "stacked"),
    ("txc_batchtopk_pre", "pre"),
    ("txc_batchtopk_post", "post"),
    ("spectral_txc", "spectral"),
)
WINDOW_TS = (2, 4, 8)                 # T for window archs (token archs are T=1)
K_POS_SWEEP = (1, 2, 4, 8, 16)        # gives the per-bench k_pos axis + hits B*=2
CAPACITY_FRACS = (0.5, 1.0, 2.0)      # {F//2, F, 2F}: deep-scarce, boundary, over-complete
_POOLED = frozenset({"pre", "stacked", "spectral"})


def capacities(F: int) -> list[int]:
    """The bench's ``d_sae`` sweep ``{F//2, F, 2F}`` (uniform across benches;
    comparability rides on the normalized metric + matched sparsity, not on an
    identical absolute ``d_sae``)."""
    return [max(1, int(F * f)) for f in CAPACITY_FRACS]


def min_d_sae(family: str, k_pos: int, T: int) -> int:
    """Smallest feasible ``d_sae`` for one ``(family, k_pos, T)`` cell."""
    return k_pos * T if family in _POOLED else k_pos


def arch_t_list(archs=FAIR_BACKBONE, window_ts=WINDOW_TS):
    """Expand ``(arch, family)`` → ``[(arch, T, family)]`` (T=1 for token archs)."""
    out = []
    for arch, fam in archs:
        if fam == "token":
            out.append((arch, 1, fam))
        else:
            out.extend((arch, T, fam) for T in window_ts)
    return out


def uniform_cells(ds: str, F: int, n_steps: int, *, seeds=(1, 2, 42),
                  k_pos_sweep=K_POS_SWEEP, archs=FAIR_BACKBONE,
                  window_ts=WINDOW_TS, L: int = 32, d_saes=None,
                  untrained: bool = True, untrained_kpos: int = 1, log=None):
    """The locked uniform grid for one datasource.

    - **trained**: every ``(arch, T)`` × ``d_sae ∈ {F//2, F, 2F}`` × dict-feasible
      ``k_pos`` × ``seed``.
    - **untrained control** (``n_steps=0``, ``untrained``): one per ``(arch, T)``
      at the ``F`` anchor, ``k_pos=untrained_kpos``, per seed.

    ``d_saes`` overrides the capacity sweep (e.g. a single anchor / a memo value).
    ``log(msg)`` (default: none) reports how many ``(d_sae, k_pos)`` combinations
    were dropped as dict-infeasible. Returns a list of pickleable cell dicts for
    :func:`explorations.synthetic.grid.run_pool`.
    """
    caps = list(d_saes) if d_saes is not None else capacities(F)
    at = arch_t_list(archs, window_ts)

    def cell(arch, T, d, k, seed, n, kind):
        return {"ds": ds, "arch": arch, "T": T, "d_sae": d, "k_pos": k,
                "seed": seed, "n_steps": n, "kind": kind, "eval_window_L": L}

    cells, dropped = [], 0
    for seed in seeds:
        for arch, T, fam in at:
            for d in caps:
                for k in k_pos_sweep:
                    if d < min_d_sae(fam, k, T):
                        dropped += 1
                        continue
                    cells.append(cell(arch, T, d, k, seed, n_steps, "trained"))
            if untrained:
                anchor = F if F in caps or d_saes is None else caps[len(caps) // 2]
                cells.append(cell(arch, T, anchor, untrained_kpos, seed, 0, "untrained"))
    if log and dropped:
        log(f"[design] {ds}: dropped {dropped} dict-infeasible (arch,T,d,k) cells "
            f"(pre/stacked/spectral need d_sae≥k_pos·T; token/post need d_sae≥k_pos)")
    return cells
