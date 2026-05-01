"""Cell-ID convention for Stage B paper-budget runs.

Cell ID format:
    <arch>__<hookpoint>__k<k_per_position>__s<seed>[__r<rank>]

The trailing `__r<rank>` is OPTIONAL. When omitted (legacy IDs from the
sprint and original sweep) the rank defaults to "meandiff" — the original
mining ranking criterion. Other valid ranks: "tstat", "ratio".

Examples:
    txc__resid_L10__k32__s42                    (rank=meandiff, legacy)
    txc__resid_L10__k32__s42__rtstat            (rank=tstat)
    tsae__ln1_L10__k16__s42__rratio             (rank=ratio)
    stacked_sae__attn_L10__k64__s1              (rank=meandiff)

The TRAINING checkpoint is shared across all rank values for the same
(arch, hp, k, seed) tuple — different rankings re-mine the same encoder.
So `ckpt_path` uses `cell.training_id` (rank stripped); per-rank artifacts
(features, B1, metric, sonnet grades) use `cell.id`.

Used by:
    - train_txc / mine_features / b2_cross_model  (--cell flag)
    - evaluate_cell.py                            (single-cell pipeline)
    - hill_climb.py                               (state, perturbations)
    - b1_steer_eval                               (--cells filter)
    - grade_sonnet.py                             (per-cell grading)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

VALID_RANKS = ("meandiff", "tstat", "ratio")


@dataclass(frozen=True)
class Cell:
    arch: str
    hookpoint_key: str          # e.g. "resid_L10"
    k_per_position: int
    seed: int
    rank: str = "meandiff"      # mining ranking criterion

    @property
    def training_id(self) -> str:
        """ID used for the training checkpoint (rank-independent)."""
        return f"{self.arch}__{self.hookpoint_key}__k{self.k_per_position}__s{self.seed}"

    @property
    def id(self) -> str:
        base = self.training_id
        return base if self.rank == "meandiff" else f"{base}__r{self.rank}"

    @classmethod
    def from_id(cls, cell_id: str) -> "Cell":
        parts = cell_id.split("__")
        # Accept 4 parts (legacy) or 5 (with rank)
        if len(parts) == 4:
            arch, hp, k_str, s_str = parts
            rank = "meandiff"
        elif len(parts) == 5:
            arch, hp, k_str, s_str, r_str = parts
            if not r_str.startswith("r"):
                raise ValueError(f"bad cell id (5th part must be r<rank>): {cell_id!r}")
            rank = r_str[1:]
            if rank not in VALID_RANKS:
                raise ValueError(f"bad cell id (rank {rank!r} not in {VALID_RANKS}): {cell_id!r}")
        else:
            raise ValueError(f"bad cell id: {cell_id!r}")
        if not k_str.startswith("k") or not s_str.startswith("s"):
            raise ValueError(f"bad cell id: {cell_id!r}")
        return cls(arch=arch, hookpoint_key=hp,
                   k_per_position=int(k_str[1:]), seed=int(s_str[1:]),
                   rank=rank)

    def with_(self, **kw) -> "Cell":
        """Return a new Cell with fields overridden (for perturbations)."""
        d = {"arch": self.arch, "hookpoint_key": self.hookpoint_key,
             "k_per_position": self.k_per_position, "seed": self.seed,
             "rank": self.rank}
        d.update(kw)
        return Cell(**d)


# ─── Filename helpers (one place so everything agrees) ──────────────────────────

def ckpt_path(cell: Cell, ckpt_dir: Path | str) -> Path:
    # Training is rank-independent — share ckpt across ranks of the same
    # (arch, hookpoint, k, seed). Use training_id, not full id.
    return Path(ckpt_dir) / f"{cell.training_id}.pt"


def sonnet_grades_path(cell: Cell, grades_dir: Path | str) -> Path:
    """Per-cell Sonnet 0-3 grade file. Mirrors b1__<id>.json naming so the
    grader output and the B1 row indices line up trivially."""
    return Path(grades_dir) / f"b1__{cell.id}.json"


def features_path(cell: Cell, features_dir: Path | str) -> Path:
    return Path(features_dir) / f"{cell.id}.npz"


def b2_path(cell: Cell, b2_dir: Path | str) -> Path:
    return Path(b2_dir) / f"{cell.id}.npz"


def b1_per_cell_path(cell: Cell, b1_dir: Path | str) -> Path:
    return Path(b1_dir) / f"b1__{cell.id}.json"


def cell_metric_path(cell: Cell, metrics_dir: Path | str) -> Path:
    return Path(metrics_dir) / f"{cell.id}.json"


def train_log_path(cell: Cell, logs_dir: Path | str) -> Path:
    return Path(logs_dir) / f"{cell.id}__train.jsonl"


# ─── Source-tag convention used in B1 results (and parsed by plotters) ──────────

def source_tag(cell: Cell, feature_id: int, mode: str) -> str:
    """B1 source tag for one (cell, feature, mode). Follows the existing
    `<arch>_<hookpoint>_f<id>_<mode>` pattern but with cell ID baked in
    so we can dedupe per-cell when k_per_position or rank varies. Rank is
    encoded only when non-default to keep legacy tags stable."""
    rank_suffix = "" if cell.rank == "meandiff" else f"__r{cell.rank}"
    return (f"{cell.arch}_{cell.hookpoint_key}"
            f"__k{cell.k_per_position}__s{cell.seed}{rank_suffix}"
            f"_f{feature_id}_{mode}")


def parse_source_tag(tag: str) -> dict | None:
    """Reverse of source_tag. Returns None for DoM-style tags (no cell)."""
    if "_f" not in tag or "__k" not in tag:
        return None
    head, _, ftail = tag.partition("_f")
    fid_str, _, mode = ftail.partition("_")
    # head: <arch>_<hookpoint>__k<k>__s<seed>
    try:
        prefix, k_part, s_part = head.split("__")
        arch_hp = prefix
        # arch may have underscores (topk_sae, stacked_sae)
        # hookpoint is <component>_L<layer> with one underscore
        # → split arch_hp on the LAST two underscores: penultimate is hookpoint component, last is L<layer>
        sub = arch_hp.rsplit("_", 2)
        if len(sub) != 3:
            return None
        arch, hp_comp, hp_layer = sub
        return {
            "arch": arch,
            "hookpoint_key": f"{hp_comp}_{hp_layer}",
            "k_per_position": int(k_part[1:]),
            "seed": int(s_part[1:]),
            "feature_id": int(fid_str),
            "mode": mode,
        }
    except Exception:
        return None
