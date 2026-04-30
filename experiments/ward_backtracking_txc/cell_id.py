"""Cell-ID convention for Stage B paper-budget runs.

Cell ID format:
    <arch>__<hookpoint>__k<k_per_position>__s<seed>

Examples:
    txc__resid_L10__k32__s42
    tsae__ln1_L10__k16__s42
    stacked_sae__attn_L10__k64__s1

Used by:
    - train_txc / mine_features / b2_cross_model  (--cell flag)
    - evaluate_cell.py                            (single-cell pipeline)
    - hill_climb.py                               (state, perturbations)
    - b1_steer_eval                               (--cells filter)
"""

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Cell:
    arch: str
    hookpoint_key: str          # e.g. "resid_L10"
    k_per_position: int
    seed: int

    @property
    def id(self) -> str:
        return f"{self.arch}__{self.hookpoint_key}__k{self.k_per_position}__s{self.seed}"

    @classmethod
    def from_id(cls, cell_id: str) -> "Cell":
        parts = cell_id.split("__")
        if len(parts) != 4:
            raise ValueError(f"bad cell id: {cell_id!r}")
        arch, hp, k_str, s_str = parts
        if not k_str.startswith("k") or not s_str.startswith("s"):
            raise ValueError(f"bad cell id: {cell_id!r}")
        return cls(arch=arch, hookpoint_key=hp,
                   k_per_position=int(k_str[1:]), seed=int(s_str[1:]))

    def with_(self, **kw) -> "Cell":
        """Return a new Cell with fields overridden (for perturbations)."""
        d = {"arch": self.arch, "hookpoint_key": self.hookpoint_key,
             "k_per_position": self.k_per_position, "seed": self.seed}
        d.update(kw)
        return Cell(**d)


# ─── Filename helpers (one place so everything agrees) ──────────────────────────

def ckpt_path(cell: Cell, ckpt_dir: Path | str) -> Path:
    return Path(ckpt_dir) / f"{cell.id}.pt"


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
    so we can dedupe per-cell when k_per_position varies."""
    return f"{cell.arch}_{cell.hookpoint_key}__k{cell.k_per_position}__s{cell.seed}_f{feature_id}_{mode}"


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
