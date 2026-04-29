"""Shared plot helpers."""

from pathlib import Path
import yaml


def load_cfg(config_arg=None):
    p = Path(config_arg) if config_arg else Path(__file__).resolve().parents[1] / "config.yaml"
    return yaml.safe_load(p.read_text())


def plots_dir(cfg) -> Path:
    p = Path(cfg["paths"]["plots_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def features_npz_path(cfg, arch: str, hookpoint_key: str) -> Path:
    """Locate the features npz for an (arch, hookpoint) cell. TXC keeps the
    legacy filename `<key>.npz`; other archs use `<arch>_<key>.npz`."""
    feat_dir = Path(cfg["paths"]["features_dir"])
    if arch == "txc":
        return feat_dir / f"{hookpoint_key}.npz"
    return feat_dir / f"{arch}_{hookpoint_key}.npz"


def b2_npz_path(cfg, arch: str, hookpoint_key: str) -> Path:
    b2_dir = Path(cfg["paths"]["b2_dir"])
    if arch == "txc":
        return b2_dir / f"{hookpoint_key}.npz"
    return b2_dir / f"{arch}_{hookpoint_key}.npz"


def iter_arch_hookpoint(cfg):
    """Yield (arch, hookpoint_dict) for every enabled cell in cfg."""
    arch_list = cfg["txc"].get("arch_list", ["txc"])
    for arch in arch_list:
        for hp in cfg["hookpoints"]:
            if hp.get("enabled", True):
                yield arch, hp
