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
