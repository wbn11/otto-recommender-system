"""Lightweight YAML config loader shared by OTTO tasks.

Returns ``{}`` when the config file or PyYAML is unavailable, so every caller
can fall back to its own hard-coded defaults and keep working.
"""
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _ROOT / "configs" / "default.yaml"


def load_config(path=None):
    cfg_path = Path(path) if path else _DEFAULT_PATH
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}
