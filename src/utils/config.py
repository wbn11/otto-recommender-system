"""Configuration helpers for legacy tasks and reproducible experiments.

The original scripts still call :func:`load_config` and therefore continue to
read ``configs/default.yaml``.  New experiment code uses
:func:`resolve_project_config`, which layers ``base.yaml``, a data profile and
an experiment override without introducing a configuration framework.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PATH = _ROOT / "configs" / "default.yaml"
_BASE_PATH = _ROOT / "configs" / "base.yaml"


def load_config(path=None):
    cfg_path = Path(path) if path else _DEFAULT_PATH
    try:
        import yaml

        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except (FileNotFoundError, ImportError):
        return {}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load one required YAML mapping.

    Unlike the legacy loader, this function fails loudly because silently
    falling back to defaults would make experiment manifests misleading.
    """

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised by environment check
        raise ImportError("PyYAML is required for experiment configuration.") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise TypeError(f"Configuration root must be a mapping: {config_path}")
    return value


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""

    merged = deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def resolve_project_config(
    experiment_path: str | Path | None = None,
    *,
    base_path: str | Path = _BASE_PATH,
) -> dict[str, Any]:
    """Resolve base -> data profile -> experiment configuration layers."""

    resolved = load_yaml(base_path)
    experiment = load_yaml(experiment_path) if experiment_path else {}
    data_mode = experiment.get("data_mode", resolved.get("data_mode", "debug"))
    profile_path = _ROOT / "configs" / "data" / f"{data_mode}.yaml"
    if profile_path.exists():
        resolved = deep_merge(resolved, load_yaml(profile_path))
    elif data_mode != "legacy":
        raise FileNotFoundError(f"Unknown data mode {data_mode!r}: {profile_path}")
    return deep_merge(resolved, experiment)


def config_hash(config: Mapping[str, Any]) -> str:
    """Return a stable short hash for a fully resolved configuration."""

    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def dump_yaml(config: Mapping[str, Any], path: str | Path) -> None:
    """Write a resolved configuration with stable key ordering."""

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise ImportError("PyYAML is required for experiment configuration.") from exc

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(dict(config), handle, sort_keys=True, allow_unicode=True)
