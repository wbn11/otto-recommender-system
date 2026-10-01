"""Validate the runtime required by the upgraded OTTO pipeline."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REQUIRED_MODULES = (
    "numpy",
    "pandas",
    "polars",
    "pyarrow",
    "yaml",
    "duckdb",
    "lightgbm",
    "psutil",
)
GPU_MODULES = ("torch", "faiss")


def _nvidia_smi() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return {"available": False, "output": None}
    try:
        result = subprocess.run(
            [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"available": False, "output": str(exc)}
    return {"available": True, "output": result.stdout.strip()}


def collect_environment(require_gpu: bool = False) -> tuple[dict[str, Any], list[str]]:
    failures: list[str] = []
    modules: dict[str, Any] = {}
    for module_name in (*REQUIRED_MODULES, *GPU_MODULES):
        try:
            module = importlib.import_module(module_name)
            modules[module_name] = {
                "available": True,
                "version": getattr(module, "__version__", None),
            }
        except (ImportError, OSError) as exc:
            modules[module_name] = {"available": False, "error": str(exc)}
            if module_name in REQUIRED_MODULES or require_gpu:
                failures.append(f"missing module: {module_name}")

    torch_info: dict[str, Any] = {}
    if modules["torch"]["available"]:
        import torch

        torch_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "cudnn_version": torch.backends.cudnn.version(),
            "device_count": torch.cuda.device_count(),
            "devices": [
                {
                    "name": torch.cuda.get_device_name(index),
                    "memory_bytes": torch.cuda.get_device_properties(index).total_memory,
                    "capability": list(torch.cuda.get_device_capability(index)),
                }
                for index in range(torch.cuda.device_count())
            ],
        }
        if require_gpu and not torch.cuda.is_available():
            failures.append("PyTorch cannot access CUDA")

    faiss_info: dict[str, Any] = {}
    if modules["faiss"]["available"]:
        import faiss

        gpu_count = faiss.get_num_gpus() if hasattr(faiss, "get_num_gpus") else 0
        faiss_info = {"gpu_count": gpu_count}
        if require_gpu and gpu_count < 1:
            failures.append("FAISS cannot access a GPU")

    report = {
        "ok": not failures,
        "platform": platform.platform(),
        "python": sys.version,
        "executable": sys.executable,
        "modules": modules,
        "torch": torch_info,
        "faiss": faiss_info,
        "nvidia_smi": _nvidia_smi(),
        "failures": failures,
    }
    return report, failures


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check OTTO Python, CUDA and FAISS dependencies.")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    report, failures = collect_environment(args.require_gpu)
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
